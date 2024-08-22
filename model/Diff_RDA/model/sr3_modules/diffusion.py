import math
import torch
from torch import device, nn, einsum
import torch.nn.functional as F
from inspect import isfunction
from functools import partial
import numpy as np
from tqdm import tqdm


def _warmup_beta(linear_start, linear_end, n_timestep, warmup_frac):
    betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    warmup_time = int(n_timestep * warmup_frac)
    betas[:warmup_time] = np.linspace(
        linear_start, linear_end, warmup_time, dtype=np.float64)
    return betas


def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == 'quad':
        betas = np.linspace(linear_start ** 0.5, linear_end ** 0.5,
                            n_timestep, dtype=np.float64) ** 2
    elif schedule == 'linear':
        betas = np.linspace(linear_start, linear_end,
                            n_timestep, dtype=np.float64)
    elif schedule == 'warmup10':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.1)
    elif schedule == 'warmup50':
        betas = _warmup_beta(linear_start, linear_end,
                             n_timestep, 0.5)
    elif schedule == 'const':
        betas = linear_end * np.ones(n_timestep, dtype=np.float64)
    elif schedule == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1. / np.linspace(n_timestep,
                                 1, n_timestep, dtype=np.float64)
    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) /
            n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise NotImplementedError(schedule)
    return betas


# gaussian diffusion trainer class

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        Unet_fn,
        image_size,
        channels=3,
        loss_type='l1',
        conditional=True,
        schedule_opt=None
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.denoise_fn = denoise_fn
        self.Unet_fn = Unet_fn
        self.loss_type = loss_type
        self.conditional = conditional
        self.ddim_timesteps = 5
        self.ddim_eta = 1.0
        if schedule_opt is not None:
            pass
            # self.set_new_noise_schedule(schedule_opt)

    def set_loss(self, device):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='sum').to(device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='sum').to(device)
        else:
            raise NotImplementedError()

    def set_new_noise_schedule(self, schedule_opt, device):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)

        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['n_timestep'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        betas = betas.detach().cpu().numpy() if isinstance(
            betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(
            np.append(1., alphas_cumprod))

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev',
                             to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod',
                             to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod',
                             to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod',
                             to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * \
            (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance',
                             to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(
            np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def predict_start_from_noise(self, x_t, t, noise):
        return self.sqrt_recip_alphas_cumprod[t] * x_t - \
            self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * \
            x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, condition_x=None):
        batch_size = x.shape[0]
        noise_level = torch.FloatTensor(
            [self.sqrt_alphas_cumprod_prev[t+1]]).repeat(batch_size, 1).to(x.device)
        if condition_x is not None:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(torch.cat([condition_x, x], dim=1), noise_level))
        else:
            x_recon = self.predict_start_from_noise(
                x, t=t, noise=self.denoise_fn(x, noise_level))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        x_denoise_recon = x_recon

        x_recon = self.Unet_fn(torch.cat([condition_x, x], dim=1), noise_level)
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance, x_recon, x_denoise_recon

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, condition_x=None):
        model_mean, model_log_variance, x_0, x_denoise_recon = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x)
        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp(), x_0, x_denoise_recon

    @torch.no_grad()
    def p_sample_loop(self, x_in, continous=False):
        device = self.betas.device
        sample_inter = (1 | (self.num_timesteps//10))
        if not self.conditional:
            shape = x_in
            img = torch.randn(shape, device=device)
            ret_img = img
            ret_x_0 = img
            ret_x_0_denoise = img
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img, x_0, x_0_denoise = self.p_sample(img, i)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
                    ret_x_0 = torch.cat([ret_x_0, x_0], dim=0)
                    ret_x_0_denoise = torch.cat([ret_x_0_denoise, x_0_denoise], dim=0)
        else:
            x = x_in
            shape = x.shape
            img = torch.randn(shape, device=device)
            ret_img = x
            ret_x_0 = x
            ret_x_0_denoise = x
            for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
                img, x_0, x_0_denoise = self.p_sample(img, i, condition_x=x)
                if i % sample_inter == 0:
                    ret_img = torch.cat([ret_img, img], dim=0)
                    ret_x_0 = torch.cat([ret_x_0, x_0], dim=0)
                    ret_x_0_denoise = torch.cat([ret_x_0_denoise, x_0_denoise], dim=0)
        if continous:
            return ret_img, ret_x_0, ret_x_0_denoise
        else:
            return ret_img[-1], ret_x_0, ret_x_0_denoise

    @torch.no_grad()
    def super_resolution(self, x_in, continous=False):
        return self.p_sample_loop(x_in, continous)

    @torch.no_grad()
    def super_resolution_ddim(self, x_in, continous=False):
        return self.p_sample_loop_ddim(x_in, continous)

    @torch.no_grad()
    def p_sample_loop_ddim(self, x_in, continous=False):
        device = self.betas.device
        ddim_timesteps = self.ddim_timesteps
        interval = self.num_timesteps // ddim_timesteps
        timestep_seq = np.arange(self.num_timesteps - 1, -1, -interval)[::-1]
        prev_timestep_seq = np.append(np.array([-1]), timestep_seq[:-1])

        sample_interval = (1 | (ddim_timesteps // 10))
        x = x_in
        shape = x.shape
        b, c, h, w = shape

        img = torch.randn(shape, device=device)
        ret_img = x
        x0_img = x
        denoise_x0_img = x

        for i in tqdm(reversed(range(ddim_timesteps)), desc='Sampling loop Diff-RDA time step', total=ddim_timesteps):
            t = torch.tensor([timestep_seq[i]] * b, dtype=torch.long, device=device)
            prev_t = torch.tensor([prev_timestep_seq[i]] * b, dtype=torch.long, device=device)

            alpha_cumprod_t = self.extract_from_tensor(self.alphas_cumprod, t, img.shape)
            alpha_cumprod_t_prev = torch.ones_like(alpha_cumprod_t) if i == 0 else self.extract_from_tensor(self.alphas_cumprod,
                                                                                                 prev_t, img.shape)
            noise_level = torch.full((b, 1), self.sqrt_alphas_cumprod_prev[t + 1], device=x.device, dtype=torch.float32)

            pred_noise = self.denoise_fn(torch.cat([x_in, img], dim=1), noise_level)
            x_recon = self.predict_start_from_noise(img, t=t, noise=pred_noise)
            x_recon.clamp_(-1., 1.)
            denoise_x0 = x_recon

            pred_x0 = self.Unet_fn(torch.cat([x_in, x_recon], dim=1), noise_level)
            pred_x0.clamp_(-1., 1.)

            sigmas_t = self.ddim_eta * torch.sqrt(
                (1 - alpha_cumprod_t_prev) / (1 - alpha_cumprod_t) * (1 - alpha_cumprod_t / alpha_cumprod_t_prev))

            x_t_prev = torch.sqrt(alpha_cumprod_t_prev) * pred_x0 + torch.sqrt(1 - alpha_cumprod_t_prev - sigmas_t ** 2) * pred_noise \
                       + sigmas_t * torch.randn_like(pred_x0)
            img = x_t_prev

            if i % sample_interval == 0:
                ret_img = torch.cat([ret_img, img], dim=0)
                x0_img = torch.cat([x0_img, x_recon], dim=0)
                denoise_x0_img = torch.cat([denoise_x0_img, denoise_x0], dim=0)

        return (ret_img, x0_img, denoise_x0_img) if continous else (ret_img[-1], x0_img, denoise_x0_img)

    def extract_from_tensor(self, alpha, t, x_shape):
        b = t.shape[0]
        out = torch.index_select(alpha.to(t.device), 0, t).float()
        out = out.view(b, *([1] * (len(x_shape) - 1)))
        return out

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            continuous_sqrt_alpha_cumprod * x_start +
            (1 - continuous_sqrt_alpha_cumprod**2).sqrt() * noise
        )

    def p_losses(self, x_in, noise=None):
        x_start = x_in['high']
        [b, c, h, w] = x_start.shape
        t = np.random.randint(1, self.num_timesteps + 1)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.random.uniform(
                self.sqrt_alphas_cumprod_prev[t-1],
                self.sqrt_alphas_cumprod_prev[t],
                size=b
            )
        ).to(x_start.device)
        continuous_sqrt_alpha_cumprod = continuous_sqrt_alpha_cumprod.view(
            b, -1)

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(
            x_start=x_start, continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1), noise=noise)

        if not self.conditional:
            pred_noise = self.denoise_fn(x_noisy, continuous_sqrt_alpha_cumprod)
        else:
            pred_noise = self.denoise_fn(torch.cat([x_in['low'], x_noisy], dim=1), continuous_sqrt_alpha_cumprod)

        x_0_from_denoise = self.predict_start_from_noise(x_noisy, t=t-1, noise=pred_noise)
        x_0_from_denoise.clamp_(-1., 1.)

        #x_0 = self.Unet_fn(torch.cat([x_in['low'], x_0_from_denoise], dim=1), continuous_sqrt_alpha_cumprod)
        x_0 = self.Unet_fn(torch.cat([x_in['low'], x_0_from_denoise.detach()], dim=1), continuous_sqrt_alpha_cumprod)
        x_0.clamp_(-1., 1.)

        loss_x0 = self.loss_func(x_0, x_start)
        loss_eps = self.loss_func(noise, pred_noise)

        loss = loss_eps + loss_x0
        return loss, loss_eps, loss_x0

    def forward(self, x, *args, **kwargs):
        return self.p_losses(x, *args, **kwargs)
