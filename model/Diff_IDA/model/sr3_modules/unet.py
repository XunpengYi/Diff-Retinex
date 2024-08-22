import math
import torch
from torch import nn
import torch.nn.functional as F
from inspect import isfunction


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

# PositionalEncoding Source： https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels, use_affine_level=False):
        super(FeatureWiseAffine, self).__init__()
        self.use_affine_level = use_affine_level
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels*(1+self.use_affine_level))
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        if self.use_affine_level:
            gamma, beta = self.noise_func(noise_embed).view(batch, -1, 1, 1).chunk(2, dim=1)
            x = (1 + gamma) * x + beta
        else:
            x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            Swish(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, noise_level_emb_dim=None, dropout=0, use_affine_level=False, norm_groups=32):
        super().__init__()
        self.noise_func = FeatureWiseAffine(
            noise_level_emb_dim, dim_out, use_affine_level)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb):
        b, c, h, w = x.shape
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)
        return h + self.res_conv(x)

class ResnetBlock_c(nn.Module):
    def __init__(self, dim, dim_out, dropout=0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head

        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        head_dim = channel // n_head

        norm = self.norm(input)
        qkv = self.qkv(norm).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + input


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, dim, dim_out, *, noise_level_emb_dim=None, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x, time_emb):
        x = self.res_block(x, time_emb)
        if(self.with_attn):
            x = self.attn(x)
        return x

class ResnetBlocWithAttn_c(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock_c(dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = SelfAttention(dim_out, norm_groups=norm_groups)

    def forward(self, x):
        x = self.res_block(x)
        if(self.with_attn):
            x = self.attn(x)
        return x

class UNet(nn.Module):
    def __init__(
        self,
        in_channel=6,
        out_channel=3,
        inner_channel=32,
        norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8),
        res_blocks=3,
        dropout=0,
        with_noise_level_emb=True,
    ):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = ind == num_mults - 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = ind == num_mults - 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks+1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)

class consist_Unet(nn.Module):
    def __init__(
        self, in_channel=3, out_channel=3, inner_channel=32, norm_groups=32,
        channel_mults=(1, 2, 4, 8, 8), res_blocks=3, dropout=0, with_noise_level_emb=True,):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = ind == num_mults - 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)

        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=True),
            ResnetBlocWithAttn(pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                               dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = ind == num_mults - 1
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))

        self.ups = nn.ModuleList(ups)
        self.final_conv = Block(pre_channel, out_channel, groups=norm_groups)

    def forward(self, x, time):
        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t)
            else:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, feats.pop()), dim=1), t)
            else:
                x = layer(x)

        return self.final_conv(x)



class Downsample_with_PixelUnshuffle(nn.Module):
    def __init__(self, n_feat):
        super(Downsample_with_PixelUnshuffle, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample_with_PixelShuffle(nn.Module):
    def __init__(self, n_feat):
        super(Upsample_with_PixelShuffle, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class UNet_banch(nn.Module):
    def __init__(
        self, in_channel=6, out_channel=3, inner_channel=32, norm_groups=32, num_blocks=[1, 1, 2, 2], refinement_block=2, dropout=0):
        super().__init__()

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(PositionalEncoding(inner_channel), nn.Linear(inner_channel, inner_channel * 4),
                Swish(), nn.Linear(inner_channel * 4, inner_channel))

        self.proj_c = nn.Conv2d(in_channel, inner_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_x = nn.Conv2d(in_channel, inner_channel, kernel_size=3, stride=1, padding=1, bias=False)

        #--------------------------------------------------------------------------------------------------------------------------#
        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                ResnetBlocWithAttn(inner_channel, inner_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                    dropout=dropout, with_attn=False))
        self.x_encoder_level1 = nn.Sequential(*layers)

        self.x_down1_2 = Downsample_with_PixelUnshuffle(inner_channel)  ## From Level 1 to Level 2

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                ResnetBlocWithAttn(inner_channel * 2 ** 1, inner_channel * 2 ** 1, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.x_encoder_level2 = nn.Sequential(*layers)

        self.x_down2_3 = Downsample_with_PixelUnshuffle(inner_channel * 2 ** 1)  ## From Level 2 to Level 3

        layers = []
        for i in range(num_blocks[2]):
            layers.append(
                ResnetBlocWithAttn(inner_channel * 2 ** 2, inner_channel * 2 ** 2, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.x_encoder_level3 = nn.Sequential(*layers)

        self.x_down3_4 = Downsample_with_PixelUnshuffle(inner_channel * 2 ** 2)  ## From Level 3 to Level 4

        layers = []
        for i in range(num_blocks[3]):
            layers.append(
                ResnetBlocWithAttn(inner_channel * 2 ** 3, inner_channel * 2 ** 3, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=True))
        self.x_encoder_level4 = nn.Sequential(*layers)

        #--------------------------------------------------------------------------------------------------------------------------#
        layers = []
        for i in range(num_blocks[0]):
            layers.append(
                ResnetBlocWithAttn_c(inner_channel, inner_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.c_encoder_level1 = nn.Sequential(*layers)

        self.c_down1_2 = Downsample_with_PixelUnshuffle(inner_channel)  ## From Level 1 to Level 2

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                ResnetBlocWithAttn_c(inner_channel * 2 ** 1, inner_channel * 2 ** 1, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.c_encoder_level2 = nn.Sequential(*layers)

        self.c_down2_3 = Downsample_with_PixelUnshuffle(inner_channel * 2 ** 1)  ## From Level 2 to Level 3

        layers = []
        for i in range(num_blocks[2]):
            layers.append(
                ResnetBlocWithAttn_c(inner_channel * 2 ** 2, inner_channel * 2 ** 2, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.c_encoder_level3 = nn.Sequential(*layers)

        self.c_down3_4 = Downsample_with_PixelUnshuffle(inner_channel * 2 ** 2)  ## From Level 3 to Level 4

        layers = []
        for i in range(num_blocks[3]):
            layers.append(
                ResnetBlocWithAttn_c(inner_channel * 2 ** 3, inner_channel * 2 ** 3, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=True))
        self.c_encoder_level4 = nn.Sequential(*layers)
        # --------------------------------------------------------------------------------------------------------------------------#

        layers = []
        for i in range(num_blocks[3]):
            layers.append(
                ResnetBlocWithAttn(inner_channel * 2 ** 3, inner_channel * 2 ** 3, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.x_decoder_level4 = nn.Sequential(*layers)

        self.up4_3 = Upsample_with_PixelShuffle(inner_channel * 2 ** 3)
        self.reduce_chan_level3 = nn.Conv2d(inner_channel * 2 ** 3, inner_channel * 2 ** 2, kernel_size=1, bias=True)

        layers = []
        for i in range(num_blocks[2]):
            layers.append(
                ResnetBlocWithAttn(inner_channel * 2 ** 2, inner_channel * 2 ** 2, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.x_decoder_level3 = nn.Sequential(*layers)

        self.up3_2 = Upsample_with_PixelShuffle(inner_channel * 2 ** 2)
        self.reduce_chan_level2 = nn.Conv2d(inner_channel * 2 ** 2, inner_channel * 2 ** 1, kernel_size=1, bias=True)

        layers = []
        for i in range(num_blocks[1]):
            layers.append(
                ResnetBlocWithAttn(inner_channel * 2 ** 1, inner_channel * 2 ** 1, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.x_decoder_level2 = nn.Sequential(*layers)

        self.up2_1 = Upsample_with_PixelShuffle(inner_channel * 2 ** 1)
        self.reduce_chan_level1 = nn.Conv2d(inner_channel * 2 ** 1, inner_channel * 2 ** 1, kernel_size=1, bias=True)

        layers = []
        for i in range(num_blocks[0]):
            layers.append(ResnetBlocWithAttn(inner_channel * 2 ** 1, inner_channel * 2 ** 1, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.x_decoder_level1 = nn.Sequential(*layers)

        layers = []
        for i in range(refinement_block):
            layers.append(ResnetBlocWithAttn(inner_channel * 2 ** 1, inner_channel * 2 ** 1, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups,
                                   dropout=dropout, with_attn=False))
        self.refinement = nn.Sequential(*layers)

        self.final_conv = Block(inner_channel * 2 ** 1, out_channel, groups=norm_groups)

    def forward(self, x, time):
        x_c = x[:, 1:2, :, :]
        x_n = x[:, 0:1, :, :]

        t = self.noise_level_mlp(time) if exists(
            self.noise_level_mlp) else None

        x_n_enc_level1 = self.proj_x(x_n)
        x_c_enc_level1 = self.proj_c(x_c)

        for layer in self.x_encoder_level1:
            if isinstance(layer, ResnetBlocWithAttn):
                x_n_enc_level1 = layer(x_n_enc_level1, t)
            else:
                x_n_enc_level1 = layer(x_n_enc_level1)
        out_x_n_enc_level1 = x_n_enc_level1

        for layer in self.c_encoder_level1:
                x_c_enc_level1 = layer(x_c_enc_level1)
        out_x_c_enc_level1 = x_c_enc_level1
        out_x_n_enc_level1 = out_x_n_enc_level1 + out_x_c_enc_level1

        inp_x_n_enc_level2 = self.x_down1_2(out_x_n_enc_level1)
        for layer in self.x_encoder_level2:
            if isinstance(layer, ResnetBlocWithAttn):
                inp_x_n_enc_level2 = layer(inp_x_n_enc_level2, t)
            else:
                inp_x_n_enc_level2 = layer(inp_x_n_enc_level2)
        out_x_n_enc_level2 = inp_x_n_enc_level2

        inp_x_c_enc_level2 = self.c_down1_2(out_x_c_enc_level1)
        for layer in self.c_encoder_level2:
                inp_x_c_enc_level2 = layer(inp_x_c_enc_level2)
        out_x_c_enc_level2 = inp_x_c_enc_level2
        out_x_n_enc_level2 = out_x_n_enc_level2 + out_x_c_enc_level2

        inp_x_n_enc_level3 = self.x_down2_3(out_x_n_enc_level2)
        for layer in self.x_encoder_level3:
            if isinstance(layer, ResnetBlocWithAttn):
                inp_x_n_enc_level3 = layer(inp_x_n_enc_level3, t)
            else:
                inp_x_n_enc_level3 = layer(inp_x_n_enc_level3)
        out_x_n_enc_level3 = inp_x_n_enc_level3

        inp_x_c_enc_level3 = self.c_down2_3(out_x_c_enc_level2)
        for layer in self.c_encoder_level3:
                inp_x_c_enc_level3 = layer(inp_x_c_enc_level3)
        out_x_c_enc_level3 = inp_x_c_enc_level3
        out_x_n_enc_level3 = out_x_n_enc_level3 + out_x_c_enc_level3

        inp_x_n_enc_level4 = self.x_down3_4(out_x_n_enc_level3)
        for layer in self.x_encoder_level4:
            if isinstance(layer, ResnetBlocWithAttn):
                inp_x_n_enc_level4 = layer(inp_x_n_enc_level4, t)
            else:
                inp_x_n_enc_level4 = layer(inp_x_n_enc_level4)
        out_x_n_enc_level4 = inp_x_n_enc_level4

        inp_x_c_enc_level4 = self.c_down3_4(out_x_c_enc_level3)
        for layer in self.c_encoder_level4:
                inp_x_c_enc_level4 = layer(inp_x_c_enc_level4)
        out_x_c_enc_level4 = inp_x_c_enc_level4
        out_x_n_enc_level4 = out_x_n_enc_level4 + out_x_c_enc_level4

        inp_x_n_dec_level4 = out_x_n_enc_level4
        for layer in self.x_decoder_level4:
            if isinstance(layer, ResnetBlocWithAttn):
                inp_x_n_dec_level4 = layer(inp_x_n_dec_level4, t)
            else:
                inp_x_n_dec_level4 = layer(inp_x_n_dec_level4)
        out_x_n_dec_level4 = inp_x_n_dec_level4

        inp_x_n_dec_level3 = self.up4_3(out_x_n_dec_level4)
        inp_dec_level3 = torch.cat([inp_x_n_dec_level3, out_x_n_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        for layer in self.x_decoder_level3:
            if isinstance(layer, ResnetBlocWithAttn):
                inp_dec_level3 = layer(inp_dec_level3, t)
            else:
                inp_dec_level3 = layer(inp_dec_level3)
        out_x_n_dec_level3 = inp_x_n_dec_level3

        inp_x_n_dec_level2 = self.up3_2(out_x_n_dec_level3)
        inp_dec_level2 = torch.cat([inp_x_n_dec_level2, out_x_n_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        for layer in self.x_decoder_level2:
            if isinstance(layer, ResnetBlocWithAttn):
                inp_dec_level2 = layer(inp_dec_level2, t)
            else:
                inp_dec_level2 = layer(inp_dec_level2)
        out_x_n_dec_level2 = inp_x_n_dec_level2

        inp_x_n_dec_level1 = self.up2_1(out_x_n_dec_level2)
        inp_dec_level1 = torch.cat([inp_x_n_dec_level1, out_x_n_enc_level1], 1)
        for layer in self.x_decoder_level1:
            if isinstance(layer, ResnetBlocWithAttn):
                inp_dec_level1 = layer(inp_dec_level1, t)
            else:
                inp_dec_level1 = layer(inp_dec_level1)
        out_x_n_dec_level1 = inp_dec_level1

        for layer in self.refinement:
            if isinstance(layer, ResnetBlocWithAttn):
                out_x_n_dec_level1 = layer(out_x_n_dec_level1, t)
            else:
                out_x_n_dec_level1 = layer(out_x_n_dec_level1)

        return self.final_conv(out_x_n_dec_level1)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LightBrightnessAdjustment(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, inner_channel=32, norm_groups=32, with_noise_level_emb=True, num_blocks=5):
        super().__init__()

        if with_noise_level_emb:
            noise_level_channel = inner_channel
            self.noise_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )
        else:
            noise_level_channel = None
            self.noise_level_mlp = None

        self.conv1 = nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)

        self.mid_blocks = nn.ModuleList([
            ResnetBlocWithAttn(
                inner_channel, inner_channel, noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=0, with_attn=False)
            for _ in range(num_blocks)
        ])

        self.final_conv = Block(inner_channel, out_channel, groups=norm_groups)

    def forward(self, x, t):
        x_res = x
        t = self.noise_level_mlp(t) if exists(self.noise_level_mlp) else None

        x = self.conv1(x)

        for block in self.mid_blocks:
            x = block(x, t)

        return torch.tanh(self.final_conv(x)) + x_res