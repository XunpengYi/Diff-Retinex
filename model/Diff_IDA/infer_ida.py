import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from tensorboardX import SummaryWriter
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/Diff_IDA_val.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true')
    parser.add_argument('-log_infer', action='store_true')

    #Change to generate results with seed
    torch.manual_seed(1)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataloader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    # model
    diffusion = Model.create_model(opt)
    logger.info('Initial Model Finished')

    diffusion.set_new_noise_schedule(
        opt['model']['beta_schedule']['val'], schedule_phase='val')
    
    logger.info('Begin Model Inference.')
    current_step = 0
    current_epoch = 0
    idx = 0

    result_path = '{}'.format(opt['path']['results'])
    os.makedirs(result_path, exist_ok=True)
    for _,  val_data_collect in enumerate(val_loader):
        idx += 1
        name = str(val_data_collect[1][0])
        val_data = val_data_collect[0]

        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        high_img = Metrics.tensor2img(visuals['high'])  # uint8
        x_0_img = Metrics.tensor2img(visuals['x_0'])  # uint8
        x0_denoise_img = Metrics.tensor2img(visuals['x_0_denoise'])  # uint8
        fake_img = Metrics.tensor2img(visuals['DDPM'])  # uint8
        fake_single = Metrics.tensor2img(visuals['DDPM'].squeeze(0)[-1])  # uint8

        img_mode = 'grid'
        if img_mode == 'single':
            single_img = visuals['DDPM_single']  # uint8
            sample_num = single_img.shape[0]
            for iter in range(0, sample_num):
                Metrics.save_img(
                    Metrics.tensor2img(single_img[iter]), '{}/{}_{}_sr_{}.png'.format(result_path, current_step, idx, iter))
        else:
            # grid img
            low_img = Metrics.tensor2img(visuals['low'])  # uint8
            Metrics.save_img(
                Metrics.tensor2img(visuals['low'][-1]), '{}/{}_{}_low.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                high_img, '{}/{}_{}_high.png'.format(result_path, current_step, name))
            Metrics.save_img(
                x_0_img, '{}/{}_{}_x_0.png'.format(result_path, current_step, name))
            Metrics.save_img(
                fake_single, '{}/{}.png'.format(result_path, name))
