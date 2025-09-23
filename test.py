import argparse
import copy
import numpy as np
import os
import json
import re

from utils.logger import load_config, make_output_dirs


def get_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default='config/params_58.json', type=str,
                        help='JSON file for configuration')

    return parser


def main(args):
    # fix the seed for reproducibility
    seed = args['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True

    log_writer = SummaryWriter(log_dir=args['path']['log_dir'])

    dataset_sim_val = SpikeDatasetSim(args, is_train=False)
    sampler_sim_val = torch.utils.data.SequentialSampler(dataset_sim_val)
    dataloader_sim_val = torch.utils.data.DataLoader(
        dataset_sim_val,
        batch_size=args['dataloader']['batch_size'],
        num_workers=args['dataloader']['num_workers'],
        persistent_workers=args['dataloader']['persistent_workers'],
        drop_last=True, pin_memory=True, sampler=sampler_sim_val)

    dataset_real = SpikeDatasetReal(args)
    sampler_real = torch.utils.data.SequentialSampler(dataset_real)
    dataloader_real = torch.utils.data.DataLoader(
        dataset_real,
        batch_size=args['dataloader']['batch_size'],
        num_workers=args['dataloader']['num_workers'],
        persistent_workers=args['dataloader']['persistent_workers'],
        drop_last=True, pin_memory=True, sampler=sampler_real)

    unet_model = unet.__dict__["unet_model"](args)

    max_psnr, max_ssim, min_lpips, max_clipiqa, max_musiq = 0, 0, 100, 0, 0

    if args['path']['model_checkpoint_path'] is not None:
        if os.path.isdir(args['path']['model_checkpoint_path']):
            checkpoint_path_list = sorted(os.listdir(args['path']['model_checkpoint_path']))
            model_checkpoint_path_list = []
            for checkpoint_path in checkpoint_path_list:
                if 'opt' not in checkpoint_path:
                    model_checkpoint_path_list.append(os.path.join(args['path']['model_checkpoint_path'], checkpoint_path))
        else:
            model_checkpoint_path_list = [args['path']['model_checkpoint_path']]

        for model_checkpoint_path in model_checkpoint_path_list:
            model = diffusion.__dict__["gaussian_diffusion"](args, unet_model)

            model_checkpoint = torch.load(model_checkpoint_path, map_location='cpu')

            msg = model.load_state_dict(model_checkpoint, strict=False)
            print("load model checkpoint msg: ", msg)

            device = torch.device(args['device'])
            model.to(device)
            model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print("Model = %s" % str(model))
            print('Number of model params: %.2fM' % (model_params / 1.e6))

            iter_step = int(re.split('_', re.split('/', model_checkpoint_path)[-1])[0][1:])

            sample_sim(args, model, dataloader_sim_val, iter_step, max_psnr, max_ssim, min_lpips, log_writer, is_test=True)

            sample_real(args, model, dataloader_real, iter_step, max_clipiqa, max_musiq, log_writer, is_test=True)


if __name__ == "__main__":
    parser = get_args_parser()
    parser = parser.parse_args()

    args = load_config(parser.config)
    args_copy = copy.deepcopy(args)
    args['opt'] = 'test'

    make_output_dirs(args)
    # save params
    with open(os.path.join(args['path']['log_dir'], "params_58.json"), 'w') as json_file:
        json.dump(args_copy, json_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_ids']
    import torch
    torch.set_num_threads(1)
    import torch.backends.cudnn as cudnn
    from torch.utils.tensorboard import SummaryWriter

    from trainer.sample import sample_sim, sample_real
    from dataset.spike_dataset import SpikeDatasetSim, SpikeDatasetReal
    from model import unet, diffusion

    main(args)
