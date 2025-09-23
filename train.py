import argparse
import copy
import numpy as np
import os
import time
import datetime
import json
import re
import random

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
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False  # True
    cudnn.deterministic = True

    log_writer = SummaryWriter(log_dir=args['path']['log_dir'])

    dataset_train = SpikeDatasetSim(args, is_train=True)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args['dataloader']['batch_size'],
        num_workers=args['dataloader']['num_workers'],
        persistent_workers=args['dataloader']['persistent_workers'],
        drop_last=True, pin_memory=True, sampler=sampler_train)

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
    model = diffusion.__dict__["gaussian_diffusion"](args, unet_model)

    if args['path']['model_checkpoint_path'] is not None:
        model_checkpoint = torch.load(args['path']['model_checkpoint_path'], map_location='cpu')
        msg = model.load_state_dict(model_checkpoint, strict=False)
        print("load model checkpoint msg: ", msg)

        iter_step = int(re.split('_', os.path.split(args['path']['model_checkpoint_path'])[-1])[0][1:])
        epoch = int(re.split('_', os.path.split(args['path']['model_checkpoint_path'])[-1])[1][1:-4]) - 1
    else:
        iter_step = args['train']['start_iter']
        epoch = 0

    device = torch.device(args['device'])
    model.to(device)
    model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print("Model = %s" % str(model))
    print('Number of model params: %.2fM' % (model_params / 1.e6))

    optim_params = list(model.parameters())
    optimizer = torch.optim.Adam(optim_params, lr=args['train']['lr'])
    if args['path']['optimizer_checkpoint_path'] is not None:
        optimizer_checkpoint = torch.load(args['path']['optimizer_checkpoint_path'], map_location='cpu')
        optimizer.load_state_dict(optimizer_checkpoint)

    start_time = time.time()

    max_psnr, max_ssim, min_lpips, max_clipiqa, max_musiq = 0, 0, 100, 0, 0
    while iter_step < args['train']['n_iter']:
        iter_step, epoch, max_psnr, max_ssim, min_lpips, max_clipiqa, max_musiq = \
            train_one_epoch(args, model, dataloader_train, dataloader_sim_val, dataloader_real,
            optimizer, iter_step, epoch, max_psnr, max_ssim, min_lpips, max_clipiqa, max_musiq, log_writer)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    print('Training time {}'.format(total_time_str))


if __name__ == "__main__":
    parser = get_args_parser()
    parser = parser.parse_args()

    args = load_config(parser.config)
    args_copy = copy.deepcopy(args)
    args['opt'] = 'train'

    make_output_dirs(args)
    # save params
    with open(os.path.join(args['path']['log_dir'], "params_58.json"), 'w') as json_file:
        json.dump(args_copy, json_file)

    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu_ids']
    import torch
    torch.set_num_threads(1)
    import torch.backends.cudnn as cudnn
    from torch.utils.tensorboard import SummaryWriter

    from dataset.spike_dataset import SpikeDatasetSim, SpikeDatasetReal
    from model import unet, diffusion
    from trainer.train_one_epoch import train_one_epoch

    main(args)
