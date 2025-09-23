import os
import numpy as np
import lpips
import pyiqa
from skimage.metrics import peak_signal_noise_ratio as psnr_metirc
from skimage.metrics import structural_similarity as ssim_metirc
import matplotlib.pyplot as plt

import torch

from utils import misc
from utils.visualize import vis_sample_sim_train, vis_sample_real_train, vis_sample_sim, vis_sample_real

clipiqa_metric = pyiqa.create_metric('clipiqa')
musiq_metric = pyiqa.create_metric('musiq')
lpips_metric = lpips.LPIPS(net='alex')


@torch.no_grad()
def sample_sim(args, model, data_loader, iter_step, max_psnr, max_ssim, min_lpips, log_writer, is_test=False):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Val sim:'

    psnr_list = []
    ssim_list = []
    lpips_list = []

    index = 0
    for data_iter_step, (spike_image, spike_image_rm_dark_trunc_A, spike_image_rm_dark_trunc_A_mask,
                         trunc_diff, gt_image, gradient_image, spike_density, mask_prop, A, Ld_div_q) in enumerate(
            metric_logger.log_every(data_loader, args['freq']['print_freq'], header)):
        spike_image = spike_image.to(args['device'], non_blocking=True)
        spike_image_rm_dark_trunc_A = spike_image_rm_dark_trunc_A.to(args['device'], non_blocking=True)
        spike_image_rm_dark_trunc_A_mask = spike_image_rm_dark_trunc_A_mask.to(args['device'], non_blocking=True)
        trunc_diff = trunc_diff.to(args['device'], non_blocking=True)
        gt_image = gt_image.to(args['device'], non_blocking=True)
        gradient_image = gradient_image.to(args['device'], non_blocking=True)
        spike_density = spike_density.to(args['device'], non_blocking=True)
        mask_prop = mask_prop.to(args['device'], non_blocking=True)
        A = A.to(args['device'], non_blocking=True)
        Ld_div_q = Ld_div_q.to(args['device'], non_blocking=True)

        image_sample, x_0_list, x_t_list = model(spike_image, spike_image_rm_dark_trunc_A_mask,
                                                 trunc_diff, spike_density, A, Ld_div_q, phase="sample")

        psnr_list_iter = []
        ssim_list_iter = []
        lpips_list_iter = []

        for i in range(args['dataloader']['batch_size']):
            _Ld_div_q, _A, _spike_density, _mask_prop = Ld_div_q[i].squeeze(0)[3:-3, :], A[i].squeeze(0)[3:-3, :],\
                spike_density[i].item(), mask_prop[i].item()
            scale_factor = args['dataset']['scale_factor'] / _spike_density
            _spike_image, _spike_image_rm_dark_trunc_A, _spike_image_rm_dark_trunc_A_mask, \
            _trunc_diff, _image_sample, _gt_image, _gradient_image = \
                spike_image[i].squeeze(0)[3:-3, :] / scale_factor, \
                spike_image_rm_dark_trunc_A[i].squeeze(0)[3:-3, :] / scale_factor, \
                spike_image_rm_dark_trunc_A_mask[i].squeeze(0)[3:-3, :] / scale_factor, \
                trunc_diff[i].squeeze(0)[3:-3, :] / scale_factor, \
                image_sample[i].squeeze(0)[3:-3, :] / scale_factor, \
                gt_image[i].squeeze(0)[3:-3, :] / scale_factor, \
                gradient_image[i].squeeze(0)[3:-3, :] / scale_factor

            _x_0_list = []
            for x_0 in x_0_list:
                _x_0 = x_0[i].squeeze(0)[3:-3, :] / scale_factor
                _x_0_list.append(_x_0)

            _x_t_list = []
            for x_t in x_t_list:
                _x_t = x_t[i].squeeze(0)[3:-3, :] / scale_factor
                _x_t_list.append(_x_t)

            lpips = lpips_metric.to(args['device'])(_gt_image, _image_sample.float()).item()
            lpips_x0_list = []
            for _x_0 in _x_0_list:
                lpips_x0 = lpips_metric.to(args['device'])(_gt_image, _x_0.float()).item()
                lpips_x0_list.append(lpips_x0)

            _spike_image, _spike_image_rm_dark_trunc_A, _spike_image_rm_dark_trunc_A_mask, \
                _image_sample, _gt_image, _gradient_image = \
                _spike_image.detach().cpu().numpy(), _spike_image_rm_dark_trunc_A.detach().cpu().numpy(), \
                _spike_image_rm_dark_trunc_A_mask.detach().cpu().numpy(), \
                _image_sample.detach().cpu().numpy(),  _gt_image.detach().cpu().numpy(), _gradient_image.detach().cpu().numpy()

            for (i, _x_0) in enumerate(_x_0_list):
                _x_0_list[i] = _x_0.detach().cpu().numpy()

            for (i, _x_t) in enumerate(_x_t_list):
                _x_t_list[i] = _x_t.detach().cpu().numpy()

            psnr = psnr_metirc(_image_sample, _gt_image, data_range=args['dataset']['light_scale'])
            ssim = ssim_metirc(_image_sample, _gt_image, data_range=args['dataset']['light_scale'])

            psnr_x0_list = []
            ssim_x0_list = []
            for _x_0 in _x_0_list:
                psnr_x0 = psnr_metirc(_x_0, _gt_image, data_range=args['dataset']['light_scale'])
                psnr_x0_list.append(psnr_x0)
                ssim_x0 = ssim_metirc(_x_0, _gt_image, data_range=args['dataset']['light_scale'])
                ssim_x0_list.append(ssim_x0)

            psnr_list_iter.append(psnr)
            ssim_list_iter.append(ssim)
            lpips_list_iter.append(lpips)

            index += 1
            if (iter_step + 1) % args['freq']['vis_val_freq'] == 0:
                if index <= 10:
                    if is_test:
                        vis_sample_sim(args, _spike_image, _image_sample, _gt_image, iter_step, index, psnr, ssim, lpips)
                    else:
                        vis_sample_sim_train(args, _spike_image, _spike_image_rm_dark_trunc_A, _spike_image_rm_dark_trunc_A_mask,
                                             _image_sample, _gt_image, _gradient_image,
                                             _x_0_list, _x_t_list, iter_step, index,
                                             _spike_density, _mask_prop,
                                             psnr, ssim, lpips, psnr_x0_list, ssim_x0_list, lpips_x0_list)

        psnr_avg_iter = sum(psnr_list_iter) / len(psnr_list_iter)
        ssim_avg_iter = sum(ssim_list_iter) / len(ssim_list_iter)
        lpips_avg_iter = sum(lpips_list_iter) / len(lpips_list_iter)

        metric_logger.meters['psnr'].update(psnr_avg_iter, n=args['dataloader']['batch_size'])
        metric_logger.meters['ssim'].update(ssim_avg_iter, n=args['dataloader']['batch_size'])
        metric_logger.meters['lpips'].update(lpips_avg_iter, n=args['dataloader']['batch_size'])

        psnr_list.append(psnr_avg_iter)
        ssim_list.append(ssim_avg_iter)
        lpips_list.append(lpips_avg_iter)

    psnr_avg = sum(psnr_list) / len(psnr_list)
    ssim_avg = sum(ssim_list) / len(ssim_list)
    lpips_avg = sum(lpips_list) / len(lpips_list)

    if psnr_avg > max_psnr:
        max_psnr = psnr_avg
    if ssim_avg > max_ssim:
        max_ssim = ssim_avg
    if lpips_avg < min_lpips:
        min_lpips = lpips_avg

    save_root = os.path.join(args['path']['vis_val_sim_dir'])
    os.rename(os.path.join(save_root, 'I{:06d}'.format(iter_step + 1)),
              os.path.join(save_root, 'I{:06d}-p_{:.4f}({:.4f})-s_{:.4f}({:.4f})-l_{:.4f}({:.4f})'.format(
                  iter_step + 1, psnr_avg, max_psnr, ssim_avg, max_ssim, lpips_avg, min_lpips)))

    log_writer.add_scalar('val_sim/psnr', psnr_avg, iter_step)
    log_writer.add_scalar('val_sim/ssim', ssim_avg, iter_step)
    log_writer.add_scalar('val_sim/lpips', lpips_avg, iter_step)
    log_writer.flush()

    print('----------------------------------------------------------------------')
    print('{}: psnr: {:.4f}({:.4f}) ssim: {:.4f}({:.4f}) lpips: {:4f}({:.4f})'.format(
        args['name'], psnr_avg, max_psnr, ssim_avg, max_ssim, lpips_avg, min_lpips))
    print('----------------------------------------------------------------------')

    return psnr_avg, ssim_avg, lpips_avg, max_psnr, max_ssim, min_lpips


@torch.no_grad()
def sample_real(args, model, data_loader, iter_step, max_clipiqa, max_musiq, log_writer, is_test=False):
    model.eval()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Val real:'

    clipiqa_list = []
    musiq_list = []

    index = 0
    for data_iter_step, (spike_image, spike_image_rm_dark_trunc_A, spike_image_rm_dark_trunc_A_mask,
                         trunc_diff, spike_density, mask_prop, A, Ld_div_q) in enumerate(
            metric_logger.log_every(data_loader, args['freq']['print_freq'], header, is_real=True)):
        spike_image = spike_image.to(args['device'], non_blocking=True)
        spike_image_rm_dark_trunc_A = spike_image_rm_dark_trunc_A.to(args['device'], non_blocking=True)
        spike_image_rm_dark_trunc_A_mask = spike_image_rm_dark_trunc_A_mask.to(args['device'], non_blocking=True)
        trunc_diff = trunc_diff.to(args['device'], non_blocking=True)
        spike_density = spike_density.to(args['device'], non_blocking=True)
        mask_prop = mask_prop.to(args['device'], non_blocking=True)
        A = A.to(args['device'], non_blocking=True)
        Ld_div_q = Ld_div_q.to(args['device'], non_blocking=True)

        image_sample, x_0_list, x_t_list = model(spike_image, spike_image_rm_dark_trunc_A_mask,
                                                 trunc_diff, spike_density, A, Ld_div_q, phase="sample")

        clipiqa_list_iter = []
        musiq_list_iter = []
        for i in range(args['dataloader']['batch_size']):
            _spike_density, _mask_prop = spike_density[i].item(), mask_prop[i].item()
            scale_factor = args['dataset']['scale_factor'] / _spike_density
            _spike_image, _spike_image_rm_dark_trunc_A, _spike_image_rm_dark_trunc_A_mask, \
            _trunc_diff, _image_sample = \
                spike_image[i].squeeze(0)[3:-3, :] / scale_factor, \
                spike_image_rm_dark_trunc_A[i].squeeze(0)[3:-3, :] / scale_factor, \
                spike_image_rm_dark_trunc_A_mask[i].squeeze(0)[3:-3, :] / scale_factor, \
                trunc_diff[i].squeeze(0)[3:-3, :] / scale_factor, \
                image_sample[i].squeeze(0)[3:-3, :] / scale_factor

            _spike_image, _spike_image_rm_dark_trunc_A, _spike_image_rm_dark_trunc_A_mask, \
            _trunc_diff, _image_sample = \
                _spike_image.detach().cpu().numpy(),  _spike_image_rm_dark_trunc_A.detach().cpu().numpy(), \
                _spike_image_rm_dark_trunc_A_mask.detach().cpu().numpy(), _trunc_diff.detach().cpu().numpy(), \
                _image_sample.detach().cpu().numpy()

            _x_0_list = []
            for x_0 in x_0_list:
                _x_0 = x_0[i].squeeze(0)[3:-3, :].detach().cpu().numpy()
                _x_0_list.append(_x_0)

            _x_t_list = []
            for x_t in x_t_list:
                _x_t = x_t[i].squeeze(0)[3:-3, :].detach().cpu().numpy()
                _x_t_list.append(_x_t)

            _image_sample = _image_sample - _image_sample.min()
            _image_sample_gamma = np.power(_image_sample, 1 / 2.2)
            _image_sample_gamma_norm = (_image_sample_gamma - _image_sample_gamma.min()) / (
                _image_sample_gamma.max() - _image_sample_gamma.min()
            )

            clipiqa = clipiqa_metric(torch.from_numpy(
                _image_sample_gamma_norm).float().unsqueeze(0).unsqueeze(0)).item()
            musiq = musiq_metric(torch.from_numpy(
                _image_sample_gamma_norm).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)).item()

            clipiqa_x0_list = []
            musiq_x0_list = []
            for _x_0 in _x_0_list:
                _x_0 = _x_0 - _x_0.min()
                _x_0_gamma = np.power(_x_0, 1 / 2.2)
                _x_0_gamma_norm = (_x_0_gamma - _x_0_gamma.min()) / (
                        _x_0_gamma.max() - _x_0_gamma.min())

                clipiqa_x0 = clipiqa_metric(torch.from_numpy(
                    _x_0_gamma_norm).float().unsqueeze(0).unsqueeze(0)).item()
                musiq_x0 = musiq_metric(torch.from_numpy(
                    _x_0_gamma_norm).float().unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)).item()

                clipiqa_x0_list.append(clipiqa_x0)
                musiq_x0_list.append(musiq_x0)

            clipiqa_list_iter.append(clipiqa)
            musiq_list_iter.append(musiq)

            index += 1

            if (iter_step + 1) % args['freq']['vis_val_freq'] == 0:
                if is_test:
                    vis_sample_real(args, _spike_image, _image_sample, iter_step, index, clipiqa, musiq)
                else:
                    vis_sample_real_train(args, _spike_image, _spike_image_rm_dark_trunc_A,
                                          _spike_image_rm_dark_trunc_A_mask, _image_sample,
                                          _x_0_list, _x_t_list, iter_step, index,
                                          _spike_density, _mask_prop,
                                          clipiqa, musiq, clipiqa_x0_list, musiq_x0_list)

        clipiqa_avg_iter = (sum(clipiqa_list_iter) / len(clipiqa_list_iter))
        musiq_avg_iter = (sum(musiq_list_iter) / len(musiq_list_iter))

        metric_logger.meters['clipiqa'].update(clipiqa_avg_iter, n=args['dataloader']['batch_size'])
        metric_logger.meters['musiq'].update(musiq_avg_iter, n=args['dataloader']['batch_size'])

        clipiqa_list.append(clipiqa_avg_iter)
        musiq_list.append(musiq_avg_iter)

    clipiqa_avg = sum(clipiqa_list) / len(clipiqa_list)
    musiq_avg = sum(musiq_list) / len(musiq_list)

    if clipiqa_avg > max_clipiqa:
        max_clipiqa = clipiqa_avg
    if musiq_avg > max_musiq:
        max_musiq = musiq_avg

    save_root = args['path']['vis_val_real_dir']
    os.rename(os.path.join(save_root, 'I{:06d}'.format(iter_step + 1)),
              os.path.join(save_root, 'I{:06d}-c_{:.4f}({:.4f})-m_{:.4f}({:.4f})'.format(
                  iter_step + 1, clipiqa_avg, max_clipiqa, musiq_avg, max_musiq)))

    log_writer.add_scalar('val_real/clipiqa', clipiqa_avg, iter_step)
    log_writer.add_scalar('val_real/musiq', musiq_avg, iter_step)
    log_writer.flush()

    print('{}: clipiqa: {:.4f}({:.4f}) musiq: {:.4f}({:.4f})'.format(
        args['name'], clipiqa_avg, max_clipiqa, musiq_avg, max_musiq))

    return clipiqa_avg, musiq_avg, max_clipiqa, max_musiq
