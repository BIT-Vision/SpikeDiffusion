import torch
import torch.nn as nn

from utils import misc
from trainer.sample import sample_sim, sample_real
from utils.misc import save_model
from utils.visualize import vis_train


def train_one_epoch(args, model, dataloader_train, dataloader_sim_val, dataloader_real,
                    optimizer, iter_step, epoch, max_psnr, max_ssim, min_lpips, max_clipiqa, max_musiq, log_writer):
    model.train()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Iter:[{}] Epoch:[{}]'.format(iter_step, epoch)

    optimizer.zero_grad()

    for data_iter_step, (spike_image, spike_image_rm_dark_trunc_A, spike_image_rm_dark_trunc_A_mask,
                         trunc_diff, gt_image, gradient_image, spike_density, mask_prop, A, Ld_div_q) in enumerate(
            metric_logger.log_every(dataloader_train, args['freq']['print_freq'], header)):

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

        loss, output_image, x_t, t = model(spike_image, spike_image_rm_dark_trunc_A_mask,
                                           trunc_diff, spike_density, A, Ld_div_q,
                                           phase="train", gt_image=gt_image)

        loss_diff = loss['loss_diff'].item()
        metric_logger.update(loss_diff=loss_diff)
        loss_total = args['experiment']['loss_diff_factor'] * loss['loss_diff']
        if (iter_step + 1) % args['freq']['train_log_freq'] == 0:
            log_writer.add_scalar('train' + '/loss_diff', loss_diff, iter_step)

        if args['experiment']['use_loss_lpips']:
            loss_lpips = loss['loss_lpips'].item()
            loss_total += args['experiment']['loss_lpips_factor'] * loss['loss_lpips']
            metric_logger.update(loss_lpips=loss_lpips)
            if (iter_step + 1) % args['freq']['train_log_freq'] == 0:
                log_writer.add_scalar('train' + '/loss_lpips', loss_lpips, iter_step)
        else:
            loss_lpips = None

        loss_total.backward()
        if args['train']['clip_grad'] is not None:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=args['train']['clip_grad'])
        optimizer.step()
        optimizer.zero_grad()

        if args['device'] == 'cuda':
            torch.cuda.synchronize()

        metric_logger.update(loss=loss_total.item())
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        if (iter_step + 1) % args['freq']['train_log_freq'] == 0:
            log_writer.add_scalar('train/loss', loss_total.item(), iter_step)
            log_writer.add_scalar('train/lr', lr, iter_step)
            log_writer.flush()

        if (iter_step + 1) % args['freq']['vis_train_freq'] == 0:
            _t, _spike_density, _mask_prop = t[0].item(), spike_density[0].item(), mask_prop[0].item()

            scale_factor = args['dataset']['scale_factor'] / _spike_density

            _gt_image, _gradient_image, _spike_image, \
            _spike_image_rm_dark_trunc_A, _spike_image_rm_dark_trunc_A_mask, _output_image, _x_t = \
                gt_image[0].squeeze(0).detach().cpu().numpy() / scale_factor, \
                gradient_image[0].squeeze(0).detach().cpu().numpy() / scale_factor, \
                spike_image[0].squeeze(0).detach().cpu().numpy() / scale_factor, \
                spike_image_rm_dark_trunc_A[0].squeeze(0).detach().cpu().numpy() / scale_factor, \
                spike_image_rm_dark_trunc_A_mask[0].squeeze(0).detach().cpu().numpy() / scale_factor, \
                output_image[0].squeeze(0).detach().cpu().numpy() / scale_factor, \
                x_t[0].squeeze(0).detach().cpu().numpy() / scale_factor

            loss_lpips_m = getattr(metric_logger, 'loss_lpips', None)
            if loss_lpips_m is not None:
                loss_lpips_avg = loss_lpips_m.global_avg
            else:
                loss_lpips_avg = None

            vis_train(args, _gt_image, _gradient_image, _spike_image,
                      _spike_image_rm_dark_trunc_A, _spike_image_rm_dark_trunc_A_mask,
                      _output_image, _x_t, _t, _spike_density, _mask_prop, epoch, iter_step,
                      loss_diff, metric_logger.loss_diff.global_avg, loss_lpips, loss_lpips_avg)

        # Val
        if (iter_step + 1) > 1 and (iter_step + 1) % args['freq']['val_freq'] == 0:
            print_str = '{}: loss_diff: {:.4f}'.format(args['name'], metric_logger.loss_diff.global_avg)
            if args['experiment']['use_loss_lpips']:
                print_str += ' loss_lpips: {:.4f}'.format(metric_logger.loss_lpips.global_avg)

            print_str += ' loss: {:.4f}'.format(metric_logger.loss.global_avg)
            print(print_str)

            psnr_avg_val, ssim_avg_val, lpips_avg_val, max_psnr, max_ssim, min_lpips = \
                sample_sim(args, model, dataloader_sim_val, iter_step, max_psnr, max_ssim, min_lpips, log_writer)

            clipiqa_avg_real, musiq_avg_real, max_clipiqa, max_musiq = \
                sample_real(args, model, dataloader_real, iter_step, max_clipiqa, max_musiq, log_writer)

        if ((iter_step + 1) > 1 and (iter_step + 1) % args['freq']['save_checkpoint_freq'] == 0):
            save_model(args, model, optimizer, iter_step + 1, epoch + 1)

        if (iter_step + 1) == args['train']['n_iter']:
            break

        iter_step += 1

    epoch += 1

    return iter_step, epoch, max_psnr, max_ssim, min_lpips, max_clipiqa, max_musiq
