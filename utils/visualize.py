import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math


def vis_train(args, gt_image, gradient_image, spike_image,
              spike_image_rm_dark_trunc_A, spike_image_rm_dark_trunc_A_mask,
              output_image, x_t, t, spike_density, mask_prop, epoch, iter_step,
              loss_diff, loss_diff_avg, loss_lpips=None, loss_lpips_avg=None):
    plt.figure()
    plt.subplots_adjust(top=0.7, bottom=0.3, left=0.2, right=0.8)
    plt.subplot(2, 4, 1)
    plt.imshow(gt_image, cmap='gray')
    plt.axis('off')
    plt.title("gt_image", fontsize=6)

    plt.subplot(2, 4, 2)
    plt.imshow(gradient_image, cmap='gray')
    plt.axis('off')
    plt.title("gradient_image", fontsize=6)

    plt.subplot(2, 4, 3)
    _spike_image = spike_image - spike_image.min()
    spike_image_gamma = np.power(_spike_image, 1 / 2.2)
    plt.imshow(spike_image_gamma, cmap='gray')
    plt.axis('off')
    plt.title("spike_image", fontsize=6)

    plt.subplot(2, 4, 5)
    spike_image_rm_dark_trunc_A_norm = (spike_image_rm_dark_trunc_A - spike_image_rm_dark_trunc_A.min()) / (
            spike_image_rm_dark_trunc_A.max() - spike_image_rm_dark_trunc_A.min()
    )
    spike_image_rm_dark_trunc_A_hist = cv2.equalizeHist((spike_image_rm_dark_trunc_A_norm * 255).astype(np.uint8))
    plt.imshow(spike_image_rm_dark_trunc_A_hist, cmap='gray')
    plt.axis('off')
    plt.title('rm_dark_trunc_A_hist d_{:.2f}'.format(spike_density), fontsize=6)

    plt.subplot(2, 4, 6)
    plt.imshow(spike_image_rm_dark_trunc_A_mask, cmap='gray')
    plt.axis('off')
    plt.title('mask p_{:.4f}'.format(mask_prop), fontsize=6)

    plt.subplot(2, 4, 7)
    _x_t = x_t - x_t.min()
    x_t_gamma = np.power(_x_t, 1 / 2.2)
    plt.imshow(x_t_gamma, cmap='gray')
    plt.axis('off')
    plt.title('xt_{}'.format(t), fontsize=6)

    plt.subplot(2, 4, 8)
    plt.imshow(output_image, cmap='gray')
    plt.axis('off')
    sub_title = 'x_0 ld_{:.4f}'.format(loss_diff)
    if args['experiment']['use_loss_lpips']:
        sub_title += ' lp_{:.4f}'.format(loss_lpips)
    plt.title(sub_title, fontsize=6)

    title = '(I{:06d}_E{:03d})-ld_{:.4f}'.format(iter_step + 1, epoch, loss_diff_avg)
    if args['experiment']['use_loss_lpips']:
        title += '_lp_{:.4f}'.format(loss_lpips_avg)
    plt.suptitle(title)
    plt.savefig(os.path.join(args['path']['vis_train_dir'], title + ".png"),
                bbox_inches='tight', pad_inches=0, dpi=400)
    plt.show()
    plt.close()


def vis_sample_sim_train(args, spike_image, spike_image_rm_dark_trunc_A, spike_image_rm_dark_trunc_A_mask,
                         image_sample, gt_image, gradient_image, x_0_list, x_t_list,
                         iter_step, index, spike_density, mask_prop, psnr, ssim, lpips,
                         psnr_x0_list, ssim_x0_list, lpips_x0_list):
    save_root = os.path.join(args['path']['vis_val_sim_dir'], 'I{:06d}'.format(iter_step + 1))
    os.makedirs(save_root, exist_ok=True)

    len_num = 2 + math.ceil(len(x_0_list) / 4) + math.ceil(len(x_t_list) / 4)

    plt.subplots_adjust(hspace=0.5)
    plt.subplot(len_num, 4, 1)
    _spike_image = spike_image - spike_image.min()
    spike_image_gamma = np.power(_spike_image, 1 / 2.2)
    plt.imshow(spike_image_gamma, cmap='gray')
    plt.axis('off')
    plt.title('spike_image_gamma', fontsize=6)

    plt.subplot(len_num, 4, 2)
    plt.imshow(gradient_image, cmap='gray')
    plt.axis('off')
    plt.title('gradient_image', fontsize=6)

    plt.subplot(len_num, 4, 3)
    spike_image_rm_dark_trunc_A_norm = (spike_image_rm_dark_trunc_A - spike_image_rm_dark_trunc_A.min()) / (
            spike_image_rm_dark_trunc_A.max() - spike_image_rm_dark_trunc_A.min()
    )
    spike_image_rm_dark_trunc_A_hist = cv2.equalizeHist((spike_image_rm_dark_trunc_A_norm * 255).astype(np.uint8))
    plt.imshow(spike_image_rm_dark_trunc_A_hist, cmap='gray')
    plt.axis('off')
    plt.title('rm_dark_trunc_A_hist d_{:.2f}'.format(spike_density), fontsize=6)

    plt.subplot(len_num, 4, 4)
    plt.imshow(spike_image_rm_dark_trunc_A_mask, cmap='gray')
    plt.axis('off')
    plt.title('mask p_{:.4f}'.format(mask_prop), fontsize=6)

    plt.subplot(len_num, 4, 5)
    _image_sample = image_sample - image_sample.min()
    image_sample_gamma = np.power(_image_sample, 1 / 2.2)
    plt.imshow(image_sample_gamma, cmap='gray')
    plt.axis('off')
    plt.title('image_sample_gamma', fontsize=6)

    plt.subplot(len_num, 4, 6)
    _gt_image = gt_image - gt_image.min()
    gt_image_gamma = np.power(_gt_image, 1 / 2.2)
    plt.imshow(gt_image_gamma, cmap='gray')
    plt.axis('off')
    plt.title('gt_image_gamma', fontsize=6)

    plt.subplot(len_num, 4, 7)
    plt.imshow(image_sample, cmap='gray')
    plt.axis('off')
    plt.title('image_sample', fontsize=6)

    plt.subplot(len_num, 4, 8)
    plt.imshow(gt_image, cmap='gray')
    plt.axis('off')
    plt.title('gt_image', fontsize=6)

    for i, x_0 in enumerate(x_0_list):
        plt.subplot(len_num, 4, i + 9)
        _x_0 = x_0 - x_0.min()
        x_0_gamma = np.power(_x_0, 1 / 2.2)
        plt.imshow(x_0_gamma, cmap='gray')
        plt.axis('off')
        plt.title('x0_{} p_{:.4f}-s_{:.4f}-l_{:.4f}'.format(
            i + 1, psnr_x0_list[i], ssim_x0_list[i], lpips_x0_list[i]), fontsize=5)

    for i, x_t in enumerate(x_t_list):
        _x_t = x_t - x_t.min()
        x_t_gamma = np.power(_x_t, 1 / 2.2)
        plt.subplot(len_num, 4, i + 4 * (2 + math.ceil(len(x_0_list) / 4)) + 1)
        plt.imshow(x_t_gamma, cmap='gray')
        plt.axis('off')
        plt.title('xt_{}'.format(i + 1), fontsize=6)

    title = "(I{:06d}_N{:02d})-p_{:.4f}-s_{:.4f}-l_{:.4f}".format(iter_step + 1, index, psnr, ssim, lpips)
    plt.suptitle(title, fontsize=8)
    plt.savefig(os.path.join(save_root, title + ".png"), bbox_inches='tight', pad_inches=0, dpi=400)
    plt.show()
    plt.close()


def vis_sample_sim(args, spike_image, image_sample, gt_image, iter_step, index, psnr, ssim, lpips):
    save_root = os.path.join(args['path']['vis_val_sim_dir'], 'I{:06d}'.format(iter_step + 1))
    os.makedirs(save_root, exist_ok=True)

    plt.imshow(image_sample, cmap='gray')
    plt.axis('off')
    title = "(I{:06d}_N{:02d})-sample-p_{:.4f}-s_{:.4f}-l_{:.4f}".format(iter_step + 1, index, psnr, ssim, lpips)
    plt.savefig(os.path.join(save_root, title + ".png"), bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
    plt.close()

    spike_image_gamma = np.power(spike_image, 1 / 2.2)
    plt.imshow(spike_image_gamma, cmap='gray')
    plt.axis('off')
    title = "(I{:06d}_N{:02d})-spike_image_gamma".format(iter_step + 1, index, psnr, ssim, lpips)
    plt.savefig(os.path.join(save_root, title + ".png"), bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
    plt.close()

    plt.imshow(gt_image, cmap='gray')
    plt.axis('off')
    title = "(I{:06d}_N{:02d})-gt".format(iter_step + 1, index, psnr, ssim, lpips)
    plt.savefig(os.path.join(save_root, title + ".png"), bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
    plt.close()


def vis_sample_real_train(args, spike_image, spike_image_rm_dark_trunc_A, spike_image_rm_dark_trunc_A_mask,
                          image_sample, x_0_list, x_t_list, iter_step, index,
                          spike_density, mask_prop,
                          clipiqa, musiq, clipiqa_x0_list, musiq_x0_list):
    save_root = os.path.join(args['path']['vis_val_real_dir'], 'I{:06d}'.format(iter_step + 1))
    os.makedirs(save_root, exist_ok=True)

    len_num = 2 + math.ceil(len(x_0_list) / 4) + math.ceil(len(x_t_list) / 4)

    plt.subplots_adjust(hspace=0.5)
    plt.subplot(len_num, 4, 1)
    _spike_image = spike_image - spike_image.min()
    spike_image_gamma = np.power(_spike_image, 1 / 2.2)
    plt.imshow(spike_image_gamma, cmap='gray')
    plt.axis('off')
    plt.title('spike_image_gamma', fontsize=6)

    plt.subplot(len_num, 4, 2)
    spike_image_rm_dark_trunc_A_norm = (spike_image_rm_dark_trunc_A - spike_image_rm_dark_trunc_A.min()) / (
            spike_image_rm_dark_trunc_A.max() - spike_image_rm_dark_trunc_A.min())
    spike_image_rm_dark_trunc_A_hist = cv2.equalizeHist((spike_image_rm_dark_trunc_A_norm * 255).astype(np.uint8))
    plt.imshow(spike_image_rm_dark_trunc_A_hist, cmap='gray')
    plt.axis('off')
    plt.title('rm_dark_trunc_A_hist d_{:.2f}'.format(spike_density), fontsize=6)

    plt.subplot(len_num, 4, 3)
    plt.imshow(spike_image_rm_dark_trunc_A_mask, cmap='gray')
    plt.axis('off')
    plt.title('mask p_{:.4f}'.format(mask_prop), fontsize=6)

    plt.subplot(len_num, 4, 5)
    _image_sample = image_sample - image_sample.min()
    image_sample_gamma = np.power(_image_sample, 1 / 2.2)
    plt.imshow(image_sample_gamma, cmap='gray')
    plt.axis('off')
    plt.title('image_sample_gamma', fontsize=6)

    plt.subplot(len_num, 4, 6)
    plt.imshow(image_sample, cmap='gray')
    plt.axis('off')
    plt.title('image_sample', fontsize=6)

    plt.subplot(len_num, 4, 7)
    image_sample_norm = (image_sample - image_sample.min()) / \
                        (image_sample.max() - image_sample.min())
    image_sample_norm_hist = cv2.equalizeHist((image_sample_norm * 255).astype(np.uint8))
    plt.imshow(image_sample_norm_hist, cmap='gray')
    plt.axis('off')
    plt.title('image_sample_hist', fontsize=6)

    for i, x_0 in enumerate(x_0_list):
        plt.subplot(len_num, 4, i + 9)
        _x_0 = x_0 - x_0.min()
        x_0_gamma = np.power(_x_0, 1 / 2.2)
        plt.imshow(x_0_gamma, cmap='gray')
        plt.axis('off')
        plt.title('x0_{} c_{:.4f}-m_{:.4f}'.format(
            i + 1, clipiqa_x0_list[i], musiq_x0_list[i]), fontsize=5)

    for i, x_t in enumerate(x_t_list):
        _x_t = x_t - x_t.min()
        x_t_gamma = np.power(_x_t, 1 / 2.2)
        plt.subplot(len_num, 4, i + 4 * (2 + math.ceil(len(x_0_list) / 4)) + 1)
        plt.imshow(x_t_gamma, cmap='gray')
        plt.axis('off')
        plt.title('xt_{}'.format(i + 1), fontsize=6)

    title = "(I{:06d}_N{:02d})-c_{:.4f}-m_{:.4f}".format(
        iter_step + 1, index, clipiqa, musiq)
    plt.suptitle(title, fontsize=8)
    plt.savefig(os.path.join(save_root, title + ".png"), bbox_inches='tight', pad_inches=0, dpi=400)
    plt.show()
    plt.close()


def vis_sample_real(args, spike_image, image_sample, iter_step, index, clipiqa, musiq):
    save_root = os.path.join(args['path']['vis_val_real_dir'], 'I{:06d}'.format(iter_step + 1))

    os.makedirs(save_root, exist_ok=True)

    _image_sample = image_sample - image_sample.min()
    image_sample_gamma = np.power(_image_sample, 1 / 2.2)
    plt.imshow(image_sample_gamma, cmap='gray')
    plt.axis('off')
    title = "(I{:06d}_N{:02d})-sample-c_{:.4f}-m_{:.4f}".format(
        iter_step + 1, index, clipiqa, musiq)
    plt.savefig(os.path.join(save_root, title + ".png"), bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
    plt.close()

    spike_image_gamma = np.power(spike_image, 1 / 2.2)
    plt.imshow(spike_image_gamma, cmap='gray')
    plt.axis('off')
    title = "(I{:06d}_N{:02d})-spike_image".format(iter_step + 1, index)
    plt.savefig(os.path.join(save_root, title + ".png"), bbox_inches='tight', pad_inches=0, dpi=200)
    plt.show()
    plt.close()
