import numpy as np
import os
from PIL import Image
import copy
import cv2
import matplotlib.pyplot as plt

from torch.utils.data import Dataset

from dataset.cal_fpn import cal_fpn_params


class SpikeDataset(Dataset):
    def __init__(self, args):

        self.args = args
        self.A, self.Ld_div_q = cal_fpn_params(args)
        self.window_length = args['dataset']['window_length']
        self.threshold = args['dataset']['threshold']
        self.height, self.width = args['dataset']['height'], args['dataset']['width']

    def find_start(self, spike_frames, window_length):
        flag_map = np.zeros((self.height, self.width))
        start_map = np.zeros((self.height, self.width))

        index = 0
        while (len(flag_map[flag_map == 0]) != 0):
            spike_frame = spike_frames[index]
            start_map[(spike_frame == 1) & (flag_map == 0)] = index
            flag_map[spike_frame == 1] = 1

            index += 1
            if index == window_length:
                start_map[flag_map == 0] = index
                break

        return start_map

    def find_end(self, spike_frames, window_length):
        flag_map = np.zeros((self.height, self.width))
        end_map = np.zeros((self.height, self.width))

        index = window_length - 1
        while (len(flag_map[flag_map == 0]) != 0):
            spike_frame = spike_frames[index]
            end_map[(spike_frame == 1) & (flag_map == 0)] = index
            flag_map[spike_frame == 1] = 1

            index -= 1
            if index == -1:
                end_map[flag_map == 0] = index
                break

        return end_map


class SpikeDatasetSim(SpikeDataset):
    def __init__(self, args, is_train=True):
        super().__init__(args)
        self.is_train = is_train
        self.light_scale = args['dataset']['light_scale']

        if is_train:
            self.dataset_root = args['dataset']['dataset_train_root']
            self.video_frames_list = []
            dataset_dir_list = sorted(os.listdir(self.dataset_root))

            for dataset_dir in dataset_dir_list:
                image_list_per_class = sorted(os.listdir(os.path.join(self.dataset_root, dataset_dir)))
                for image_name in image_list_per_class:
                    self.video_frames_list.append(
                        os.path.join(self.dataset_root, dataset_dir, image_name))
            self.len = len(self.video_frames_list)
        else:
            self.dataset_root = os.path.join(args['dataset']['dataset_val_root'])
            val_images_list = sorted(os.listdir(self.dataset_root))
            self.val_images_list = []

            for val_images in val_images_list:
                self.val_images_list.append(os.path.join(self.dataset_root, val_images))
            self.len = len(self.val_images_list)

    def __getitem__(self, index):
        if self.is_train:
            if self.args['experiment']['fpn_rand']:
                A = self.A
                A_flattened = A.flatten()
                np.random.shuffle(A_flattened)
                A = A_flattened.reshape(self.A.shape)

                Ld_div_q = self.Ld_div_q
                Ld_div_q_flattened = Ld_div_q.flatten()
                np.random.shuffle(Ld_div_q_flattened)
                Ld_div_q = Ld_div_q_flattened.reshape(self.Ld_div_q.shape)

            else:
                A = self.A
                Ld_div_q = self.Ld_div_q

            video_frame_path = self.video_frames_list[index]
            image = Image.open(video_frame_path).convert("L")
            image = np.array(image)

            # crop
            image = image[image.shape[0] // 7: image.shape[0] * 7 // 8, image.shape[1] // 8: image.shape[1] * 7 // 8]
            if np.random.randint(0, 2) == 0:
                # 裁剪小，后放大
                factor = np.random.randint(50 // 1.5, 50)
                crop_height = factor * 5
                crop_width = factor * 8
                start_height = np.random.randint(0, image.shape[0] - crop_height)
                start_width = np.random.randint(0, image.shape[1] - crop_width)
                image = image[start_height:start_height + crop_height,
                        start_width:start_width + crop_width]
                image = cv2.resize(image, (self.width, self.height),
                                   interpolation=cv2.INTER_LANCZOS4)
            else:
                # 裁剪大，后缩小
                factor = np.random.randint(50, min(50 * 1.5, image.shape[0] // 5, image.shape[1] // 8))
                crop_height = factor * 5
                crop_width = factor * 8
                start_height = np.random.randint(0, image.shape[0] - crop_height)
                start_width = np.random.randint(0, image.shape[1] - crop_width)
                image = image[start_height:start_height + crop_height,
                        start_width:start_width + crop_width]
                image = cv2.resize(image, (self.width, self.height),
                                   interpolation=cv2.INTER_AREA)

            if image.mean() == 0:
                image += 50

            # gamma
            gamma_scale_1 = np.random.uniform(self.args['dataset']['gamma_scale_range1'][0],
                                              self.args['dataset']['gamma_scale_range1'][1])
            image_gamma = np.power(image, 1 / gamma_scale_1)
            image_gamma = image_gamma * image.astype(np.float32).mean() / image_gamma.mean()

            # gradient
            if np.random.randint(0, 2) == 0:
                gradient_type = np.random.choice(['diagonal', 'circle'])

                if gradient_type == 'diagonal':
                    x_rand = np.random.randint(self.width // 4, self.width // 2)
                    y_rand = np.random.randint(self.height // 4, self.height // 2)
                    if np.random.randint(0, 2) == 0:
                        angle = np.random.uniform(0, 90)
                        x = np.concatenate((np.zeros((x_rand)),
                                            np.linspace(0, self.threshold, self.width - x_rand)))
                        y = np.concatenate((np.zeros((y_rand)),
                                            np.linspace(0, self.threshold, self.height - y_rand)))
                    else:
                        angle = np.random.uniform(180, 270)
                        x = np.concatenate((np.linspace(0, self.threshold, self.width - x_rand),
                                            np.ones((x_rand)) * self.threshold))
                        y = np.concatenate((np.linspace(0, self.threshold, self.height - y_rand),
                                            np.ones((y_rand)) * self.threshold))

                    radian = np.deg2rad(angle)
                    X, Y = np.meshgrid(x, y)
                    gradient_image = X * np.cos(radian) + Y * np.sin(radian)

                elif gradient_type == 'circle':
                    center_x = np.random.uniform(0, self.width)
                    center_y = np.random.uniform(0, self.height)
                    radius = np.random.uniform(self.height / 2, self.width / 2)
                    X, Y = np.meshgrid(np.arange(self.width),
                                       np.arange(self.height))
                    distance = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
                    gradient_image = np.clip(self.threshold - (distance / radius) * self.threshold, 0, self.threshold)

                else:
                    raise ValueError

                _range = np.random.uniform(self.threshold / 2, self.threshold)
                gradient_image = (gradient_image - gradient_image.min()) / (gradient_image.max() - gradient_image.min())
                gradient_image = (gradient_image * _range).astype(np.float32)

                image_add_gradient = image_gamma.astype(np.float32) + gradient_image
                image_add_gradient = image_add_gradient * image.astype(np.float32).mean() / image_add_gradient.mean()

                # plt.imshow(gradient_image, cmap='gray')
                # plt.axis('off')
                # plt.show()
                # plt.close()

                gamma_scale_2 = np.random.uniform(self.args['dataset']['gamma_scale_range2'][0],
                                                  self.args['dataset']['gamma_scale_range2'][1])
                image_add_gradient_gamma = np.power(image_add_gradient, 1 / gamma_scale_2)
                image_add_gradient_gamma = image_add_gradient_gamma * image.astype(
                    np.float32).mean() / image_add_gradient_gamma.mean()

                image = image_add_gradient_gamma

            else:
                gradient_image = np.zeros((self.height, self.width))
                image = image_gamma

            light_intensity = self.light_scale / self.threshold

            image = image * light_intensity
            gt_image = image

            A_rand = np.random.normal(loc=1, scale=0.02, size=A.shape)
            dark_rand = np.random.uniform(0.95, 1.05) + np.clip(
                np.random.normal(loc=0, scale=0.27, size=(Ld_div_q).shape) ** 9, -3, 3)
            dark_rand = np.clip(dark_rand, 0, dark_rand.max())

            image_possion = np.random.poisson(image * self.window_length)
            integrator_init = np.random.random(
                (self.height, self.width)) * self.threshold
            integrator_sum = image_possion * (A * A_rand) + \
                             (Ld_div_q * dark_rand) * self.threshold * self.window_length

            # -------------------- spike_image --------------------
            spike_image = np.floor(
                (integrator_sum + integrator_init) / self.threshold) * self.threshold / self.window_length
            spike_image = np.maximum(spike_image, np.zeros(spike_image.shape))

            # -------------------- spike_image_ws --------------------
            spike_image_ws = np.zeros(integrator_sum.shape)

            # s=0时，直接赋为 th/w+1
            spike_image_ws[spike_image * self.window_length == 0] = self.threshold / (self.window_length + 1)
            # print("s0:", len(spike_image_ws[spike_image * self.window_length == 0]))

            # s=1时，取 th/max(n,w-n)
            integrator_sum[integrator_sum < 1] = 1
            spike_image_ws[np.round(spike_image * self.window_length) == self.threshold] = \
                (self.threshold / (self.window_length *
                                   np.maximum(self.threshold - integrator_init,
                                              integrator_sum + integrator_init - self.threshold) / integrator_sum))[
                    np.round(spike_image * self.window_length) == self.threshold]
            # print("s1:", len(spike_image_ws[np.round(spike_image * self.window_length) == self.threshold]))

            # s=2时，加强泊松噪声
            interval_avg = np.ones(integrator_sum.shape) * self.window_length * self.threshold / integrator_sum
            interval_num = np.maximum(np.floor((integrator_sum + integrator_init) / self.threshold) - 1, 1)
            window_length_randw = interval_avg * interval_num
            integrator_sum_randw = np.random.poisson(image * window_length_randw) * (A * A_rand) + \
                                   (Ld_div_q * dark_rand) * self.threshold * window_length_randw
            spike_image_ws[np.round(spike_image * self.window_length) > self.threshold] = (
                    integrator_sum_randw / window_length_randw)[np.round(spike_image * self.window_length) > self.threshold]
            # print("s>1:", len(spike_image_ws[np.round(spike_image * self.window_length) > self.threshold]))

            Ld_div_q = Ld_div_q * self.threshold

        else:
            val_images = np.load(self.val_images_list[index])
            if val_images.shape[0] == 6:
                gt_image, gradient_image, spike_image, spike_image_ws, A, Ld_div_q = val_images[0], val_images[1], \
                    val_images[2], val_images[3], val_images[4], val_images[5]
            else:
                gt_image, spike_frames = val_images[0], val_images[1: self.window_length + 1]
                spike_image = spike_frames.mean(axis=0) * self.threshold

                start_map = self.find_start(spike_frames, self.window_length)
                end_map = self.find_end(spike_frames, self.window_length)
                length_map = end_map - start_map

                spike_frames_sum = spike_frames.sum(axis=0) - 1
                spike_frames_sum[length_map < 0] = 1
                spike_frames_sum[length_map == 0] = 1

                length_map[length_map < 0] = self.window_length + 1
                length_map[length_map == 0] = np.maximum(start_map[length_map == 0], self.window_length - end_map[length_map == 0]) + 1

                spike_image_ws = (spike_frames_sum / length_map) * self.threshold

                gradient_image = np.zeros((self.height, self.width))
                A, Ld_div_q = self.A, self.Ld_div_q * self.threshold

        trunc_diff = spike_image - spike_image_ws

        # -------------------- spike_density --------------------
        spike_image_rm_dark_trunc_A = (spike_image_ws - Ld_div_q) / A
        spike_image_rm_dark_trunc_A_mask = copy.deepcopy(spike_image_rm_dark_trunc_A)
        add_trunc_noise_num = len(
            spike_image_rm_dark_trunc_A_mask[np.round(spike_image * self.window_length) <= self.threshold])
        spike_image_rm_dark_trunc_A_mask[np.round(spike_image * self.window_length) <= self.threshold] = 0
        spike_image_rm_dark_trunc_A_mask[np.round(spike_image * self.window_length) > self.threshold] = 1

        mask_prop = add_trunc_noise_num / (self.height * self.width)
        add_trunc_noise_dense = self.args['dataset']['n'] * mask_prop / self.window_length
        spike_density = (spike_image_rm_dark_trunc_A.mean() - add_trunc_noise_dense).astype(np.float32)

        gt_image = np.pad(gt_image, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        gradient_image = np.pad(gradient_image, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        spike_image = np.pad(spike_image, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        spike_image_rm_dark_trunc_A = np.pad(spike_image_rm_dark_trunc_A, ((3, 3), (0, 0)), mode="reflect").astype(
            np.float32)
        spike_image_rm_dark_trunc_A_mask = np.pad(spike_image_rm_dark_trunc_A_mask, ((3, 3), (0, 0)),
                                                  mode="reflect").astype(np.float32)
        trunc_diff = np.pad(trunc_diff, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        A = np.pad(A, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        Ld_div_q = np.pad(Ld_div_q, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)

        scale_factor = self.args['dataset']['scale_factor'] / spike_density
        spike_image_scale = spike_image * scale_factor
        spike_image_rm_dark_trunc_A_scale = spike_image_rm_dark_trunc_A * scale_factor
        spike_image_rm_dark_trunc_A_mask_scale = spike_image_rm_dark_trunc_A_mask * scale_factor
        trunc_diff_scale = trunc_diff * scale_factor
        gt_image_scale = gt_image * scale_factor
        Ld_div_q = Ld_div_q * scale_factor

        data = {
            'spike_image': spike_image_scale[np.newaxis, :, :],
            'spike_image_rm_dark_trunc_A': spike_image_rm_dark_trunc_A_scale[np.newaxis, :, :],
            'spike_image_rm_dark_trunc_A_mask': spike_image_rm_dark_trunc_A_mask_scale[np.newaxis, :, :],
            'trunc_diff': trunc_diff_scale[np.newaxis, :, :],
            'gt_image': gt_image_scale[np.newaxis, :, :],
            'gradient_image': gradient_image[np.newaxis, :, :],
            'spike_density': spike_density,
            'mask_prop': mask_prop,
            "A": A[np.newaxis, :, :],
            "Ld_div_q": Ld_div_q[np.newaxis, :, :]
        }

        return data

    def __len__(self):
        return self.len


class SpikeDatasetReal(SpikeDataset):
    def __init__(self, args):
        super().__init__(args)
        self.dataset_root = os.path.join(args['dataset']['dataset_real_root'])

        self.spike_file_list = []
        spike_file_list = sorted(os.listdir(self.dataset_root))
        for spike_file in spike_file_list:
            self.spike_file_list.append(os.path.join(self.dataset_root, spike_file))

        self.len = len(self.spike_file_list)

    def __getitem__(self, index):
        spike_file_path = self.spike_file_list[index]

        spike_frames = np.load(spike_file_path)[:self.window_length]
        spike_image = spike_frames.mean(axis=0) * self.threshold

        start_map = self.find_start(spike_frames, self.window_length)
        end_map = self.find_end(spike_frames, self.window_length)
        length_map = end_map - start_map

        spike_frames_sum = spike_frames.sum(axis=0) - 1
        spike_frames_sum[length_map < 0] = 1
        spike_frames_sum[length_map == 0] = 1

        length_map[length_map < 0] = self.window_length + 1
        length_map[length_map == 0] = np.maximum(start_map[length_map == 0],
                                                 self.window_length - end_map[length_map == 0]) + 1

        spike_image_ws = (spike_frames_sum / length_map) * self.threshold

        trunc_diff = spike_image - spike_image_ws

        spike_image_rm_dark_trunc_A = (spike_image_ws - self.Ld_div_q * self.threshold) / self.A

        spike_image_rm_dark_trunc_A_mask = copy.deepcopy(spike_image_rm_dark_trunc_A)
        add_trunc_noise_num = len(
            spike_image_rm_dark_trunc_A_mask[np.round(spike_image * self.window_length) <= self.threshold])
        spike_image_rm_dark_trunc_A_mask[np.round(spike_image * self.window_length) <= self.threshold] = 0
        spike_image_rm_dark_trunc_A_mask[np.round(spike_image * self.window_length) > self.threshold] = 1

        mask_prop = add_trunc_noise_num / (self.height * self.width)
        add_trunc_noise_dense = self.args['dataset']['n'] * mask_prop / self.window_length
        spike_density = (spike_image_rm_dark_trunc_A.mean() - add_trunc_noise_dense).astype(np.float32)

        spike_image = np.pad(spike_image, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        spike_image_rm_dark_trunc_A = np.pad(spike_image_rm_dark_trunc_A, ((3, 3), (0, 0)), mode="reflect").astype(
            np.float32)
        spike_image_rm_dark_trunc_A_mask = np.pad(spike_image_rm_dark_trunc_A_mask, ((3, 3), (0, 0)),
                                                  mode="reflect").astype(np.float32)
        trunc_diff = np.pad(trunc_diff, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        A = np.pad(self.A, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)
        Ld_div_q = np.pad(self.Ld_div_q * self.threshold, ((3, 3), (0, 0)), mode="reflect").astype(np.float32)

        scale_factor = self.args['dataset']['scale_factor'] / spike_density
        spike_image_scale = spike_image * scale_factor
        spike_image_rm_dark_trunc_A_scale = spike_image_rm_dark_trunc_A * scale_factor
        spike_image_rm_dark_trunc_A_mask_scale = spike_image_rm_dark_trunc_A_mask * scale_factor
        trunc_diff_scale = trunc_diff * scale_factor
        Ld_div_q = Ld_div_q * scale_factor

        data = {
            'spike_image': spike_image_scale[np.newaxis, :, :],
            'spike_image_rm_dark_trunc_A': spike_image_rm_dark_trunc_A_scale[np.newaxis, :, :],
            'spike_image_rm_dark_trunc_A_mask': spike_image_rm_dark_trunc_A_mask_scale[np.newaxis, :, :],
            'trunc_diff': trunc_diff_scale[np.newaxis, :, :],
            'spike_density': spike_density,
            'mask_prop': mask_prop,
            'A': A[np.newaxis, :, :],
            'Ld_div_q': Ld_div_q[np.newaxis, :, :]
        }

        return data

    def __len__(self):
        return self.len
