import math
import lpips
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

lpips_metric = lpips.LPIPS(net='alex')


class GaussianDiffusion(nn.Module):
    def __init__(self, args, model, loss_type='l1',
                 Alpha_schedule_opt=None, alpha_schedule_opt=None, beta_schedule_opt=None, gamma_schedule_opt=None):
        super().__init__()

        self.args = args
        self.model = model

        self.window_length = args['dataset']['window_length']

        self.Alpha_schedule_opt = Alpha_schedule_opt
        self.alpha_schedule_opt = alpha_schedule_opt
        self.beta_schedule_opt = beta_schedule_opt
        self.gamma_schedule_opt = gamma_schedule_opt

        self.loss_type = loss_type
        self.set_loss()

    def set_loss(self):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(self.args['device'])
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(self.args['device'])
        else:
            raise ValueError

    def init_base_params(self):
        self.T = self.args['diffusion']['n_timestep']
        self.register_buffer('k', torch.tensor(self.args['diffusion']['k']))  # k

    def init_ratio_params(self, A, ratio_schedule_opt, start_value=0.001, end_value=0.999):
        n_timestep = self.T
        schedule = ratio_schedule_opt['schedule']
        power = ratio_schedule_opt['p']

        b, h, w = self.args['dataloader']['batch_size'], self.args['dataset']['height'] + 6, self.args['dataset'][
            'width']

        if schedule == 'exp':
            power_timestep = torch.linspace(0, 1, n_timestep + 1, dtype=torch.float64)[1:].to(
                self.args['device']) ** power  # t1 = (t / T) ^ p
            # power_timestep_minus_one = torch.linspace(0, 1, n_timestep + 1, dtype=torch.float64)[:-1].to(
            #     self.args['device']) ** power  # t2 = (t-1 / T) ^ p

            a = torch.pow(A.unsqueeze(-1).repeat(1, 1, 1, 1, n_timestep), power_timestep)  # a(t)
            a_prev = torch.cat((torch.ones([b, 1, h, w, 1], dtype=torch.float64,
                device=self.args['device']), a[:, :, :, :, :-1]), dim=-1)  # a(t-1)
            # Alphas = torch.pow(A.unsqueeze(-1).repeat(1, 1, 1, 1, n_timestep),
            #                    (power_timestep - power_timestep_minus_one))  # A(t) = A^(t1 - t2)
            Alphas = a / a_prev
            Alphas_cumprod = a
            Alphas_cumprod_prev = torch.cat(
                (torch.ones(Alphas_cumprod.shape[:-1], device=self.args['device']).unsqueeze(-1),
                 Alphas_cumprod[:, :, :, :, :-1]), dim=-1)  # A-(t-1)

        else:
            raise ValueError

        return Alphas, Alphas_cumprod, Alphas_cumprod_prev

    def init_residual_params(self, residual_schedule_opt, start_value=0.001, end_value=0.999):
        n_timestep = self.T
        schedule = residual_schedule_opt['schedule']
        power = residual_schedule_opt['p']

        b, h, w = self.args['dataloader']['batch_size'], self.args['dataset']['height'] + 6, self.args['dataset'][
            'width']

        if schedule == 'resshift' or schedule == 'resshift_sqrt':
            increaser = math.exp(
                1 / (n_timestep - 1) * math.log(end_value / start_value))  # b0 = e^(1 / (T-1) * log(/nT / /n1))
            base = torch.ones([n_timestep, ], dtype=torch.float64, device=self.args['device']) * increaser  # b0
            power_timestep = torch.linspace(0, 1, n_timestep).to(self.args['device']) ** power  # (t-1 / T-1)^p
            power_timestep *= (n_timestep - 1)  # βt = (t-1 / T-1)^p * (T-1)
            sqrt_etas = torch.pow(base, power_timestep) * start_value  # /nt = /n1 * bo^βt

            sqrt_etas = sqrt_etas.repeat(b, 1, h, w, 1)  # /n(t)

        else:
            increaser = math.exp(
                1 / (n_timestep - 1) * math.log(start_value / end_value))  # b0 = e^(1 / (T-1) * log(/nT / /n1))
            base = torch.ones([n_timestep, ], dtype=torch.float64, device=self.args['device']) * increaser  # b0
            power_timestep = torch.linspace(1, 0, n_timestep).to(self.args['device']) ** power  # (t-1 / T-1)^p
            power_timestep *= (n_timestep - 1)  # βt = (t-1 / T-1)^p * (T-1)
            sqrt_etas = torch.pow(base, power_timestep) * end_value  # /nt = /n1 * bo^βt

            sqrt_etas = sqrt_etas.repeat(b, 1, h, w, 1)  # /n(t)

        if schedule == 'resshift' or schedule == 'resshift_reverse':
            etas = sqrt_etas ** 2

            etas_prev = torch.cat((torch.zeros(etas.shape[:-1], device=self.args['device']).unsqueeze(-1),
                 etas[:, :, :, :, :-1]), dim=-1) # n(t-1)

            if self.args['diffusion']['algorithm'] == 'a' or self.args['diffusion']['algorithm'] == 'ab':
                alphas = etas - etas_prev  # a(t) = n(t) - n(t-1)
            else:
                alphas = etas - self.Alphas * etas_prev

            alphas_cumsum = etas  # a-(t)
            alphas_cumsum_prev = torch.cat(
                (torch.zeros(alphas_cumsum.shape[:-1], device=self.args['device']).unsqueeze(-1),
                 alphas_cumsum[:, :, :, :, :-1]), dim=-1)

        else:
            sqrt_etas_prev = torch.cat((torch.zeros(sqrt_etas.shape[:-1], device=self.args['device']).unsqueeze(-1),
                sqrt_etas[:, :, :, :, :-1]), dim=-1)  # /n(t-1)

            if self.args['diffusion']['algorithm'] == 'a' or self.args['diffusion']['algorithm'] == 'ab':
                alphas = sqrt_etas - sqrt_etas_prev  # a(t) = /n(t) - /n(t-1)
            else:
                alphas = sqrt_etas - self.Alphas * sqrt_etas_prev

            alphas_cumsum = sqrt_etas  # a-(t)
            alphas_cumsum_prev = torch.cat(
                (torch.zeros(alphas_cumsum.shape[:-1], device=self.args['device']).unsqueeze(-1),
                 alphas_cumsum[:, :, :, :, :-1]), dim=-1)

        return alphas, alphas_cumsum, alphas_cumsum_prev

    def init_a_sample_params(self):
        self.register_buffer('posterior_deno',
                             self.gammas ** 2 + self.gammas_cumsum_prev ** 2)  # deno(t) = γ(t)^2 + γ-(t-1)^2

        self.register_buffer('posterior_var',
                             self.k ** 2 * (self.gammas ** 2 * self.gammas_cumsum_prev ** 2) / self.posterior_deno)  # k^2 * γ(t)^2 * γ-(t-1)^2 / deno
        self.register_buffer('posterior_var_clipped', torch.cat((self.posterior_var[:, :, :, :, 1:2],
                                                                 self.posterior_var[:, :, :, :, 1:]), dim=-1))
        self.register_buffer('posterior_log_var_clipped', torch.log(self.posterior_var_clipped))

        self.register_buffer('posterior_mean_coef1', self.gammas_cumsum_prev ** 2 / self.posterior_deno)  # γ-(t-1)^2 / deno
        self.register_buffer('posterior_mean_coef2', self.gammas ** 2 / self.posterior_deno)  # γ(t)^2 / deno
        self.register_buffer('posterior_mean_coef3', (self.alphas_cumsum_prev * self.gammas ** 2 -
                                                      self.alphas * self.gammas_cumsum_prev ** 2) / self.posterior_deno)  # (α(t-1)*γ(t)^2 - α(t)*γ(t-1)^2) / deno)

    def init_ab_sample_params(self):
        self.register_buffer('posterior_deno',
                             self.gammas_cumsum_prev ** 2 + self.gammas ** 2)  # deno(t) = γ-(t-1)^2 + γ(t)^2

        self.register_buffer('posterior_var',
                             self.k ** 2 * self.gammas ** 2 * self.gammas_cumsum_prev ** 2 / self.posterior_deno)  # k^2 * γ(t)^2 * γ-(t-1)^2 / deno(t))
        self.register_buffer('posterior_var_clipped', torch.cat((self.posterior_var[:, :, :, :, 1:2],
                                                                 self.posterior_var[:, :, :, :, 1:]), dim=-1))
        self.register_buffer('posterior_log_var_clipped', torch.log(self.posterior_var_clipped))

        self.register_buffer('posterior_mean_coef1',
                             self.gammas_cumsum_prev ** 2 / self.posterior_deno)  # γ-(t-1)^2 / deno(t)
        self.register_buffer('posterior_mean_coef2',
                             self.gammas ** 2 / self.posterior_deno)  # γ(t)^2 / deno(t)
        self.register_buffer('posterior_mean_coef3',
                             (self.alphas_cumsum_prev * self.gammas ** 2 - self.alphas * self.gammas_cumsum_prev ** 2) /
                             self.posterior_deno)  # a-(t-1) * γ(t)^2 - a(t) * γ-(t-1)^2 / deno(t)
        self.register_buffer('posterior_mean_coef4',
                             (self.betas_cumsum_prev * self.gammas ** 2 - self.betas * self.gammas_cumsum_prev ** 2) /
                             self.posterior_deno)  # β-(t-1) * γ(t)^2 - β(t) * γ-(t-1)^2 / deno(t)

    def init_Aa_sample_params(self):
        self.register_buffer('posterior_deno',
                             self.Alphas ** 2 * self.gammas_cumsum_prev ** 2 + self.gammas ** 2)  # deno(t) = A(t)^2 * γ-(t-1)^2 + γ(t)^2

        self.register_buffer('posterior_var',
                             self.k ** 2 * self.gammas ** 2 * self.gammas_cumsum_prev ** 2 / self.posterior_deno)  # k^2 * γ(t)^2 * γ-(t-1)^2 / deno(t))
        self.register_buffer('posterior_var_clipped', torch.cat((self.posterior_var[:, :, :, :, 1:2],
                                                                 self.posterior_var[:, :, :, :, 1:]), dim=-1))
        self.register_buffer('posterior_log_var_clipped', torch.log(self.posterior_var_clipped))

        self.register_buffer('posterior_mean_coef1',
                             self.Alphas * self.gammas_cumsum_prev ** 2 / self.posterior_deno)  # A(t) * γ-(t-1)^2 / deno(t)
        self.register_buffer('posterior_mean_coef2',
                             self.Alphas_cumprod_prev * self.gammas ** 2 / self.posterior_deno)  # A-(t-1) * γ(t)^2 / deno(t)
        self.register_buffer('posterior_mean_coef3',
                             (self.alphas_cumsum_prev * self.gammas ** 2 - self.Alphas * self.alphas * self.gammas_cumsum_prev ** 2) /
                             self.posterior_deno)  # a-(t-1) * γ(t)^2 - A(t) * a(t) * γ-(t-1)^2 / deno(t)

    def init_Aab_sample_params(self):
        self.register_buffer('posterior_deno',
                             self.Alphas ** 2 * self.gammas_cumsum_prev ** 2 + self.gammas ** 2)  # deno(t) = A(t)^2 * γ-(t-1)^2 + γ(t)^2

        self.register_buffer('posterior_var',
                             self.k ** 2 * self.gammas ** 2 * self.gammas_cumsum_prev ** 2 / self.posterior_deno)  # k^2 * γ(t)^2 * γ-(t-1)^2 / deno(t))
        self.register_buffer('posterior_var_clipped', torch.cat((self.posterior_var[:, :, :, :, 1:2],
                                                                 self.posterior_var[:, :, :, :, 1:]), dim=-1))
        self.register_buffer('posterior_log_var_clipped', torch.log(self.posterior_var_clipped))

        self.register_buffer('posterior_mean_coef1',
                             self.Alphas * self.gammas_cumsum_prev ** 2 / self.posterior_deno)  # A(t) * γ-(t-1)^2 / deno(t)
        self.register_buffer('posterior_mean_coef2',
                             self.Alphas_cumprod_prev * self.gammas ** 2 / self.posterior_deno)  # A-(t-1) * γ(t)^2 / deno(t)
        self.register_buffer('posterior_mean_coef3',
                             (self.alphas_cumsum_prev * self.gammas ** 2 - self.Alphas * self.alphas * self.gammas_cumsum_prev ** 2) /
                             self.posterior_deno)  # a-(t-1) * γ(t)^2 - A(t) * a(t) * γ-(t-1)^2 / deno(t)
        self.register_buffer('posterior_mean_coef4',
                             (self.betas_cumsum_prev * self.gammas ** 2 - self.Alphas * self.betas * self.gammas_cumsum_prev ** 2) /
                             self.posterior_deno)  # β-(t-1) * γ(t)^2 - A(t) * β(t) * γ-(t-1)^2 / deno(t)

    def extract(self, v, t, x_shape):
        h, w = x_shape[-2], x_shape[-1]
        out = torch.gather(v, index=t.repeat(1, 1, h, w, 1).permute(4, 1, 2, 3, 0), dim=-1).squeeze(-1).float()

        return out

    def train_forward(self, spike_image, spike_image_rm_dark_trunc_A_mask,
                      gt_image, trunc_diff, spike_density, A, Ld_div_q):
        self.init_base_params()
        if self.args['diffusion']['algorithm'] == 'a':
            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

        elif self.args['diffusion']['algorithm'] == 'ab':
            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            betas, betas_cumsum, betas_cumsum_prev = self.init_residual_params(self.beta_schedule_opt)
            self.register_buffer('betas', betas)  # β(t)
            self.register_buffer('betas_cumsum', betas_cumsum)  # β-(t)
            self.register_buffer('betas_cumsum_prev', betas_cumsum_prev)  # β-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

        elif self.args['diffusion']['algorithm'] == 'Aa':
            Alphas, Alphas_cumprod, Alphas_cumprod_prev = self.init_ratio_params(A, self.Alpha_schedule_opt)
            self.register_buffer('Alphas', Alphas)  # A(t)
            self.register_buffer('Alphas_cumprod', Alphas_cumprod)  # A-(t)
            self.register_buffer('Alphas_cumprod_prev', Alphas_cumprod_prev)  # A-(t-1)

            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

        elif self.args['diffusion']['algorithm'] == 'Aab':
            Alphas, Alphas_cumprod, Alphas_cumprod_prev = self.init_ratio_params(A, self.Alpha_schedule_opt)
            self.register_buffer('Alphas', Alphas)  # A(t)
            self.register_buffer('Alphas_cumprod', Alphas_cumprod)  # A-(t)
            self.register_buffer('Alphas_cumprod_prev', Alphas_cumprod_prev)  # A-(t-1)

            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            betas, betas_cumsum, betas_cumsum_prev = self.init_residual_params(self.beta_schedule_opt)
            self.register_buffer('betas', betas)  # β(t)
            self.register_buffer('betas_cumsum', betas_cumsum)  # β-(t)
            self.register_buffer('betas_cumsum_prev', betas_cumsum_prev)  # β-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

        else:
            raise ValueError

        x_0 = gt_image
        y = spike_image

        noise = torch.randn_like(x_0, device=x_0.device)
        t = torch.randint(self.T, size=(x_0.shape[0],), device=x_0.device)

        scale_factor = self.args['dataset']['scale_factor'] / spike_density.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        spike_density = (0.1 / spike_density).float()

        if self.args['diffusion']['algorithm'] == 'a':
            x_t = x_0 + self.extract(self.alphas_cumsum, t, x_0.shape) * (y - x_0) + \
                  self.extract(self.k * self.gammas_cumsum, t, x_0.shape) * noise
        elif self.args['diffusion']['algorithm'] == 'ab':
            if self.args['diffusion']['dark_add_trunc']:
                x_t = x_0 + self.extract(self.alphas_cumsum, t, x_0.shape) * (Ld_div_q + trunc_diff) + \
                      self.extract(self.betas_cumsum, t, x_0.shape) * (y - x_0 - (Ld_div_q + trunc_diff)) + \
                      self.extract(self.k * self.gammas_cumsum, t, x_0.shape) * noise
            else:
                x_t = x_0 + self.extract(self.alphas_cumsum, t, x_0.shape) * Ld_div_q + \
                      self.extract(self.betas_cumsum, t, x_0.shape) * (y - A * x_0 - Ld_div_q) + \
                      self.extract(self.k * self.gammas_cumsum, t, x_0.shape) * noise
        elif self.args['diffusion']['algorithm'] == 'Aa':
            x_t = self.extract(self.Alphas_cumprod, t, x_0.shape) * x_0 + \
                  self.extract(self.alphas_cumsum, t, x_0.shape) * (y - A * x_0) + \
                  self.extract(self.k * self.gammas_cumsum, t, x_0.shape) * noise
        elif self.args['diffusion']['algorithm'] == 'Aab':
            if self.args['diffusion']['dark_add_trunc']:
                x_t = self.extract(self.Alphas_cumprod, t, x_0.shape) * x_0 + \
                      self.extract(self.alphas_cumsum, t, x_0.shape) * (Ld_div_q + trunc_diff) + \
                      self.extract(self.betas_cumsum, t, x_0.shape) * (y - A * x_0 - (Ld_div_q + trunc_diff)) + \
                      self.extract(self.k * self.gammas_cumsum, t, x_0.shape) * noise
            else:
                x_t = self.extract(self.Alphas_cumprod, t, x_0.shape) * x_0 + \
                      self.extract(self.alphas_cumsum, t, x_0.shape) * Ld_div_q + \
                      self.extract(self.betas_cumsum, t, x_0.shape) * (y - A * x_0 - Ld_div_q) + \
                      self.extract(self.k * self.gammas_cumsum, t, x_0.shape) * noise
        else:
            raise ValueError

        x_t_ = x_t

        if self.args['experiment']['use_rm_dark_trunc_A_mask_con']:
            x_t_ = torch.cat((x_t_, spike_image_rm_dark_trunc_A_mask), dim=1)

            # _spike_image_rm_dark_trunc_A_mask = spike_image_rm_dark_trunc_A_mask[0][0].detach().cpu().numpy()
            # _spike_image_rm_dark_trunc_A_mask = _spike_image_rm_dark_trunc_A_mask - _spike_image_rm_dark_trunc_A_mask.min()
            # _spike_image_rm_dark_trunc_A_mask_gamma = np.power(_spike_image_rm_dark_trunc_A_mask, 1 / 3)
            # plt.imshow(_spike_image_rm_dark_trunc_A_mask_gamma, cmap='gray')
            # plt.axis('off')
            # plt.show()
            # plt.close()

        if self.args['experiment']['use_dark_con']:
            x_t_ = torch.cat((x_t_, Ld_div_q), dim=1)

            # _Ld_div_q = Ld_div_q[0][0].detach().cpu().numpy()
            # _Ld_div_q = _Ld_div_q - _Ld_div_q.min()
            # _Ld_div_q_gamma = np.power(_Ld_div_q, 1 / 3)
            # plt.imshow(_Ld_div_q_gamma, cmap='gray')
            # plt.axis('off')
            # plt.show()
            # plt.close()

        x_output = self.model(x_t_.float(), t, spike_density=spike_density)

        loss = {}
        loss['loss_diff'] = self.loss_func(gt_image, x_output)

        if self.args['experiment']['use_loss_lpips']:
            loss['loss_lpips'] = lpips_metric.to(self.args['device'])(gt_image, x_output).mean()

        return loss, x_output, x_t, t

    def sample_forward(self, spike_image, spike_image_rm_dark_trunc_A_mask,
                       trunc_diff, spike_density, A, Ld_div_q):
        self.init_base_params()
        if self.args['diffusion']['algorithm'] == 'a':
            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

            self.init_a_sample_params()

        elif self.args['diffusion']['algorithm'] == 'ab':
            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            betas, betas_cumsum, betas_cumsum_prev = self.init_residual_params(self.beta_schedule_opt)
            self.register_buffer('betas', betas)  # β(t)
            self.register_buffer('betas_cumsum', betas_cumsum)  # β-(t)
            self.register_buffer('betas_cumsum_prev', betas_cumsum_prev)  # β-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

            self.init_ab_sample_params()

        elif self.args['diffusion']['algorithm'] == 'Aa':
            Alphas, Alphas_cumprod, Alphas_cumprod_prev = self.init_ratio_params(A, self.Alpha_schedule_opt)
            self.register_buffer('Alphas', Alphas)  # A(t)
            self.register_buffer('Alphas_cumprod', Alphas_cumprod)  # A-(t)
            self.register_buffer('Alphas_cumprod_prev', Alphas_cumprod_prev)  # A-(t-1)

            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

            self.init_Aa_sample_params()

        elif self.args['diffusion']['algorithm'] == 'Aab':
            Alphas, Alphas_cumprod, Alphas_cumprod_prev = self.init_ratio_params(A, self.Alpha_schedule_opt)
            self.register_buffer('Alphas', Alphas)  # A(t)
            self.register_buffer('Alphas_cumprod', Alphas_cumprod)  # A-(t)
            self.register_buffer('Alphas_cumprod_prev', Alphas_cumprod_prev)  # A-(t-1)

            alphas, alphas_cumsum, alphas_cumsum_prev = self.init_residual_params(self.alpha_schedule_opt)
            self.register_buffer('alphas', alphas)  # a(t)
            self.register_buffer('alphas_cumsum', alphas_cumsum)  # a-(t)
            self.register_buffer('alphas_cumsum_prev', alphas_cumsum_prev)  # a-(t-1)

            betas, betas_cumsum, betas_cumsum_prev = self.init_residual_params(self.beta_schedule_opt)
            self.register_buffer('betas', betas)  # β(t)
            self.register_buffer('betas_cumsum', betas_cumsum)  # β-(t)
            self.register_buffer('betas_cumsum_prev', betas_cumsum_prev)  # β-(t-1)

            gammas, gammas_cumsum, gammas_cumsum_prev = self.init_residual_params(self.gamma_schedule_opt)
            self.register_buffer('gammas', gammas)  # γ(t)
            self.register_buffer('gammas_cumsum', gammas_cumsum)  # γ-(t)
            self.register_buffer('gammas_cumsum_prev', gammas_cumsum_prev)  # γ-(t-1)

            self.init_Aab_sample_params()
        else:
            raise ValueError

        y = spike_image

        spike_density = (0.1 / spike_density).float()

        noise = torch.randn_like(y)
        t = torch.ones(y.shape[0], dtype=torch.long, device=spike_image.device) * (self.T - 1)

        x_t = y + self.extract(self.k * self.gammas_cumsum, t, y.shape) * noise
        
        x_0_list = []
        x_t_list = []
        x_t_list.append(x_t)
        for time_step in reversed(range(self.T)):
            t = torch.ones(x_t.shape[0], dtype=torch.long, device=spike_image.device) * time_step

            with torch.no_grad():
                x_t_ = x_t

                if self.args['experiment']['use_rm_dark_trunc_A_mask_con']:
                    x_t_ = torch.cat((x_t_, spike_image_rm_dark_trunc_A_mask), dim=1)

                if self.args['experiment']['use_dark_con']:
                    x_t_ = torch.cat((x_t_, Ld_div_q), dim=1)
                
                x_0 = self.model(x_t_.float(), t, spike_density=spike_density)
                x_0_list.append(x_0)

                if self.args['diffusion']['algorithm'] == 'a':
                    posterior_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
                                     self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_0 + \
                                     self.extract(self.posterior_mean_coef3, t, x_t.shape) * (y - x_0)
                elif self.args['diffusion']['algorithm'] == 'ab':
                    if self.args['diffusion']['dark_add_trunc']:
                        posterior_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
                                         self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_0 + \
                                         self.extract(self.posterior_mean_coef3, t, x_t.shape) * (
                                                 Ld_div_q + trunc_diff) + \
                                         self.extract(self.posterior_mean_coef4, t, x_t.shape) * (
                                                 y - x_0 - (Ld_div_q + trunc_diff))
                    else:
                        posterior_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
                                         self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_0 + \
                                         self.extract(self.posterior_mean_coef3, t, x_t.shape) * Ld_div_q + \
                                         self.extract(self.posterior_mean_coef4, t, x_t.shape) * (
                                                 y - x_0 - Ld_div_q)
                elif self.args['diffusion']['algorithm'] == 'Aa':
                    posterior_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
                                     self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_0 + \
                                     self.extract(self.posterior_mean_coef3, t, x_t.shape) * (y - A * x_0)
                elif self.args['diffusion']['algorithm'] == 'Aab':
                    if self.args['diffusion']['dark_add_trunc']:
                        posterior_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
                                         self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_0 + \
                                         self.extract(self.posterior_mean_coef3, t, x_t.shape) * (
                                                 Ld_div_q + trunc_diff) + \
                                         self.extract(self.posterior_mean_coef4, t, x_t.shape) * (
                                                 y - A * x_0 - (Ld_div_q + trunc_diff))
                    else:
                        posterior_mean = self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_t + \
                                         self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_0 + \
                                         self.extract(self.posterior_mean_coef3, t, x_t.shape) * Ld_div_q + \
                                         self.extract(self.posterior_mean_coef4, t, x_t.shape) * (
                                                 y - A * x_0 - Ld_div_q)

                else:
                    raise ValueError

                posterior_log_var = self.extract(self.posterior_log_var_clipped, t, x_t.shape)

                # no noise when t == 0
                if time_step > 0:
                    noise = torch.randn_like(x_t)
                else:
                    noise = 0

                x_t = posterior_mean + torch.exp(0.5 * posterior_log_var) * noise
                if time_step > 0:
                    x_t_list.append(x_t)

        x_0 = x_t

        return x_0, x_0_list, x_t_list

    def forward(self, spike_image, spike_image_rm_dark_trunc_A_mask,
                trunc_diff, spike_density, A, Ld_div_q, phase='train', gt_image=None):
        if phase == 'train':
            return self.train_forward(spike_image, spike_image_rm_dark_trunc_A_mask,
                                      gt_image, trunc_diff, spike_density, A, Ld_div_q)

        elif phase == 'sample':
            return self.sample_forward(spike_image, spike_image_rm_dark_trunc_A_mask,
                                       trunc_diff, spike_density, A, Ld_div_q)

        else:
            raise ValueError


def gaussian_diffusion(args, model):
    model = GaussianDiffusion(args, model,
        Alpha_schedule_opt=args['diffusion']['Alpha_schedule'],
        alpha_schedule_opt=args['diffusion']['alpha_schedule'],
        beta_schedule_opt=args['diffusion']['beta_schedule'],
        gamma_schedule_opt=args['diffusion']['gamma_schedule'],
        loss_type=args['diffusion']['loss_type']
    )

    return model
