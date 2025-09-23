import math
from inspect import isfunction

import torch
from torch import nn


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


# PositionalEncoding Source: https://github.com/lmnt-com/wavegrad/blob/master/src/wavegrad/model.py
class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, con_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=con_level.dtype, device=con_level.device) / count
        encoding = con_level.unsqueeze(-1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat([torch.sin(encoding), torch.cos(encoding)], dim=-1)

        return encoding


class FeatureWiseAffine(nn.Module):
    def __init__(self, args, in_channels, out_channels):
        super(FeatureWiseAffine, self).__init__()
        self.args = args

        self.con_func = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x, con_emb):
        batch = x.shape[0]

        x = x + self.con_func(con_emb).view(batch, -1, 1, 1)

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
    def __init__(self, args, dim, dim_out, noise_level_emb_dim=None, dropout=0, norm_groups=32):
        super().__init__()
        self.args = args
        self.noise_func = FeatureWiseAffine(args, noise_level_emb_dim, dim_out)

        if args['experiment']['use_spike_density_value']:
            self.spike_density_func = FeatureWiseAffine(args, noise_level_emb_dim, dim_out)

        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb, spike_density_emb=None):
        h = self.block1(x)
        h = self.noise_func(h, time_emb)
        h = self.block2(h)

        if spike_density_emb is not None:
            h = self.spike_density_func(h, spike_density_emb)

        h = h + self.res_conv(x)

        return h


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=1, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)

    def forward(self, x):
        batch, channel, height, width = x.shape
        n_head = self.n_head
        head_dim = channel // n_head

        x = self.norm(x)
        qkv = self.qkv(x).view(batch, n_head, head_dim * 3, height, width)
        query, key, value = qkv.chunk(3, dim=2)  # bhdyx

        attn = torch.einsum(
            "bnchw, bncyx -> bnhwyx", query, key
        ).contiguous() / math.sqrt(channel)
        attn = attn.view(batch, n_head, height, width, -1)
        attn = torch.softmax(attn, -1)
        attn = attn.view(batch, n_head, height, width, height, width)

        out = torch.einsum("bnhwyx, bncyx -> bnchw", attn, value).contiguous()
        out = self.out(out.view(batch, channel, height, width))

        return out + x


class ResnetBlocWithAttn(nn.Module):
    def __init__(self, args, dim, dim_out, *, noise_level_emb_dim=None,
                 norm_groups=32, with_self_attn=False, with_cross_attn=False, n_head=1, dropout=0):
        super().__init__()
        self.with_self_attn = with_self_attn
        self.with_cross_attn = with_cross_attn

        self.res_block = ResnetBlock(args, dim, dim_out, noise_level_emb_dim, norm_groups=norm_groups, dropout=dropout)
        if with_self_attn:
            self.self_attn = SelfAttention(dim_out, n_head=n_head, norm_groups=norm_groups)

    def forward(self, x, time_emb, spike_density_emb=None):
        x = self.res_block(x, time_emb, spike_density_emb=spike_density_emb)
        if self.with_self_attn:
            x = self.self_attn(x)

        return x


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, activation='relu', norm=None):
        super(ConvLayer, self).__init__()

        bias = False if norm == 'BN' else True
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        if activation is not None:
            self.activation = getattr(torch, activation, 'relu')
        else:
            self.activation = None

        self.norm = norm
        if norm == 'BN':
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == 'IN':
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):
        out = self.conv2d(x)

        if self.norm in ['BN', 'IN']:
            out = self.norm_layer(out)

        if self.activation is not None:
            out = self.activation(out)

        return out


class UNet(nn.Module):
    def __init__(self, args, in_channel=1, out_channel=1, inner_channel=32,
                channel_mults=(1, 2, 4, 8, 8), res_blocks=3, attn_ind=(2, 3, 4),
                norm_groups=32, dropout=0):
        super().__init__()

        self.args = args

        noise_level_channel = inner_channel
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(inner_channel),
            nn.Linear(inner_channel, inner_channel * 4),
            Swish(),
            nn.Linear(inner_channel * 4, inner_channel)
        )

        # spike_density
        if args['experiment']['use_spike_density_value']:
            self.spike_density_level_mlp = nn.Sequential(
                PositionalEncoding(inner_channel),
                nn.Linear(inner_channel, inner_channel * 4),
                Swish(),
                nn.Linear(inner_channel * 4, inner_channel)
            )

        num_mults = len(channel_mults)
        self.num_mults = num_mults

        pre_channel = inner_channel
        feat_channels = [pre_channel]

        # downs
        downs = [nn.Conv2d(in_channel, inner_channel, kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (ind in attn_ind)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(ResnetBlocWithAttn(
                    args, pre_channel, channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, with_self_attn=use_attn, dropout=dropout))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
        self.downs = nn.ModuleList(downs)

        # mid
        self.mid = nn.ModuleList([
            ResnetBlocWithAttn(args, pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, with_self_attn=True, dropout=dropout),
            ResnetBlocWithAttn(args, pre_channel, pre_channel, noise_level_emb_dim=noise_level_channel,
                               norm_groups=norm_groups, with_self_attn=False, with_cross_attn=False, dropout=dropout)
        ])

        # ups
        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (ind in attn_ind)
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(ResnetBlocWithAttn(
                    args, pre_channel+feat_channels.pop(), channel_mult, noise_level_emb_dim=noise_level_channel,
                    norm_groups=norm_groups, with_self_attn=use_attn, with_cross_attn=False, dropout=dropout))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight.data, gain=1)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)

    def forward(self, x_t, time, spike_density=None):
        t_emb = self.noise_level_mlp(time)
        if self.args['experiment']['use_spike_density_value']:
            spike_density_emb = self.spike_density_level_mlp(spike_density)
        else:
            spike_density_emb = None

        x = x_t

        # UNet
        unet_feats = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t_emb, spike_density_emb=spike_density_emb)
            else:
                x = layer(x)
            unet_feats.append(x)

        for layer in self.mid:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(x, t_emb, spike_density_emb=spike_density_emb)
            else:
                x = layer(x)

        i = 0
        for layer in self.ups:
            if isinstance(layer, ResnetBlocWithAttn):
                x = layer(torch.cat((x, unet_feats.pop()), dim=1), t_emb,
                          spike_density_emb=spike_density_emb)
                i += 1
            else:
                x = layer(x)

        return self.final_conv(x)


def unet_model(args):
    model = UNet(args,
        in_channel=args['unet']['in_channel'], out_channel=args['unet']['out_channel'],
        inner_channel=args['unet']['inner_channel'], channel_mults=args['unet']['channel_multiplier'],
        res_blocks=args['unet']['res_blocks'], attn_ind=args['unet']['attn_ind'],
        norm_groups=args['unet']['norm_groups'], dropout=args['unet']['dropout']
    )

    return model
