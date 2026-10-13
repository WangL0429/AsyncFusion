import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from asyncfusion.utils import get_root_logger
import datetime

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)

def make_layer(basic_block, num_basic_block, **kwarg):
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ResidualBlockNoBN3D(nn.Module):

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN3D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlockNoBN2D(nn.Module):

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN2D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale

class ResidualBlockNoBN(nn.Module):

    def __init__(self, num_feat=64, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out

class ResidualBlockNoBN_DW(nn.Module):

    def __init__(self, num_feat=64, pytorch_init=False):
        super(ResidualBlockNoBN_DW, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 2, 3, 1, 1, bias=True, groups=num_feat * 2)
        self.conv3 = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv3(self.lrelu(self.conv2(self.lrelu(self.conv1(x)))))
        return identity + out

class Upsample(nn.Sequential):

    def __init__(self, scale, num_feat):
        m = []
        if scale & scale - 1 == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    assert x.size()[-2:] == flow.size()[1:3]
    b, c, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()
    grid.requires_grad = False
    vgrid = grid + flow
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)
    return output

def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = (int(flow_h * sizes[0]), int(flow_w * sizes[1]))
    elif size_type == 'shape':
        output_h, output_w = (sizes[0], sizes[1])
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')
    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow

def pixel_unshuffle(x, scale):
    b, c, hh, hw = x.size()
    out_channel = c * scale ** 2
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)

class PixelShufflePack(nn.Module):

    def __init__(self, in_channels, out_channels, scale_factor, upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(self.in_channels, self.out_channels * scale_factor * scale_factor, self.upsample_kernel, padding=(self.upsample_kernel - 1) // 2)
        default_init_weights([self.upsample_conv], 1)

    def forward(self, x):
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x

class ResidualBlocksWithInputConv(nn.Module):

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()
        main = []
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        main.append(make_layer(ResidualBlockNoBN, num_blocks, num_feat=out_channels))
        self.main = nn.Sequential(*main)

    def forward(self, feat):
        return self.main(feat)
