import math
import torch
from torch import autograd as autograd
from math import exp
from torch import nn as nn
from torch.nn import functional as F
import numpy as np
import torch.fft
from asyncfusion.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')

def RGB2YCrCb(rgb_image, with_CbCr=False):
    R, G, B = (rgb_image[:, 0:1, :], rgb_image[:, 1:2, :], rgb_image[:, 2:3, :])
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = -0.169 * R - 0.331 * G + 0.5 * B + 128 / 255.0
    Cr = 0.5 * R - 0.419 * G - 0.081 * B + 128 / 255.0
    Y, Cb, Cr = (Y.clamp(0.0, 1.0), Cb.clamp(0.0, 1.0), Cr.clamp(0.0, 1.0))
    if with_CbCr:
        return (Y, torch.cat([Cb, Cr], dim=1))
    return (Y, Cb, Cr)

@LOSS_REGISTRY.register()
class TemporalConsistency_loss(nn.Module):

    def __init__(self):
        super(TemporalConsistency_loss, self).__init__()

    def forward(self, V_f, V_ir):
        assert V_f.shape == V_ir.shape, 'The shapes of V_f and V_ir must be the same.'
        diff_f = V_f[:, 1:] - V_f[:, :-1]
        diff_ir = V_ir[:, 1:] - V_ir[:, :-1]
        loss = F.l1_loss(diff_f, diff_ir)
        return loss

@LOSS_REGISTRY.register()
class Fidelity_loss(nn.Module):

    def __init__(self):
        super(Fidelity_loss, self).__init__()
        print('Using Fidelity_loss() as loss function ~')
        self.grad_operator = Gradient()
        self.loss_func = nn.L1Loss(reduction='mean')

    def forward(self, img_rec, img_ref, type='ir'):
        Y_rec, _, _ = RGB2YCrCb(img_rec)
        Y_ref, _, _ = RGB2YCrCb(img_ref)
        grad_rec_x, grad_rec_y = self.grad_operator(Y_rec)
        grad_ref_x, grad_ref_y = self.grad_operator(Y_ref)
        loss_intensity = 10 * self.loss_func(img_rec, img_ref)
        loss_grad = 10 * self.loss_func(grad_rec_x, grad_ref_x) + 10 * self.loss_func(grad_rec_y, grad_ref_y)
        loss_fidelity = loss_intensity + loss_grad
        return {f'{type}_loss_intensity': loss_intensity, f'{type}_loss_grad': loss_grad, f'{type}_loss_fid': loss_fidelity}

@LOSS_REGISTRY.register()
class Fusion_loss(nn.Module):

    def __init__(self):
        super(Fusion_loss, self).__init__()
        print('Building Fusion_loss() as loss function ~')
        self.loss_func_ssim = L_SSIM(window_size=13)
        self.loss_func_grad = L_Gradient_Max()
        self.loss_func_max = L_Intensity_Max()
        self.loss_func_consist = L_Intensity_Consist()
        self.loss_func_color = L_Color()

    def forward(self, img_f, img_ir, img_vi, int_weight=15, consist_weight=1, ssim_ir_weight=1, ssim_weight=1, ir_compose=0.8, color_weight=100, grad_weight=1, fft_weight=0.05, regular=False):
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(img_vi)
        Y_ir, _, _ = RGB2YCrCb(img_ir)
        Y_f, Cb_f, Cr_f = RGB2YCrCb(img_f)
        loss_ssim = ssim_weight * (self.loss_func_ssim(img_vi, img_f) + ssim_ir_weight * self.loss_func_ssim(Y_ir, Y_f))
        loss_max = int_weight * self.loss_func_max(Y_f, Y_vi, Y_ir)
        loss_consist = consist_weight * self.loss_func_consist(Y_f, Y_vi, Y_ir, ir_compose)
        loss_color = color_weight * self.loss_func_color(Cb_f, Cr_f, Cb_vi, Cr_vi)
        loss_grad = grad_weight * self.loss_func_grad(Y_f, Y_vi, Y_ir)
        total_loss = loss_ssim + loss_max + loss_consist + loss_color + loss_grad
        return {'loss_intensity_max': loss_max, 'loss_color': loss_color, 'loss_grad': loss_grad, 'loss_intensity_consist': loss_consist, 'loss_ssim': loss_ssim, 'loss_fusion': total_loss}

class L_Intensity_Max(nn.Module):

    def __init__(self):
        super(L_Intensity_Max, self).__init__()

    def forward(self, img_f, img_vi, img_ir):
        img_max = torch.max(img_vi, img_ir)
        return F.l1_loss(img_max, img_f)

class L_Color(nn.Module):

    def __init__(self):
        super(L_Color, self).__init__()

    def forward(self, Cb_f, Cr_f, Cb_vi, Cr_vi):
        return F.l1_loss(Cb_f, Cb_vi) + F.l1_loss(Cr_f, Cr_vi)

class L_Intensity_Consist(nn.Module):

    def __init__(self):
        super(L_Intensity_Consist, self).__init__()

    def forward(self, img_f, img_vi, img_ir, ir_compose, consist_mode='l1'):
        loss_func = F.mse_loss if consist_mode == 'l2' else F.l1_loss
        return loss_func(img_vi, img_f) + ir_compose * loss_func(img_ir, img_f)

class L_Gradient_Max(nn.Module):

    def __init__(self):
        super(L_Gradient_Max, self).__init__()
        self.grad_operator = Gradient()

    def forward(self, img_f, img_vi, img_ir):
        grad_f_x, grad_f_y = self.grad_operator(img_f)
        grad_vi_x, grad_vi_y = self.grad_operator(img_vi)
        grad_ir_x, grad_ir_y = self.grad_operator(img_ir)
        loss = F.l1_loss(grad_f_x, torch.max(grad_vi_x, grad_ir_x)) + F.l1_loss(grad_f_y, torch.max(grad_vi_y, grad_ir_y))
        return loss

class Gradient(nn.Module):

    def __init__(self):
        super(Gradient, self).__init__()
        self.sobel_x = nn.Parameter(torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3), requires_grad=False)
        self.sobel_y = nn.Parameter(torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).view(1, 1, 3, 3), requires_grad=False)

    def forward(self, img):
        img = F.pad(img, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(img, self.sobel_x)
        grad_y = F.conv2d(img, self.sobel_y)
        return (torch.abs(grad_x), torch.abs(grad_y))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=24, window=None, size_average=True, val_range=None):
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1
        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range
    padd = 0
    _, channel, height, width = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)
    ssim_map = (2 * mu1_mu2 + C1) * v1 / ((mu1_sq + mu2_sq + C1) * v2)
    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)
    return 1 - ret

class L_SSIM(torch.nn.Module):

    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(L_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        _, channel, _, _ = img1.size()
        _, channel_2, _, _ = img2.size()
        if channel != channel_2 and channel == 1:
            img1 = torch.concat([img1, img1, img1], dim=1)
            channel = 3
        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device).type(img1.dtype)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

@LOSS_REGISTRY.register()
class L1Loss(nn.Module):

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)
