import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .confidence import _conf_at

class ConfidenceWeightedTemporalModeling(nn.Module):

    def __init__(self, in_channels=3, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=(1, 1, 1)):
        super(ConfidenceWeightedTemporalModeling, self).__init__()
        self.module = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding), nn.LeakyReLU(negative_slope=0.2, inplace=True), nn.Conv3d(out_channels, out_channels, kernel_size, stride, padding))
        self.kt = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size

    def forward(self, x, weight=None):
        if weight is None:
            return self.module(x)
        b, c, t, h, w = x.shape
        m = _conf_at(weight, b, t, h, w, x.dtype).clamp(min=0.001)
        m = rearrange(m, 'b t c hh ww -> b c t hh ww')
        ones = torch.ones(1, 1, self.kt, 1, 1, device=x.device, dtype=x.dtype)
        pad = (self.kt // 2, 0, 0)
        cnt_full = F.conv3d(torch.ones_like(m), ones, padding=pad)
        cnt_real = F.conv3d(m, ones, padding=pad)
        ratio = cnt_full / cnt_real.clamp(min=1e-06)
        h1 = self.module[0](x * m) * ratio
        h1 = self.module[1](h1)
        return self.module[2](h1)
CWTM = ConfidenceWeightedTemporalModeling
