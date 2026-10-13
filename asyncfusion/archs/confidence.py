import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def _conf_at(conf, b, t, h, w, dtype):
    if conf is None:
        return None
    conf = conf.to(dtype)
    if conf.dim() == 2:
        return conf.view(conf.shape[0], conf.shape[1], 1, 1, 1).expand(b, t, 1, h, w)
    c = rearrange(conf, 'b t c hh ww -> (b t) c hh ww')
    if c.shape[-2:] != (h, w):
        c = F.interpolate(c, size=(h, w), mode='bilinear', align_corners=False)
    return rearrange(c, '(b t) c h w -> b t c h w', b=b, t=t)
