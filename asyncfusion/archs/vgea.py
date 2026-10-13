import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .confidence import _conf_at
from .motion_util import warp_by_flow
from .transformer_blocks import Cross_TransformerBlock

class VisibleGuidedEvidenceAggregation(nn.Module):

    def __init__(self, in_channels, num_heads, ffn_expansion_factor, bias, LayerNorm_type, embed_dim, group):
        super(VisibleGuidedEvidenceAggregation, self).__init__()
        self.cross = Cross_TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group)
        self.gate = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, f_ir, f_vi, conf, b, t, flow=None, gate=None):
        if conf is None and flow is None:
            return f_ir
        ir = rearrange(f_ir, '(b t) c h w -> b t c h w', b=b, t=t)
        vi = rearrange(f_vi, '(b t) c h w -> b t c h w', b=b, t=t)
        h, w = ir.shape[-2:]
        if flow is not None and gate is not None:
            fl = rearrange(flow, 'b t c hh ww -> (b t) c hh ww')
            gt = rearrange(gate, 'b t c hh ww -> (b t) c hh ww')
            if fl.shape[-2:] != (h, w):
                sy, sx = (h / fl.shape[-2], w / fl.shape[-1])
                fl = F.interpolate(fl, size=(h, w), mode='bilinear', align_corners=False)
                fl = torch.stack((fl[:, 0] * sx, fl[:, 1] * sy), dim=1)
                gt = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=False)
            aligned = warp_by_flow(f_ir, fl)
            f_ir = f_ir + self.beta * gt * (aligned - f_ir)
            ir = rearrange(f_ir, '(b t) c h w -> b t c h w', b=b, t=t)
        if conf is None:
            return f_ir
        cf = _conf_at(conf, b, t, h, w, ir.dtype)
        out = []
        for i in range(t):
            ctx = vi[:, i]
            borrowed = self.cross(ir[:, i], ctx)
            need = 1.0 - cf[:, i]
            out.append(ir[:, i] + self.gate * need * borrowed)
        return rearrange(torch.stack(out, dim=1), 'b t c h w -> (b t) c h w')
VGEA = VisibleGuidedEvidenceAggregation
