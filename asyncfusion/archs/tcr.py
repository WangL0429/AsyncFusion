import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from .confidence import _conf_at
from .transformer_blocks import Co_TransformerBlock

class TemporalContextRefinementBlock(nn.Module):

    def __init__(self, in_channels=192, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', embed_dim=64, group=4):
        super(TemporalContextRefinementBlock, self).__init__()
        self.co_transformer = Co_TransformerBlock(dim=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group)

    def forward(self, x, conf=None):
        b, t, c, h, w = x.shape
        cf = _conf_at(conf, b, t, h, w, x.dtype)
        updated_frames = []
        for i in range(t):
            curr_frame = x[:, i, :, :, :]
            if i == 0:
                prev_frame = curr_frame
                next_frame = x[:, i + 1, :, :, :]
                pi, ni = (i, i + 1)
            elif i == t - 1:
                prev_frame = x[:, i - 1, :, :, :]
                next_frame = curr_frame
                pi, ni = (i - 1, i)
            else:
                prev_frame = x[:, i - 1, :, :, :]
                next_frame = x[:, i + 1, :, :, :]
                pi, ni = (i - 1, i + 1)
            assert prev_frame.shape == curr_frame.shape == next_frame.shape, f'Shape mismatch: {prev_frame.shape}, {curr_frame.shape}, {next_frame.shape}'
            cp = cf[:, pi] if cf is not None else None
            cn = cf[:, ni] if cf is not None else None
            updated_frame = self.co_transformer(curr_frame, prev_frame, next_frame, cp, cn)
            updated_frames.append(updated_frame)
        updated_x = torch.stack(updated_frames, dim=1)
        return updated_x

class TemporalContextRefinement(nn.Module):

    def __init__(self, in_channels=192, num_layers=4, num_heads=4, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias', embed_dim=64, group=4):
        super(TemporalContextRefinement, self).__init__()
        self.layers = nn.ModuleList([TemporalContextRefinementBlock(in_channels=in_channels, num_heads=num_heads, ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type, embed_dim=embed_dim, group=group) for _ in range(num_layers)])

    def forward(self, x, conf=None):
        for layer in self.layers:
            x = layer(x, conf)
        return x
TCR = TemporalContextRefinement
