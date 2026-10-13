import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

def _gray(x):
    if x.shape[1] == 1:
        return x
    w = x.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x[:, :3] * w).sum(dim=1, keepdim=True)

@torch.no_grad()
def pyramid_flow(src, dst, levels=3, radius=2):
    n, _, h, w = src.shape
    flow = src.new_zeros(n, 2, h // 2 ** (levels - 1), w // 2 ** (levels - 1))
    for lvl in reversed(range(levels)):
        s = 2 ** lvl
        a = F.avg_pool2d(src, s) if s > 1 else src
        bimg = F.avg_pool2d(dst, s) if s > 1 else dst
        hh, ww = a.shape[-2:]
        if flow.shape[-2:] != (hh, ww):
            flow = F.interpolate(flow, size=(hh, ww), mode='bilinear', align_corners=False) * 2.0
        a_w = warp_by_flow(a, flow)
        best = a_w.new_zeros(n, 2, hh, ww)
        best_cost = F.avg_pool2d((a_w - bimg).abs().mean(dim=1, keepdim=True), 5, stride=1, padding=2)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                shifted = torch.roll(bimg, shifts=(dy, dx), dims=(-2, -1))
                cost = (a_w - shifted).abs().mean(dim=1, keepdim=True)
                cost = F.avg_pool2d(cost, 5, stride=1, padding=2)
                cand = a_w.new_tensor([dx, dy]).view(1, 2, 1, 1).expand(n, 2, hh, ww)
                take = cost < best_cost
                best = torch.where(take, cand, best)
                best_cost = torch.where(take, cost, best_cost)
        flow = flow + best
    return flow

def warp_by_flow(x, flow):
    n, _, h, w = x.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=x.device, dtype=x.dtype), torch.arange(w, device=x.device, dtype=x.dtype), indexing='ij')
    gx = (gx.unsqueeze(0) + flow[:, 0]) / max(w - 1, 1) * 2 - 1
    gy = (gy.unsqueeze(0) + flow[:, 1]) / max(h - 1, 1) * 2 - 1
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)

@torch.no_grad()
def flow_reliability(src, dst, flow, thresh=1.0):
    back = pyramid_flow(dst, src)
    resid = (flow + warp_by_flow(back, flow)).norm(dim=1, keepdim=True)
    return 1.0 - (resid / thresh).clamp(0, 1)
