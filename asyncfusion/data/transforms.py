import cv2
import random
import torch
import numpy as np

def paired_random_crop_multi(img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs, gt_patch_size, scale, gt_path=None):
    if not isinstance(img_ir_gts, list):
        img_ir_gts = [img_ir_gts]
    if not isinstance(img_ir_lqs, list):
        img_ir_lqs = [img_ir_lqs]
    if not isinstance(img_vi_gts, list):
        img_vi_gts = [img_vi_gts]
    if not isinstance(img_vi_lqs, list):
        img_vi_lqs = [img_vi_lqs]
    input_type = 'Tensor' if torch.is_tensor(img_ir_gts[0]) else 'Numpy'
    if input_type == 'Tensor':
        h_lq_ir, w_lq_ir = img_ir_lqs[0].size()[-2:]
        h_gt_ir, w_gt_ir = img_ir_gts[0].size()[-2:]
        h_lq_vi, w_lq_vi = img_vi_lqs[0].size()[-2:]
        h_gt_vi, w_gt_vi = img_vi_gts[0].size()[-2:]
    else:
        h_lq_ir, w_lq_ir = img_ir_lqs[0].shape[0:2]
        h_gt_ir, w_gt_ir = img_ir_gts[0].shape[0:2]
        h_lq_vi, w_lq_vi = img_vi_lqs[0].shape[0:2]
        h_gt_vi, w_gt_vi = img_vi_gts[0].shape[0:2]
    lq_patch_size = gt_patch_size // scale
    if h_gt_ir != h_lq_ir * scale or w_gt_ir != w_lq_ir * scale:
        raise ValueError(f'Scale mismatches for IR. GT ({h_gt_ir}, {w_gt_ir}) is not {scale}x multiplication of LQ ({h_lq_ir}, {w_lq_ir}).')
    if h_gt_vi != h_lq_vi * scale or w_gt_vi != w_lq_vi * scale:
        raise ValueError(f'Scale mismatches for VI. GT ({h_gt_vi}, {w_gt_vi}) is not {scale}x multiplication of LQ ({h_lq_vi}, {w_lq_vi}).')
    if h_lq_ir < lq_patch_size or w_lq_ir < lq_patch_size:
        raise ValueError(f'LQ IR ({h_lq_ir}, {w_lq_ir}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')
    if h_lq_vi < lq_patch_size or w_lq_vi < lq_patch_size:
        raise ValueError(f'LQ VI ({h_lq_vi}, {w_lq_vi}) is smaller than patch size ({lq_patch_size}, {lq_patch_size}). Please remove {gt_path}.')
    top = random.randint(0, h_lq_ir - lq_patch_size)
    left = random.randint(0, w_lq_ir - lq_patch_size)
    if input_type == 'Tensor':
        img_ir_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_ir_lqs]
        img_vi_lqs = [v[:, :, top:top + lq_patch_size, left:left + lq_patch_size] for v in img_vi_lqs]
    else:
        img_ir_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_ir_lqs]
        img_vi_lqs = [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in img_vi_lqs]
    top_gt_ir, left_gt_ir = (int(top * scale), int(left * scale))
    top_gt_vi, left_gt_vi = (int(top * scale), int(left * scale))
    if input_type == 'Tensor':
        img_ir_gts = [v[:, :, top_gt_ir:top_gt_ir + gt_patch_size, left_gt_ir:left_gt_ir + gt_patch_size] for v in img_ir_gts]
        img_vi_gts = [v[:, :, top_gt_vi:top_gt_vi + gt_patch_size, left_gt_vi:left_gt_vi + gt_patch_size] for v in img_vi_gts]
    else:
        img_ir_gts = [v[top_gt_ir:top_gt_ir + gt_patch_size, left_gt_ir:left_gt_ir + gt_patch_size, ...] for v in img_ir_gts]
        img_vi_gts = [v[top_gt_vi:top_gt_vi + gt_patch_size, left_gt_vi:left_gt_vi + gt_patch_size, ...] for v in img_vi_gts]
    if len(img_ir_gts) == 1:
        img_ir_gts = img_ir_gts[0]
    if len(img_ir_lqs) == 1:
        img_ir_lqs = img_ir_lqs[0]
    if len(img_vi_gts) == 1:
        img_vi_gts = img_vi_gts[0]
    if len(img_vi_lqs) == 1:
        img_vi_lqs = img_vi_lqs[0]
    return (img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs)
