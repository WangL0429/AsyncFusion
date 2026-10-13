import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import os
from natsort import natsorted
from asyncfusion.data.transforms import paired_random_crop_multi
from asyncfusion.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from asyncfusion.utils.registry import DATASET_REGISTRY

def _frame_names(directory):
    suffixes = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    return natsorted((p.name for p in Path(directory).iterdir() if p.is_file() and (not p.name.startswith('.')) and (p.suffix.lower() in suffixes)))

@DATASET_REGISTRY.register()
class FusionVideoDataset(data.Dataset):

    def __init__(self, opt):
        super(FusionVideoDataset, self).__init__()
        self.opt = opt
        self.ir_gt_root = Path(opt['dataroot_gt_ir'])
        self.ir_lq_root = Path(opt['dataroot_lq_ir'])
        self.vi_gt_root = Path(opt['dataroot_gt_vi'])
        self.vi_lq_root = Path(opt['dataroot_lq_vi'])
        self.num_frame = opt['num_frame']
        self.keys = []
        folder_list = natsorted((p.name for p in self.ir_lq_root.iterdir() if p.is_dir() and (not p.name.startswith('.'))))
        for folder in folder_list:
            sub_ir_gt_folder = self.ir_gt_root / folder
            sub_ir_lq_folder = self.ir_lq_root / folder
            sub_vi_gt_folder = self.vi_gt_root / folder
            sub_vi_lq_folder = self.vi_lq_root / folder
            if not (sub_ir_gt_folder.exists() and sub_ir_lq_folder.exists() and sub_vi_gt_folder.exists() and sub_vi_lq_folder.exists()):
                print(f'[Skip] Missing folder under: {folder}')
                continue
            ir_gt_count = len(_frame_names(sub_ir_gt_folder))
            ir_lq_count = len(_frame_names(sub_ir_lq_folder))
            vi_gt_count = len(_frame_names(sub_vi_gt_folder))
            vi_lq_count = len(_frame_names(sub_vi_lq_folder))
            if not ir_gt_count == ir_lq_count == vi_gt_count == vi_lq_count:
                print(f"Inconsistent file counts in folder '{folder}': IR_GT={ir_gt_count}, IR_LQ={ir_lq_count}, VI_GT={vi_gt_count}, VI_LQ={vi_lq_count}")
                continue
            file_list = _frame_names(sub_ir_lq_folder)
            frame_num = len(file_list)
            for i in range(frame_num):
                self.keys.append(f'{folder}/{i}/{frame_num}')
        self.file_client = None
        self.io_backend_opt = opt['io_backend'].copy()
        self.interval_list = opt.get('interval_list', [1])
        self.random_reverse = opt.get('random_reverse', False)
        interval_str = ','.join((str(x) for x in self.interval_list))
        logger = get_root_logger()
        logger.info(f'Temporal augmentation interval list: [{interval_str}]; random reverse is {self.random_reverse}.')

    def __getitem__(self, index):
        if self.file_client is None:
            backend_type = self.io_backend_opt.get('type')
            io_kwargs = {k: v for k, v in self.io_backend_opt.items() if k != 'type'}
            self.file_client = FileClient(backend_type, **io_kwargs)
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        key = self.keys[index]
        clip_name, center_idx, frame_num = key.split('/')
        center_idx = int(center_idx)
        frame_num = int(frame_num)
        interval = random.choice(self.interval_list)
        start_frame_idx = center_idx
        if start_frame_idx > frame_num - self.num_frame:
            start_frame_idx = random.randint(0, max(0, frame_num - self.num_frame))
        end_frame_idx = start_frame_idx + self.num_frame
        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))
        if len(neighbor_list) < self.num_frame:
            neighbor_list = list(range(start_frame_idx, start_frame_idx + self.num_frame))
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()
        img_ir_lqs, img_vi_lqs = ([], [])
        img_ir_gts, img_vi_gts = ([], [])
        lq_names = _frame_names(self.ir_lq_root / clip_name)
        for neighbor in neighbor_list:
            frame_name = lq_names[neighbor]
            img_ir_lq_path = os.path.join(str(self.ir_lq_root), clip_name, frame_name)
            img_vi_lq_path = os.path.join(str(self.vi_lq_root), clip_name, frame_name)
            img_ir_gt_path = os.path.join(str(self.ir_gt_root), clip_name, frame_name)
            img_vi_gt_path = os.path.join(str(self.vi_gt_root), clip_name, frame_name)
            ir_lq_bytes = self.file_client.get(img_ir_lq_path, 'lq')
            vi_lq_bytes = self.file_client.get(img_vi_lq_path, 'lq')
            img_ir_lqs.append(imfrombytes(ir_lq_bytes, float32=True))
            img_vi_lqs.append(imfrombytes(vi_lq_bytes, float32=True))
            ir_gt_bytes = self.file_client.get(img_ir_gt_path, 'gt')
            vi_gt_bytes = self.file_client.get(img_vi_gt_path, 'gt')
            img_ir_gts.append(imfrombytes(ir_gt_bytes, float32=True))
            img_vi_gts.append(imfrombytes(vi_gt_bytes, float32=True))
        img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs = paired_random_crop_multi(img_ir_gts, img_ir_lqs, img_vi_gts, img_vi_lqs, gt_size, scale, img_ir_gt_path)
        data_len = len(img_ir_lqs)
        img_ir_lqs.extend(img_ir_gts)
        img_vi_lqs.extend(img_vi_gts)
        img_ir_results = img_ir_lqs
        img_vi_results = img_vi_lqs
        img_ir_results = img2tensor(img_ir_results)
        img_vi_results = img2tensor(img_vi_results)
        img_ir_lqs = torch.stack(img_ir_results[:data_len], dim=0)
        img_ir_gts = torch.stack(img_ir_results[data_len:data_len * 2], dim=0)
        img_vi_lqs = torch.stack(img_vi_results[:data_len], dim=0)
        img_vi_gts = torch.stack(img_vi_results[data_len:data_len * 2], dim=0)
        return {'lq_ir': img_ir_lqs, 'lq_vi': img_vi_lqs, 'gt_ir': img_ir_gts, 'gt_vi': img_vi_gts, 'key': key, 'frame_names': [lq_names[i] for i in neighbor_list]}

    def __len__(self):
        return len(self.keys)
