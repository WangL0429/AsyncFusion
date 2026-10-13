import copy
import json
from pathlib import Path
import torch
import torch.utils.data as data
from asyncfusion.data.fusion_video_dataset import FusionVideoDataset
from asyncfusion.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class AsyncFusionDataset(data.Dataset):

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.conditions = opt['async_conditions']
        self.num_frame = opt['num_frame']
        self.subs, self.lags = ([], [])
        for cond_dir in self.conditions:
            cond_dir = Path(cond_dir)
            sub_opt = copy.deepcopy(dict(opt))
            sub_opt['dataroot_lq_ir'] = str(cond_dir / opt['async_ir_lq_mod'])
            sub_opt['dataroot_lq_vi'] = str(cond_dir / opt['async_vi_lq_mod'])
            sub_opt['dataroot_gt_ir'] = opt['dataroot_gt_ir']
            sub_opt['dataroot_gt_vi'] = opt['dataroot_gt_vi']
            sub_opt.pop('async_conditions', None)
            self.subs.append(FusionVideoDataset(sub_opt))
            payload = json.loads((cond_dir / 'schedule.json').read_text())
            self.lags.append({clip: {e['n']: abs(e['n'] - e['ir_used']) for e in entries} for clip, entries in payload['clips'].items()})
        self.per_sub = [len(s) for s in self.subs]

    def __len__(self):
        return sum(self.per_sub)

    def __getitem__(self, index):
        k = 0
        while index >= self.per_sub[k]:
            index -= self.per_sub[k]
            k += 1
        out = self.subs[k][index]
        t = out['lq_ir'].shape[0]
        clip_name, center_idx, frame_num = out['key'].split('/')
        per_clip = self.lags[k][clip_name]
        out['ir_lag'] = torch.tensor([float(per_clip[int(Path(name).stem)]) for name in out['frame_names']], dtype=torch.float32)
        out['condition'] = str(self.conditions[k])
        return out
