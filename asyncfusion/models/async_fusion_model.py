import torch
from collections import OrderedDict
from einops import rearrange
from asyncfusion.models.fusion_model import FusionModel
from asyncfusion.utils.registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class AsyncFusionModel(FusionModel):

    def init_training_settings(self):
        super().init_training_settings()
        train_opt = self.opt['train']
        self.w_oracle = train_opt.get('oracle_weight', 0.0)
        self.w_motion = train_opt.get('motion_weight', 0.0)
        self.motion_lambda = train_opt.get('motion_lambda', 10.0)
        self.net_oracle = None
        if self.w_oracle > 0:
            from copy import deepcopy
            from asyncfusion.archs import build_network
            self.net_oracle = build_network(deepcopy(self.opt['network_g']))
            self.net_oracle = self.net_oracle.to(self.device)
            path = self.opt['path'].get('pretrain_network_oracle', None)
            if not path:
                raise ValueError('Set path.pretrain_network_oracle to a compatible synchronous teacher checkpoint.')
            if path:
                self.load_network(self.net_oracle, path, self.opt['path'].get('strict_load_g', True), 'params')
            self.net_oracle.eval()
            for p in self.net_oracle.parameters():
                p.requires_grad = False

    def load_network(self, net, load_path, strict=True, param_key='params'):
        import torch as _torch
        obj = _torch.load(load_path, map_location=lambda s, l: s, weights_only=False)
        if param_key is not None and param_key in obj:
            obj = obj[param_key]
        obj = {k[7:] if k.startswith('module.') else k: v for k, v in obj.items()}
        bare = self.get_bare_model(net)
        cur = bare.state_dict()
        changed = False
        for k, v in list(obj.items()):
            if k in cur and v.numel() == 1 and (cur[k].numel() > 1):
                obj[k] = v.reshape(1).expand_as(cur[k]).clone()
                changed = True
        if changed:
            from asyncfusion.utils import get_root_logger
            get_root_logger().info('broadcast scalar lag_tau from the checkpoint to per-scale')
        self._print_different_keys_loading(net, obj, strict)
        bare.load_state_dict(obj, strict=strict)

    def feed_data(self, data):
        super().feed_data(data)
        lag = data.get('ir_lag')
        if lag is not None:
            lag = lag.to(self.device)
            if lag.dim() == 1:
                lag = lag.unsqueeze(0)
        self.ir_lag = lag
        self.sync_ir = self.gt_ir if hasattr(self, 'gt_ir') else None
        self.sync_vi = self.gt_vi if hasattr(self, 'gt_vi') else None

    @staticmethod
    def _motion_weight(x, lam):
        d = (x[:, 1:] - x[:, :-1]).abs().mean(dim=2, keepdim=True)
        d = torch.cat([d[:, :1], d], dim=1)
        return 1.0 + lam * d

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.results = self.net_g(self.lq_ir, self.lq_vi, ir_lag=self.ir_lag)
        self.output = self.results['fusion']
        self.rec_ir = self.results['ir']
        self.rec_vi = self.results['vi']
        l_total, loss_dict = self._base_losses()
        if self.w_motion > 0:
            w = self._motion_weight(self.gt_vi, self.motion_lambda)
            l_motion = self.w_motion * (w * (self.output - self.gt_vi).abs()).mean()
            l_total = l_total + l_motion
            loss_dict['l_motion'] = l_motion
        if self.net_oracle is not None:
            with torch.no_grad():
                oracle_out = self.net_oracle(self.sync_ir, self.sync_vi, ir_lag=None)
                target = oracle_out['fusion']
            w = self._motion_weight(self.gt_vi, self.motion_lambda) if self.w_motion > 0 else 1.0
            l_oracle = self.w_oracle * (w * (self.output - target).abs()).mean()
            l_total = l_total + l_oracle
            loss_dict['l_oracle'] = l_oracle
        l_total.backward()
        self.optimizer_g.step()
        self.log_dict = self.reduce_loss_dict(loss_dict)
        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def _base_losses(self):
        l_total = 0
        loss_dict = OrderedDict()
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt_vi)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        if self.cri_fus:
            l_fusion = self.cri_fus(rearrange(self.output, 'b t c h w -> (b t) c h w'), rearrange(self.gt_ir, 'b t c h w -> (b t) c h w'), rearrange(self.gt_vi, 'b t c h w -> (b t) c h w'))
            l_total += l_fusion['loss_fusion']
            loss_dict.update(l_fusion)
        if self.cri_fid:
            l_fid_ir = self.cri_fid(rearrange(self.rec_ir, 'b t c h w -> (b t) c h w'), rearrange(self.gt_ir, 'b t c h w -> (b t) c h w'), type='ir')
            l_fid_vi = self.cri_fid(rearrange(self.rec_vi, 'b t c h w -> (b t) c h w'), rearrange(self.gt_vi, 'b t c h w -> (b t) c h w'), type='vi')
            l_total += l_fid_ir['ir_loss_fid'] + l_fid_vi['vi_loss_fid']
            loss_dict.update(l_fid_ir)
            loss_dict.update(l_fid_vi)
        if self.cri_consist:
            l_cons_ir = self.weight_consist * self.cri_consist(self.rec_ir, self.gt_ir)
            l_cons_vi = self.weight_consist * self.cri_consist(self.rec_vi, self.gt_vi)
            l_cons_fu = self.weight_consist * (self.cri_consist(self.output, self.gt_ir) + self.cri_consist(self.output, self.gt_vi))
            l_total += l_cons_ir + l_cons_vi + l_cons_fu
            loss_dict['l_cons_ir'] = l_cons_ir
            loss_dict['l_cons_vi'] = l_cons_vi
            loss_dict['l_cons_fusion'] = l_cons_fu
        return (l_total, loss_dict)
