from .training_model import TrainingModel
from asyncfusion.archs import build_network
from asyncfusion.losses import build_loss
from asyncfusion.utils import get_root_logger

class FusionModel(TrainingModel):

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']
        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)
            self.net_g_ema.eval()
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
            self.weight_pix = train_opt['pixel_opt'].get('weight', 1.0)
        else:
            self.cri_pix = None
        if train_opt.get('fusion_opt'):
            self.cri_fus = build_loss(train_opt['fusion_opt']).to(self.device)
            self.weight_fus = train_opt['fusion_opt'].get('weight', 1.0)
        else:
            self.cri_fus = None
        if train_opt.get('fidelity_opt'):
            self.cri_fid = build_loss(train_opt['fidelity_opt']).to(self.device)
            self.weight_fid = train_opt['fidelity_opt'].get('weight', 1.0)
        else:
            self.cri_fid = None
        if train_opt.get('consistency_opt'):
            self.cri_consist = build_loss(train_opt['consistency_opt']).to(self.device)
            self.weight_consist = train_opt['consistency_opt'].get('weight', 10)
        else:
            self.cri_consist = None
        if self.cri_pix is None and self.cri_fus is None:
            raise ValueError('Both pixel and fusion losses are None.')
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq_ir = data['lq_ir'].to(self.device)
        self.lq_vi = data['lq_vi'].to(self.device)
        if 'gt_ir' in data:
            self.gt_ir = data['gt_ir'].to(self.device)
        if 'gt_vi' in data:
            self.gt_vi = data['gt_vi'].to(self.device)
