from copy import deepcopy
from asyncfusion.utils import get_root_logger
from asyncfusion.utils.registry import LOSS_REGISTRY
from .losses import Fidelity_loss, Fusion_loss, L1Loss, TemporalConsistency_loss
__all__ = ['Fidelity_loss', 'Fusion_loss', 'L1Loss', 'TemporalConsistency_loss', 'build_loss']

def build_loss(opt):
    opt = deepcopy(opt)
    loss_type = opt.pop('type')
    loss = LOSS_REGISTRY.get(loss_type)(**opt)
    logger = get_root_logger()
    logger.info(f'Loss [{loss.__class__.__name__}] is created.')
    return loss
