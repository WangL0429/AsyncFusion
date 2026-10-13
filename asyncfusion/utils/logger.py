import datetime
import logging
import time
from .dist_util import get_dist_info, master_only
initialized_logger = {}

class MessageLogger:

    def __init__(self, opt, start_iter=1, tb_logger=None, total_iters=None):
        self.exp_name = opt['name']
        self.interval = opt['logger']['print_freq']
        self.start_iter = start_iter
        self.max_iters = total_iters
        self.use_tb_logger = opt['logger']['use_tb_logger']
        self.tb_logger = tb_logger
        self.start_time = time.time()
        self.logger = get_root_logger()

    @master_only
    def __call__(self, log_vars):
        epoch = log_vars.pop('epoch')
        current_iter = log_vars.pop('iter')
        lrs = log_vars.pop('lrs')
        message = f'[{self.exp_name}][epoch:{epoch:3d}, iter:{current_iter:8,d}, lr:('
        for v in lrs:
            message += f'{v:.3e},'
        message += ')] '
        if 'time' in log_vars.keys():
            iter_time = log_vars.pop('time')
            data_time = log_vars.pop('data_time')
            total_time = time.time() - self.start_time
            time_sec_avg = total_time / (current_iter - self.start_iter + 1)
            eta_sec = time_sec_avg * (self.max_iters - current_iter - 1)
            eta_str = str(datetime.timedelta(seconds=int(eta_sec)))
            message += f'[eta: {eta_str}, '
            message += f'time (data): {iter_time:.3f} ({data_time:.3f})] '
        for k, v in log_vars.items():
            message += f'{k}: {v:.3e} '
            if self.use_tb_logger and 'debug' not in self.exp_name:
                if k.startswith('l_'):
                    self.tb_logger.add_scalar(f'losses/{k}', v, current_iter)
                else:
                    self.tb_logger.add_scalar(k, v, current_iter)
        self.logger.info(message)

@master_only
def init_tb_logger(log_dir):
    from torch.utils.tensorboard import SummaryWriter
    tb_logger = SummaryWriter(log_dir=log_dir)
    return tb_logger

@master_only
def init_wandb_logger(opt):
    import wandb
    logger = get_root_logger()
    project = opt['logger']['wandb']['project']
    resume_id = opt['logger']['wandb'].get('resume_id')
    if resume_id:
        wandb_id = resume_id
        resume = 'allow'
        logger.warning(f'Resume wandb logger with id={wandb_id}.')
    else:
        wandb_id = wandb.util.generate_id()
        resume = 'never'
    wandb.init(id=wandb_id, resume=resume, name=opt['name'], config=opt, project=project, sync_tensorboard=True)
    logger.info(f'Use wandb logger with id={wandb_id}; project={project}.')

def get_root_logger(logger_name='asyncfusion', log_level=logging.INFO, log_file=None):
    logger = logging.getLogger(logger_name)
    if logger_name in initialized_logger:
        return logger
    format_str = '%(asctime)s %(levelname)s: %(message)s'
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format_str))
    logger.addHandler(stream_handler)
    logger.propagate = False
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    elif log_file is not None:
        logger.setLevel(log_level)
        file_handler = logging.FileHandler(log_file, 'w')
        file_handler.setFormatter(logging.Formatter(format_str))
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
    initialized_logger[logger_name] = True
    return logger

def get_env_info():
    import torch
    import torchvision
    from asyncfusion.version import __version__
    msg = '\n                ____                _       _____  ____\n               / __ ) ____ _ _____ (_)_____/ ___/ / __ \\\n              / __  |/ __ `// ___// // ___/\\__ \\ / /_/ /\n             / /_/ // /_/ /(__  )/ // /__ ___/ // _, _/\n            /_____/ \\__,_//____//_/ \\___//____//_/ |_|\n     ______                   __   __                 __      __\n    / ____/____   ____   ____/ /  / /   __  __ _____ / /__   / /\n   / / __ / __ \\ / __ \\ / __  /  / /   / / / // ___// //_/  / /\n  / /_/ // /_/ // /_/ // /_/ /  / /___/ /_/ // /__ / /<    /_/\n  \\____/ \\____/ \\____/ \\____/  /_____/\\____/ \\___//_/|_|  (_)\n    '
    msg += f'\nVersion Information: \n\tAsyncFusion: {__version__}\n\tPyTorch: {torch.__version__}\n\tTorchVision: {torchvision.__version__}'
    return msg
