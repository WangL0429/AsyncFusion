import datetime
import logging
import time
import torch
from os import path as osp
from asyncfusion.data import build_dataloader, build_dataset
from asyncfusion.data.data_sampler import EnlargedSampler
from asyncfusion.data.prefetch_dataloader import CPUPrefetcher, CUDAPrefetcher
from asyncfusion.models import build_model
from asyncfusion.utils import MessageLogger, check_resume, get_env_info, get_root_logger, get_time_str, init_tb_logger, init_wandb_logger, make_exp_dirs, mkdir_and_rename, scandir
from asyncfusion.utils.options import dict2str, parse_options
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
torch.backends.cudnn.benchmark = True

def init_tb_loggers(opt):
    if opt['logger'].get('wandb') is not None and opt['logger']['wandb'].get('project') is not None and ('debug' not in opt['name']):
        assert opt['logger'].get('use_tb_logger') is True, 'should turn on tensorboard when using wandb'
        init_wandb_logger(opt)
    tb_logger = None
    if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name']:
        current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        log_dir = osp.join(opt['root_path'], 'tb_logger', opt['name'], current_time)
        tb_logger = init_tb_logger(log_dir=log_dir)
    return tb_logger

def create_train_dataloader(opt, logger):
    dataset_opt = opt['datasets']['train']
    enlarge_ratio = dataset_opt.get('dataset_enlarge_ratio', 1)
    train_set = build_dataset(dataset_opt)
    sampler = EnlargedSampler(train_set, opt['world_size'], opt['rank'], enlarge_ratio)
    loader = build_dataloader(train_set, dataset_opt, num_gpu=opt['num_gpu'], dist=opt['dist'], sampler=sampler, seed=opt['manual_seed'])
    global_batch = dataset_opt['batch_size_per_gpu'] * (opt['world_size'] if opt['dist'] else max(opt['num_gpu'], 1))
    if len(train_set) < global_batch:
        raise ValueError('The training dataset must contain at least one complete batch.')
    total_epochs = int(opt['train']['total_epoch'])
    if total_epochs < 1:
        raise ValueError('total_epoch must be positive.')
    iterations = len(loader)
    total_iters = total_epochs * iterations
    logger.info(f'Training clips/windows: {len(train_set)}; global batch: {global_batch}; iterations per epoch: {iterations}; epochs: {total_epochs}; total iterations: {total_iters}')
    return loader, sampler, total_epochs, total_iters


def load_resume_state(opt):
    resume_state_path = None
    if opt['auto_resume']:
        state_path = opt['path']['training_states']
        if osp.isdir(state_path):
            states = list(scandir(state_path, suffix='state', recursive=False, full_path=False))
            if len(states) != 0:
                states = [float(v.split('.state')[0]) for v in states]
                resume_state_path = osp.join(state_path, f'{max(states):.0f}.state')
                opt['path']['resume_state'] = resume_state_path
    elif opt['path'].get('resume_state'):
        resume_state_path = opt['path']['resume_state']
    if resume_state_path is None:
        resume_state = None
    else:
        device = 'cuda' if opt['num_gpu'] else 'cpu'
        resume_state = torch.load(resume_state_path, map_location=device, weights_only=False)
        check_resume(opt, resume_state['iter'])
    return resume_state

def train_pipeline(root_path):
    opt = parse_options(root_path, is_train=True)
    opt['root_path'] = root_path
    resume_state = load_resume_state(opt)
    if resume_state is None:
        make_exp_dirs(opt)
        if opt['logger'].get('use_tb_logger') and 'debug' not in opt['name'] and (opt['rank'] == 0):
            mkdir_and_rename(osp.join(opt['root_path'], 'tb_logger', opt['name']))
    log_file = osp.join(opt['path']['log'], f"train_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(logger_name='asyncfusion', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))
    tb_logger = init_tb_loggers(opt)
    train_loader, train_sampler, total_epochs, total_iters = create_train_dataloader(opt, logger)
    if resume_state:
        model = build_model(opt)
        model.resume_training(resume_state)
        logger.info(f"Resuming training from epoch: {resume_state['epoch']}, iter: {resume_state['iter']}.")
        current_iter = resume_state['iter']
        start_epoch = current_iter // len(train_loader)
    else:
        model = build_model(opt)
        start_epoch = 0
        current_iter = 0
    msg_logger = MessageLogger(opt, current_iter, tb_logger, total_iters=total_iters)
    prefetch_mode = opt['datasets']['train'].get('prefetch_mode')
    if prefetch_mode is None or prefetch_mode == 'cpu':
        prefetcher = CPUPrefetcher(train_loader)
    elif prefetch_mode == 'cuda':
        prefetcher = CUDAPrefetcher(train_loader, opt)
        logger.info(f'Use {prefetch_mode} prefetch dataloader')
        if opt['datasets']['train'].get('pin_memory') is not True:
            raise ValueError('Please set pin_memory=True for CUDAPrefetcher.')
    else:
        raise ValueError(f"Wrong prefetch_mode {prefetch_mode}.Supported ones are: None, 'cuda', 'cpu'.")
    logger.info(f'Start training from epoch: {start_epoch}, iter: {current_iter}')
    data_time, iter_time = (time.time(), time.time())
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs):
        train_sampler.set_epoch(epoch)
        prefetcher.reset()
        train_data = prefetcher.next()
        if resume_state is not None and epoch == start_epoch:
            for _ in range(current_iter % len(train_loader)):
                train_data = prefetcher.next()
        while train_data is not None:
            data_time = time.time() - data_time
            current_iter += 1
            if current_iter > total_iters:
                break
            model.update_learning_rate(current_iter, warmup_iter=opt['train'].get('warmup_iter', -1))
            model.feed_data(train_data)
            model.optimize_parameters(current_iter)
            iter_time = time.time() - iter_time
            if current_iter % opt['logger']['print_freq'] == 0:
                log_vars = {'epoch': epoch, 'iter': current_iter}
                log_vars.update({'lrs': model.get_current_learning_rate()})
                log_vars.update({'time': iter_time, 'data_time': data_time})
                log_vars.update(model.get_current_log())
                msg_logger(log_vars)
            if current_iter % opt['logger']['save_checkpoint_freq'] == 0:
                logger.info('Saving models and training states.')
                model.save(epoch, current_iter)
            data_time = time.time()
            iter_time = time.time()
            train_data = prefetcher.next()
    consumed_time = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    logger.info(f'End of training. Time consumed: {consumed_time}')
    logger.info('Save the latest model.')
    model.save(epoch=-1, current_iter=-1)
    if tb_logger:
        tb_logger.close()
if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
