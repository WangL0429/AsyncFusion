import argparse
import os
import random
import torch
import yaml
from collections import OrderedDict
from os import path as osp
from asyncfusion.utils import set_random_seed
from asyncfusion.utils.dist_util import get_dist_info, init_dist

def ordered_yaml():
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))
    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return (Loader, Dumper)

def resolve_project_paths(opt, root_path):
    root_path = osp.abspath(root_path)
    data_root = os.environ.get('ASYNCFUSION_DATA_ROOT', osp.join(root_path, 'data'))

    def expand(value):
        if isinstance(value, list):
            return [expand(item) for item in value]
        if not isinstance(value, str):
            return value
        value = value.replace('${ASYNCFUSION_DATA_ROOT}', data_root)
        value = osp.expanduser(osp.expandvars(value))
        return osp.normpath(value if osp.isabs(value) else osp.join(root_path, value))
    for dataset in opt.get('datasets', {}).values():
        for key, value in list(dataset.items()):
            if key.startswith('dataroot_') or key.startswith('meta_info_file') or key in ('manifest', 'async_conditions'):
                if value is not None:
                    dataset[key] = expand(value)
    for key, value in list(opt.get('path', {}).items()):
        if value is not None and ('pretrain_network' in key or key in ('resume_state', 'save_folder', 'visualization')):
            opt['path'][key] = expand(value)
    return opt

def parse(opt_path, root_path, is_train=True, debug=False):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
    opt = resolve_project_paths(opt, root_path)
    if debug and (not opt['name'].startswith('debug')):
        opt['name'] = 'debug_' + opt['name']
    opt['is_train'] = is_train
    if opt['num_gpu'] == 'auto':
        opt['num_gpu'] = torch.cuda.device_count()
    for phase, dataset in opt['datasets'].items():
        phase = phase.split('_')[0]
        dataset['phase'] = phase
        if 'scale' in opt:
            dataset['scale'] = opt['scale']
        if dataset.get('dataroot_gt') is not None:
            dataset['dataroot_gt'] = osp.expanduser(dataset['dataroot_gt'])
        if dataset.get('dataroot_lq') is not None:
            dataset['dataroot_lq'] = osp.expanduser(dataset['dataroot_lq'])
    for key, val in opt['path'].items():
        if val is not None and ('resume_state' in key or 'pretrain_network' in key):
            opt['path'][key] = osp.expanduser(val)
    if not is_train:
        raise ValueError('Only training options are supported.')
    if set(opt['datasets']) != {'train'}:
        raise ValueError('Provide only datasets.train in the training configuration.')
    experiments_root = osp.join(root_path, 'experiments', opt['name'])
    opt['path']['experiments_root'] = experiments_root
    opt['path']['models'] = osp.join(experiments_root, 'models')
    opt['path']['training_states'] = osp.join(experiments_root, 'training_states')
    opt['path']['log'] = experiments_root
    if 'debug' in opt['name']:
        opt['logger']['print_freq'] = 1
        opt['logger']['save_checkpoint_freq'] = 8
    return opt

def dict2str(opt, indent_level=1):
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg

def parse_options(root_path, is_train=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, required=True, help='Path to option YAML file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none', help='job launcher')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    opt = parse(args.opt, root_path, is_train=is_train, debug=args.debug)
    opt['auto_resume'] = args.auto_resume
    if args.launcher == 'none':
        opt['dist'] = False
        print('Disable distributed.', flush=True)
    else:
        opt['dist'] = True
        if args.launcher == 'slurm' and 'dist_params' in opt:
            init_dist(args.launcher, **opt['dist_params'])
        else:
            init_dist(args.launcher)
    opt['rank'], opt['world_size'] = get_dist_info()
    seed = opt.get('manual_seed')
    if seed is None:
        seed = random.randint(1, 10000)
        opt['manual_seed'] = seed
    set_random_seed(seed + opt['rank'])
    return opt
