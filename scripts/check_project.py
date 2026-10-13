"""Check code, configuration, and optional local video training inputs."""
import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'options/train/train_asyncfusion.yml')
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    for path in ROOT.rglob('*.py'):
        if not any(part.startswith('.') for part in path.relative_to(ROOT).parts):
            ast.parse(path.read_text(), filename=str(path))
    from asyncfusion.utils.options import parse
    from asyncfusion.utils.registry import ARCH_REGISTRY, DATASET_REGISTRY, MODEL_REGISTRY
    import asyncfusion.archs
    import asyncfusion.data
    import asyncfusion.models
    opt = parse(str(args.config), str(ROOT), is_train=True)
    ARCH_REGISTRY.get(opt['network_g']['type'])
    MODEL_REGISTRY.get(opt['model_type'])
    DATASET_REGISTRY.get(opt['datasets']['train']['type'])
    if args.check_inputs:
        dataset = opt['datasets']['train']
        required = [Path(dataset[key]) for key in ('dataroot_gt_ir', 'dataroot_gt_vi')]
        required += [Path(condition) / 'schedule.json' for condition in dataset['async_conditions']]
        if opt['train'].get('oracle_weight', 0) > 0:
            teacher = opt['path'].get('pretrain_network_oracle')
            if not teacher:
                raise ValueError('A synchronous teacher checkpoint path is required.')
            required.append(Path(teacher))
        initializer = opt['path'].get('pretrain_network_g')
        if initializer:
            required.append(Path(initializer))
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError('Missing local training inputs:\n' + '\n'.join(missing))
        from asyncfusion.data import build_dataset
        train_set = build_dataset(dataset)
        if len(train_set) == 0:
            raise ValueError('The video training dataset is empty.')
        sample = train_set[0]
        assert sample['ir_lag'].shape[0] == sample['lq_ir'].shape[0]
        print(f'Video training input check passed: {len(train_set)} windows.')
    print('Python syntax and training configuration checks passed.')


if __name__ == '__main__':
    main()
