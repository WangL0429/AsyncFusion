import argparse
import json
import os
import shutil
from pathlib import Path
MODALITIES = ('infrared_Enhance', 'visible_Enhance', 'infrared_noise', 'visible_Blur')
SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

def _inventory(source):
    inventory = {}
    for split in ('train',):
        root = source / split
        if not root.exists():
            continue
        mods = [m for m in MODALITIES if (root / m).is_dir()]
        if not {'infrared_Enhance', 'visible_Enhance'}.issubset(mods):
            raise ValueError(f'{root}: clean IR and VI directories are required')
        clips = sorted((p.name for p in (root / 'infrared_Enhance').iterdir() if p.is_dir()))
        if not clips:
            raise ValueError(f'No clips in {root}')
        for mod in mods:
            found = sorted((p.name for p in (root / mod).iterdir() if p.is_dir()))
            if found != clips:
                raise ValueError(f'Clip mismatch for {root / mod}')
        rows = []
        for clip in clips:
            names = sorted((p.name for p in (root / mods[0] / clip).iterdir() if p.is_file() and (not p.name.startswith('.')) and (p.suffix.lower() in SUFFIXES)))
            if not names:
                raise ValueError(f'No frames in {root / mods[0] / clip}')
            if any((not Path(n).stem.isdigit() for n in names)):
                raise ValueError(f'{clip}: filenames must have numeric frame indices')
            names.sort(key=lambda n: int(Path(n).stem))
            ids = [int(Path(n).stem) for n in names]
            if ids != list(range(ids[0], ids[0] + len(ids))):
                raise ValueError(f'{clip}: numeric frame indices must be consecutive')
            for mod in mods:
                found = {p.name for p in (root / mod / clip).iterdir() if p.is_file() and (not p.name.startswith('.')) and (p.suffix.lower() in SUFFIXES)}
                if found != set(names):
                    raise ValueError(f'Frame pairing mismatch for {root / mod / clip}')
            rows.append((clip, names, ids))
        inventory[split] = (mods, rows)
    if not inventory:
        raise ValueError(f'No train split under {source}')
    return inventory

def prepare_async_views(data_root, output_root, rates, policy='nearest', link_mode='symlink'):
    source = Path(data_root).resolve()
    output = Path(output_root).resolve()
    rates = sorted(set(rates))
    if not rates or any((type(r) is not int or r < 1 for r in rates)):
        raise ValueError('rates must be positive integers')
    if policy not in ('nearest', 'previous'):
        raise ValueError('Unknown sampling policy')
    if link_mode not in ('symlink', 'hardlink', 'copy'):
        raise ValueError('Unknown link mode')
    if output == source or source in output.parents:
        raise ValueError('Output must be outside the source dataset')
    inventory = _inventory(source)
    plans = []
    for rate in rates:
        cond = 'sync' if rate == 1 else f'r{rate}_p0_' + ('noncausal_nearest' if policy == 'nearest' else 'causal_previous')
        for split, (mods, clips) in inventory.items():
            target = output / cond / split
            if target.exists():
                raise FileExistsError(f'Refusing to overwrite {target}')
            plans.append((rate, split, mods, clips, target))
    completed = []
    for rate, split, mods, clips, target in plans:
        target.mkdir(parents=True)
        try:
            schedule = {}
            for clip, names, ids in clips:
                observations = list(range(0, len(names), rate))
                used = [min(observations, key=lambda j: (abs(j - i), j)) if policy == 'nearest' else i // rate * rate for i in range(len(names))]
                schedule[clip] = [dict(n=n, ir_used=ids[j]) for n, j in zip(ids, used)]
                for mod in mods:
                    folder = target / mod / clip
                    folder.mkdir(parents=True)
                    for i, name in enumerate(names):
                        j = used[i] if mod.startswith('infrared') else i
                        old = source / split / mod / clip / names[j]
                        new = folder / name
                        if link_mode == 'symlink':
                            new.symlink_to(old)
                        elif link_mode == 'hardlink':
                            os.link(old, new)
                        else:
                            shutil.copyfile(old, new)
            document = dict(version=1, rate=rate, phase=0, policy=policy, clips=schedule)
            (target / 'schedule.json').write_text(json.dumps(document, indent=2) + '\n')
        except Exception:
            shutil.rmtree(target)
            raise
        completed.append(str(target))
    return completed

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Synchronized video dataset root containing train/')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--rates', type=int, nargs='+', default=[1, 2, 3, 5, 8])
    parser.add_argument('--policy', choices=['nearest', 'previous'], default='nearest')
    parser.add_argument('--link-mode', choices=['symlink', 'hardlink', 'copy'], default='symlink')
    args = parser.parse_args()
    for path in prepare_async_views(args.data_root, args.output_root, args.rates, args.policy, args.link_mode):
        print(path)
if __name__ == '__main__':
    main()
