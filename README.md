# AsyncFusion

## Multi-modal Video Fusion under Frame-Rate Mismatch

Training code for **AsyncFusion**, an asynchronous infrared-visible video fusion network. The network combines confidence-weighted temporal modeling, visible-guided evidence aggregation, and temporal context refinement to handle different effective frame rates across the two video streams.

## Abstract

Existing multi-modal video fusion methods typically assume that the two modalities are temporally synchronized on a frame-by-frame basis. In practical imaging systems, however, differences in acquisition characteristics across modalities often result in different effective frame rates, leading to asynchronous observations. Frame-rate mismatch in multi-modal video fusion induces motion-dependent temporal artifacts, including displaced thermal structures, suppressed motion, and progressively degraded fusion quality. We present the first systematic study of this setting, and propose AsyncFusion, a multi-modal video fusion network that fuses streams captured at different frame rates into spatio-temporally coherent video. Specifically, AsyncFusion incorporates a timestamp-aware attenuation mechanism that dynamically decays stale thermal features proportional to their temporal latency. To compensate for information loss without generating explicit intermediate frames, the network employs a cross-modal temporal attention mechanism, enabling thermal representations to attend directly across time to the dense visible stream. Furthermore, a motion-weighted objective function prevents the network from converging to trivial solutions such as scene freezing. Crucially, both mechanisms naturally deactivate under synchronized conditions, ensuring seamless generalization without compromising synchronous fusion performance. Consequently, AsyncFusion yields superior temporal stability and motion preservation across diverse frame-rate ratios compared to frame-repetition or interpolation baselines.

## Method

| Module | Full name | Implementation |
| --- | --- | --- |
| CWTM | Confidence-Weighted Temporal Modeling | `asyncfusion/archs/cwtm.py` |
| VGEA | Visible-Guided Evidence Aggregation | `asyncfusion/archs/vgea.py` |
| TCR | Temporal Context Refinement | `asyncfusion/archs/tcr.py` |

The training objective combines fusion quality, infrared/visible reconstruction fidelity, temporal consistency, motion weighting, and distillation from a frozen synchronous teacher. See [module details](docs/MODULES.md).

## Installation

Use Python 3.10–3.12 and a compatible PyTorch/TorchVision pair. The CUDA training environment requires PyTorch with CUDA support.

```bash
git clone https://github.com/WangL0429/AsyncFusion.git
cd AsyncFusion
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

For a CPU environment, install `requirements-cpu.txt` in place of `requirements.txt` and set `num_gpu: 0` in the training configuration.

## Prepare video training data

Training consumes synchronized infrared and visible video clips extracted into consecutive frames. Each clip must have the same frame filenames in both modalities. Filenames use consecutive numeric indices; every clip must contain at least `num_frame` frames.

```text
data/
└── M3SVD/
    └── train/
        ├── infrared_Enhance/
        │   └── clip_001/
        │       ├── 00000.png
        │       ├── 00001.png
        │       └── ...
        └── visible_Enhance/
            └── clip_001/
                ├── 00000.png
                ├── 00001.png
                └── ...
```

Set the dataset location and generate the asynchronous training views:

```bash
export ASYNCFUSION_DATA_ROOT=/path/to/data
python scripts/prepare_async_video.py \
  --data-root "$ASYNCFUSION_DATA_ROOT/M3SVD" \
  --output-root "$ASYNCFUSION_DATA_ROOT/M3SVD_async" \
  --rates 1 2 3 5 8 \
  --policy nearest \
  --link-mode symlink
```

The preparation script writes one `schedule.json` per training condition. Each record stores the query frame index `n` and the selected infrared observation index `ir_used`; the loader derives the lag as `abs(n - ir_used)`. Original synchronized frames remain the training targets.

`nearest` sampling may select a future infrared observation and prefers the older observation in a tie. Use `previous` for causal sampling, then update `datasets.train.async_conditions` to the resulting `causal_previous` directories. The script refuses to overwrite existing condition directories. Use `--link-mode copy` when symbolic links are unsuitable.

## Train

Edit [the training configuration](options/train/train_asyncfusion.yml):

- Set `num_gpu`, `batch_size_per_gpu`, and `num_worker_per_gpu` for your hardware.
- Provide a compatible synchronous teacher checkpoint at `path.pretrain_network_oracle`. The teacher uses the same network architecture and is frozen during training. Checkpoint keys must match the configured network; a `params` state dictionary is supported.
- Optionally set `path.pretrain_network_g` to initialize the trainable network from a compatible checkpoint. Its default value is null.
- Adjust `total_epoch` and the cosine scheduler period for your dataset. Total iterations equal `total_epoch * len(train_loader)`; the default configuration uses four epochs and a scheduler period of 108560 iterations.

Model checkpoints and datasets must be obtained separately and stored locally.

Check the project and local training inputs:

```bash
python scripts/check_project.py --check-inputs
```

Start training on one GPU after setting `num_gpu: 1`:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_asyncfusion.yml
```

For the default four-GPU configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  train.py -opt options/train/train_asyncfusion.yml --launcher pytorch
```

Training logs, checkpoints, and optimizer states are saved under `experiments/AsyncFusion/`. TensorBoard logs are saved under `tb_logger/AsyncFusion/` when enabled.

```bash
tensorboard --logdir tb_logger/AsyncFusion
```

To resume from the latest saved optimizer state:

```bash
python train.py -opt options/train/train_asyncfusion.yml --auto_resume
```

Use the same distributed launcher when resuming a distributed run.

## Project layout

```text
AsyncFusion/
├── asyncfusion/
│   ├── archs/           # Video fusion network and CWTM/VGEA/TCR
│   ├── data/            # Video clip loading and lag schedules
│   ├── losses/          # Video training objectives
│   ├── models/          # Trainable network and frozen teacher
│   ├── utils/           # Training utilities
│   └── train.py         # Training loop
├── options/train/train_asyncfusion.yml
├── scripts/
│   ├── prepare_async_video.py
│   └── check_project.py
├── docs/MODULES.md
└── train.py
```

## Acknowledgments

See [third-party notices](docs/THIRD_PARTY_NOTICES.md) for the training infrastructure and Transformer components incorporated into this project.
