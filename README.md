# Simple GCN Baseline for MPI-INF-3DHP

This mini-project packages the lightweight training and visualization utilities we built on top of the original AttnGCN codebase. It provides two entry points:

- `main_simple_gcn_3dhp.py`: trains a 3-layer graph convolutional network on MPI-INF-3DHP 2D→3D pose lifting using the preprocessed `.npz` splits from the parent project.
- `visualize_simple_gcn_3dhp.py`: runs inference with a saved checkpoint and renders predictions versus ground truth as an animation (falls back to GIF when ffmpeg is not available).

## Dataset

Processed MPI-INF-3DHP dataset is available via the [google drive link](https://drive.google.com/file/d/1yHAwqSxp9x2RgCvRJEQfg1t17FYuMb8c/view?usp=sharing)

## Project Layout

```
simple_gcn_baseline/
├── main_simple_gcn_3dhp.py
├── visualize_simple_gcn_3dhp.py
└── common/
    ├── __init__.py
    ├── camera.py
    ├── fusion.py
    ├── generator_3dhp.py
    ├── loss.py
    ├── skeleton.py
    ├── utils.py
    └── visualization.py
```

The `common` package contains the minimal dataset and rendering helpers required by the scripts. It expects the MPI-INF-3DHP training/test `.npz` files to live alongside the original repository root (e.g. `data_train_3dhp.npz`, `data_test_3dhp.npz`).

## Training

```bash
python main_simple_gcn_3dhp.py \
  --data-root /path/to/AttnGCN \
  --mode 3dhp \
  --batch-size 64 \
  --epochs 60 \
  --use-cuda
```

Useful flags:

- `--subset <fraction>`: train on a fractional subset of frames (e.g. `0.25` for 25%).
- `--save-path <file>`: store the best-performing checkpoint.

## Visualization

```bash
python visualize_simple_gcn_3dhp.py \
  --data-root /path/to/AttnGCN \
  --checkpoint checkpoints/simple_gcn.pth \
  --output demo.mp4 \
  --use-cuda
```

When ffmpeg is unavailable the script automatically saves a GIF next to the requested filename.

## Requirements

- Python 3.8+
- PyTorch
- NumPy, Matplotlib
- Access to the preprocessed MPI-INF-3DHP `.npz` files produced by the AttnGCN preprocessing scripts.

## Notes

- The dataset loader provided here keeps the root-centred 3D poses from the preprocessing pipeline; global translations must be reintroduced by adjusting the `.npz` files if absolute world coordinates are required.
- The scripts assume the directory structure of the original AttnGCN project; adjust `--data-root` if you relocate the `.npz` assets.

