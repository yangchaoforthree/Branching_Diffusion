# Branching Diffusion for Point Processes

## Overview

This repository implements the branching diffusion for point-process generation. Please follow our paper "Branching Diffusion for Point Process in Time and Space"

## Repository Structure

- `train.py`: training entry point and MMD-based checkpoint selection.
- `evaluate.py`: checkpoint evaluation with MMD, STSD, and WD.
- `sample.py`: sequence generation from a saved checkpoint.
- `configs/`: YAML experiment configuration files.
- `data/`: local point-process datasets.
- `branching_diffusion/`: process, denoiser, data, metric, checkpoint, and
  training modules.

## Installation

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux or macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Dataset Format

Datasets are trusted local `.npy`, `.pkl`, or `.pickle` files containing a
collection of temporal point-process sequences. The recommended schema is a
mapping from sequence id to sequence dictionary:

```python
{
    0: {
        "event_time": np.ndarray,
        "event_type": np.ndarray,
    },
    1: {
        "event_time": np.ndarray,
        "event_type": np.ndarray,
    },
}
```

`event_time` is required. If `event_type` is absent, the loader fills zeros;
the temporal model does not use marks in the forward process, reverse denoiser,
loss, sampling, or metrics. Variable-length sequences are padded at batch time
with numeric sentinels.

The included example dataset is:

```text
data/three_peak_ipp_40T_2000seqs.npy
```

## Configuration

`configs/syn_tpp_1.yaml`:

```yaml
dataset: three_peak_ipp_40T_2000seqs.npy
max_epochs: 100
batch_size: 32
sinkhorn_iterations: 100
diffusion_steps: 5
learning_rate: 0.001
optimizer: Adam
```

Command-line values override YAML values where a CLI override exists. Other
algorithm settings, including proposal oversampling, model width, encoder
depth, reference rate scale, gradient clipping fallback, and metric defaults,
are fixed constants in `branching_diffusion/constants.py`.

## Training

CPU:

```powershell
python train.py `
  --config configs\syn_tpp_1.yaml `
  --data-root data `
  --output-dir outputs `
  --run-name syn_tpp_1_seed_42_cpu `
  --seed 42 `
  --gpu -1
```

GPU:

```powershell
python train.py `
  --config configs\syn_tpp_1.yaml `
  --data-root data `
  --output-dir outputs `
  --run-name syn_tpp_1_seed_42_gpu `
  --seed 42 `
  --gpu 0
```

Custom dataset:

```powershell
python train.py --config configs\syn_tpp_1.yaml --data-root data --dataset custom_dataset.pkl --run-name custom_seed_42 --seed 42 --gpu -1
```

## Evaluation

```powershell
python evaluate.py `
  --checkpoint outputs\seed_42\reverse_denoiser.pt `
  --config configs\syn_tpp_1.yaml `
  --data-root data `
  --output-json outputs\seed_42\evaluation_metrics.json `
  --seed 42 `
  --gpu -1 `
  --num-sequences 200
```

The JSON output stores dataset metadata, checkpoint path, evaluation seed,
device, real/generated sequence counts, MMD bandwidth, STSD, WD, and sequence
length statistics.

## Sampling

```powershell
python sample.py `
  --checkpoint outputs\seed_42\reverse_denoiser.pt `
  --output outputs\seed_42\samples.npy `
  --num-sequences 20 `
  --seed 42 `
  --gpu -1
```

The sample file is a NumPy object array whose entries are one-dimensional event
time arrays.

## Checkpoint Compatibility

New checkpoints store `reverse_denoiser_state`, `diffusion_steps`,
`reference_mass`, and `reference_rate_scale`. `load_checkpoint` also accepts
older checkpoint metadata and older denoiser state-dict keys through an
explicit one-to-one mapping. The migration does not reshape, reinitialize, or
modify tensors.

## Reproducibility

Use `--seed` to set NumPy and PyTorch random states before data loading,
training, sampling, or evaluation. Different hardware, PyTorch builds, and
CUDA kernels can still introduce numerical differences.
