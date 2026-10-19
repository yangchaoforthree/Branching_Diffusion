# Temporal Branching Diffusion for Point Processes

## Overview

This repository implements a temporal-only instance of branching diffusion for
point-process generation. The code follows the paper terminology while keeping
the implemented algorithm unchanged: a homogeneous temporal reference process,
a forward branching diffusion chain, a permutation-equivariant reverse
branching denoiser, and an entropic UOT/Sinkhorn training loss.

The implementation operates on one-dimensional event times. Marks and spatial
coordinates may appear in dataset files, but the forward process, reverse
denoiser, loss, sampling, and metrics in this repository use event times only.

## Method

### Reference Process

`BranchingForwardProcess` defines a homogeneous temporal reference process on
the time horizon `[0, T]`. The reference event rate is
`reference_mass / T`, where `reference_mass` is taken from the training split
mean event count. The terminal initialization used by sampling is produced by
`sample_terminal_poisson_process`, which runs the full forward process from an
empty counting measure and therefore samples the terminal Poisson reference
distribution implemented by this code path.

### Forward Branching Diffusion

The forward noising process is implemented by
`BranchingForwardProcess.forward_noising_step`. Each step applies:

1. `apply_transport_step`: Gaussian temporal perturbation with reflection
   implemented as clamping to `[0, T]`. Because the reference intensity is
   homogeneous in this implementation, there is no learned spatial transport
   field in the forward step.
2. `apply_birth_death_step`: independent survival/death sampling using the
   per-step survival probability derived from the mass-retention schedule.
3. `sample_reference_births`: homogeneous Poisson reference births with rate
   derived from the schedule drop and the reference event rate.

`sample_forward_trajectory` composes these steps and returns all intermediate
step dictionaries. The output support is a tensor of event times; no marks,
spatial coordinates, or parent-offspring identifiers are represented in the
current code.

### Reverse Branching Denoiser

`ReverseBranchingDenoiser` is a DeepSets-style permutation-equivariant denoiser.
For a current event set and diffusion step, it predicts:

- retention logits for current events through
  `predict_retention_probability_logits`;
- reverse transport as a time displacement through `predict_reverse_transport`;
- birth-proposal acceptance logits through
  `predict_birth_probability_logits`;
- paired retention and birth logits through `predict_reverse_net_growth`.

`reverse_denoising_step` samples retention/deletion decisions, refines retained
event times, samples Poisson birth proposals from the reference process, accepts
birth proposals, and sorts the resulting event times. `sample_reverse_trajectory`
runs all reverse steps from `N_K` to `N_0`, and `sample_event_sequences`
combines terminal initialization with the reverse trajectory sampler.

### Training Objective

`construct_soft_predicted_measure` creates the soft predicted measure used for
training. Its support is the concatenation of time-refined current events and
birth proposal times. Its event mass is the concatenation of sigmoid retention
probabilities and sigmoid birth probabilities.

`train_branching_diffusion_model` samples a random diffusion step, generates
`N_k` and `N_{k-1}` with the forward process, constructs the soft predicted
measure, normalizes predicted event mass, and calls
`compute_diffusion_step_sinkhorn_loss`. The loss uses the direct-domain
entropic UOT/Sinkhorn implementation in
`branching_diffusion/entropic_unbalanced_ot.py`
with the repository constants for entropic blur, mass variation penalty, cost
scale, transport mode, and bias correction.

### Evaluation

`evaluate.py` samples one generated sequence set and computes all reported
metrics on that same set:

- counting-process MMD from `compute_counting_process_mmd`;
- spatio-temporal Sinkhorn divergence from
  `compute_temporal_sinkhorn_divergence`, specialized here to time-only
  supports;
- event-count Wasserstein distance from
  `compute_event_count_wasserstein_distance`.

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
