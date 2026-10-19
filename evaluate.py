"""Evaluate a temporal branching diffusion checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from branching_diffusion.checkpoints import load_checkpoint
from branching_diffusion.configuration import load_experiment_config
from branching_diffusion.constants import DEFAULT_SEED, EVALUATION_SEQUENCE_COUNT
from branching_diffusion.data import create_dataloaders
from branching_diffusion.metrics import collect_test_sequences, compute_all_evaluation_metrics
from branching_diffusion.reverse_denoising_sampler import sample_event_sequences
from train import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse evaluation command-line arguments.

    Purpose:
        Defines checkpoint, dataset, output, seed, device, and sample-count
        arguments for one evaluation run.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description="Evaluate a temporal branching diffusion checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--config", type=Path, default=Path("configs/syn_tpp_1.yaml"), help="YAML file containing the default dataset name.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Directory containing local point-process datasets.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset filename overriding checkpoint and YAML metadata.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global seed used before evaluation sampling.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index. Use -1 for CPU.")
    parser.add_argument("--num-sequences", type=int, default=EVALUATION_SEQUENCE_COUNT, help="Number of real and generated sequences used for metrics.")
    return parser.parse_args()


def main() -> None:
    """Run evaluation once and compute all metrics on one generated set.

    Purpose:
        Loads a checkpoint, samples generated event sequences, and evaluates
        counting-process MMD, spatio-temporal Sinkhorn divergence, and
        event-count Wasserstein distance.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    config = load_experiment_config(args.config)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.gpu)
    forward_process, reverse_denoiser, checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_dataset = checkpoint.get("train_args", {}).get("dataset_name") or checkpoint.get("train_args", {}).get("dataset")
    dataset_name = args.dataset or checkpoint_dataset or config.dataset
    if args.dataset is not None and checkpoint_dataset is not None and args.dataset != checkpoint_dataset:
        print(
            f"Warning: CLI dataset '{args.dataset}' differs from checkpoint dataset "
            f"'{checkpoint_dataset}'. The CLI value is used."
        )
    dataloaders = create_dataloaders(
        data_root=args.data_root,
        dataset_name=dataset_name,
        batch_size=config.batch_size,
        seed=args.seed,
    )
    test_sequences = collect_test_sequences(
        dataloaders["test"]["loader"],
        pad_value=forward_process.pad_value,
        max_sequences=args.num_sequences,
    )
    generated = sample_event_sequences(
        forward_process,
        reverse_denoiser,
        num_sequences=args.num_sequences,
        device=device,
    )
    num_samples = min(len(test_sequences), len(generated))
    metrics = compute_all_evaluation_metrics(
        test_sequences[:num_samples],
        generated[:num_samples],
        time_horizon=forward_process.time_horizon,
    )
    dataset_path = (args.data_root / dataset_name).resolve()
    result = {
        "dataset": {
            "name": dataset_name,
            "path": str(dataset_path),
        },
        "checkpoint": str(args.checkpoint.resolve()),
        "evaluation": {
            "seed": int(args.seed),
            "num_real_sequences": int(num_samples),
            "num_generated_sequences": int(num_samples),
            "device": str(device),
        },
        "metrics": metrics,
    }
    output_path = args.output_json or (args.checkpoint.parent / "evaluation_result.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Evaluation Results")
    print("------------------")
    print(f"Dataset: {dataset_path}")
    print(f"Checkpoint: {args.checkpoint.resolve()}")
    print(f"Evaluation seed: {args.seed}")
    print(f"Number of real sequences: {num_samples}")
    print(f"Number of generated sequences: {num_samples}")
    print(f"Device: {device}")
    print(f"MMD:  {metrics['mmd']:.10f}")
    print(f"STSD: {metrics['stsd']:.10f}")
    print(f"WD:   {metrics['wd']:.10f}")


if __name__ == "__main__":
    main()
