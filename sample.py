"""Sample event sequences from a temporal branching diffusion checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from branching_diffusion.checkpoints import load_checkpoint
from branching_diffusion.constants import DEFAULT_SEED
from branching_diffusion.reverse_denoising_sampler import sample_event_sequences
from train import resolve_device


def parse_args() -> argparse.Namespace:
    """Parse sampling command-line arguments.

    Purpose:
        Defines checkpoint, output, sample-count, seed, and device arguments.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(description="Sample generated temporal event sequences.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--output", type=Path, required=True, help="Output NumPy file for sampled sequences.")
    parser.add_argument("--num-sequences", type=int, default=20, help="Number of sequences to sample.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global seed used before sampling.")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU index. Use -1 for CPU.")
    return parser.parse_args()


def main() -> None:
    """Generate sequences and save them as a NumPy object array.

    Purpose:
        Loads a checkpoint, samples generated temporal event sequences, and
        writes them to an object-dtype NumPy file.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = resolve_device(args.gpu)
    forward_process, reverse_denoiser, _ = load_checkpoint(args.checkpoint, device)
    sequences = sample_event_sequences(forward_process, reverse_denoiser, args.num_sequences, device)
    output_data = np.array([seq.numpy() for seq in sequences], dtype=object)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, output_data, allow_pickle=True)


if __name__ == "__main__":
    main()
