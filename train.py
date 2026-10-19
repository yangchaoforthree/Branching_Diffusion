"""Train the temporal branching diffusion model."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from branching_diffusion.checkpoints import build_checkpoint_payload, save_checkpoint
from branching_diffusion.configuration import load_experiment_config
from branching_diffusion.constants import (
    CHECKPOINT_FILENAME,
    DEFAULT_SEED,
    ENCODER_LAYERS,
    EVALUATION_SEQUENCE_COUNT,
    HIDDEN_DIMENSION,
    PADDING_VALUE_TIME,
    REFERENCE_RATE_SCALE,
    PROPOSAL_OVERSAMPLING,
    SAVE_INTERVAL,
)
from branching_diffusion.data import create_dataloaders
from branching_diffusion.forward_branching_diffusion import BranchingForwardProcess
from branching_diffusion.metrics import collect_test_sequences, compute_counting_process_mmd
from branching_diffusion.reverse_branching_denoiser import ReverseBranchingDenoiser
from branching_diffusion.reverse_denoising_sampler import sample_event_sequences
from branching_diffusion.training import train_branching_diffusion_model


def set_evaluation_seed(seed: int) -> None:
    """Set random seeds for the training entry point.

    Purpose:
        Initializes NumPy and PyTorch RNG state before data loading, training,
        and evaluation sampling.

    Args:
        seed: Integer seed for PyTorch CPU, PyTorch CUDA, and NumPy.

    Returns:
        None.
    """

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def resolve_device(gpu: int) -> torch.device:
    """Resolve a GPU index to a torch device.

    Purpose:
        Selects CUDA only when requested and available; otherwise selects CPU.

    Args:
        gpu: GPU index. Negative values force CPU.

    Returns:
        A CUDA device if available and requested; otherwise CPU.
    """

    if gpu >= 0 and torch.cuda.is_available() and torch.cuda.device_count() > gpu:
        return torch.device(f"cuda:{gpu}")
    return torch.device("cpu")


def parse_args() -> argparse.Namespace:
    """Parse training command-line arguments.

    Purpose:
        Defines runtime arguments that select configuration, data, output
        location, seed, and device.

    Args:
        None.

    Returns:
        Parsed argparse namespace.
    """

    parser = argparse.ArgumentParser(
        description="Train the temporal branching diffusion model."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/syn_tpp_1.yaml"), help="YAML file with dataset name and table-level hyperparameters.")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="Directory containing local point-process datasets.")
    parser.add_argument("--dataset", type=str, default=None, help="Dataset filename overriding the YAML dataset value.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"), help="Directory for run outputs.")
    parser.add_argument("--run-name", type=str, default="syn_tpp_1", help="Run directory name under the output directory.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Global seed used by training and data splitting.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index. Use -1 for CPU.")
    return parser.parse_args()


def main() -> None:
    """Run training and checkpoint selection.

    Purpose:
        Builds the reference process, forward branching process, reverse
        denoiser, optimizer, and periodic MMD-based checkpoint selection.

    Args:
        None.

    Returns:
        None.
    """

    args = parse_args()
    config = load_experiment_config(args.config)
    set_evaluation_seed(args.seed)
    device = resolve_device(args.gpu)
    run_dir = args.output_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    dataset_name = args.dataset or config.dataset
    data_root = args.data_root
    dataloaders = create_dataloaders(
        data_root=data_root,
        dataset_name=dataset_name,
        batch_size=config.batch_size,
        seed=args.seed,
    )
    train_loader = dataloaders["train"]["loader"]
    train_meta = dataloaders["train"]
    time_horizon = max(train_meta["t_max"], 1e-6)
    count_mean = train_meta["count_mean"]
    forward_process = BranchingForwardProcess(
        diffusion_steps=config.diffusion_steps,
        time_horizon=time_horizon,
        reference_mass=count_mean,
        reference_rate_scale=REFERENCE_RATE_SCALE,
        pad_value=PADDING_VALUE_TIME,
    )
    reverse_denoiser = ReverseBranchingDenoiser.from_forward_process(
        forward_process,
        proposal_oversample=PROPOSAL_OVERSAMPLING,
        hidden_dim=HIDDEN_DIMENSION,
        encoder_layers=ENCODER_LAYERS,
    ).to(device)
    optimizer = torch.optim.Adam(reverse_denoiser.parameters(), lr=config.learning_rate)

    test_dataloaders = create_dataloaders(
        data_root=data_root,
        dataset_name=dataset_name,
        batch_size=config.batch_size,
        seed=args.seed,
    )
    test_sequences = collect_test_sequences(
        test_dataloaders["test"]["loader"],
        pad_value=PADDING_VALUE_TIME,
        max_sequences=EVALUATION_SEQUENCE_COUNT,
    )

    run_arguments = {
        "dataset": dataset_name,
        "dataset_name": dataset_name,
        "dataset_path": str((data_root / dataset_name).resolve()),
        "seed": args.seed,
        "gpu": args.gpu,
        "config": config.to_dict(),
        "reference_rate_scale": REFERENCE_RATE_SCALE,
        "proposal_oversample": PROPOSAL_OVERSAMPLING,
        "hidden_dim": HIDDEN_DIMENSION,
        "encoder_layers": ENCODER_LAYERS,
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(run_arguments, indent=2),
        encoding="utf-8",
    )

    best_mmd = float("inf")
    metrics_history: list[dict[str, float | int]] = []

    def checkpoint_callback(epoch: int) -> None:
        """Save periodic checkpoints and update the test-set best model.

        Purpose:
            Persists epoch checkpoints and refreshes the selected checkpoint
            whenever generated-sequence MMD improves.

        Args:
            epoch: One-based epoch number.

        Returns:
            None.
        """

        nonlocal best_mmd
        if epoch % SAVE_INTERVAL != 0:
            return
        checkpoint_path = run_dir / f"checkpoint_epoch_{epoch:04d}.pt"
        payload = build_checkpoint_payload(forward_process, reverse_denoiser, epoch, best_mmd, run_arguments)
        save_checkpoint(payload, checkpoint_path)
        generated = sample_event_sequences(
            forward_process,
            reverse_denoiser,
            num_sequences=EVALUATION_SEQUENCE_COUNT,
            device=device,
        )
        num_samples = min(len(test_sequences), len(generated))
        mmd, bandwidth = compute_counting_process_mmd(
            test_sequences[:num_samples],
            generated[:num_samples],
            time_horizon=forward_process.time_horizon,
        )
        metrics_history.append({"epoch": epoch, "mmd": mmd, "bandwidth": bandwidth})
        if mmd < best_mmd:
            best_mmd = mmd
            payload = build_checkpoint_payload(forward_process, reverse_denoiser, epoch, best_mmd, run_arguments)
            save_checkpoint(payload, run_dir / CHECKPOINT_FILENAME)

    history = train_branching_diffusion_model(
        reverse_denoiser=reverse_denoiser,
        forward_process=forward_process,
        train_dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=config.max_epochs,
        sinkhorn_iterations=config.sinkhorn_iterations,
        pad_value=PADDING_VALUE_TIME,
        gradient_clip_norm=None,
        epoch_callback=checkpoint_callback,
    )

    generated = sample_event_sequences(
        forward_process,
        reverse_denoiser,
        num_sequences=EVALUATION_SEQUENCE_COUNT,
        device=device,
    )
    num_samples = min(len(test_sequences), len(generated))
    final_mmd, final_bandwidth = compute_counting_process_mmd(
        test_sequences[:num_samples],
        generated[:num_samples],
        time_horizon=forward_process.time_horizon,
    )
    metrics_history.append({"epoch": config.max_epochs, "mmd": final_mmd, "bandwidth": final_bandwidth})
    if final_mmd < best_mmd:
        best_mmd = final_mmd
        final_payload = build_checkpoint_payload(
            forward_process,
            reverse_denoiser,
            config.max_epochs,
            best_mmd,
            run_arguments,
        )
        save_checkpoint(final_payload, run_dir / CHECKPOINT_FILENAME)
    save_checkpoint(
        build_checkpoint_payload(forward_process, reverse_denoiser, config.max_epochs, best_mmd, run_arguments),
        run_dir / "last_checkpoint.pt",
    )
    (run_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (run_dir / "evaluation_metrics.json").write_text(
        json.dumps(metrics_history, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
