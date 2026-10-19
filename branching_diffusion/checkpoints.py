"""Checkpoint helpers with explicit terminology migration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from branching_diffusion.forward_branching_diffusion import BranchingForwardProcess
from branching_diffusion.reverse_branching_denoiser import ReverseBranchingDenoiser, map_checkpoint_state_dict_keys


def build_checkpoint_payload(
    forward_process: BranchingForwardProcess,
    reverse_denoiser: ReverseBranchingDenoiser,
    epoch: int,
    best_mmd: float,
    run_arguments: dict[str, Any],
) -> dict[str, Any]:
    """Build a checkpoint payload.

    Purpose:
        Serializes model weights and construction metadata using the paper
        terminology for new checkpoints.

    Args:
        forward_process: Forward process carrying schedule metadata.
        reverse_denoiser: Reverse denoiser whose weights are saved.
        epoch: One-based epoch associated with the checkpoint.
        best_mmd: Best test-set MMD tracked by the callback behavior.
        run_arguments: JSON-safe runtime arguments and configuration values.

    Returns:
        A dictionary containing model state and construction metadata.
    """

    return {
        "epoch": epoch,
        "best_mmd": best_mmd,
        "reverse_denoiser_state": reverse_denoiser.state_dict(),
        "forward_config": {
            "diffusion_steps": forward_process.diffusion_steps,
            "time_horizon": forward_process.time_horizon,
            "reference_mass": forward_process.reference_event_rate * forward_process.time_horizon,
            "reference_rate_scale": forward_process.reference_rate_scale,
            "pad_value": forward_process.pad_value,
        },
        "reverse_config": {
            "proposal_oversample": reverse_denoiser.proposal_oversample,
            "hidden_dim": reverse_denoiser.hidden_dim,
            "encoder_layers": reverse_denoiser.encoder_layers,
        },
        "train_args": dict(run_arguments),
    }


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Save a checkpoint payload to disk.

    Args:
        payload: Checkpoint dictionary.
        path: Output path. Parent directories are created if needed.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def migrate_forward_config_fields(forward_config: dict[str, Any]) -> dict[str, Any]:
    """Migrate checkpoint forward-process metadata to paper field names.

    Purpose:
        Supports checkpoints written before the terminology refactor without
        substring replacement or inferred tensor changes.

    Args:
        forward_config: Raw checkpoint forward configuration dictionary.

    Returns:
        Dictionary with `diffusion_steps`, `time_horizon`, `reference_mass`,
        `reference_rate_scale`, and `pad_value`.
    """

    diffusion_steps = (
        forward_config["diffusion_steps"]
        if "diffusion_steps" in forward_config
        else forward_config["num_steps"]
    )
    reference_mass = (
        forward_config["reference_mass"]
        if "reference_mass" in forward_config
        else forward_config["mean_count"]
    )
    reference_rate_scale = (
        forward_config["reference_rate_scale"]
        if "reference_rate_scale" in forward_config
        else forward_config["poisson_rate_scale"]
    )
    return {
        "diffusion_steps": diffusion_steps,
        "time_horizon": forward_config["time_horizon"],
        "reference_mass": reference_mass,
        "reference_rate_scale": reference_rate_scale,
        "pad_value": forward_config["pad_value"],
    }


def migrate_checkpoint_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Migrate checkpoint payload fields to current paper terminology.

    Purpose:
        Converts previous checkpoint metadata and state-dict keys in memory,
        preserving every tensor value exactly.

    Args:
        checkpoint: Raw checkpoint dictionary loaded with `torch.load`.

    Returns:
        New checkpoint dictionary containing `reverse_denoiser_state` and
        migrated `forward_config` fields.
    """

    migrated_checkpoint = dict(checkpoint)
    source_state = checkpoint.get("reverse_denoiser_state", checkpoint.get("reverse_model_state"))
    if source_state is None:
        raise KeyError("Checkpoint is missing reverse denoiser state.")
    migrated_state = map_checkpoint_state_dict_keys(source_state)
    for source_key, source_tensor in source_state.items():
        target_key = next(
            (
                mapped_key
                for mapped_key, mapped_tensor in migrated_state.items()
                if mapped_tensor is source_tensor
            ),
            None,
        )
        if target_key is None or not torch.equal(source_tensor, migrated_state[target_key]):
            raise ValueError(f"Checkpoint tensor changed during key migration: {source_key}")
    migrated_checkpoint["reverse_denoiser_state"] = migrated_state
    migrated_checkpoint["forward_config"] = migrate_forward_config_fields(checkpoint["forward_config"])
    return migrated_checkpoint


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[BranchingForwardProcess, ReverseBranchingDenoiser, dict[str, Any]]:
    """Load a checkpoint and reconstruct model objects.

    Purpose:
        Loads current checkpoints and migrated checkpoints whose metadata or
        denoiser weight keys used earlier public names.

    Args:
        checkpoint_path: Path to a checkpoint written by `save_checkpoint`.
        device: Device used for loading model tensors.

    Returns:
        `(forward_process, reverse_denoiser, checkpoint)` with the reverse
        denoiser moved to `device` and switched to eval mode.
    """

    raw_checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint = migrate_checkpoint_payload(raw_checkpoint)
    forward_config = checkpoint["forward_config"]
    forward_process = BranchingForwardProcess(
        diffusion_steps=forward_config["diffusion_steps"],
        time_horizon=forward_config["time_horizon"],
        reference_mass=forward_config["reference_mass"],
        reference_rate_scale=forward_config["reference_rate_scale"],
        pad_value=forward_config["pad_value"],
    )
    reverse_config = checkpoint["reverse_config"]
    reverse_denoiser = ReverseBranchingDenoiser.from_forward_process(
        forward_process,
        proposal_oversample=reverse_config["proposal_oversample"],
        hidden_dim=reverse_config["hidden_dim"],
        encoder_layers=reverse_config["encoder_layers"],
    )
    reverse_denoiser.load_state_dict(checkpoint["reverse_denoiser_state"])
    reverse_denoiser.to(device)
    reverse_denoiser.eval()
    return forward_process, reverse_denoiser, checkpoint


__all__ = [
    "build_checkpoint_payload",
    "load_checkpoint",
    "migrate_checkpoint_payload",
    "migrate_forward_config_fields",
    "save_checkpoint",
]
