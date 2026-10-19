"""Training loop for the reverse branching denoiser."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from branching_diffusion.constants import (
    GRADIENT_CLIP_FALLBACK,
    MASS_KL_PENALTY,
    SINKHORN_EPSILON,
    SINKHORN_POWER,
    TIME_COST_SCALE,
    TRANSPORT_MODE,
    USE_BIAS_CORRECTION,
)
from branching_diffusion.forward_branching_diffusion import BranchingForwardProcess
from branching_diffusion.entropic_unbalanced_ot import compute_diffusion_step_sinkhorn_loss
from branching_diffusion.reverse_branching_denoiser import ReverseBranchingDenoiser
from branching_diffusion.reverse_denoising import construct_soft_predicted_measure


def train_branching_diffusion_model(
    reverse_denoiser: ReverseBranchingDenoiser,
    forward_process: BranchingForwardProcess,
    train_dataloader: object,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_epochs: int = 100,
    sinkhorn_iterations: int = 50,
    pad_value: float = -1.0,
    gradient_clip_norm: float | None = None,
    epoch_callback: Callable[[int], None] | None = None,
) -> list[dict[str, Any]]:
    """Train the reverse branching denoiser with a diffusion-step UOT loss.

    Purpose:
        Samples a random forward diffusion step for each sequence, constructs
        a soft predicted previous measure with the reverse denoiser, and
        minimizes the entropic UOT/Sinkhorn diffusion-step loss.

    Args:
        reverse_denoiser: Reverse branching denoiser optimized by this loop.
        forward_process: Forward branching process used to sample `N_k` and
            `N_{k-1}` training pairs.
        train_dataloader: Dataloader yielding `event_time` tensors with shape
            `(batch_size, max_events)`.
        optimizer: Optimizer updated once for every batch with at least one
            valid sequence.
        device: Device used for training tensors.
        num_epochs: Number of epochs to run.
        sinkhorn_iterations: Number of direct-domain Sinkhorn scaling
            iterations. The YAML configuration sets this to 100 by request.
        pad_value: Numeric padding sentinel used for batch masking.
        gradient_clip_norm: Optional clipping norm. If `None`, the effective
            norm is `1.0`.
        epoch_callback: Optional callback invoked after each epoch with the
            one-based epoch number.

    Returns:
        A list of epoch metric dictionaries with average loss and skip counts.
    """

    if gradient_clip_norm is None:
        gradient_clip_norm = GRADIENT_CLIP_FALLBACK
    history: list[dict[str, Any]] = []
    reverse_denoiser.to(device)
    reverse_denoiser.train()

    for epoch_index in range(num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0
        batch_count = 0
        total_valid_samples = 0
        skipped_samples = 0
        total_transport = 0.0
        total_entropy = 0.0
        total_kl = 0.0

        for batch in train_dataloader:
            optimizer.zero_grad(set_to_none=True)
            batch_loss = torch.tensor(0.0, device=device)
            valid_samples = 0
            event_times = batch["event_time"].to(device)
            batch_size = event_times.shape[0]

            for batch_index in range(batch_size):
                initial_event_times = event_times[batch_index]
                initial_event_times = initial_event_times[initial_event_times != pad_value]
                if initial_event_times.numel() == 0:
                    skipped_samples += 1
                    continue
                try:
                    if initial_event_times.numel() < 2:
                        skipped_samples += 1
                        continue
                    diffusion_step_count = int(
                        torch.randint(
                            1,
                            forward_process.diffusion_steps + 1,
                            (1,),
                            device=device,
                        ).item()
                    )
                    diffusion_step_index = diffusion_step_count - 1
                    forward_trajectory = forward_process.sample_forward_trajectory(
                        initial_event_times,
                        num_steps=diffusion_step_count,
                    )
                    current_measure = forward_trajectory[-1]["next_event_times"]
                    if current_measure.numel() == 0:
                        skipped_samples += 1
                        continue
                    previous_measure = (
                        forward_trajectory[-2]["next_event_times"]
                        if diffusion_step_count > 1
                        else initial_event_times
                    )
                    if previous_measure.numel() < 2:
                        skipped_samples += 1
                        continue

                    birth_proposal_times = reverse_denoiser.sample_birth_proposals(device=device)
                    predicted_measure, predicted_event_mass = construct_soft_predicted_measure(
                        reverse_denoiser,
                        current_event_times=current_measure,
                        birth_proposal_times=birth_proposal_times,
                        diffusion_step_index=diffusion_step_index,
                    )
                    if predicted_measure.numel() < 2:
                        skipped_samples += 1
                        continue
                    target_event_mass = torch.ones_like(previous_measure) / previous_measure.numel()
                    predicted_measure_mass = predicted_event_mass.sum()
                    if (
                        predicted_measure_mass < 1e-10
                        or torch.isnan(predicted_measure_mass)
                        or torch.isinf(predicted_measure_mass)
                    ):
                        skipped_samples += 1
                        continue
                    normalized_predicted_event_mass = predicted_event_mass / predicted_measure_mass
                    if (
                        torch.isnan(normalized_predicted_event_mass).any()
                        or torch.isinf(normalized_predicted_event_mass).any()
                    ):
                        skipped_samples += 1
                        continue

                    loss_result = compute_diffusion_step_sinkhorn_loss(
                        x=previous_measure.unsqueeze(-1),
                        y=predicted_measure.unsqueeze(-1),
                        w_x=target_event_mass,
                        w_y=normalized_predicted_event_mass,
                        eps=SINKHORN_EPSILON,
                        lam=MASS_KL_PENALTY,
                        max_iters=sinkhorn_iterations,
                        p=SINKHORN_POWER,
                        debiased=False,
                        return_components=True,
                        transport_mode=TRANSPORT_MODE,
                        use_bias_correction=USE_BIAS_CORRECTION,
                        cost_delta=TIME_COST_SCALE,
                    )
                    loss_sample = loss_result["total"]
                    if torch.isnan(loss_sample) or torch.isinf(loss_sample) or loss_sample < 0:
                        skipped_samples += 1
                        continue
                    batch_loss = batch_loss + loss_sample
                    valid_samples += 1
                    total_valid_samples += 1
                    total_transport += float(loss_result["transport"].detach().cpu())
                    total_entropy += float(loss_result["entropy"].detach().cpu())
                    total_kl += float((loss_result["kl_p"] + loss_result["kl_q"]).detach().cpu())
                except Exception:
                    skipped_samples += 1
                    continue

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                if torch.isnan(batch_loss) or torch.isinf(batch_loss):
                    skipped_samples += valid_samples
                    continue
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(reverse_denoiser.parameters(), gradient_clip_norm)
                optimizer.step()
                epoch_loss += float(batch_loss.detach().cpu())
                batch_count += 1

        avg_loss = epoch_loss / batch_count if batch_count > 0 else float("inf")
        elapsed = time.time() - epoch_start
        mean_transport = total_transport / total_valid_samples if total_valid_samples else 0.0
        mean_entropy = total_entropy / total_valid_samples if total_valid_samples else 0.0
        mean_kl = total_kl / total_valid_samples if total_valid_samples else 0.0
        history.append(
            {
                "epoch": epoch_index + 1,
                "loss": avg_loss,
                "valid_samples": total_valid_samples,
                "skipped_samples": skipped_samples,
                "transport": mean_transport,
                "entropy": mean_entropy,
                "kl": mean_kl,
                "elapsed_seconds": elapsed,
            }
        )
        if epoch_callback is not None:
            epoch_callback(epoch_index + 1)

    return history


__all__ = ["train_branching_diffusion_model"]
