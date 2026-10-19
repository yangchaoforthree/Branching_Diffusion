"""Sampling utilities for terminal initialization and reverse denoising."""

from __future__ import annotations

import torch

from branching_diffusion.forward_branching_diffusion import BranchingForwardProcess
from branching_diffusion.reverse_branching_denoiser import ReverseBranchingDenoiser


def sample_terminal_poisson_process(
    forward_process: BranchingForwardProcess,
    device: torch.device,
) -> torch.Tensor:
    """Sample the terminal noising state from the reference process.

    Purpose:
        Draws the temporal terminal initialization used before reverse
        denoising. The implementation obtains this by applying all forward
        noising steps to an empty counting measure.

    Args:
        forward_process: Forward process containing schedules and rates.
        device: Device used for all random sampling.

    Returns:
        A sorted event-time tensor after running all forward noising steps from
        an empty initial set.
    """

    terminal_event_times = torch.empty(0, dtype=torch.float32, device=device)
    for diffusion_step_index in range(forward_process.diffusion_steps):
        if terminal_event_times.numel() > 0:
            terminal_event_times = forward_process.apply_transport_step(
                terminal_event_times,
                forward_process.transport_step_schedule[diffusion_step_index],
            )
        if terminal_event_times.numel() > 0:
            surviving_event_times, _, _ = forward_process.apply_birth_death_step(
                terminal_event_times,
                forward_process.survival_probabilities[diffusion_step_index],
            )
            terminal_event_times = surviving_event_times
        reference_birth_times = forward_process.sample_reference_births(
            reference_birth_rate=forward_process.reference_birth_rates[diffusion_step_index],
            device=device,
        )
        if reference_birth_times.numel() > 0:
            terminal_event_times = torch.cat([terminal_event_times, reference_birth_times], dim=0)
        if terminal_event_times.numel() > 0:
            terminal_event_times = torch.sort(terminal_event_times)[0]
    return terminal_event_times


def sample_reverse_trajectory(
    forward_process: BranchingForwardProcess,
    reverse_denoiser: ReverseBranchingDenoiser,
    terminal_event_times: torch.Tensor,
) -> torch.Tensor:
    """Sample a complete reverse denoising trajectory.

    Purpose:
        Runs the reverse chain from `N_K` to a generated event sequence.

    Args:
        forward_process: Forward process carrying the diffusion-step count.
        reverse_denoiser: Reverse branching denoiser.
        terminal_event_times: One-dimensional terminal event-time tensor.

    Returns:
        One-dimensional generated event-time tensor after all reverse steps.
    """

    current_event_times = terminal_event_times
    for diffusion_step_index in range(forward_process.diffusion_steps - 1, -1, -1):
        current_event_times, _ = reverse_denoiser.reverse_denoising_step(
            current_event_times,
            diffusion_step_index=diffusion_step_index,
            deterministic=False,
        )
    return current_event_times


def sample_event_sequences(
    forward_process: BranchingForwardProcess,
    reverse_denoiser: ReverseBranchingDenoiser,
    num_sequences: int,
    device: torch.device,
) -> list[torch.Tensor]:
    """Generate temporal event sequences with the reverse chain.

    Purpose:
        Samples terminal Poisson reference processes and denoises each one into
        a generated temporal event sequence.

    Args:
        forward_process: Forward process used to sample terminal noising states.
        reverse_denoiser: Reverse branching denoiser.
        num_sequences: Number of sequences to sample.
        device: Device for model execution and random sampling.

    Returns:
        A list of CPU event-time tensors.
    """

    generated_sequences: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(num_sequences):
            terminal_event_times = sample_terminal_poisson_process(forward_process, device)
            generated_event_times = sample_reverse_trajectory(
                forward_process,
                reverse_denoiser,
                terminal_event_times,
            )
            generated_sequences.append(generated_event_times.cpu())
    return generated_sequences


__all__ = ["sample_event_sequences", "sample_reverse_trajectory", "sample_terminal_poisson_process"]
