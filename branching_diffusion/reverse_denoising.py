"""Reverse denoising process operations used by training and sampling."""

from __future__ import annotations

import torch

from branching_diffusion.reverse_branching_denoiser import ReverseBranchingDenoiser


def construct_soft_predicted_measure(
    reverse_denoiser: ReverseBranchingDenoiser,
    current_event_times: torch.Tensor,
    birth_proposal_times: torch.Tensor,
    diffusion_step_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Construct the soft predicted previous counting measure.

    Purpose:
        Builds the differentiable predicted measure used by the entropic UOT
        training objective. Current events receive retention-probability mass
        after time refinement, and birth proposals receive birth-probability
        mass.

    Args:
        reverse_denoiser: Reverse branching denoiser.
        current_event_times: Current noisy event times `N_k` with shape
            `(num_current_events,)`, floating dtype, and units of event time.
        birth_proposal_times: Candidate reverse birth times with shape
            `(num_birth_proposals,)`, floating dtype, and units of event time.
        diffusion_step_index: Zero-based diffusion step index.

    Returns:
        `(predicted_event_times, predicted_event_mass)`. The support tensor
        concatenates time-refined current events followed by birth proposal
        events. The mass tensor concatenates sigmoid retention probabilities
        followed by sigmoid birth probabilities. No sorting or normalization
        is applied.
    """

    device = current_event_times.device
    if current_event_times.numel() > 0:
        retention_logits = reverse_denoiser.predict_retention_probability_logits(
            current_event_times,
            diffusion_step_index,
        )
        retention_probabilities = torch.sigmoid(retention_logits)
        time_displacement = reverse_denoiser.predict_reverse_transport(
            current_event_times,
            diffusion_step_index,
        )
        denoised_event_times = torch.clamp(
            current_event_times + time_displacement,
            min=0.0,
            max=reverse_denoiser.time_horizon,
        )
    else:
        retention_probabilities = torch.empty(0, device=device)
        denoised_event_times = current_event_times

    if birth_proposal_times.numel() > 0:
        birth_logits = reverse_denoiser.predict_birth_probability_logits(
            birth_proposal_times,
            diffusion_step_index,
        )
        birth_probabilities = torch.sigmoid(birth_logits)
    else:
        birth_probabilities = torch.empty(0, device=device)

    predicted_event_times = torch.cat([denoised_event_times, birth_proposal_times], dim=0)
    predicted_event_mass = torch.cat([retention_probabilities, birth_probabilities], dim=0)
    return predicted_event_times, predicted_event_mass


__all__ = ["construct_soft_predicted_measure"]
