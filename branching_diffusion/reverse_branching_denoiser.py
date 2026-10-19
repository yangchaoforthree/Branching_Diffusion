"""Permutation-equivariant reverse branching denoiser."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def build_permutation_equivariant_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build the MLP blocks used by the permutation-equivariant denoiser.

    Purpose:
        Creates the pointwise MLP stack used inside the DeepSets-style encoder.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Hidden layer width.
        output_dim: Output feature dimension.
        num_layers: Number of linear layers. Must be at least one.
        activation: Optional activation instance. Defaults to `nn.GELU()`.

    Returns:
        `nn.Sequential` containing linear layers and activations. Inputs and
        outputs are two-dimensional tensors with trailing dimension
        `input_dim` and `output_dim`, respectively.
    """

    if num_layers < 1:
        raise ValueError("num_layers must be at least 1.")
    layers: list[nn.Module] = []
    dimensions: list[int] = [input_dim]
    if num_layers == 1:
        dimensions.append(output_dim)
    else:
        dimensions.extend([hidden_dim] * (num_layers - 1))
        dimensions.append(output_dim)
    activation_layer = activation if activation is not None else nn.GELU()
    for layer_index in range(len(dimensions) - 1):
        layers.append(nn.Linear(dimensions[layer_index], dimensions[layer_index + 1]))
        if layer_index < len(dimensions) - 2:
            layers.append(activation_layer)
    return nn.Sequential(*layers)


@dataclass
class ReverseDenoisingStepInfo:
    """Summary of one reverse denoising step.

    Purpose:
        Records cardinality diagnostics for a sampled reverse step.

    Args:
        diffusion_step_index: Zero-based reverse diffusion step index.
        num_input_events: Number of input events in the current measure.
        num_retained_events: Number of retained and time-refined events.
        num_birth_events: Number of accepted birth proposals.
        num_output_events: Number of events after union and sorting.

    Returns:
        Dataclass instances expose the five integer fields above.
    """

    diffusion_step_index: int
    num_input_events: int
    num_retained_events: int
    num_birth_events: int
    num_output_events: int


class ReverseBranchingDenoiser(nn.Module):
    """Permutation-equivariant temporal reverse branching denoiser.

    Purpose:
        Approximates the paper's reverse denoising process on temporal event
        sets. For each reverse step it predicts retention probabilities for
        current events, birth probabilities for proposal events, and time
        transport refinements for retained events.

    Args:
        diffusion_steps: Number of forward/reverse diffusion steps `K`.
        time_horizon: Observation horizon `T` in event-time units.
        reference_event_rate: Homogeneous reference event rate inherited from
            the forward process.
        reference_rate_scale: Multiplier applied to the reference event rate.
        proposal_oversample: Poisson birth-proposal oversampling multiplier.
        hidden_dim: Hidden width used by embeddings, DeepSets layers, and
            prediction heads.
        encoder_layers: Number of MLP layers in the point encoder.
        pad_value: Temporal padding sentinel retained for checkpoint metadata.
    """

    def __init__(
        self,
        diffusion_steps: int,
        time_horizon: float,
        reference_event_rate: float,
        reference_rate_scale: float = 1.0,
        proposal_oversample: float = 1.5,
        hidden_dim: int = 128,
        encoder_layers: int = 2,
        pad_value: float = -1.0,
    ) -> None:
        """Initialize embeddings, DeepSets blocks, and prediction heads."""

        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.encoder_layers = int(encoder_layers)
        self.diffusion_steps = int(diffusion_steps)
        self.time_horizon = float(time_horizon)
        self.reference_event_rate = float(reference_event_rate)
        self.reference_rate_scale = float(reference_rate_scale)
        self.reference_birth_proposal_rate = self.reference_event_rate * self.reference_rate_scale
        self.proposal_oversample = float(proposal_oversample)
        self.pad_value = float(pad_value)
        self.time_feature_dim = 4
        event_role_dim = hidden_dim // 4
        diffusion_step_embedding_dim = hidden_dim // 4
        self.diffusion_step_embedding = nn.Embedding(diffusion_steps + 1, diffusion_step_embedding_dim)
        self.event_role_embedding = nn.Embedding(2, event_role_dim)
        self.point_feature_dim = self.time_feature_dim + diffusion_step_embedding_dim + event_role_dim
        self.deepsets_phi = build_permutation_equivariant_mlp(
            input_dim=self.point_feature_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=encoder_layers,
        )
        self.deepsets_rho = build_permutation_equivariant_mlp(
            input_dim=hidden_dim * 2,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            num_layers=max(1, encoder_layers - 1),
        )
        self.retention_probability_head = nn.Linear(hidden_dim, 1)
        self.birth_probability_head = nn.Linear(hidden_dim, 1)
        self.time_refinement_head = nn.Linear(hidden_dim, 1)

    @classmethod
    def from_forward_process(
        cls,
        forward_process: object,
        reference_rate_scale: float | None = None,
        proposal_oversample: float = 2.0,
        hidden_dim: int = 128,
        encoder_layers: int = 2,
    ) -> "ReverseBranchingDenoiser":
        """Construct the denoiser from a forward branching process.

        Purpose:
            Shares the reference event rate, time horizon, diffusion-step
            count, and padding metadata with the forward process.

        Args:
            forward_process: Object exposing `diffusion_steps`,
                `time_horizon`, `reference_event_rate`,
                `reference_rate_scale`, and `pad_value`.
            reference_rate_scale: Optional override for the reference rate
                multiplier.
            proposal_oversample: Poisson birth-proposal oversampling
                multiplier.
            hidden_dim: Hidden width for embeddings and MLP layers.
            encoder_layers: Number of encoder MLP layers.

        Returns:
            A `ReverseBranchingDenoiser` sharing forward-process metadata.
        """

        scale = (
            forward_process.reference_rate_scale
            if reference_rate_scale is None
            else reference_rate_scale
        )
        return cls(
            diffusion_steps=forward_process.diffusion_steps,
            time_horizon=forward_process.time_horizon,
            reference_event_rate=forward_process.reference_event_rate,
            reference_rate_scale=scale,
            proposal_oversample=proposal_oversample,
            hidden_dim=hidden_dim,
            encoder_layers=encoder_layers,
            pad_value=forward_process.pad_value,
        )

    def time_features(self, event_times: torch.Tensor) -> torch.Tensor:
        """Build four-channel temporal event features.

        Purpose:
            Encodes event time using normalized time, reverse normalized time,
            sine, and cosine features.

        Args:
            event_times: One-dimensional floating tensor with shape
                `(num_events,)` and values in event-time units.

        Returns:
            Floating tensor with shape `(num_events, 4)` on the same device.
            Empty input returns shape `(0, 4)`.
        """

        if event_times.numel() == 0:
            return event_times.new_zeros((0, self.time_feature_dim))
        normalized_event_times = event_times / max(self.time_horizon, 1e-6)
        return torch.stack(
            [
                normalized_event_times,
                1.0 - normalized_event_times,
                torch.sin(2.0 * math.pi * normalized_event_times),
                torch.cos(2.0 * math.pi * normalized_event_times),
            ],
            dim=-1,
        )

    def encode_events(
        self,
        event_times: torch.Tensor,
        diffusion_step_index: int,
        event_role: int,
    ) -> torch.Tensor:
        """Encode current or proposal events with DeepSets aggregation.

        Purpose:
            Implements the permutation-equivariant event encoder used before
            retention, birth, and time-refinement predictions.

        Args:
            event_times: One-dimensional floating tensor with shape
                `(num_events,)`.
            diffusion_step_index: Zero-based diffusion step index.
            event_role: Integer role id, `0` for current events and `1` for
                birth proposal events.

        Returns:
            Tensor with shape `(num_events, hidden_dim)`. Empty inputs return
            a tensor with shape `(0, hidden_dim)` on the input device.
        """

        if event_times.numel() == 0:
            return torch.zeros((0, self.hidden_dim), device=event_times.device)
        device = event_times.device
        temporal_features = self.time_features(event_times)
        diffusion_step_features = self.diffusion_step_embedding(
            torch.full((event_times.shape[0],), diffusion_step_index, dtype=torch.long, device=device)
        )
        event_role_features = self.event_role_embedding(
            torch.full((event_times.shape[0],), event_role, dtype=torch.long, device=device)
        )
        event_features = torch.cat(
            [temporal_features, diffusion_step_features, event_role_features],
            dim=-1,
        )
        local_event_embedding = self.deepsets_phi(event_features)
        global_event_embedding = local_event_embedding.mean(dim=0, keepdim=True)
        expanded_global_embedding = global_event_embedding.expand(local_event_embedding.shape[0], -1)
        combined_event_embedding = torch.cat(
            [local_event_embedding, expanded_global_embedding],
            dim=-1,
        )
        return self.deepsets_rho(combined_event_embedding)

    def predict_retention_probability_logits(
        self,
        current_event_times: torch.Tensor,
        diffusion_step_index: int,
    ) -> torch.Tensor:
        """Predict retention logits for current events.

        Purpose:
            Exposes the denoiser's reverse deletion/retention quantity for the
            current measure.

        Args:
            current_event_times: One-dimensional floating tensor with shape
                `(num_current_events,)`.
            diffusion_step_index: Zero-based reverse diffusion step index.

        Returns:
            One-dimensional tensor with shape `(num_current_events,)`.
        """

        current_event_embedding = self.encode_events(
            current_event_times,
            diffusion_step_index=diffusion_step_index,
            event_role=0,
        )
        if current_event_embedding.numel() == 0:
            return torch.empty(0, device=current_event_times.device)
        return self.retention_probability_head(current_event_embedding).squeeze(-1)

    def predict_reverse_transport(
        self,
        retained_event_times: torch.Tensor,
        diffusion_step_index: int,
    ) -> torch.Tensor:
        """Predict reverse transport as a time displacement for retained events.

        Purpose:
            Implements the temporal-only reverse transport update used by the
            code's time-refinement step.

        Args:
            retained_event_times: One-dimensional floating tensor with shape
                `(num_retained_events,)`.
            diffusion_step_index: Zero-based reverse diffusion step index.

        Returns:
            One-dimensional time-displacement tensor with shape
            `(num_retained_events,)` in event-time units.
        """

        retained_event_embedding = self.encode_events(
            retained_event_times,
            diffusion_step_index=diffusion_step_index,
            event_role=0,
        )
        if retained_event_embedding.numel() == 0:
            return torch.empty(0, device=retained_event_times.device)
        return self.time_refinement_head(retained_event_embedding).squeeze(-1)

    def predict_birth_probability_logits(
        self,
        birth_proposal_times: torch.Tensor,
        diffusion_step_index: int,
    ) -> torch.Tensor:
        """Predict birth acceptance logits for proposal events.

        Purpose:
            Exposes the denoiser's reverse birth quantity for proposal events.

        Args:
            birth_proposal_times: One-dimensional floating tensor with shape
                `(num_birth_proposals,)`.
            diffusion_step_index: Zero-based reverse diffusion step index.

        Returns:
            One-dimensional tensor with shape `(num_birth_proposals,)`.
        """

        birth_proposal_embedding = self.encode_events(
            birth_proposal_times,
            diffusion_step_index=diffusion_step_index,
            event_role=1,
        )
        if birth_proposal_embedding.numel() == 0:
            return torch.empty(0, device=birth_proposal_times.device)
        return self.birth_probability_head(birth_proposal_embedding).squeeze(-1)

    def predict_reverse_net_growth(
        self,
        current_event_times: torch.Tensor,
        birth_proposal_times: torch.Tensor,
        diffusion_step_index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict reverse retention and birth logits.

        Purpose:
            Provides the temporal implementation's discrete counterpart of the
            paper's reverse net-growth field: retention logits for current
            events and birth logits for proposal events.

        Args:
            current_event_times: Tensor with shape `(num_current_events,)`.
            birth_proposal_times: Tensor with shape `(num_birth_proposals,)`.
            diffusion_step_index: Zero-based reverse diffusion step index.

        Returns:
            `(retention_logits, birth_logits)`, both one-dimensional tensors.
        """

        return (
            self.predict_retention_probability_logits(current_event_times, diffusion_step_index),
            self.predict_birth_probability_logits(birth_proposal_times, diffusion_step_index),
        )

    def sample_birth_proposals(self, device: torch.device) -> torch.Tensor:
        """Sample reverse birth proposal event times from the reference process.

        Purpose:
            Draws the paper's proposal pool from the homogeneous temporal
            reference distribution with oversampling factor `Lambda`.

        Args:
            device: Torch device used for sampling and returned storage.

        Returns:
            One-dimensional `float32` tensor of proposal times on
            `[0, time_horizon]`.
        """

        proposal_mean = (
            self.reference_birth_proposal_rate * self.time_horizon * self.proposal_oversample
        )
        num_birth_proposals = torch.poisson(
            torch.tensor(proposal_mean, device=device, dtype=torch.float32)
        ).to(torch.int64).item()
        if num_birth_proposals <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        return torch.rand(num_birth_proposals, device=device, dtype=torch.float32) * self.time_horizon

    def reverse_denoising_step(
        self,
        current_event_times: torch.Tensor,
        diffusion_step_index: int,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, ReverseDenoisingStepInfo]:
        """Run one reverse denoising step.

        Purpose:
            Applies reverse deletion/retention, reverse time refinement, and
            reverse birth proposal acceptance for one diffusion step.

        Args:
            current_event_times: One-dimensional tensor of current noisy event
                times with shape `(num_current_events,)`.
            diffusion_step_index: Zero-based reverse diffusion step index.
            deterministic: If true, converts probabilities to threshold
                decisions at `0.5`; if false, samples Bernoulli decisions.

        Returns:
            `(previous_event_times, step_info)`. `previous_event_times` is a
            sorted `float32` tensor on the same device. `step_info` contains
            integer cardinality diagnostics.
        """

        device = current_event_times.device
        num_input_events = current_event_times.numel()
        retention_logits = self.predict_retention_probability_logits(
            current_event_times,
            diffusion_step_index,
        )
        retention_probabilities = torch.sigmoid(retention_logits)

        if retention_probabilities.numel() == 0:
            retention_mask = torch.zeros(0, dtype=torch.bool, device=device)
        elif deterministic:
            retention_mask = retention_probabilities >= 0.5
        else:
            retention_mask = torch.bernoulli(retention_probabilities).to(torch.bool)
        retained_event_times = current_event_times[retention_mask]

        if retained_event_times.numel() > 0:
            time_displacement = self.predict_reverse_transport(
                retained_event_times,
                diffusion_step_index,
            )
            denoised_event_times = torch.clamp(
                retained_event_times + time_displacement,
                min=0.0,
                max=self.time_horizon,
            )
        else:
            denoised_event_times = retained_event_times

        birth_proposal_times = self.sample_birth_proposals(device=device)
        if birth_proposal_times.numel() > 0:
            birth_logits = self.predict_birth_probability_logits(
                birth_proposal_times,
                diffusion_step_index,
            )
            birth_probabilities = torch.sigmoid(birth_logits)
            if deterministic:
                birth_mask = birth_probabilities >= 0.5
            else:
                birth_mask = torch.bernoulli(birth_probabilities).to(torch.bool)
            birth_event_times = birth_proposal_times[birth_mask]
        else:
            birth_event_times = birth_proposal_times

        if denoised_event_times.numel() == 0 and birth_event_times.numel() == 0:
            return (
                torch.empty(0, device=device, dtype=torch.float32),
                ReverseDenoisingStepInfo(diffusion_step_index, num_input_events, 0, 0, 0),
            )
        predicted_event_times = torch.cat([denoised_event_times, birth_event_times], dim=0)
        previous_event_times = predicted_event_times[torch.argsort(predicted_event_times)]
        step_info = ReverseDenoisingStepInfo(
            diffusion_step_index=diffusion_step_index,
            num_input_events=num_input_events,
            num_retained_events=int(denoised_event_times.numel()),
            num_birth_events=int(birth_event_times.numel()),
            num_output_events=int(previous_event_times.numel()),
        )
        return previous_event_times, step_info


EXPLICIT_CHECKPOINT_KEY_MAPPING = {
    "step_embedding.weight": "diffusion_step_embedding.weight",
    "point_role_embedding.weight": "event_role_embedding.weight",
    "keep_head.weight": "retention_probability_head.weight",
    "keep_head.bias": "retention_probability_head.bias",
    "add_head.weight": "birth_probability_head.weight",
    "add_head.bias": "birth_probability_head.bias",
    "time_refine_head.weight": "time_refinement_head.weight",
    "time_refine_head.bias": "time_refinement_head.bias",
    "point_type_embedding.weight": "event_role_embedding.weight",
}
"""Explicit old-to-current checkpoint key mapping for denoiser weights."""


def map_checkpoint_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Map old checkpoint keys to paper-terminology denoiser keys.

    Purpose:
        Provides exact checkpoint compatibility without substring
        replacement, tensor reshaping, or tensor reinitialization.

    Args:
        state_dict: Model state dictionary whose values are tensors.

    Returns:
        New dictionary with keys renamed according to
        `EXPLICIT_CHECKPOINT_KEY_MAPPING`. Tensor objects are reused exactly.
    """

    mapped_state_dict: dict[str, torch.Tensor] = {}
    for source_key, source_tensor in state_dict.items():
        target_key = EXPLICIT_CHECKPOINT_KEY_MAPPING.get(source_key, source_key)
        if target_key in mapped_state_dict:
            raise ValueError(f"Duplicate checkpoint target key after mapping: {target_key}")
        mapped_state_dict[target_key] = source_tensor
    return mapped_state_dict


__all__ = [
    "EXPLICIT_CHECKPOINT_KEY_MAPPING",
    "ReverseBranchingDenoiser",
    "ReverseDenoisingStepInfo",
    "build_permutation_equivariant_mlp",
    "map_checkpoint_state_dict_keys",
]
