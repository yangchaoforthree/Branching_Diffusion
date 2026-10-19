"""Forward branching diffusion process for temporal point processes."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


class BranchingForwardProcess:
    """Forward branching diffusion process on one-dimensional event times.

    Purpose:
        Implements the paper's temporal forward noising process as an
        operator-splitting chain: transport perturbation, independent
        survival/death sampling, and homogeneous reference births.

    Args:
        diffusion_steps: Number of forward noising steps `K`.
        time_horizon: Observation horizon `T` in event-time units.
        reference_mass: Expected event count used to define the homogeneous
            reference event rate. Must be nonnegative.
        reference_rate_scale: Multiplier applied to the reference event rate
            for the per-step reference birth intensity.
        mass_retention_schedule: Optional schedule `rho_0, ..., rho_K` for
            expected mass retention. If omitted, uses a linear schedule from
            `1.0` to `0.0`.
        transport_step_schedule: Optional list of `K` step sizes for Gaussian
            temporal perturbation. If omitted, each step uses `0.1 / K`.
        pad_value: Numeric temporal padding sentinel retained for metadata.
    """

    def __init__(
        self,
        diffusion_steps: int,
        time_horizon: float,
        reference_mass: float,
        reference_rate_scale: float = 1.0,
        mass_retention_schedule: Sequence[float] | None = None,
        transport_step_schedule: Sequence[float] | None = None,
        pad_value: float = -1.0,
    ) -> None:
        """Initialize schedules, reference event rate, and birth intensities."""

        if diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        self.diffusion_steps = int(diffusion_steps)
        self.time_horizon = float(time_horizon)
        self.pad_value = float(pad_value)
        self.reference_event_rate = float(reference_mass) / max(self.time_horizon, 1e-6)
        self.reference_rate_scale = float(reference_rate_scale)
        if mass_retention_schedule is None:
            self.mass_retention_schedule = torch.linspace(
                1.0,
                0.0,
                steps=self.diffusion_steps + 1,
                dtype=torch.float32,
            )
        else:
            self.mass_retention_schedule = torch.tensor(mass_retention_schedule, dtype=torch.float32)
        if transport_step_schedule is None:
            self.transport_step_schedule = [0.1 / self.diffusion_steps] * self.diffusion_steps
        else:
            self.transport_step_schedule = list(transport_step_schedule)
        self.survival_probabilities = self.compute_survival_probabilities()
        self.reference_birth_rates = self.compute_reference_birth_rates()

    def compute_survival_probabilities(self) -> list[float]:
        """Compute per-step survival probabilities from the mass schedule.

        Purpose:
            Converts the paper's retention schedule into Bernoulli survival
            probabilities used by the birth-death step.

        Args:
            None.

        Returns:
            A list of `K` Python floats. Entry `k` equals
            `rho_{k+1} / rho_k` when `rho_k > 0`, otherwise `0.0`.
        """

        survival_probabilities: list[float] = []
        for diffusion_step_index in range(1, len(self.mass_retention_schedule)):
            previous_mass_retention = self.mass_retention_schedule[diffusion_step_index - 1]
            survival_probabilities.append(
                0.0
                if previous_mass_retention <= 0
                else float(self.mass_retention_schedule[diffusion_step_index] / previous_mass_retention)
            )
        return survival_probabilities

    def compute_reference_birth_rates(self) -> list[float]:
        """Compute homogeneous reference birth rates for all forward steps.

        Purpose:
            Implements the reference-birth intensity associated with the
            decrease in retained data mass between adjacent diffusion steps.

        Args:
            None.

        Returns:
            A list of `K` Python floats with units events per event-time unit.
            Entry `k` is
            `(rho_k - rho_{k+1}) * reference_event_rate * reference_rate_scale`.
        """

        reference_birth_rates: list[float] = []
        for diffusion_step_index in range(1, len(self.mass_retention_schedule)):
            retention_drop = max(
                float(
                    self.mass_retention_schedule[diffusion_step_index - 1]
                    - self.mass_retention_schedule[diffusion_step_index]
                ),
                0.0,
            )
            reference_birth_rates.append(
                retention_drop * self.reference_event_rate * self.reference_rate_scale
            )
        return reference_birth_rates

    def apply_transport_step(self, event_times: torch.Tensor, step_size: float) -> torch.Tensor:
        """Apply the temporal transport perturbation.

        Purpose:
            Applies the paper's transport step for a homogeneous temporal
            reference process, where the reference transport field vanishes
            and only Gaussian perturbation remains.

        Args:
            event_times: One-dimensional `float32` or floating tensor with
                shape `(num_events,)`, defined on `[0, time_horizon]`.
            step_size: Nonnegative diffusion step size controlling the
                perturbation variance.

        Returns:
            Tensor with the same shape, dtype, and device as `event_times`,
            computed as `event_times + randn_like(event_times) *
            sqrt(2 * step_size)` and clamped to `[0, time_horizon]`.
        """

        if event_times.numel() == 0:
            return event_times
        temporal_noise = torch.randn_like(event_times) * math.sqrt(2.0 * step_size)
        return torch.clamp(event_times + temporal_noise, 0.0, self.time_horizon)

    @staticmethod
    def apply_birth_death_step(
        parent_event_times: torch.Tensor,
        survival_probability: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply independent death sampling to parent events.

        Purpose:
            Implements the death/survival part of the paper's birth-death
            step for the current temporal implementation. Offspring from
            surviving parents are not parameterized separately in this code;
            reference births are sampled by `sample_reference_births`.

        Args:
            parent_event_times: One-dimensional floating tensor with shape
                `(num_parent_events,)`.
            survival_probability: Bernoulli probability in `[0, 1]` for
                retaining each parent event.

        Returns:
            `(surviving_event_times, deleted_event_times, survival_mask)`.
            The first two tensors have the same dtype and device as
            `parent_event_times`; `survival_mask` is boolean with shape
            `(num_parent_events,)`.
        """

        if parent_event_times.numel() > 0:
            survival_mask = torch.rand_like(parent_event_times) < survival_probability
            surviving_event_times = parent_event_times[survival_mask]
            deleted_event_times = parent_event_times[~survival_mask]
        else:
            survival_mask = torch.zeros_like(parent_event_times, dtype=torch.bool)
            surviving_event_times = parent_event_times
            deleted_event_times = parent_event_times
        return surviving_event_times, deleted_event_times, survival_mask

    def sample_reference_births(self, reference_birth_rate: float, device: torch.device) -> torch.Tensor:
        """Sample homogeneous reference birth events.

        Purpose:
            Samples the paper's reference births from the temporal terminal
            Poisson reference process restricted to one forward step.

        Args:
            reference_birth_rate: Event intensity per unit time for the current
                forward step.
            device: Torch device for random sampling and returned storage.

        Returns:
            One-dimensional `float32` tensor of event times on `[0,
            time_horizon]`. Cardinality is sampled from
            `Poisson(reference_birth_rate * time_horizon)`.
        """

        terminal_event_count = torch.poisson(
            torch.tensor(reference_birth_rate * self.time_horizon, device=device)
        ).to(torch.int64).item()
        if terminal_event_count <= 0:
            return torch.empty(0, device=device, dtype=torch.float32)
        return torch.rand(terminal_event_count, device=device, dtype=torch.float32) * self.time_horizon

    def forward_noising_step(
        self,
        current_event_times: torch.Tensor,
        diffusion_step_index: int,
    ) -> dict[str, torch.Tensor]:
        """Run one forward noising step.

        Purpose:
            Applies transport, survival/death sampling, and reference
            births in the same order as the implemented training path.

        Args:
            current_event_times: One-dimensional event-time tensor with shape
                `(num_events,)`.
            diffusion_step_index: Zero-based index in `[0, K - 1]`.

        Returns:
            Dictionary with tensor keys `next_event_times`,
            `transported_event_times`, `surviving_event_times`,
            `deleted_event_times`, `reference_birth_times`,
            `survival_mask`, and `survival_labels`. The next measure is the
            concatenation of surviving parent events and reference births; it
            is not sorted by this method.
        """

        transport_step_size = self.transport_step_schedule[diffusion_step_index]
        survival_probability = self.survival_probabilities[diffusion_step_index]
        reference_birth_rate = self.reference_birth_rates[diffusion_step_index]
        device = current_event_times.device
        transported_event_times = self.apply_transport_step(
            current_event_times,
            transport_step_size,
        )
        surviving_event_times, deleted_event_times, survival_mask = self.apply_birth_death_step(
            transported_event_times,
            survival_probability,
        )
        reference_birth_times = self.sample_reference_births(reference_birth_rate, device=device)
        next_event_times = torch.cat([surviving_event_times, reference_birth_times], dim=0)
        survival_labels = torch.cat(
            [
                torch.ones(surviving_event_times.shape[0], dtype=torch.float32, device=device),
                torch.zeros(reference_birth_times.shape[0], dtype=torch.float32, device=device),
            ],
            dim=0,
        )
        return {
            "next_event_times": next_event_times,
            "transported_event_times": transported_event_times,
            "surviving_event_times": surviving_event_times,
            "deleted_event_times": deleted_event_times,
            "reference_birth_times": reference_birth_times,
            "survival_mask": survival_mask,
            "survival_labels": survival_labels,
        }

    def sample_forward_trajectory(
        self,
        initial_event_times: torch.Tensor,
        num_steps: int,
    ) -> list[dict[str, torch.Tensor]]:
        """Sample a forward noising trajectory.

        Purpose:
            Runs the forward Markov chain from an initial counting measure for
            the requested number of steps.

        Args:
            initial_event_times: One-dimensional event-time tensor with shape
                `(num_initial_events,)`.
            num_steps: Number of forward steps to execute from step zero.

        Returns:
            List of per-step dictionaries returned by
            `forward_noising_step`. The `next_event_times` tensor from each
            step is passed directly to the next step.
        """

        forward_trajectory: list[dict[str, torch.Tensor]] = []
        current_event_times = initial_event_times
        for diffusion_step_index in range(num_steps):
            step_state = self.forward_noising_step(current_event_times, diffusion_step_index)
            forward_trajectory.append(step_state)
            current_event_times = step_state["next_event_times"]
        return forward_trajectory

    def sample_noisy_measure_given_initial(
        self,
        initial_event_times: torch.Tensor,
        step_count: int,
    ) -> torch.Tensor:
        """Sample a noisy counting measure from the initial event times.

        Purpose:
            Convenience wrapper for obtaining the event times after
            `step_count` forward noising steps.

        Args:
            initial_event_times: One-dimensional event-time tensor.
            step_count: Number of forward noising steps to run.

        Returns:
            One-dimensional tensor of event times after `step_count` steps.
        """

        if step_count == 0:
            return initial_event_times
        forward_trajectory = self.sample_forward_trajectory(initial_event_times, num_steps=step_count)
        return forward_trajectory[-1]["next_event_times"]


__all__ = ["BranchingForwardProcess"]
