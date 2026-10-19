"""Evaluation metrics for temporal point-process sequence generation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from scipy.stats import wasserstein_distance

from branching_diffusion.entropic_unbalanced_ot import compute_debiased_sinkhorn_divergence


def unpad_sequence(times: torch.Tensor, pad_value: float) -> torch.Tensor:
    """Remove numeric padding from one batched sequence.

    Args:
        times: One-dimensional padded event-time tensor.
        pad_value: Numeric padding sentinel.

    Returns:
        Event times where `times != pad_value`.
    """

    return times[times != pad_value]


def collect_test_sequences(
    test_dataloader: object,
    pad_value: float,
    max_sequences: int | None = None,
) -> list[torch.Tensor]:
    """Collect nonempty test sequences using numeric padding removal.

    Args:
        test_dataloader: DataLoader yielding dictionaries with `event_time`.
        pad_value: Numeric padding sentinel.
        max_sequences: Optional maximum number of sequences to keep.

    Returns:
        A list of CPU tensors. Empty sequences are skipped.
    """

    test_sequences: list[torch.Tensor] = []
    for batch in test_dataloader:
        event_time = batch["event_time"]
        for batch_index in range(event_time.shape[0]):
            times = unpad_sequence(event_time[batch_index], pad_value)
            if times.numel() > 0:
                test_sequences.append(times.cpu())
        if max_sequences is not None and len(test_sequences) >= max_sequences:
            test_sequences = test_sequences[:max_sequences]
            break
    return test_sequences


def match_shapes(
    x_sequences: list[np.ndarray],
    y_sequences: list[np.ndarray],
    time_horizon: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad two event-sequence sets to the matched metric array shape.

    Args:
        x_sequences: First list of one-dimensional event-time arrays.
        y_sequences: Second list of one-dimensional event-time arrays.
        time_horizon: Padding value and event-time horizon.

    Returns:
        Two arrays of shape `(num_sequences, max_valid_events)`.
    """

    max_x = max([(x < time_horizon).sum() for x in x_sequences])
    max_y = max([(y < time_horizon).sum() for y in y_sequences])
    max_size = max(max_x, max_y)
    new_x = np.ones((len(x_sequences), max_size)) * time_horizon
    new_y = np.ones((len(y_sequences), max_size)) * time_horizon
    for index, x in enumerate(x_sequences):
        valid = x[x < time_horizon]
        new_x[index, : len(valid)] = valid
    for index, y in enumerate(y_sequences):
        valid = y[y < time_horizon]
        new_y[index, : len(valid)] = valid
    return new_x, new_y


def counting_distance(x: np.ndarray, y_array: np.ndarray, time_horizon: float) -> np.ndarray:
    """Compute the counting-process sequence distance.

    Args:
        x: One sequence array with shape `(length,)`.
        y_array: Matched sequence array with shape `(num_sequences, length)`.
        time_horizon: Sentinel and endpoint used by the distance.

    Returns:
        A one-dimensional array of distances from `x` to each row of
        `y_array`.
    """

    x_copy, y_copy = x.copy(), y_array.copy()
    x_copy = x_copy[None].repeat(y_copy.shape[0], 0)
    x_len = (x_copy < time_horizon).sum(-1)
    y_len = (y_copy < time_horizon).sum(-1)
    to_swap = x_len > y_len
    x_copy[to_swap], y_copy[to_swap] = y_copy[to_swap].copy(), x_copy[to_swap].copy()
    mask_x = x_copy < time_horizon
    mask_y = y_copy < time_horizon
    result = (np.abs(x_copy - y_copy) * mask_x).sum(-1)
    result += ((time_horizon - y_copy) * (~mask_x & mask_y)).sum(-1)
    return result


def gaussian_kernel(distances: np.ndarray, sigma2: float) -> np.ndarray:
    """Apply the RBF kernel to counting distances.

    Args:
        distances: Counting-distance values.
        sigma2: Squared bandwidth.

    Returns:
        Kernel values `exp(-distance / (2 * sigma2))`.
    """

    return np.exp(-distances / (2 * sigma2))


def compute_counting_process_mmd(
    real_sequences: list[torch.Tensor],
    generated_sequences: list[torch.Tensor],
    time_horizon: float,
    sigma: Optional[float] = None,
) -> tuple[float, float]:
    """Compute biased empirical MMD over counting-process distance.

    Args:
        real_sequences: Real event-time tensors.
        generated_sequences: Generated event-time tensors.
        time_horizon: Observation horizon and padding sentinel for matching.
        sigma: Optional bandwidth. If omitted, the median of all
            within-real, cross, and within-generated distances, with fallback
            `1.0` when the median is zero.

    Returns:
        `(mmd, sigma_used)`, both Python floats.
    """

    x_list = [times.cpu().numpy() for times in real_sequences]
    y_list = [times.cpu().numpy() for times in generated_sequences]
    x_array, y_array = match_shapes(x_list, y_list, time_horizon)
    x_array = x_array / time_horizon
    y_array = y_array / time_horizon
    normalized_horizon = 1.0

    x_x_distances = []
    for x_item in x_array:
        x_x_distances.append(counting_distance(x_item, x_array, time_horizon=normalized_horizon))
    x_x_d = np.concatenate(x_x_distances)

    x_y_distances = []
    for x_item in x_array:
        x_y_distances.append(counting_distance(x_item, y_array, time_horizon=normalized_horizon))
    x_y_d = np.concatenate(x_y_distances)

    y_y_distances = []
    for y_item in y_array:
        y_y_distances.append(counting_distance(y_item, y_array, time_horizon=normalized_horizon))
    y_y_d = np.concatenate(y_y_distances)

    if sigma is None:
        sigma = np.median(np.concatenate([x_x_d, x_y_d, y_y_d]))
        if sigma == 0:
            sigma = 1.0
    sigma2 = sigma**2
    expectation_xx = np.mean(gaussian_kernel(x_x_d, sigma2))
    expectation_xy = np.mean(gaussian_kernel(x_y_d, sigma2))
    expectation_yy = np.mean(gaussian_kernel(y_y_d, sigma2))
    mmd_squared = expectation_xx - 2 * expectation_xy + expectation_yy
    return float(np.sqrt(max(0.0, mmd_squared))), float(sigma)


def compute_event_count_wasserstein_distance(
    real_sequences: list[torch.Tensor],
    generated_sequences: list[torch.Tensor],
) -> float:
    """Compute event-count Wasserstein distance.

    Purpose:
        Compares the generated and target event-count distributions while
        ignoring event times and marks.

    Args:
        real_sequences: Real event-time tensors. Only `numel()` is used.
        generated_sequences: Generated event-time tensors. Only `numel()` is
            used.

    Returns:
        A Python float from `scipy.stats.wasserstein_distance` applied to the
        two empirical event-count arrays.

    Raises:
        ValueError: Propagated by SciPy if either empirical distribution is
            empty.

    Randomness:
        This function is deterministic.
    """

    real_lengths = np.array([times.numel() for times in real_sequences])
    generated_lengths = np.array([times.numel() for times in generated_sequences])
    return float(wasserstein_distance(real_lengths, generated_lengths))


def compute_temporal_sinkhorn_divergence(
    real_sequences: list[torch.Tensor],
    generated_sequences: list[torch.Tensor],
    time_horizon: float,
    sigma_t: float = 1.0,
    eps: float = 0.01,
    lam: float = 100.0,
) -> float:
    """Compute temporal Sinkhorn divergence over paired sequences.

    Args:
        real_sequences: Real event-time tensors.
        generated_sequences: Generated event-time tensors paired with
            `real_sequences` by `zip`.
        time_horizon: Observation horizon used to normalize event times to
            `[0, 1]`.
        sigma_t: Preserved public scale parameter. The reference temporal-only
            implementation accepts this argument but does not use it in the
            computation.
        eps: Entropic regularization parameter for the debiased Sinkhorn
            divergence.
        lam: Unbalanced mass penalty for the debiased Sinkhorn divergence.

    Returns:
        The mean debiased Sinkhorn divergence over nonempty paired sequences.
        If every pair contains an empty sequence, returns `0.0`.
    """

    del sigma_t
    horizon = float(time_horizon.item() if isinstance(time_horizon, torch.Tensor) else time_horizon)
    total_divergence = 0.0
    valid_pairs = 0
    for real_times, generated_times in zip(real_sequences, generated_sequences):
        if real_times.numel() == 0 or generated_times.numel() == 0:
            continue
        real_times_2d = (real_times / horizon).unsqueeze(1)
        generated_times_2d = (generated_times / horizon).unsqueeze(1)
        real_weights = torch.ones(
            real_times_2d.shape[0],
            device=real_times.device,
        ) / real_times_2d.shape[0]
        generated_weights = torch.ones(
            generated_times_2d.shape[0],
            device=generated_times.device,
        ) / generated_times_2d.shape[0]
        try:
            divergence = compute_debiased_sinkhorn_divergence(
                x=real_times_2d,
                y=generated_times_2d,
                w_x=real_weights,
                w_y=generated_weights,
                eps=eps,
                lam=lam,
                p=2.0,
            )
            total_divergence += divergence.item()
            valid_pairs += 1
        except Exception:
            continue
    if valid_pairs == 0:
        return 0.0
    return total_divergence / valid_pairs


def compute_all_evaluation_metrics(
    real_sequences: list[torch.Tensor],
    generated_sequences: list[torch.Tensor],
    time_horizon: float,
    sigma: Optional[float] = None,
) -> dict[str, float]:
    """Compute all public evaluation metrics from one generated sample set.

    Args:
        real_sequences: Real event-time tensors.
        generated_sequences: Generated event-time tensors.
        time_horizon: Observation horizon used by temporal metrics.
        sigma: Optional MMD bandwidth. If omitted, the median heuristic is
            used.

    Returns:
        A JSON-safe dictionary containing `mmd`, `mmd_bandwidth`, `stsd`, `wd`,
        sequence count statistics, and length-difference metadata.
    """

    mmd, bandwidth = compute_counting_process_mmd(
        real_sequences,
        generated_sequences,
        time_horizon=time_horizon,
        sigma=sigma,
    )
    wd = compute_event_count_wasserstein_distance(real_sequences, generated_sequences)
    stsd = compute_temporal_sinkhorn_divergence(
        real_sequences,
        generated_sequences,
        time_horizon=time_horizon,
    )
    real_lengths = [times.numel() for times in real_sequences]
    generated_lengths = [times.numel() for times in generated_sequences]
    return {
        "mmd": float(mmd),
        "mmd_bandwidth": float(bandwidth),
        "stsd": float(stsd),
        "wd": float(wd),
        "real_length_mean": float(np.mean(real_lengths)),
        "real_length_std": float(np.std(real_lengths)),
        "generated_length_mean": float(np.mean(generated_lengths)),
        "generated_length_std": float(np.std(generated_lengths)),
        "length_mean_difference": float(abs(np.mean(generated_lengths) - np.mean(real_lengths))),
    }


__all__ = [
    "collect_test_sequences",
    "compute_all_evaluation_metrics",
    "compute_counting_process_mmd",
    "compute_temporal_sinkhorn_divergence",
    "compute_event_count_wasserstein_distance",
    "counting_distance",
    "gaussian_kernel",
    "match_shapes",
    "unpad_sequence",
]
