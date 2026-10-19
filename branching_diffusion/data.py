"""Temporal point-process dataset loading with preserved preprocessing behavior."""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from branching_diffusion.constants import (
    PADDING_VALUE_TIME,
    PADDING_VALUE_TYPE,
    TRAIN_RATIO,
    VALIDATION_RATIO,
)


SUPPORTED_DATASET_EXTENSIONS = {".npy", ".pkl", ".pickle"}
"""Dataset file extensions accepted by the public data loader."""


def resolve_dataset_path(data_root: Path, dataset_name: str) -> Path:
    """Resolve and validate a configured point-process dataset path.

    Args:
        data_root: Directory containing local temporal point-process datasets.
        dataset_name: Dataset filename from YAML or the command line. The value
            must be a filename rather than an arbitrary path.

    Returns:
        An absolute path to an existing `.npy`, `.pkl`, or `.pickle` file.
    """

    if Path(dataset_name).name != dataset_name:
        raise ValueError("dataset_name must be a filename, not a path.")
    dataset_path = (data_root / dataset_name).expanduser().resolve()
    if dataset_path.suffix.lower() not in SUPPORTED_DATASET_EXTENSIONS:
        raise ValueError("Dataset extension must be .npy, .pkl, or .pickle.")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    return dataset_path


def _load_serialized_dataset(dataset_path: Path) -> Any:
    """Read a serialized local dataset container.

    Args:
        dataset_path: Existing `.npy`, `.pkl`, or `.pickle` file.

    Returns:
        The top-level Python object loaded from the file.
    """

    suffix = dataset_path.suffix.lower()
    if suffix == ".npy":
        loaded = np.load(str(dataset_path), allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == ():
            return loaded.item()
        return loaded
    if suffix in {".pkl", ".pickle"}:
        with dataset_path.open("rb") as handle:
            return pickle.load(handle)
    raise ValueError("Dataset extension must be .npy, .pkl, or .pickle.")


def _iter_sequence_objects(raw: Any) -> list[Any]:
    """Convert a supported top-level container into sequence objects.

    Args:
        raw: Object returned by `_load_serialized_dataset`.

    Returns:
        A list preserving the container iteration order.

    Raises:
        ValueError: If the top-level object cannot be interpreted as a
            sequence collection or contains no sequences.
    """

    if isinstance(raw, dict):
        sequences = list(raw.values())
    elif isinstance(raw, np.ndarray):
        if raw.shape == ():
            return _iter_sequence_objects(raw.item())
        sequences = list(raw.tolist() if raw.dtype == object else raw)
    elif isinstance(raw, (list, tuple)):
        sequences = list(raw)
    else:
        raise ValueError("Dataset must be a dictionary, list, tuple, or NumPy array of sequences.")
    if not sequences:
        raise ValueError("Dataset must contain at least one sequence.")
    return sequences


def _as_one_dimensional_array(value: Any, field_name: str, sequence_index: int) -> np.ndarray:
    """Convert one sequence field into a finite one-dimensional array.

    Args:
        value: Array-like field value.
        field_name: Name used in validation errors.
        sequence_index: Zero-based index of the sequence being validated.

    Returns:
        A one-dimensional NumPy array.
    """

    try:
        array = np.asarray(value)
    except Exception as exc:
        raise ValueError(f"Sequence {sequence_index} field {field_name} is not array-like.") from exc
    if array.ndim != 1:
        raise ValueError(f"Sequence {sequence_index} field {field_name} must be one-dimensional.")
    numeric = array.astype(float if field_name == "event_time" else int, copy=False)
    if not np.isfinite(numeric.astype(float)).all():
        raise ValueError(f"Sequence {sequence_index} field {field_name} contains NaN or Inf.")
    return numeric


def _normalize_sequence(seq: Any, sequence_index: int) -> dict[str, list[float]]:
    """Normalize one supported sequence object without changing padding policy.

    Args:
        seq: Sequence object. Preferred schema is a dictionary with
            `event_time` and optional `event_type` arrays. Dictionaries mapping
            mark ids to time arrays and plain array-like time lists are also
            accepted for input-format compatibility.
        sequence_index: Zero-based index used for error reporting.

    Returns:
        A dictionary with `event_time` as floats and `event_type` as integers.
        Values are sorted by event time, and source padding remains present.
    """

    if isinstance(seq, dict) and "event_time" in seq:
        event_times = _as_one_dimensional_array(seq["event_time"], "event_time", sequence_index)
        if "event_type" in seq:
            event_types = _as_one_dimensional_array(seq["event_type"], "event_type", sequence_index)
        else:
            event_types = np.zeros(event_times.shape[0], dtype=int)
    elif isinstance(seq, dict):
        event_times_list: list[float] = []
        event_types_list: list[int] = []
        for event_type, times in seq.items():
            times_array = _as_one_dimensional_array(times, "event_time", sequence_index)
            event_times_list.extend(float(item) for item in times_array)
            event_types_list.extend([int(event_type)] * len(times_array))
        event_times = np.asarray(event_times_list, dtype=float)
        event_types = np.asarray(event_types_list, dtype=int)
    elif isinstance(seq, (list, tuple, np.ndarray)):
        event_times = _as_one_dimensional_array(seq, "event_time", sequence_index)
        event_types = np.zeros(event_times.shape[0], dtype=int)
    else:
        raise ValueError(f"Sequence {sequence_index} has unsupported structure.")

    if event_times.shape[0] != event_types.shape[0]:
        raise ValueError(f"Sequence {sequence_index} event_time and event_type lengths differ.")
    paired = sorted(zip(event_times.tolist(), event_types.tolist()), key=lambda item: item[0])
    return {
        "event_time": [float(t) for t, _ in paired],
        "event_type": [int(k) for _, k in paired],
    }


def load_temporal_point_process_dataset(dataset_path: Path) -> list[dict[str, list[float]]]:
    """Load a temporal point-process dataset from a local `.npy` or pickle file.

    Args:
        dataset_path: Path to a supported serialized dataset.

    Returns:
        A list of normalized sequence dictionaries. Each sequence has
        `event_time` and `event_type` lists, and source padding is preserved.
    """

    dataset_path = dataset_path.expanduser().resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {dataset_path}")
    raw = _load_serialized_dataset(dataset_path)
    return [_normalize_sequence(seq, index) for index, seq in enumerate(_iter_sequence_objects(raw))]


def load_raw_sequences(data_path: Path) -> list[dict[str, list[float]]]:
    """Load normalized sequence dictionaries from a supported dataset file.

    Args:
        data_path: Path to a trusted local `.npy`, `.pkl`, or `.pickle` file.

    Returns:
        Sequence dictionaries with sorted `event_time` and `event_type` lists.
        Existing numeric padding remains present until batch-level masking.
    """

    return load_temporal_point_process_dataset(data_path)


def split_sequences(
    sequences: list[dict[str, list[float]]],
    train_ratio: float = TRAIN_RATIO,
    validation_ratio: float = VALIDATION_RATIO,
) -> tuple[list[dict[str, list[float]]], list[dict[str, list[float]]], list[dict[str, list[float]]]]:
    """Split sequences with floor rounding and global `torch.randperm`.

    Args:
        sequences: Loaded sequence dictionaries in source order.
        train_ratio: Fraction used for the training split before floor
            rounding. The default is 0.8.
        validation_ratio: Fraction used for the middle split before floor
            rounding. The default is 0.1.

    Returns:
        Three lists for train, middle, and test splits.
    """

    total = len(sequences)
    if total == 0:
        raise ValueError("Sequence list is empty and cannot be split.")

    train_count = int(math.floor(total * train_ratio))
    validation_count = int(math.floor(total * validation_ratio))
    test_count = total - train_count - validation_count
    if test_count <= 0:
        test_count = 1
        train_count = max(train_count - 1, 1)

    shuffled_indices = torch.randperm(total).tolist()
    shuffled = [sequences[idx] for idx in shuffled_indices]
    train_set = shuffled[:train_count]
    validation_set = shuffled[train_count : train_count + validation_count]
    test_set = shuffled[train_count + validation_count :]

    if not train_set:
        if validation_set:
            train_set.append(validation_set.pop())
        elif test_set:
            train_set.append(test_set.pop())
    if not test_set and len(train_set) > 1:
        test_set.append(train_set.pop())

    return train_set, validation_set, test_set


class EventSequenceDataset(Dataset):
    """Torch dataset wrapper preserving stored-sequence metadata statistics.

    Args:
        sequences: Sequence dictionaries with `event_time` and `event_type`.

    Attributes:
        event_counts: Per-sequence stored lengths, including source padding.
        max_time: Maximum stored event time in the split.
        max_length: Maximum stored sequence length in the split.
        mean_count: Mean stored length in the split.
        num_marks: Number of observed marks, including source mark values.
    """

    def __init__(self, sequences: list[dict[str, list[float]]]) -> None:
        """Initialize the dataset and compute stored-sequence metadata."""

        self.sequences = sequences
        self.event_counts = [len(seq["event_time"]) for seq in sequences]
        self.max_time = self._compute_max_time()
        self.max_length = max(self.event_counts) if self.event_counts else 0
        self.mean_count = float(np.mean(self.event_counts)) if self.event_counts else 0.0
        self.num_marks = self._compute_num_marks()

    def __len__(self) -> int:
        """Return the number of stored sequences."""

        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Return one sequence sample.

        Args:
            index: Zero-based sequence index.

        Returns:
            A dictionary containing `event_time` as `float32` and `event_type`
            as `long`. Source padding remains present.
        """

        sequence = self.sequences[index]
        event_time = torch.tensor(sequence["event_time"], dtype=torch.float32)
        event_type = torch.tensor(sequence["event_type"], dtype=torch.long)
        return {"event_time": event_time, "event_type": event_type}

    def _compute_max_time(self) -> float:
        """Compute the maximum stored event time over the split."""

        max_time = 0.0
        for seq in self.sequences:
            if seq["event_time"]:
                max_time = max(max_time, max(seq["event_time"]))
        return max_time

    def _compute_num_marks(self) -> int:
        """Compute the number of observed mark ids from stored event types."""

        max_mark = -1
        for seq in self.sequences:
            if seq["event_type"]:
                max_mark = max(max_mark, max(seq["event_type"]))
        return max_mark + 1 if max_mark >= 0 else 0


def collate_event_sequences(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad variable-length event sequences with configured sentinels.

    Args:
        batch: List of dataset samples with one-dimensional `event_time` and
            `event_type` tensors.

    Returns:
        A dictionary with `event_time` and `event_type` tensors of shape
        `(batch_size, max_length)` and a `lengths` tensor of shape
        `(batch_size,)`. The `lengths` values include source padding when the
        source file already stored it.
    """

    batch_size = len(batch)
    lengths = torch.tensor([seq["event_time"].shape[0] for seq in batch], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded_times = torch.full(
        (batch_size, max_len), fill_value=PADDING_VALUE_TIME, dtype=torch.float32
    )
    padded_types = torch.full(
        (batch_size, max_len), fill_value=PADDING_VALUE_TYPE, dtype=torch.long
    )
    for idx, sample in enumerate(batch):
        length = sample["event_time"].shape[0]
        padded_times[idx, :length] = sample["event_time"]
        padded_types[idx, :length] = sample["event_type"]
    return {"event_time": padded_times, "event_type": padded_types, "lengths": lengths}


def create_dataloaders(
    data_root: Path,
    dataset_name: str,
    batch_size: int,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """Create train, middle, and test dataloaders.

    Args:
        data_root: Directory containing the supported dataset file.
        dataset_name: Dataset filename.
        batch_size: Batch size passed to every `DataLoader`.
        seed: Seed passed to global `torch.manual_seed` before loading and
            splitting.

    Returns:
        A dictionary keyed by `train`, `val`, and `test`. Each entry contains
        the loader plus metadata fields `t_max`, `length_max`,
        `count_mean`, and `num_marks`.
    """

    torch.manual_seed(seed)
    sequences = load_raw_sequences(resolve_dataset_path(data_root, dataset_name))
    train_set, validation_set, test_set = split_sequences(sequences)
    datasets = {
        "train": EventSequenceDataset(train_set),
        "val": EventSequenceDataset(validation_set),
        "test": EventSequenceDataset(test_set),
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_event_sequences,
        )
        for split, ds in datasets.items()
    }
    meta = {}
    for split_name, dataset in datasets.items():
        meta[split_name] = {
            "t_max": float(dataset.max_time),
            "length_max": int(dataset.max_length),
            "count_mean": float(dataset.mean_count),
            "num_marks": int(dataset.num_marks),
        }
    return {split: {"loader": loaders[split], **meta[split]} for split in loaders}


__all__ = [
    "EventSequenceDataset",
    "collate_event_sequences",
    "create_dataloaders",
    "load_raw_sequences",
    "load_temporal_point_process_dataset",
    "resolve_dataset_path",
    "split_sequences",
]
