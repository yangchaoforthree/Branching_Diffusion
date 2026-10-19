"""YAML configuration for the dataset name and hyperparameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration loaded from YAML.

    Args:
        dataset: Dataset filename resolved under the configured data root.
            Supported extensions are `.npy` and `.pkl`.
        max_epochs: Number of training epochs.
        batch_size: Batch size for all dataloaders.
        sinkhorn_iterations: Number of direct-domain Sinkhorn iterations.
        diffusion_steps: Number of forward and reverse diffusion steps.
        learning_rate: Adam learning rate.
        optimizer: Optimizer name. Only `Adam` is accepted.
    """

    dataset: str
    max_epochs: int
    batch_size: int
    sinkhorn_iterations: int
    diffusion_steps: int
    learning_rate: float
    optimizer: str

    def validate(self) -> None:
        """Validate configuration values without modifying them.

        Raises:
            ValueError: If a value is invalid for the supported training path.
        """

        if not self.dataset:
            raise ValueError("dataset must be a non-empty filename.")
        if Path(self.dataset).name != self.dataset:
            raise ValueError("dataset must be a filename, not a path.")
        if Path(self.dataset).suffix.lower() not in {".npy", ".pkl", ".pickle"}:
            raise ValueError("dataset must use a supported extension: .npy, .pkl, or .pickle.")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if self.sinkhorn_iterations <= 0:
            raise ValueError("sinkhorn_iterations must be positive.")
        if self.diffusion_steps <= 0:
            raise ValueError("diffusion_steps must be positive.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.optimizer != "Adam":
            raise ValueError("Only the Adam optimizer is supported.")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary representation.

        Returns:
            A dictionary containing exactly the dataclass fields.
        """

        return asdict(self)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load and validate the YAML experiment configuration.

    Args:
        path: Path to a YAML file containing the dataset filename and the
            public table-level hyperparameters.

    Returns:
        A validated `ExperimentConfig` instance.
    """

    if not path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    if "diffusion_steps" not in raw and "reverse_steps" in raw:
        raw = dict(raw)
        raw["diffusion_steps"] = raw.pop("reverse_steps")
    expected = set(ExperimentConfig.__dataclass_fields__.keys())
    actual = set(raw.keys())
    if actual != expected:
        raise ValueError(
            "Configuration must contain exactly these keys: "
            + ", ".join(sorted(expected))
        )
    config = ExperimentConfig(**raw)
    config.validate()
    return config
