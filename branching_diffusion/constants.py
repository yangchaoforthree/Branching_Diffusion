"""Fixed algorithm settings for the temporal branching diffusion path."""

from __future__ import annotations

DATASET_FILENAME = "three_peak_ipp_40T_2000seqs.npy"
"""Default dataset filename."""

PADDING_VALUE_TIME = -1.0
"""Temporal padding sentinel used by dataset collation and training masks."""

PADDING_VALUE_TYPE = -1
"""Mark padding sentinel retained in event-type batches."""

TRAIN_RATIO = 0.8
"""Train split ratio used before floor rounding."""

VALIDATION_RATIO = 0.1
"""Middle split ratio used before assigning the remaining sequences to test."""

REFERENCE_RATE_SCALE = 1.0
"""Default multiplier for homogeneous Poisson reference births."""

PROPOSAL_OVERSAMPLING = 2.5
"""Candidate-birth oversampling multiplier used by the verified training setup."""

HIDDEN_DIMENSION = 128
"""Reverse denoiser hidden width used by the training entry point."""

ENCODER_LAYERS = 2
"""Number of DeepSets MLP layers used by the training entry point."""

SINKHORN_EPSILON = 0.01
"""Blur parameter passed to the direct-domain Sinkhorn solver."""

MASS_KL_PENALTY = 100.0
"""Unbalanced mass KL penalty passed to the Sinkhorn solver."""

TRANSPORT_MODE = "uot"
"""Transport mode used by the training loss."""

USE_BIAS_CORRECTION = True
"""Whether the training loss uses the bias-corrected Sinkhorn scalar branch."""

TIME_COST_SCALE = 1.0
"""Temporal cost scaling denominator."""

SINKHORN_POWER = 2.0
"""Temporal ground-cost power for `torch.cdist(..., p) ** p`."""

GRADIENT_CLIP_FALLBACK = 1.0
"""Effective clipping norm used when no explicit clip value is supplied."""

SAVE_INTERVAL = 5
"""Callback interval for epoch checkpoints and test-set MMD selection."""

EVALUATION_SEQUENCE_COUNT = 200
"""Default number of generated and real sequences used by evaluation."""

DEFAULT_SEED = 42
"""Default seed used by the training and data-loading path."""

CHECKPOINT_FILENAME = "reverse_denoiser.pt"
"""Best reverse-denoiser checkpoint filename used by training outputs."""
