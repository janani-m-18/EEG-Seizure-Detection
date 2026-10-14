"""Training-only, EEG-safe minority oversampling for raw multichannel windows.

Conventional SMOTE interpolates feature vectors.  Applying it directly to
``(channels, samples)`` EEG tensors can distort waveform morphology and timing,
so Batch A deliberately uses transformed copies of minority-class windows
instead.  This module must only receive a training partition after a
recording-aware split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class AugmentationConfig:
    """Controls deterministic EEG-specific minority oversampling."""

    enabled: bool = False
    time_shift_seconds: float = 0.5
    amplitude_scale_range: tuple[float, float] = (0.8, 1.2)
    noise_snr_db_range: tuple[float, float] = (15.0, 25.0)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.time_shift_seconds < 0:
            raise ValueError("time_shift_seconds must be non-negative")
        if len(self.amplitude_scale_range) != 2 or not 0 < self.amplitude_scale_range[0] <= self.amplitude_scale_range[1]:
            raise ValueError("amplitude_scale_range must be positive and ordered")
        if len(self.noise_snr_db_range) != 2 or not 0 < self.noise_snr_db_range[0] <= self.noise_snr_db_range[1]:
            raise ValueError("noise_snr_db_range must be positive and ordered")


@dataclass(frozen=True, slots=True)
class AugmentedEEGData:
    """The original training windows plus explicitly identified derived copies."""

    segments: np.ndarray
    labels: np.ndarray
    is_augmented: np.ndarray
    source_indices: np.ndarray

    @property
    def augmented_count(self) -> int:
        return int(self.is_augmented.sum())


def _zero_padded_shift(segment: np.ndarray, shift_samples: int) -> np.ndarray:
    """Shift a window in time without circular wrap-around."""
    shifted = np.zeros_like(segment)
    shift_samples = max(-(segment.shape[1] - 1), min(segment.shape[1] - 1, shift_samples))
    if shift_samples > 0:
        shifted[:, shift_samples:] = segment[:, :-shift_samples]
    elif shift_samples < 0:
        shifted[:, :shift_samples] = segment[:, -shift_samples:]
    else:
        shifted[:] = segment
    return shifted


def augment_eeg_segment(segment: np.ndarray, sampling_frequency: float, config: AugmentationConfig,
                        rng: np.random.Generator) -> np.ndarray:
    """Apply one existing EEG-safe augmentation transform deterministically."""
    values = np.asarray(segment, dtype=np.float32)
    if values.ndim != 2 or sampling_frequency <= 0:
        raise ValueError("segment must be a 2D EEG window and sampling_frequency must be positive")
    max_shift = round(config.time_shift_seconds * sampling_frequency)
    shift = int(rng.integers(-max_shift, max_shift + 1)) if max_shift else 0
    transformed = _zero_padded_shift(values, shift)
    transformed *= np.float32(rng.uniform(*config.amplitude_scale_range))
    signal_power = float(np.mean(np.square(transformed, dtype=np.float64)))
    snr_db = float(rng.uniform(*config.noise_snr_db_range))
    noise_std = np.sqrt(signal_power / (10 ** (snr_db / 10))) if signal_power else 0.0
    if noise_std:
        transformed += rng.normal(0.0, noise_std, size=transformed.shape).astype(np.float32)
    return transformed


def augment_training_segments(
    segments: np.ndarray,
    labels: np.ndarray,
    sampling_frequency: float,
    config: AugmentationConfig | None = None,
) -> AugmentedEEGData:
    """Balance a training partition with transformed minority-class EEG copies.

    No validation/test data, recording identifiers, or normalization statistics
    are accepted here; callers must split recordings first and pass only the
    already independently normalized training-window array.
    """
    policy = config or AugmentationConfig()
    values = np.asarray(segments, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if values.ndim != 3 or len(values) == 0:
        raise ValueError("segments must be non-empty with shape (items, channels, samples)")
    if targets.shape != (len(values),):
        raise ValueError("labels must contain one value per segment")
    if sampling_frequency <= 0:
        raise ValueError("sampling_frequency must be positive")

    originals = np.arange(len(values), dtype=np.int64)
    unchanged = AugmentedEEGData(values.copy(), targets.copy(), np.zeros(len(values), dtype=bool), originals)
    if not policy.enabled:
        return unchanged
    labels_present, counts = np.unique(targets, return_counts=True)
    if len(labels_present) < 2 or counts.min() == counts.max():
        return unchanged

    minority_label = int(labels_present[np.argmin(counts)])
    minority_indices = np.flatnonzero(targets == minority_label)
    extra_count = int(counts.max() - counts.min())
    rng = np.random.default_rng(policy.seed)
    additions: list[np.ndarray] = []
    sources: list[int] = []
    for source_index in rng.choice(minority_indices, size=extra_count, replace=True):
        additions.append(augment_eeg_segment(values[source_index], sampling_frequency, policy, rng))
        sources.append(int(source_index))
    return AugmentedEEGData(
        segments=np.concatenate((values, np.stack(additions))).astype(np.float32, copy=False),
        labels=np.concatenate((targets, np.full(extra_count, minority_label, dtype=np.int64))),
        is_augmented=np.concatenate((np.zeros(len(values), dtype=bool), np.ones(extra_count, dtype=bool))),
        source_indices=np.concatenate((originals, np.asarray(sources, dtype=np.int64))),
    )
