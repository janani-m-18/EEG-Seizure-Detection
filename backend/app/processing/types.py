"""Typed data objects for the EEG loading, preprocessing, and segmentation pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import mne
import numpy as np


@dataclass(frozen=True, slots=True)
class EEGProcessingRequest:
    """A local, supported EEG file to process in this offline pipeline."""

    input_reference: str
    path: Path
    label: str | int | None = None

    def __post_init__(self) -> None:
        if not self.input_reference.strip():
            raise ValueError("input_reference must not be blank")


@dataclass(frozen=True, slots=True)
class PreprocessingConfig:
    """Configurable, non-clinical EEG preprocessing settings.

    The defaults implement the Batch A development-data preparation policy.
    Legacy pipelines can opt into the former FIR/no-CAR settings explicitly.
    """

    l_freq: float | None = 0.5
    h_freq: float | None = 40.0
    filter_order: int = 4
    filter_method: Literal["iir_butterworth", "fir"] = "iir_butterworth"
    notch_frequency: float | tuple[float, ...] | None = 50.0
    common_average_reference: bool = True
    peak_to_peak_rejection_volts: float | None = 500e-6
    artifact_channel_policy: Literal["preserve", "drop"] = "preserve"
    normalization: Literal["zscore", "none"] = "zscore"

    def __post_init__(self) -> None:
        if self.l_freq is not None and self.l_freq <= 0:
            raise ValueError("l_freq must be greater than zero")
        if self.h_freq is not None and self.h_freq <= 0:
            raise ValueError("h_freq must be greater than zero")
        if self.l_freq is not None and self.h_freq is not None and self.l_freq >= self.h_freq:
            raise ValueError("l_freq must be lower than h_freq")
        if self.filter_order < 1:
            raise ValueError("filter_order must be at least one")
        if self.notch_frequency is not None:
            frequencies = (self.notch_frequency,) if isinstance(self.notch_frequency, (int, float)) else self.notch_frequency
            if not frequencies or any(frequency <= 0 for frequency in frequencies):
                raise ValueError("notch_frequency values must be greater than zero")
        if self.artifact_channel_policy not in {"preserve", "drop"}:
            raise ValueError("artifact_channel_policy must be 'preserve' or 'drop'")
        if self.peak_to_peak_rejection_volts is not None and self.peak_to_peak_rejection_volts <= 0:
            raise ValueError("peak_to_peak_rejection_volts must be greater than zero")


@dataclass(frozen=True, slots=True)
class SegmentationConfig:
    """Windowing parameters expressed in seconds."""

    window_seconds: float = 4.0
    stride_seconds: float | None = 2.0
    include_partial_window: bool = False

    def __post_init__(self) -> None:
        if self.window_seconds <= 0:
            raise ValueError("window_seconds must be greater than zero")
        if self.stride_seconds is not None and self.stride_seconds <= 0:
            raise ValueError("stride_seconds must be greater than zero")

    @property
    def effective_stride_seconds(self) -> float:
        return self.stride_seconds if self.stride_seconds is not None else self.window_seconds


@dataclass(frozen=True, slots=True)
class LoadedEEG:
    input_reference: str
    raw: mne.io.BaseRaw
    source_path: Path


@dataclass(frozen=True, slots=True)
class FilteredEEG:
    input_reference: str
    raw: mne.io.BaseRaw


@dataclass(frozen=True, slots=True)
class NoiseHandledEEG:
    input_reference: str
    raw: mne.io.BaseRaw
    rejected_channel_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class NormalizedEEG:
    input_reference: str
    data: np.ndarray
    channel_names: tuple[str, ...]
    sampling_frequency: float
    rejected_channel_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SegmentMetadata:
    index: int
    start_sample: int
    stop_sample: int
    start_seconds: float
    stop_seconds: float
    label: str | int | None = None


@dataclass(frozen=True, slots=True)
class SegmentedEEG:
    input_reference: str
    segments: np.ndarray
    channel_names: tuple[str, ...]
    sampling_frequency: float
    metadata: tuple[SegmentMetadata, ...]
    rejected_channel_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProcessedEEG:
    input_reference: str
    segmented: SegmentedEEG
    preprocessing_config: PreprocessingConfig
    segmentation_config: SegmentationConfig
