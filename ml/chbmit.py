"""CHB-MIT dataset discovery, annotation parsing, and real-data preparation.

No data is downloaded or bundled by this module.  Point ``CHBMITConfig`` at an
unpacked PhysioNet CHB-MIT directory (containing ``chb01``, ``chb02``, ...).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np

from backend.app.processing import EEGProcessingPipeline, EEGProcessingRequest
from backend.app.processing.types import ProcessedEEG


class CHBMITDatasetError(RuntimeError):
    """Raised when the configured CHB-MIT dataset is absent or malformed."""


@dataclass(frozen=True, slots=True)
class CHBMITConfig:
    """Location and label policy for an unpacked CHB-MIT corpus."""

    dataset_root: Path
    annotation_glob: str = "*-summary.txt"
    positive_overlap_seconds: float = 0.0

    @classmethod
    def from_environment(cls) -> "CHBMITConfig":
        value = os.getenv("CHBMIT_DATASET_ROOT")
        if not value:
            raise CHBMITDatasetError(
                "CHBMIT_DATASET_ROOT is not set. Download and unpack CHB-MIT, then set it to that directory."
            )
        return cls(Path(value).expanduser())

    def __post_init__(self) -> None:
        if self.positive_overlap_seconds < 0:
            raise ValueError("positive_overlap_seconds cannot be negative")


@dataclass(frozen=True, slots=True)
class SeizureInterval:
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds <= self.start_seconds:
            raise ValueError("seizure intervals must have 0 <= start < end")


@dataclass(frozen=True, slots=True)
class CHBMITRecording:
    patient_id: str
    recording_id: str
    path: Path
    seizure_intervals: tuple[SeizureInterval, ...] = ()

    @property
    def has_seizure(self) -> bool:
        return bool(self.seizure_intervals)


@dataclass(frozen=True, slots=True)
class RealEEGSegments:
    """Homogeneous model input with provenance retained for leakage-safe splits."""

    segments: np.ndarray
    labels: np.ndarray
    patient_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    channel_names: tuple[str, ...]
    sampling_frequency: float
    segment_start_seconds: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.segments.ndim != 3 or len(self.segments) == 0:
            raise ValueError("segments must be non-empty with shape (items, channels, samples)")
        if self.labels.shape != (len(self.segments),):
            raise ValueError("labels must contain one value per segment")
        if len(self.patient_ids) != len(self.segments) or len(self.recording_ids) != len(self.segments):
            raise ValueError("patient_ids and recording_ids must match segment count")
        if self.segment_start_seconds is not None and len(self.segment_start_seconds) != len(self.segments):
            raise ValueError("segment_start_seconds must match segment count when provided")
        if set(np.unique(self.labels)) - {0, 1}:
            raise ValueError("CHB-MIT segment labels must be binary (0 or 1)")


_FILE_RE = re.compile(r"^\s*File Name:\s*(\S+)", re.IGNORECASE)
_START_RE = re.compile(r"^\s*Seizure (\d+ )?Start Time:\s*([0-9.]+)\s*seconds", re.IGNORECASE)
_END_RE = re.compile(r"^\s*Seizure (\d+ )?End Time:\s*([0-9.]+)\s*seconds", re.IGNORECASE)


def parse_summary_annotations(path: Path) -> dict[str, tuple[SeizureInterval, ...]]:
    """Parse standard CHB-MIT ``*-summary.txt`` seizure entries by EDF filename."""
    if not path.is_file():
        raise CHBMITDatasetError(f"CHB-MIT summary file does not exist: {path}")
    annotations: dict[str, list[SeizureInterval]] = {}
    current_name: str | None = None
    starts: list[float] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _FILE_RE.match(line)
        if match:
            current_name, starts = match.group(1), []
            annotations.setdefault(current_name, [])
            continue
        start_match = _START_RE.match(line)
        if start_match and current_name is not None:
            starts.append(float(start_match.group(2)))
            continue
        end_match = _END_RE.match(line)
        if end_match and current_name is not None:
            if not starts:
                raise CHBMITDatasetError(f"Seizure end without start in {path}")
            annotations[current_name].append(SeizureInterval(starts.pop(0), float(end_match.group(2))))
    if starts:
        raise CHBMITDatasetError(f"Unmatched seizure start time in {path}")
    return {name: tuple(intervals) for name, intervals in annotations.items()}


def discover_chbmit_recordings(config: CHBMITConfig) -> tuple[CHBMITRecording, ...]:
    """Discover EDFs and their patient-summary annotations; never invents recordings."""
    root = config.dataset_root
    if not root.is_dir():
        raise CHBMITDatasetError(
            f"CHB-MIT dataset directory is unavailable: {root}. Set CHBMIT_DATASET_ROOT to an unpacked corpus."
        )
    edfs = sorted(root.glob("chb*/*.edf"))
    if not edfs:
        raise CHBMITDatasetError(f"No CHB-MIT EDF files found below {root} (expected chb*/**.edf)")
    by_patient: dict[str, dict[str, tuple[SeizureInterval, ...]]] = {}
    for summary in root.glob(f"chb*/{config.annotation_glob}"):
        by_patient.setdefault(summary.parent.name.lower(), {}).update(parse_summary_annotations(summary))
    recordings = []
    for path in edfs:
        patient_id = path.parent.name.lower()
        intervals = by_patient.get(patient_id, {}).get(path.name, ())
        recordings.append(CHBMITRecording(patient_id, path.stem, path, intervals))
    return tuple(recordings)


def _window_label(start: float, stop: float, intervals: Iterable[SeizureInterval], minimum_overlap: float) -> int:
    return int(any(max(0.0, min(stop, item.end_seconds) - max(start, item.start_seconds)) > minimum_overlap for item in intervals))


class CHBMITSegmentBuilder:
    """Runs discovered recordings through Steps 6-8 and labels each fixed window."""

    def __init__(self, pipeline: EEGProcessingPipeline, *, required_channels: tuple[str, ...] | None = None) -> None:
        self.pipeline = pipeline
        self.required_channels = required_channels

    def build(self, recordings: Iterable[CHBMITRecording], *, minimum_overlap_seconds: float = 0.0) -> RealEEGSegments:
        all_segments: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        patients: list[str] = []
        recording_ids: list[str] = []
        segment_start_seconds: list[float] = []
        reference_channels: tuple[str, ...] | None = self.required_channels
        sfreq: float | None = None
        for recording in recordings:
            processed: ProcessedEEG = self.pipeline.process(
                EEGProcessingRequest(recording.recording_id, recording.path)
            )
            segmented = processed.segmented
            channels = segmented.channel_names
            if reference_channels is None:
                reference_channels = channels
            if channels != reference_channels:
                raise CHBMITDatasetError(
                    f"Channel mismatch in {recording.path.name}; expected {reference_channels}, got {channels}. "
                    "Configure a consistent montage/preprocessing policy before training."
                )
            if sfreq is None:
                sfreq = segmented.sampling_frequency
            elif segmented.sampling_frequency != sfreq:
                raise CHBMITDatasetError("Sampling-frequency mismatch across recordings; resampling is not implemented.")
            labels = np.asarray([
                _window_label(item.start_seconds, item.stop_seconds, recording.seizure_intervals, minimum_overlap_seconds)
                for item in segmented.metadata
            ], dtype=np.int64)
            all_segments.append(segmented.segments.astype(np.float32, copy=False))
            all_labels.append(labels)
            patients.extend([recording.patient_id] * len(labels))
            recording_ids.extend([recording.recording_id] * len(labels))
            segment_start_seconds.extend(item.start_seconds for item in segmented.metadata)
        if not all_segments:
            raise CHBMITDatasetError("No recordings were supplied for real-data preparation")
        return RealEEGSegments(np.concatenate(all_segments), np.concatenate(all_labels), tuple(patients),
                               tuple(recording_ids), reference_channels or (), sfreq or 0.0,
                               tuple(segment_start_seconds))
