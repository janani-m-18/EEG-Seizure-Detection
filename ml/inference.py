"""EDF inference using the Phase 16 development checkpoint and preprocessing policy."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from backend.app.processing import EEGProcessingRequest, SegmentedEEG
from ml.phase16_training import (
    PHASE16_CHANNEL_NAMES,
    build_phase16_model,
    phase16_pipeline,
)
from ml.training import load_checkpoint


class EEGInferenceError(ValueError):
    """Raised when an EDF or model artifact cannot be used for Phase 16-compatible inference."""


@dataclass(frozen=True, slots=True)
class PreparedEEG:
    """A validated Phase 16-compatible EDF with its source duration retained."""

    segmented: SegmentedEEG
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class SegmentPrediction:
    index: int
    start_seconds: float
    stop_seconds: float
    prediction: int
    non_seizure_probability: float
    seizure_probability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index, "start_seconds": self.start_seconds, "stop_seconds": self.stop_seconds,
            "prediction": "seizure" if self.prediction else "non_seizure",
            "probabilities": {"non_seizure": self.non_seizure_probability, "seizure": self.seizure_probability},
        }


@dataclass(frozen=True, slots=True)
class EEGInferenceResult:
    recording: str
    sampling_frequency: float
    channel_names: tuple[str, ...]
    segments: tuple[SegmentPrediction, ...]

    @property
    def recording_prediction(self) -> int:
        """Aggregate with an any-positive rule: seizure if any segment predicts seizure."""
        return int(any(item.prediction for item in self.segments))

    @property
    def maximum_seizure_probability(self) -> float:
        return max(item.seizure_probability for item in self.segments)

    def as_dict(self) -> dict[str, object]:
        return {
            "recording": self.recording, "sampling_frequency": self.sampling_frequency,
            "channel_names": list(self.channel_names),
            "recording_prediction": "seizure" if self.recording_prediction else "non_seizure",
            "aggregation_rule": "seizure when any segment predicts seizure; maximum_seizure_probability is also reported",
            "maximum_seizure_probability": self.maximum_seizure_probability,
            "segments": [item.as_dict() for item in self.segments],
        }


class EEGInferenceService:
    """Load the real Phase 16 checkpoint and infer labels/probabilities from EDF windows."""

    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        if not checkpoint_path.is_file():
            raise EEGInferenceError(f"Trained checkpoint does not exist: {checkpoint_path}")
        self.device = torch.device(device)
        self.model = build_phase16_model()
        try:
            self.checkpoint = load_checkpoint(checkpoint_path, self.model, device=device)
        except Exception as error:
            raise EEGInferenceError(f"Could not load Phase 16 checkpoint '{checkpoint_path.name}': {error}") from error
        self.model.to(self.device).eval()
        self.checkpoint_path = checkpoint_path
        self.pipeline = phase16_pipeline()

    def prepare_edf(self, path: Path) -> SegmentedEEG:
        """Run the same EDF loading, preprocessing, and segmentation policy as Phase 16."""
        return self.prepare_edf_details(path).segmented

    def prepare_edf_details(self, path: Path) -> PreparedEEG:
        """Prepare an EDF while retaining source-recording duration for API consumers."""
        if not path.exists():
            raise EEGInferenceError(f"EEG file does not exist: {path}")
        if not path.is_file():
            raise EEGInferenceError(f"EEG input must be a file: {path}")
        try:
            request = EEGProcessingRequest(path.stem, path)
            loaded = self.pipeline.load(request)
            duration_seconds = float(loaded.raw.n_times / loaded.raw.info["sfreq"])
            if float(loaded.raw.info["sfreq"]) != 256.0:
                raise EEGInferenceError(f"Expected a 256 Hz Phase 16 input, got {loaded.raw.info['sfreq']} Hz")
            filtered = self.pipeline.filter(loaded)
            noise_handled = self.pipeline.handle_noise(filtered)
            normalized = self.pipeline.normalize(noise_handled)
            segmented = self.pipeline.segment(normalized)
        except EEGInferenceError:
            raise
        except Exception as error:
            raise EEGInferenceError(f"Could not process EEG file '{path.name}': {error}") from error
        if segmented.sampling_frequency != 256.0:
            raise EEGInferenceError(f"Expected a 256 Hz Phase 16 input, got {segmented.sampling_frequency} Hz")
        if segmented.channel_names != PHASE16_CHANNEL_NAMES:
            raise EEGInferenceError("EDF channel layout is incompatible with the fixed Phase 16 CHB01 channel order")
        return PreparedEEG(segmented, duration_seconds)

    def predict_segments(self, segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return actual class IDs and softmax probabilities for validated model-shaped EEG windows."""
        values = np.asarray(segments, dtype=np.float32)
        if values.ndim != 3 or values.shape[0] == 0:
            raise EEGInferenceError("segments must be a non-empty array shaped (items, channels, samples)")
        if values.shape[1:] != (len(PHASE16_CHANNEL_NAMES), 1024):
            raise EEGInferenceError(
                f"Expected segments shaped (items, {len(PHASE16_CHANNEL_NAMES)}, 1024), got {values.shape}"
            )
        if not np.isfinite(values).all():
            raise EEGInferenceError("segments must contain only finite EEG values")
        with torch.no_grad():
            logits = self.model(torch.from_numpy(values).to(self.device))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        return probabilities.argmax(axis=1).astype(np.int64), probabilities

    def predict_edf(self, path: Path) -> EEGInferenceResult:
        """Produce per-window and recording-level results from a real EDF file."""
        segmented = self.prepare_edf(path)
        return self.predict_prepared_segments(path.stem, segmented)

    def predict_prepared_segments(self, recording: str, segmented: SegmentedEEG) -> EEGInferenceResult:
        """Predict an already Phase 16-compatible segmented recording without reprocessing it."""
        if segmented.sampling_frequency != 256.0 or segmented.channel_names != PHASE16_CHANNEL_NAMES:
            raise EEGInferenceError("Prepared EEG is incompatible with the fixed Phase 16 input contract")
        predictions, probabilities = self.predict_segments(segmented.segments)
        items = tuple(
            SegmentPrediction(meta.index, meta.start_seconds, meta.stop_seconds, int(prediction),
                              float(probability[0]), float(probability[1]))
            for meta, prediction, probability in zip(segmented.metadata, predictions, probabilities, strict=True)
        )
        return EEGInferenceResult(recording, segmented.sampling_frequency, segmented.channel_names, items)
