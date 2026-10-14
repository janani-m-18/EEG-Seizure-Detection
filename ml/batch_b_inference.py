"""EDF inference and temporal event detection for the Kaggle-trained CNN-BiLSTM checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from backend.app.processing import EEGProcessingRequest, SegmentedEEG, SegmentMetadata
from backend.app.processing.loader import EEGLoadError, select_required_channels
from ml.batch_a import batch_a_pipeline
from ml.gradcam import GradCAMExplanation, explain_batch_b_segment
from ml.inference import EEGInferenceError
from ml.kaggle_model import KaggleCNNBiLSTM
from ml.phase16_training import PHASE16_CHANNEL_NAMES


@dataclass(frozen=True, slots=True)
class EEGQualityReport:
    status: str
    sampling_frequency: float
    channel_count: int
    duration_seconds: float
    missing_canonical_channels: tuple[str, ...]
    flatline_channels: tuple[str, ...]
    extreme_amplitude_channels: tuple[str, ...]
    warnings: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "sampling_frequency": self.sampling_frequency,
            "channel_count": self.channel_count,
            "duration_seconds": round(self.duration_seconds, 2),
            "missing_canonical_channels": list(self.missing_canonical_channels),
            "flatline_channels": list(self.flatline_channels),
            "extreme_amplitude_channels": list(self.extreme_amplitude_channels),
            "warnings": list(self.warnings),
        }


@dataclass(frozen=True, slots=True)
class TemporalSeizureEvent:
    event_index: int
    start_segment: int
    end_segment: int
    start_seconds: float
    end_seconds: float
    duration_seconds: float
    consecutive_windows: int
    peak_probability: float
    mean_probability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "event_index": self.event_index,
            "event_id": self.event_index,
            "start_segment": self.start_segment,
            "end_segment": self.end_segment,
            "start_window_index": self.start_segment,
            "stop_window_index": self.end_segment,
            "start_seconds": round(self.start_seconds, 2),
            "end_seconds": round(self.end_seconds, 2),
            "stop_seconds": round(self.end_seconds, 2),
            "duration_seconds": round(self.duration_seconds, 2),
            "consecutive_windows": self.consecutive_windows,
            "window_count": self.consecutive_windows,
            "peak_probability": round(self.peak_probability, 4),
            "max_probability": round(self.peak_probability, 4),
            "mean_probability": round(self.mean_probability, 4),
        }


@dataclass(frozen=True, slots=True)
class BatchBPreparedEEG:
    segmented: SegmentedEEG
    duration_seconds: float
    quality_report: EEGQualityReport | None = None


@dataclass(frozen=True, slots=True)
class BatchBSegmentPrediction:
    index: int
    start_seconds: float
    stop_seconds: float
    prediction: int
    non_seizure_probability: float
    seizure_probability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "index": self.index,
            "start_seconds": self.start_seconds,
            "stop_seconds": self.stop_seconds,
            "prediction": "seizure" if self.prediction else "non_seizure",
            "probabilities": {
                "non_seizure": self.non_seizure_probability,
                "seizure": self.seizure_probability,
            },
        }


def extract_temporal_events(
    probabilities: np.ndarray,
    metadata: tuple[SegmentMetadata, ...],
    threshold: float,
    persistence: int,
) -> tuple[TemporalSeizureEvent, ...]:
    """Identify contiguous runs of windows meeting threshold for >= persistence steps."""
    events: list[TemporalSeizureEvent] = []
    binary = (probabilities >= threshold).astype(bool)
    cur_start: int | None = None
    cur_len = 0
    event_counter = 0

    for i, is_pos in enumerate(binary):
        if is_pos:
            if cur_start is None:
                cur_start = i
            cur_len += 1
        else:
            if cur_start is not None and cur_len >= persistence:
                end_idx = cur_start + cur_len - 1
                segment_probs = probabilities[cur_start:end_idx + 1]
                events.append(
                    TemporalSeizureEvent(
                        event_index=event_counter,
                        start_segment=cur_start,
                        end_segment=end_idx,
                        start_seconds=metadata[cur_start].start_seconds,
                        end_seconds=metadata[end_idx].stop_seconds,
                        duration_seconds=metadata[end_idx].stop_seconds - metadata[cur_start].start_seconds,
                        consecutive_windows=cur_len,
                        peak_probability=float(np.max(segment_probs)),
                        mean_probability=float(np.mean(segment_probs)),
                    )
                )
                event_counter += 1
            cur_start = None
            cur_len = 0

    if cur_start is not None and cur_len >= persistence:
        end_idx = cur_start + cur_len - 1
        segment_probs = probabilities[cur_start:end_idx + 1]
        events.append(
            TemporalSeizureEvent(
                event_index=event_counter,
                start_segment=cur_start,
                end_segment=end_idx,
                start_seconds=metadata[cur_start].start_seconds,
                end_seconds=metadata[end_idx].stop_seconds,
                duration_seconds=metadata[end_idx].stop_seconds - metadata[cur_start].start_seconds,
                consecutive_windows=cur_len,
                peak_probability=float(np.max(segment_probs)),
                mean_probability=float(np.mean(segment_probs)),
            )
        )

    return tuple(events)


@dataclass(frozen=True, slots=True)
class BatchBInferenceResult:
    recording: str
    sampling_frequency: float
    channel_names: tuple[str, ...]
    segments: tuple[BatchBSegmentPrediction, ...]
    events: tuple[TemporalSeizureEvent, ...]
    threshold: float
    persistence: int
    longest_consecutive_positive_windows: int
    positive_window_count: int
    max_seizure_probability: float
    quality_report: EEGQualityReport | None = None

    @property
    def recording_prediction(self) -> int:
        return int(len(self.events) > 0)


class BatchBInferenceService:
    """Use the Kaggle-trained CNN-BiLSTM checkpoint for robust EDF inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cpu",
        threshold: float = 0.50,
        persistence: int = 1,
        inference_batch_size: int = 64,
    ) -> None:

        if not checkpoint_path.is_file():
            raise EEGInferenceError(
                f"Batch B checkpoint does not exist: {checkpoint_path}"
            )

        if not 0 < threshold < 1 or inference_batch_size < 1 or persistence < 1:
            raise EEGInferenceError(
                "Batch B threshold, persistence, and batch size must be valid positive values"
            )

        self.device = torch.device(device)
        self.threshold = threshold
        self.persistence = persistence
        self.inference_batch_size = inference_batch_size

        # Exact architecture used for the Kaggle-trained checkpoint.
        self.model = KaggleCNNBiLSTM()

        # Load checkpoint.
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=False,
            )

            if "model_state_dict" not in checkpoint:
                raise KeyError(
                    "checkpoint does not contain 'model_state_dict'"
                )

            state_dict = checkpoint["model_state_dict"]

            if state_dict and all(
                key.startswith("module.")
                for key in state_dict
            ):
                state_dict = {
                    key.removeprefix("module."): value
                    for key, value in state_dict.items()
                }

            self.model.load_state_dict(
                state_dict,
                strict=True,
            )

        except (RuntimeError, KeyError, TypeError) as error:
            raise EEGInferenceError(
                f"Could not load Kaggle CNN-BiLSTM checkpoint "
                f"'{checkpoint_path.name}': {error}"
            ) from error

        self.model.to(self.device)
        self.model.eval()

        self.checkpoint = checkpoint
        self.checkpoint_path = checkpoint_path

        # proposal-aligned Batch A preprocessing pipeline.
        self.pipeline = batch_a_pipeline()

    def prepare_edf_details(
        self,
        path: Path,
    ) -> BatchBPreparedEEG:
        """Load, assess quality, preprocess and segment an EDF file."""

        if not path.is_file():
            raise EEGInferenceError(
                f"EEG file does not exist or is not a file: {path}"
            )

        try:
            request = EEGProcessingRequest(
                path.stem,
                path,
            )

            loaded = self.pipeline.load(request)

            duration_seconds = float(
                loaded.raw.n_times
                / loaded.raw.info["sfreq"]
            )

            if float(loaded.raw.info["sfreq"]) != 256.0:
                raise EEGInferenceError(
                    f"Expected a 256 Hz Batch B input, "
                    f"got {loaded.raw.info['sfreq']} Hz"
                )

            try:
                loaded = select_required_channels(
                    loaded,
                    PHASE16_CHANNEL_NAMES,
                )
            except EEGLoadError as error:
                raise EEGInferenceError(
                    "EDF channel layout is incompatible with the "
                    "fixed 23-channel canonical configuration"
                ) from error

            # Assess signal quality across the 23 selected channels
            raw_data = loaded.raw.get_data()
            ptp = np.ptp(raw_data, axis=1)
            var = np.var(raw_data, axis=1)
            flatline = tuple(ch for ch, v in zip(loaded.raw.ch_names, var) if v < 1e-12)
            extreme = tuple(ch for ch, p in zip(loaded.raw.ch_names, ptp) if p > 1000e-6)
            warnings = []
            if flatline:
                warnings.append(f"Near-flatline signal on channel(s): {', '.join(flatline)}")
            if extreme:
                warnings.append(f"High amplitude artifact (>1000 uV) on channel(s): {', '.join(extreme)}")

            quality_report = EEGQualityReport(
                status="warning" if warnings else "clean",
                sampling_frequency=float(loaded.raw.info["sfreq"]),
                channel_count=len(loaded.raw.ch_names),
                duration_seconds=duration_seconds,
                missing_canonical_channels=(),
                flatline_channels=flatline,
                extreme_amplitude_channels=extreme,
                warnings=tuple(warnings),
            )

            # Preprocessing: bandpass (0.5-40 Hz), notch (50 Hz), CAR, normalize (z-score)
            filtered = self.pipeline.filter(loaded)
            noise_handled = self.pipeline.handle_noise(filtered)
            normalised = self.pipeline.normalize(noise_handled)

            # 4-second windows with 2-second stride
            segmented = self.pipeline.segment(normalised)

        except EEGInferenceError:
            raise

        except Exception as error:
            raise EEGInferenceError(
                f"Could not process EEG file '{path.name}': {error}"
            ) from error

        if segmented.sampling_frequency != 256.0:
            raise EEGInferenceError(
                f"Expected a 256 Hz Batch B input, got {segmented.sampling_frequency} Hz"
            )

        if segmented.channel_names != PHASE16_CHANNEL_NAMES:
            raise EEGInferenceError(
                "EDF channel layout is incompatible with the fixed 23-channel configuration"
            )

        return BatchBPreparedEEG(
            segmented=segmented,
            duration_seconds=duration_seconds,
            quality_report=quality_report,
        )

    def prepare_edf(
        self,
        path: Path,
    ) -> SegmentedEEG:
        return self.prepare_edf_details(path).segmented

    def predict_segments(
        self,
        segments: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return binary segment predictions and seizure probabilities."""

        values = np.asarray(
            segments,
            dtype=np.float32,
        )

        expected = (
            len(PHASE16_CHANNEL_NAMES),
            1024,
        )

        if (
            values.ndim != 3
            or len(values) == 0
            or values.shape[1:] != expected
        ):
            raise EEGInferenceError(
                f"Expected segments shaped (items, {expected[0]}, {expected[1]})"
            )

        if not np.isfinite(values).all():
            raise EEGInferenceError(
                "segments must contain only finite EEG values"
            )

        scores: list[np.ndarray] = []

        with torch.no_grad():
            for start in range(
                0,
                len(values),
                self.inference_batch_size,
            ):
                batch = torch.from_numpy(
                    values[
                        start:
                        start + self.inference_batch_size
                    ]
                ).to(self.device)

                probabilities = self.model(batch)
                scores.append(probabilities.cpu().numpy())

        seizure_probabilities = np.concatenate(scores).astype(np.float64)

        predictions = (
            seizure_probabilities >= self.threshold
        ).astype(np.int64)

        return predictions, seizure_probabilities

    def predict_window(
        self,
        window_data: np.ndarray,
    ) -> float:
        """Inference for a single (23, 1024) window for real-time / live streaming."""
        arr = np.asarray(window_data, dtype=np.float32)
        if arr.shape != (len(PHASE16_CHANNEL_NAMES), 1024):
            raise EEGInferenceError(
                f"Expected window shape ({len(PHASE16_CHANNEL_NAMES)}, 1024), got {arr.shape}"
            )
        tensor = torch.from_numpy(arr[None, ...]).to(self.device)
        with torch.no_grad():
            prob = self.model(tensor).item()
        return float(prob)

    def predict_prepared_segments(
        self,
        recording: str,
        segmented: SegmentedEEG,
        quality_report: EEGQualityReport | None = None,
    ) -> BatchBInferenceResult:

        if (
            segmented.sampling_frequency != 256.0
            or segmented.channel_names != PHASE16_CHANNEL_NAMES
        ):
            raise EEGInferenceError(
                "Prepared EEG is incompatible with the fixed Batch B input contract"
            )

        predictions, probabilities = self.predict_segments(
            segmented.segments
        )

        items = tuple(
            BatchBSegmentPrediction(
                meta.index,
                meta.start_seconds,
                meta.stop_seconds,
                int(prediction),
                float(1.0 - probability),
                float(probability),
            )
            for meta, prediction, probability in zip(
                segmented.metadata,
                predictions,
                probabilities,
                strict=True,
            )
        )

        # Longest consecutive run
        binary = (probabilities >= self.threshold).astype(bool)
        max_run = cur_run = 0
        for b in binary:
            if b:
                cur_run += 1
                if cur_run > max_run:
                    max_run = cur_run
            else:
                cur_run = 0

        # Temporal event intervals
        events = extract_temporal_events(
            probabilities,
            segmented.metadata,
            threshold=self.threshold,
            persistence=self.persistence,
        )

        return BatchBInferenceResult(
            recording=recording,
            sampling_frequency=segmented.sampling_frequency,
            channel_names=segmented.channel_names,
            segments=items,
            events=events,
            threshold=self.threshold,
            persistence=self.persistence,
            longest_consecutive_positive_windows=max_run,
            positive_window_count=int(np.sum(binary)),
            max_seizure_probability=float(np.max(probabilities)) if len(probabilities) > 0 else 0.0,
            quality_report=quality_report,
        )

    def explain_segment(
        self,
        segmented: SegmentedEEG,
        segment_index: int,
        *,
        target_class: int | None = None,
    ) -> GradCAMExplanation:
        """Generate Grad-CAM for one prepared segment."""

        if not 0 <= segment_index < len(segmented.segments):
            raise EEGInferenceError(
                "segment_index is outside the prepared recording"
            )

        metadata = segmented.metadata[segment_index]

        try:
            return explain_batch_b_segment(
                self.model,
                segmented.segments[segment_index],
                channel_names=segmented.channel_names,
                sampling_frequency=segmented.sampling_frequency,
                segment_index=segment_index,
                segment_start_seconds=metadata.start_seconds,
                segment_end_seconds=metadata.stop_seconds,
                checkpoint_path=self.checkpoint_path,
                target_class=target_class,
                device=str(self.device),
            )

        except (ValueError, RuntimeError) as error:
            raise EEGInferenceError(
                f"Could not generate Batch B Grad-CAM: {error}"
            ) from error