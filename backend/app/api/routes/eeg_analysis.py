"""EDF upload, real CNN-BiLSTM analysis, and live streaming endpoint."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path, PurePath
from tempfile import NamedTemporaryFile
from typing import Any

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from backend.app.core.config import settings
from ml.batch_b_inference import BatchBInferenceService
from ml.inference import EEGInferenceError
from ml.phase16_training import PHASE16_CHANNEL_NAMES

router = APIRouter(prefix="/eeg", tags=["eeg-analysis"])
_UPLOAD_CHUNK_BYTES = 1024 * 1024


@lru_cache(maxsize=1)
def _inference_service(
    checkpoint_path: str,
    threshold: float,
    persistence: int,
) -> BatchBInferenceService:
    """Cache the configured Batch B checkpoint instead of reloading per request."""
    return BatchBInferenceService(
        Path(checkpoint_path),
        threshold=threshold,
        persistence=persistence,
    )


def _error(status_code: int, detail: str) -> HTTPException:
    return HTTPException(status_code=status_code, detail=detail)


class StreamWindowRequest(BaseModel):
    """Payload for real-time single-window analysis."""

    window_samples: list[list[float]] = Field(
        ...,
        description="23 channels x 1024 samples normalized EEG array",
    )
    current_run: int = Field(
        default=0,
        ge=0,
        description="Current consecutive positive window count from previous stream step",
    )
    timestamp_seconds: float = Field(
        default=0.0,
        ge=0.0,
        description="Current recording/stream timestamp in seconds",
    )


@router.post("/stream_window")
async def stream_window(request: StreamWindowRequest) -> dict[str, Any]:
    """Analyze a single real-time (23, 1024) EEG window without reloading the model."""
    try:
        service = _inference_service(
            str(settings.model_checkpoint_path),
            threshold=settings.seizure_threshold,
            persistence=settings.seizure_persistence,
        )
    except EEGInferenceError:
        raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, "The configured model is unavailable.")

    arr = np.array(request.window_samples, dtype=np.float32)
    expected_shape = (len(PHASE16_CHANNEL_NAMES), 1024)
    if arr.shape != expected_shape:
        raise _error(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"Expected window shape {expected_shape}, got {arr.shape}",
        )

    prob = service.predict_window(arr)
    is_positive = prob >= service.threshold
    updated_run = (request.current_run + 1) if is_positive else 0
    event_triggered = updated_run >= service.persistence

    return {
        "status": "success",
        "timestamp_seconds": request.timestamp_seconds,
        "seizure_probability": round(prob, 4),
        "threshold": service.threshold,
        "is_positive": is_positive,
        "current_run": updated_run,
        "persistence_required": service.persistence,
        "event_triggered": event_triggered,
        "monitor_state": "SEIZURE EVENT DETECTED" if event_triggered else ("SUSPECT ACTIVITY" if is_positive else "MONITORING"),
    }


@router.post("/analyze")
async def analyze_eeg(file: UploadFile = File(...)) -> dict[str, Any]:
    """Analyze one uploaded EDF and return real model/XAI results without retaining the file."""
    safe_name = Path(PurePath(file.filename or "upload.edf").name).name
    if Path(safe_name).suffix.lower() != ".edf":
        raise _error(status.HTTP_400_BAD_REQUEST, "Please upload an EDF file.")

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(prefix="eeg-upload-", suffix=".edf", delete=False) as temporary_file:
            temporary_path = Path(temporary_file.name)
            size = 0
            while chunk := await file.read(_UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > settings.analysis_max_upload_bytes:
                    raise _error(status.HTTP_413_CONTENT_TOO_LARGE, "Uploaded EDF exceeds the configured size limit.")
                temporary_file.write(chunk)
        if size == 0:
            raise _error(status.HTTP_400_BAD_REQUEST, "Uploaded EDF file is empty.")

        try:
            service = _inference_service(
                str(settings.model_checkpoint_path),
                threshold=settings.seizure_threshold,
                persistence=settings.seizure_persistence,
            )
        except EEGInferenceError:
            raise _error(status.HTTP_503_SERVICE_UNAVAILABLE, "The configured development model is unavailable.")

        try:
            prepared = service.prepare_edf_details(temporary_path)
            result = service.predict_prepared_segments(
                safe_name,
                prepared.segmented,
                quality_report=prepared.quality_report,
            )
        except EEGInferenceError as error:
            message = str(error)
            if "256 Hz" in message:
                raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "EEG sampling rate is incompatible with the current model (expected 256 Hz).")
            if "channel layout" in message or "canonical" in message:
                raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "EEG channel configuration is incompatible with the fixed 23-channel canonical configuration.")
            raise _error(status.HTTP_422_UNPROCESSABLE_CONTENT, "Unable to read and process the EEG recording.")

        representative_index = int(np.argmax([item.seizure_probability for item in result.segments]))
        metadata = prepared.segmented.metadata[representative_index]

        try:
            explanation = service.explain_segment(prepared.segmented, representative_index)
        except Exception:
            raise _error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Model explanation could not be generated.")

        representative = result.segments[representative_index]
        is_seizure = bool(result.recording_prediction)
        all_probs = [item.seizure_probability for item in result.segments]
        mean_prob = float(np.mean(all_probs)) if all_probs else 0.0
        median_prob = float(np.median(all_probs)) if all_probs else 0.0

        return {
            "status": "success",
            "file_name": safe_name,
            "recording": {
                "duration_seconds": prepared.duration_seconds,
                "sampling_rate": prepared.segmented.sampling_frequency,
                "channels": list(prepared.segmented.channel_names),
                "segment_count": len(result.segments),
            },
            "quality_report": prepared.quality_report.as_dict() if prepared.quality_report else None,
            "prediction": {
                "label": "seizure" if is_seizure else "non_seizure",
                "is_seizure": is_seizure,
                "recording_result": "SEIZURE EVENT DETECTED" if is_seizure else "NO SUSTAINED SEIZURE EVENT DETECTED",
                "highest_single_window_probability": round(representative.seizure_probability, 4),
                "highest_window_score": round(representative.seizure_probability, 4),
                "mean_window_score": round(mean_prob, 4),
                "median_window_score": round(median_prob, 4),
                "threshold": result.threshold,
                "persistence_required": result.persistence,
                "positive_window_count": result.positive_window_count,
                "longest_positive_run": result.longest_consecutive_positive_windows,
                "event_count": len(result.events),
                "probabilities": {
                    "non_seizure": representative.non_seizure_probability,
                    "seizure": representative.seizure_probability,
                },
                "representative_segment_index": representative_index,
                "aggregation_rule": (
                    f"A sustained seizure event is detected if at least {result.persistence} consecutive 4-second windows "
                    f"have probability >= {result.threshold:.2f}. "
                    f"Highest window score describes a single local 4-second receptive field, NOT the recording probability."
                ),
            },
            "temporal_events": [item.as_dict() for item in result.events],
            "segments_summary": [
                {
                    "index": item.index,
                    "start_seconds": item.start_seconds,
                    "stop_seconds": item.stop_seconds,
                    "seizure_probability": round(item.seizure_probability, 4),
                    "is_positive": item.seizure_probability >= result.threshold,
                }
                for item in result.segments
            ],
            "segments": [item.as_dict() for item in result.segments],
            "waveform": {
                "segment_index": representative_index,
                "start_seconds": metadata.start_seconds,
                "stop_seconds": metadata.stop_seconds,
                "sampling_rate": prepared.segmented.sampling_frequency,
                "channel_names": list(prepared.segmented.channel_names),
                "samples": prepared.segmented.segments[representative_index].tolist(),
            },
            "explainability": explanation.as_dict(),
            "model_info": {
                "architecture": "CNN-BiLSTM (KaggleCNNBiLSTM)",
                "checkpoint": settings.model_checkpoint_path.name,
                "sampling_frequency_hz": 256.0,
                "channels": 23,
                "window_length_seconds": 4.0,
                "stride_seconds": 2.0,
                "threshold": result.threshold,
                "persistence": result.persistence,
                "disclaimer": "Academic major project research prototype — not approved for autonomous clinical diagnosis.",
            },
        }
    finally:
        await file.close()
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
