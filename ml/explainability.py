"""Integrated Gradients explanations for the trained Phase 16 CNN-BiLSTM checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from ml.inference import EEGInferenceError, EEGInferenceService
from ml.phase16_training import PHASE16_CHANNEL_NAMES


@dataclass(frozen=True, slots=True)
class ImportantRegion:
    start_seconds: float
    stop_seconds: float
    peak_importance: float

    def as_dict(self) -> dict[str, float]:
        return {"start_seconds": self.start_seconds, "stop_seconds": self.stop_seconds,
                "peak_importance": self.peak_importance}


@dataclass(frozen=True, slots=True)
class EEGExplanation:
    prediction: int
    non_seizure_probability: float
    seizure_probability: float
    segment_start_seconds: float
    segment_stop_seconds: float
    channel_names: tuple[str, ...]
    sampling_frequency: float
    attribution: np.ndarray
    temporal_importance: np.ndarray
    channel_importance: np.ndarray
    important_regions: tuple[ImportantRegion, ...]

    def as_dict(self, *, include_attribution: bool = True) -> dict[str, object]:
        result: dict[str, object] = {
            "prediction": "seizure" if self.prediction else "non_seizure",
            "probabilities": {"non_seizure": self.non_seizure_probability, "seizure": self.seizure_probability},
            "segment_start_seconds": self.segment_start_seconds, "segment_stop_seconds": self.segment_stop_seconds,
            "channel_names": list(self.channel_names), "sampling_frequency": self.sampling_frequency,
            "attribution_shape": list(self.attribution.shape),
            "temporal_importance": self.temporal_importance.tolist(),
            "channel_importance": self.channel_importance.tolist(),
            "important_regions": [region.as_dict() for region in self.important_regions],
            "method": "Integrated Gradients from a deterministic all-zero baseline over the actual predicted-class logit",
        }
        if include_attribution:
            result["attribution"] = self.attribution.tolist()
        return result


def _important_regions(temporal_importance: np.ndarray, *, sampling_frequency: float,
                       segment_start_seconds: float, max_regions: int = 5) -> tuple[ImportantRegion, ...]:
    threshold = float(np.quantile(temporal_importance, 0.95))
    mask = temporal_importance >= threshold
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(np.int8))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    regions = [ImportantRegion(segment_start_seconds + start / sampling_frequency,
                               segment_start_seconds + stop / sampling_frequency,
                               float(temporal_importance[start:stop].max()))
               for start, stop in zip(starts, stops, strict=True)]
    return tuple(sorted(regions, key=lambda item: item.peak_importance, reverse=True)[:max_regions])


def explain_segment(service: EEGInferenceService, segment: np.ndarray, *, sampling_frequency: float,
                    segment_start_seconds: float, segment_stop_seconds: float,
                    integration_steps: int = 32) -> EEGExplanation:
    """Explain a real model prediction with Integrated Gradients from a zero baseline.

    Phase 16 channel-wise z-score normalization makes zero a deterministic
    channel-mean baseline.  Attributions integrate gradients of the actual
    predicted-class logit along the straight path from that baseline to input.
    """
    if integration_steps < 2:
        raise EEGInferenceError("integration_steps must be at least two")
    values = np.asarray(segment, dtype=np.float32)
    if values.shape != (len(PHASE16_CHANNEL_NAMES), 1024) or not np.isfinite(values).all():
        raise EEGInferenceError(f"Expected one finite segment shaped ({len(PHASE16_CHANNEL_NAMES)}, 1024)")
    predictions, probabilities = service.predict_segments(values[None, ...])
    predicted_class = int(predictions[0])
    inputs = torch.from_numpy(values[None, ...]).to(service.device)
    baseline = torch.zeros_like(inputs)
    difference = inputs - baseline
    gradients = torch.zeros_like(inputs)
    service.model.eval()
    for alpha in torch.linspace(0.0, 1.0, integration_steps, device=service.device):
        scaled = (baseline + alpha * difference).detach().requires_grad_(True)
        service.model.zero_grad(set_to_none=True)
        score = service.model(scaled)[0, predicted_class]
        gradients += torch.autograd.grad(score, scaled, retain_graph=False)[0]
    attribution = (difference * gradients / integration_steps).squeeze(0).detach().cpu().numpy()
    temporal_importance = np.mean(np.abs(attribution), axis=0)
    channel_importance = np.mean(np.abs(attribution), axis=1)
    return EEGExplanation(predicted_class, float(probabilities[0, 0]), float(probabilities[0, 1]),
                          segment_start_seconds, segment_stop_seconds, PHASE16_CHANNEL_NAMES, sampling_frequency,
                          attribution, temporal_importance, channel_importance,
                          _important_regions(temporal_importance, sampling_frequency=sampling_frequency,
                                             segment_start_seconds=segment_start_seconds))


def explain_edf_segment(service: EEGInferenceService, path: Path, segment_index: int, *, integration_steps: int = 32) -> EEGExplanation:
    """Run the real Phase 16 preprocessing pipeline then explain one EDF segment."""
    segmented = service.prepare_edf(path)
    if not 0 <= segment_index < len(segmented.segments):
        raise EEGInferenceError(f"segment_index must be between 0 and {len(segmented.segments) - 1}")
    metadata = segmented.metadata[segment_index]
    return explain_segment(service, segmented.segments[segment_index], sampling_frequency=segmented.sampling_frequency,
                           segment_start_seconds=metadata.start_seconds, segment_stop_seconds=metadata.stop_seconds,
                           integration_steps=integration_steps)


def save_attribution_visualization(segment: np.ndarray, explanation: EEGExplanation, path: Path) -> Path:
    """Save an actual EEG waveform and Integrated Gradients heatmap for future frontend use."""
    import matplotlib.pyplot as plt

    values = np.asarray(segment, dtype=np.float32)
    if values.shape != explanation.attribution.shape:
        raise EEGInferenceError("segment and attribution must have identical channel/sample dimensions")
    seconds = explanation.segment_start_seconds + np.arange(values.shape[1]) / explanation.sampling_frequency
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True, height_ratios=(1.1, 1, 1.6))
    axes[0].plot(seconds, values.mean(axis=0), linewidth=0.7)
    axes[0].set_ylabel("Mean normalized EEG")
    axes[0].set_title("Development-checkpoint attribution: actual EEG segment")
    axes[1].plot(seconds, explanation.temporal_importance, linewidth=0.8)
    axes[1].set_ylabel("Mean |IG attribution|")
    for region in explanation.important_regions:
        axes[1].axvspan(region.start_seconds, region.stop_seconds, alpha=0.25)
    image = axes[2].imshow(np.abs(explanation.attribution), aspect="auto", origin="lower",
                           extent=(seconds[0], seconds[-1], 0, values.shape[0]))
    axes[2].set_ylabel("EEG channel index")
    axes[2].set_xlabel("Recording-relative time (seconds)")
    figure.colorbar(image, ax=axes[2], label="|Integrated Gradients attribution|")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def save_explanation_json(explanation: EEGExplanation, path: Path) -> Path:
    """Save the complete, model-derived attribution for later visualization or audit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(explanation.as_dict(), indent=2), encoding="utf-8")
    return path
