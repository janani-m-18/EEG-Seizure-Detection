"""Held-out Phase 17 evaluation for the fixed CHB01 development checkpoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from ml.chbmit import CHBMITConfig, CHBMITSegmentBuilder, discover_chbmit_recordings
from ml.inference import EEGInferenceService
from ml.phase16_training import phase16_pipeline


HELD_OUT_RECORDINGS = ("chb01_01", "chb01_08", "chb01_13")


def binary_metrics_with_support(targets: np.ndarray, predictions: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    """Return binary metrics without pretending an absent class has a defined score."""
    actual = np.asarray(targets, dtype=np.int64)
    predicted = np.asarray(predictions, dtype=np.int64)
    if actual.ndim != 1 or predicted.ndim != 1 or len(actual) != len(predicted):
        raise ValueError("targets and predictions must be equally sized one-dimensional arrays")
    if len(actual) == 0 or not np.isin(actual, (0, 1)).all() or not np.isin(predicted, (0, 1)).all():
        raise ValueError("binary metrics require non-empty labels limited to 0 and 1")
    matrix = np.zeros((2, 2), dtype=np.int64)
    for truth, output in zip(actual, predicted, strict=True):
        matrix[truth, output] += 1
    per_class: dict[str, dict[str, float | int | None]] = {}
    for label, name in enumerate(("non_seizure", "seizure")):
        tp, actual_support, predicted_support = matrix[label, label], int(matrix[label, :].sum()), int(matrix[:, label].sum())
        precision = None if predicted_support == 0 else float(tp / predicted_support)
        recall = None if actual_support == 0 else float(tp / actual_support)
        f1 = None if precision is None or recall is None else (0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall)))
        per_class[name] = {"support": actual_support, "predicted_support": predicted_support,
                           "precision": precision, "recall": recall, "f1": f1}
    metric_values = [per_class[name][key] for name in ("non_seizure", "seizure") for key in ("precision", "recall", "f1")]
    return {
        "accuracy": float(np.trace(matrix) / matrix.sum()),
        "per_class": per_class,
        "precision_macro": None if any(value is None for value in metric_values[0::3]) else float(np.mean(metric_values[0::3])),
        "recall_macro": None if any(value is None for value in metric_values[1::3]) else float(np.mean(metric_values[1::3])),
        "f1_macro": None if any(value is None for value in metric_values[2::3]) else float(np.mean(metric_values[2::3])),
        "undefined_metrics_reason": (
            "Seizure precision, recall, F1, and macro metrics are undefined because the held-out annotations "
            "contain no seizure windows." if per_class["seizure"]["support"] == 0 else None
        ),
    }, matrix


def evaluate_phase17(dataset_root: Path, checkpoint_path: Path, output_dir: Path, *, docs_dir: Path | None = None) -> dict[str, Any]:
    """Run genuine predictions on the untouched, all-negative held-out recordings only."""
    recordings = discover_chbmit_recordings(CHBMITConfig(dataset_root))
    selected = tuple(item for item in recordings if item.recording_id in HELD_OUT_RECORDINGS)
    if tuple(item.recording_id for item in selected) != HELD_OUT_RECORDINGS:
        raise ValueError(f"Expected held-out recordings {HELD_OUT_RECORDINGS}, found {[item.recording_id for item in selected]}")
    data = CHBMITSegmentBuilder(phase16_pipeline()).build(selected)
    service = EEGInferenceService(checkpoint_path)
    predictions, probabilities = service.predict_segments(data.segments)
    metrics, matrix = binary_metrics_with_support(data.labels, predictions)
    summary = [
        {"recording_id": recording_id, "segment_index": index, "prediction": "seizure" if prediction else "non_seizure",
         "non_seizure_probability": float(probability[0]), "seizure_probability": float(probability[1])}
        for recording_id, index, prediction, probability in zip(data.recording_ids, range(len(data.labels)), predictions, probabilities, strict=True)
    ]
    report: dict[str, Any] = {
        "phase": 17,
        "purpose": "single-patient CHB01 development checkpoint evaluation; not clinical or patient-independent performance",
        "recordings": list(HELD_OUT_RECORDINGS), "recording_count": len(HELD_OUT_RECORDINGS),
        "segment_count": len(data.labels), "segment_shape": list(data.segments.shape),
        "class_distribution": {"non_seizure": int((data.labels == 0).sum()), "seizure": int((data.labels == 1).sum())},
        "metrics": metrics, "confusion_matrix": matrix.tolist(),
        "checkpoint_path": str(checkpoint_path), "preprocessing": "Phase 16 fixed 23-channel policy and 4-second/30-second-start segmentation",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "classification_report.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "prediction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "evaluation_metadata.json").write_text(json.dumps({key: report[key] for key in report if key not in {"metrics", "confusion_matrix"}}, indent=2), encoding="utf-8")
    np.save(output_dir / "confusion_matrix.npy", matrix)
    if docs_dir is not None:
        docs_dir.mkdir(parents=True, exist_ok=True)
        text = (
            "# Phase 17 Held-Out Development Evaluation\n\n"
            f"Evaluated recordings: {', '.join(HELD_OUT_RECORDINGS)} ({len(data.labels)} windows).\n\n"
            f"Class support: non-seizure {int((data.labels == 0).sum())}; seizure {int((data.labels == 1).sum())}.\n\n"
            f"Confusion matrix (rows actual, columns predicted; non-seizure then seizure): `{matrix.tolist()}`.\n\n"
            f"Accuracy: {metrics['accuracy']}. Seizure precision/recall/F1 and macro metrics: undefined. "
            "The selected held-out recordings contain no annotated seizure windows, so this is not a meaningful "
            "seizure-detection test set and does not support clinical or patient-independent claims.\n"
        )
        (docs_dir / "held_out_evaluation.md").write_text(text, encoding="utf-8")
    return report
