"""Batch B CNN--BiLSTM training and evaluation for Batch A CHB01 windows.

This is a development-only workflow.  It preserves the Phase 16 model and its
inference contract, writes a separate checkpoint, and never uses held-out data
for augmentation, class balancing, threshold selection, or early stopping.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from ml.augmentation import AugmentationConfig, augment_training_segments
from ml.batch_a import BATCH_A_AUGMENTATION, batch_a_pipeline, batch_a_recordings
from ml.chbmit import CHBMITSegmentBuilder, RealEEGSegments
from ml.evaluation import GroupSplit, GroupSplitConfig, split_real_eeg
from ml.model import BatchBCNNBiLSTMClassifier, BatchBModelConfig
from ml.training import EEGSegmentDataset, set_reproducible_seed


BATCH_B_PREPROCESSING_ID = "batch-a-chb01-preprocessing-v1-0.5-40hz-car-zscore-4s-2s"
BATCH_B_MODEL_ID = "batch-b-cnn64-128-256-bilstm128-64-dense64-binary-v1"
BATCH_B_DEVELOPMENT_CHECKPOINT_TYPE = "batch_b_chb01_development"
BATCH_B_FINAL_CHECKPOINT_TYPE = "batch_b_final_chbmit"
BATCH_B_FINAL_V2_CHECKPOINT_TYPE = "batch_b_final_chbmit_v2"
BATCH_B_MODEL_CONFIG = BatchBModelConfig()
BATCH_B_AUGMENTATION = replace(BATCH_A_AUGMENTATION, enabled=True)


class BatchBCompatibilityError(ValueError):
    """Raised when a checkpoint is not compatible with the Batch B contract."""


@dataclass(frozen=True, slots=True)
class BatchBTrainingConfig:
    """Required Batch B training configuration, with an adjustable CPU run length."""

    seed: int = 1
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    threshold: float = 0.5
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if not 0 < self.threshold < 1:
            raise ValueError("threshold must be between zero and one")


@dataclass(frozen=True, slots=True)
class BatchBPreparedData:
    """Full observed counts plus the bounded real tensor set retained for CPU work."""

    data: RealEEGSegments
    full_segment_count: int
    full_label_counts: dict[str, int]
    retained_segment_count: int
    retained_label_counts: dict[str, int]
    max_non_seizure_windows_per_recording: int


def build_batch_b_model(config: BatchBModelConfig | None = None) -> BatchBCNNBiLSTMClassifier:
    """Construct the standalone Batch B binary-probability model."""
    return BatchBCNNBiLSTMClassifier(config or BATCH_B_MODEL_CONFIG)


def prepare_batch_b_development_data(
    dataset_root: Path,
    *,
    max_non_seizure_windows_per_recording: int = 16,
) -> BatchBPreparedData:
    """Process all local CHB01 recordings and retain a bounded real CPU subset.

    Every available seizure window is retained.  The first deterministic set of
    non-seizure windows is retained per recording only after real Batch A
    preprocessing and annotation-based window labeling.  The complete observed
    window/label counts are returned, so the bounded run cannot be mistaken for
    training on the entire 50%-overlap data tensor.
    """
    if max_non_seizure_windows_per_recording < 1:
        raise ValueError("max_non_seizure_windows_per_recording must be positive")
    recordings = batch_a_recordings(dataset_root)
    builder = CHBMITSegmentBuilder(batch_a_pipeline())
    selected_segments: list[np.ndarray] = []
    selected_labels: list[np.ndarray] = []
    selected_patients: list[str] = []
    selected_recordings: list[str] = []
    selected_starts: list[float] = []
    full_count = full_non_seizure = full_seizure = 0
    channel_names: tuple[str, ...] | None = None
    sampling_frequency: float | None = None
    for recording in recordings:
        prepared = builder.build((recording,))
        if channel_names is None:
            channel_names, sampling_frequency = prepared.channel_names, prepared.sampling_frequency
        full_count += len(prepared.labels)
        full_non_seizure += int((prepared.labels == 0).sum())
        full_seizure += int((prepared.labels == 1).sum())
        negative_indices = np.flatnonzero(prepared.labels == 0)[:max_non_seizure_windows_per_recording]
        positive_indices = np.flatnonzero(prepared.labels == 1)
        indices = np.sort(np.concatenate((negative_indices, positive_indices)))
        selected_segments.append(prepared.segments[indices].copy())
        selected_labels.append(prepared.labels[indices].copy())
        selected_patients.extend(np.asarray(prepared.patient_ids, dtype=object)[indices].tolist())
        selected_recordings.extend(np.asarray(prepared.recording_ids, dtype=object)[indices].tolist())
        if prepared.segment_start_seconds is None:
            raise RuntimeError("Batch B preparation requires segment start-time provenance")
        selected_starts.extend(np.asarray(prepared.segment_start_seconds)[indices].tolist())

    data = RealEEGSegments(
        np.concatenate(selected_segments),
        np.concatenate(selected_labels),
        tuple(selected_patients),
        tuple(selected_recordings),
        channel_names or (),
        sampling_frequency or 0.0,
        tuple(float(item) for item in selected_starts),
    )
    return BatchBPreparedData(
        data=data,
        full_segment_count=full_count,
        full_label_counts={"non_seizure": full_non_seizure, "seizure": full_seizure},
        retained_segment_count=len(data.labels),
        retained_label_counts={"non_seizure": int((data.labels == 0).sum()), "seizure": int((data.labels == 1).sum())},
        max_non_seizure_windows_per_recording=max_non_seizure_windows_per_recording,
    )


def _binary_metrics(targets: np.ndarray, probabilities: np.ndarray, threshold: float) -> tuple[dict[str, Any], np.ndarray]:
    """Compute genuine binary metrics; absent-class metrics are represented by ``None``."""
    actual = np.asarray(targets, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if actual.ndim != 1 or scores.shape != actual.shape or len(actual) == 0:
        raise ValueError("targets and probabilities must be equally sized non-empty vectors")
    if not np.isin(actual, (0, 1)).all() or not np.isfinite(scores).all():
        raise ValueError("binary metrics require finite probabilities and labels limited to 0 and 1")
    predicted = (scores >= threshold).astype(np.int64)
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    seizure_support, non_seizure_support = tp + fn, tn + fp
    sensitivity = None if seizure_support == 0 else float(tp / seizure_support)
    specificity = None if non_seizure_support == 0 else float(tn / non_seizure_support)
    precision = None if tp + fp == 0 else float(tp / (tp + fp))
    f1 = None if sensitivity is None or precision is None else (0.0 if sensitivity + precision == 0 else float(2 * sensitivity * precision / (sensitivity + precision)))
    auc_roc: float | None = None
    if seizure_support and non_seizure_support:
        positive = scores[actual == 1]
        negative = scores[actual == 0]
        comparisons = (positive[:, None] > negative).sum() + 0.5 * (positive[:, None] == negative).sum()
        auc_roc = float(comparisons / (len(positive) * len(negative)))
    undefined: list[str] = []
    if sensitivity is None:
        undefined.append("sensitivity and seizure F1 are undefined because no seizure labels are present")
    if auc_roc is None:
        undefined.append("AUC-ROC is undefined because both classes are required")
    return {
        "accuracy": float((tp + tn) / len(actual)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "auc_roc": auc_roc,
        "support": {"non_seizure": non_seizure_support, "seizure": seizure_support},
        "confusion_counts": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "undefined_metrics_reason": "; ".join(undefined) if undefined else None,
    }, np.asarray([[tn, fp], [fn, tp]], dtype=np.int64)


def _run_epoch(
    model: BatchBCNNBiLSTMClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(optimizer is not None)
    losses: list[float] = []
    targets: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs = inputs.to(device)
            binary_labels = labels.to(device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model.forward_logits(inputs)
            loss = criterion(logits, binary_labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()) * len(labels))
            targets.append(labels.cpu().numpy())
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return sum(losses) / len(loader.dataset), np.concatenate(targets), np.concatenate(probabilities)


def _checkpoint_payload(model: BatchBCNNBiLSTMClassifier, optimizer: Adam, training: BatchBTrainingConfig, epoch: int,
                        *, checkpoint_type: str = BATCH_B_DEVELOPMENT_CHECKPOINT_TYPE,
                        experiment_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "checkpoint_type": checkpoint_type,
        "compatibility_id": BATCH_B_MODEL_ID,
        "preprocessing_id": BATCH_B_PREPROCESSING_ID,
        "model_config": asdict(model.config),
        "training_config": asdict(training),
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "class_weighting": "not used; training-only EEG-specific minority oversampling is configured separately",
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "experiment_metadata": experiment_metadata or {},
    }


def save_batch_b_checkpoint(path: Path, model: BatchBCNNBiLSTMClassifier, optimizer: Adam,
                            training: BatchBTrainingConfig, epoch: int,
                            *, checkpoint_type: str = BATCH_B_DEVELOPMENT_CHECKPOINT_TYPE,
                            experiment_metadata: dict[str, Any] | None = None) -> None:
    """Save a Batch B checkpoint with architecture and preprocessing identifiers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_checkpoint_payload(model, optimizer, training, epoch, checkpoint_type=checkpoint_type,
                                   experiment_metadata=experiment_metadata), path)


def load_batch_b_checkpoint(path: Path, model: BatchBCNNBiLSTMClassifier, *, device: str = "cpu") -> dict[str, Any]:
    """Load only a checkpoint created for this model and Batch A preprocessing contract."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("checkpoint_type") not in {
        BATCH_B_DEVELOPMENT_CHECKPOINT_TYPE, BATCH_B_FINAL_CHECKPOINT_TYPE, BATCH_B_FINAL_V2_CHECKPOINT_TYPE,
    }:
        raise BatchBCompatibilityError("checkpoint is not a compatible Batch B development or final checkpoint")
    if checkpoint.get("compatibility_id") != BATCH_B_MODEL_ID:
        raise BatchBCompatibilityError("checkpoint model compatibility identifier does not match Batch B")
    if checkpoint.get("preprocessing_id") != BATCH_B_PREPROCESSING_ID:
        raise BatchBCompatibilityError("checkpoint preprocessing configuration is incompatible with Batch A")
    if checkpoint.get("model_config") != asdict(model.config):
        raise BatchBCompatibilityError("checkpoint model configuration does not match the requested Batch B model")
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def train_batch_b(
    model: BatchBCNNBiLSTMClassifier,
    data: RealEEGSegments,
    training: BatchBTrainingConfig,
    split_config: GroupSplitConfig,
    *,
    output_dir: Path,
    augmentation: AugmentationConfig | None = None,
    best_checkpoint_name: str = "batch_b_best.pt",
    final_checkpoint_name: str = "batch_b_final.pt",
    checkpoint_type: str = BATCH_B_DEVELOPMENT_CHECKPOINT_TYPE,
) -> dict[str, Any]:
    """Train Batch B on only the already split training recordings."""
    split = split_real_eeg(data, split_config)
    set_reproducible_seed(training.seed)
    original_labels = data.labels[split.train_indices]
    augmented = augment_training_segments(
        data.segments[split.train_indices], original_labels, data.sampling_frequency,
        augmentation or BATCH_B_AUGMENTATION,
    )
    train_dataset = EEGSegmentDataset(augmented.segments, augmented.labels)
    untouched_dataset = EEGSegmentDataset(data.segments, data.labels)
    generator = torch.Generator().manual_seed(training.seed)
    train_loader = DataLoader(train_dataset, batch_size=training.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(Subset(untouched_dataset, split.validation_indices.tolist()), batch_size=training.batch_size, shuffle=False)
    device = torch.device(training.device)
    model.to(device)
    optimizer = Adam(model.parameters(), lr=training.learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, final_path = output_dir / best_checkpoint_name, output_dir / final_checkpoint_name
    history: list[dict[str, Any]] = []
    best_validation_loss, best_epoch, no_improvement = float("inf"), 0, 0
    for epoch in range(1, training.epochs + 1):
        train_loss, train_targets, train_probabilities = _run_epoch(model, train_loader, criterion, device, optimizer)
        validation_loss, validation_targets, validation_probabilities = _run_epoch(model, validation_loader, criterion, device)
        train_metrics, _ = _binary_metrics(train_targets, train_probabilities, training.threshold)
        validation_metrics, _ = _binary_metrics(validation_targets, validation_probabilities, training.threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                        "train_metrics": train_metrics, "validation_metrics": validation_metrics})
        if validation_loss < best_validation_loss:
            best_validation_loss, best_epoch, no_improvement = validation_loss, epoch, 0
            save_batch_b_checkpoint(best_path, model, optimizer, training, epoch, checkpoint_type=checkpoint_type)
        else:
            no_improvement += 1
            if no_improvement >= training.early_stopping_patience:
                break
    save_batch_b_checkpoint(final_path, model, optimizer, training, history[-1]["epoch"], checkpoint_type=checkpoint_type)
    report = {
        "model_id": BATCH_B_MODEL_ID,
        "preprocessing_id": BATCH_B_PREPROCESSING_ID,
        "model": asdict(model.config),
        "training": asdict(training),
        "loss": "BCEWithLogitsLoss",
        "optimizer": "Adam",
        "class_weighting": "not used; training-only EEG-specific minority oversampling is configured separately",
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "split": {
            "group_by": split_config.group_by,
            "train_recording_ids": list(split.train_groups),
            "validation_recording_ids": list(split.validation_groups),
            "held_out_test_recording_ids": list(split.test_groups),
            "train_segment_count_before_augmentation": len(split.train_indices),
            "train_segment_count": len(train_dataset),
            "validation_segment_count": len(split.validation_indices),
            "held_out_test_segment_count": len(split.test_indices),
            "train_label_counts_before_augmentation": {"non_seizure": int((original_labels == 0).sum()), "seizure": int((original_labels == 1).sum())},
            "train_label_counts": {"non_seizure": int((augmented.labels == 0).sum()), "seizure": int((augmented.labels == 1).sum())},
        },
        "augmentation": {"enabled": bool((augmentation or BATCH_B_AUGMENTATION).enabled),
                         "derived_training_windows": augmented.augmented_count,
                         "validation_and_held_out_test_augmented": False},
        "best_checkpoint_path": str(best_path),
        "final_checkpoint_path": str(final_path),
    }
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def evaluate_batch_b(
    model: BatchBCNNBiLSTMClassifier,
    data: RealEEGSegments,
    split: GroupSplit,
    training: BatchBTrainingConfig,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    """Evaluate the untouched held-out recording partition at segment and recording level."""
    indices = split.test_indices
    dataset = EEGSegmentDataset(data.segments, data.labels)
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=training.batch_size, shuffle=False)
    device = torch.device(training.device)
    model.to(device).eval()
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, _ in loader:
            probabilities.append(model(inputs.to(device)).cpu().numpy())
    scores = np.concatenate(probabilities)
    labels = data.labels[indices]
    metrics, matrix = _binary_metrics(labels, scores, training.threshold)
    recording_ids = np.asarray(data.recording_ids, dtype=object)[indices]
    patient_ids = np.asarray(data.patient_ids, dtype=object)[indices]
    starts_all = data.segment_start_seconds or tuple(float("nan") for _ in data.labels)
    start_seconds = np.asarray(starts_all, dtype=np.float64)[indices]
    predicted = (scores >= training.threshold).astype(np.int64)
    record_ids = tuple(dict.fromkeys(recording_ids.tolist()))
    record_labels = np.asarray([int(np.any(labels[recording_ids == item] == 1)) for item in record_ids])
    record_scores = np.asarray([float(scores[recording_ids == item].max()) for item in record_ids])
    record_metrics, record_matrix = _binary_metrics(record_labels, record_scores, training.threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "held_out_confusion_matrix.npy", matrix)
    np.savez(
        output_dir / "held_out_segment_predictions.npz",
        probabilities=scores, predictions=predicted, ground_truth=labels,
        recording_ids=recording_ids.astype(str), patient_ids=patient_ids.astype(str), segment_start_seconds=start_seconds,
    )
    report = {
        "evaluation_level": "segment and separately aggregated recording",
        "threshold": training.threshold,
        "held_out_recording_ids": list(record_ids),
        "held_out_patient_ids": list(dict.fromkeys(patient_ids.tolist())),
        "segment_level": {"metrics": metrics, "confusion_matrix": matrix.tolist(), "segment_count": len(labels)},
        "recording_level": {"aggregation_rule": "maximum segment seizure probability per recording", "metrics": record_metrics,
                            "confusion_matrix": record_matrix.tolist(), "recording_count": len(record_ids)},
        "prediction_artifact": str(output_dir / "held_out_segment_predictions.npz"),
        "confusion_matrix_artifact": str(output_dir / "held_out_confusion_matrix.npy"),
    }
    (output_dir / "held_out_evaluation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_batch_b_development(
    dataset_root: Path,
    output_dir: Path,
    *,
    epochs: int = 1,
    max_non_seizure_windows_per_recording: int = 16,
) -> dict[str, Any]:
    """Run the documented bounded real CPU development workflow and held-out evaluation."""
    prepared = prepare_batch_b_development_data(
        dataset_root,
        max_non_seizure_windows_per_recording=max_non_seizure_windows_per_recording,
    )
    training = BatchBTrainingConfig(epochs=epochs)
    split_config = GroupSplitConfig(group_by="recording", validation_fraction=0.2, test_fraction=0.2, seed=training.seed)
    # Seed before model construction as well as before loader creation/training.
    set_reproducible_seed(training.seed)
    model = build_batch_b_model()
    training_report = train_batch_b(model, prepared.data, training, split_config, output_dir=output_dir)
    restored = build_batch_b_model()
    checkpoint = load_batch_b_checkpoint(Path(training_report["best_checkpoint_path"]), restored, device=training.device)
    evaluation = evaluate_batch_b(restored, prepared.data, split_real_eeg(prepared.data, split_config), training, output_dir=output_dir)
    report = {
        "prepared_data": {
            "full_segment_count": prepared.full_segment_count,
            "full_label_counts": prepared.full_label_counts,
            "retained_segment_count": prepared.retained_segment_count,
            "retained_label_counts": prepared.retained_label_counts,
            "max_non_seizure_windows_per_recording": prepared.max_non_seizure_windows_per_recording,
            "segment_shape": list(prepared.data.segments.shape),
            "sampling_frequency": prepared.data.sampling_frequency,
        },
        "training": training_report,
        "evaluation": evaluation,
        "checkpoint_reload": {"loaded": True, "epoch": checkpoint["epoch"]},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "batch_b_development_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
