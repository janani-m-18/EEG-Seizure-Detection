"""Leakage-safe splitting, training, and evaluation for prepared real EEG segments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from ml.augmentation import AugmentationConfig, augment_training_segments
from ml.chbmit import RealEEGSegments
from ml.training import EEGSegmentDataset, TrainingConfig, save_checkpoint, set_reproducible_seed


@dataclass(frozen=True, slots=True)
class GroupSplitConfig:
    group_by: Literal["patient", "recording"] = "patient"
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    seed: int = 42

    def __post_init__(self) -> None:
        if not 0 < self.validation_fraction < 1 or not 0 < self.test_fraction < 1:
            raise ValueError("validation_fraction and test_fraction must be between zero and one")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be below one")


@dataclass(frozen=True, slots=True)
class GroupSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray
    train_groups: tuple[str, ...]
    validation_groups: tuple[str, ...]
    test_groups: tuple[str, ...]


def split_real_eeg(data: RealEEGSegments, config: GroupSplitConfig) -> GroupSplit:
    """Assign complete patients or recordings to one split, never individual windows."""
    groups = np.asarray(data.patient_ids if config.group_by == "patient" else data.recording_ids)
    unique = np.unique(groups)
    if len(unique) < 3:
        raise ValueError(f"at least three distinct {config.group_by} groups are required for train/validation/test splitting")
    rng = np.random.default_rng(config.seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    test_count = max(1, round(len(shuffled) * config.test_fraction))
    validation_count = max(1, round(len(shuffled) * config.validation_fraction))
    if test_count + validation_count >= len(shuffled):
        validation_count = len(shuffled) - test_count - 1
    test_groups = tuple(shuffled[:test_count])
    validation_groups = tuple(shuffled[test_count:test_count + validation_count])
    train_groups = tuple(shuffled[test_count + validation_count:])
    def indices(selected: tuple[str, ...]) -> np.ndarray:
        return np.flatnonzero(np.isin(groups, selected))
    result = GroupSplit(indices(train_groups), indices(validation_groups), indices(test_groups), train_groups, validation_groups, test_groups)
    if not len(result.train_indices) or not len(result.validation_indices) or not len(result.test_indices):
        raise ValueError("group split produced an empty partition")
    return result


def classification_metrics(targets: np.ndarray, predictions: np.ndarray, *, num_classes: int = 2) -> tuple[dict[str, float], np.ndarray]:
    """Compute macro precision/recall/F1 and an integer confusion matrix without sklearn."""
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for actual, predicted in zip(targets, predictions, strict=True):
        if 0 <= actual < num_classes and 0 <= predicted < num_classes:
            matrix[actual, predicted] += 1
    precision, recall, f1 = [], [], []
    for label in range(num_classes):
        tp = matrix[label, label]
        p_denom, r_denom = matrix[:, label].sum(), matrix[label, :].sum()
        p, r = (tp / p_denom if p_denom else 0.0), (tp / r_denom if r_denom else 0.0)
        precision.append(p); recall.append(r); f1.append(2 * p * r / (p + r) if p + r else 0.0)
    return {"accuracy": float(np.trace(matrix) / matrix.sum()) if matrix.sum() else 0.0,
            "precision_macro": float(np.mean(precision)), "recall_macro": float(np.mean(recall)),
            "f1_macro": float(np.mean(f1))}, matrix


def _run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, optimizer: torch.optim.Optimizer | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(optimizer is not None)
    losses, targets, predictions = 0.0, [], []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            if optimizer is not None: optimizer.zero_grad()
            logits = model(inputs); loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward(); optimizer.step()
            losses += loss.item() * len(labels)
            targets.append(labels.cpu().numpy()); predictions.append(logits.argmax(1).cpu().numpy())
    return losses / len(loader.dataset), np.concatenate(targets), np.concatenate(predictions)


def train_and_evaluate_real_data(model: nn.Module, data: RealEEGSegments, training: TrainingConfig,
                                 split_config: GroupSplitConfig, *, output_dir: Path) -> dict[str, object]:
    """Train on a group split, save checkpoint/metrics/matrix, then evaluate held-out test data."""
    split = split_real_eeg(data, split_config)
    set_reproducible_seed(training.seed)
    dataset = EEGSegmentDataset(data.segments, data.labels)
    generator = torch.Generator().manual_seed(training.seed)
    loaders = [DataLoader(Subset(dataset, indices.tolist()), batch_size=training.batch_size, shuffle=shuffle,
                          generator=generator if shuffle else None) for indices, shuffle in
               ((split.train_indices, True), (split.validation_indices, False), (split.test_indices, False))]
    device = torch.device(training.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    criterion = nn.CrossEntropyLoss()
    history = []
    for epoch in range(1, training.epochs + 1):
        train_loss, train_target, train_pred = _run_epoch(model, loaders[0], criterion, device, optimizer)
        val_loss, val_target, val_pred = _run_epoch(model, loaders[1], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, **{f"train_{k}": v for k, v in classification_metrics(train_target, train_pred)[0].items()},
                        "validation_loss": val_loss, **{f"validation_{k}": v for k, v in classification_metrics(val_target, val_pred)[0].items()}})
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = training.checkpoint_path or output_dir / "cnn_bilstm_chbmit.pt"
    save_checkpoint(checkpoint_path, model, optimizer, training, training.epochs)
    test_loss, test_target, test_pred = _run_epoch(model, loaders[2], criterion, device)
    test_metrics, matrix = classification_metrics(test_target, test_pred)
    np.save(output_dir / "confusion_matrix.npy", matrix)
    report = {"history": history, "test": {"loss": test_loss, **test_metrics}, "split": {
        "group_by": split_config.group_by, "train_groups": list(split.train_groups),
        "validation_groups": list(split.validation_groups), "test_groups": list(split.test_groups)},
        "checkpoint_path": str(checkpoint_path), "confusion_matrix_path": str(output_dir / "confusion_matrix.npy")}
    (output_dir / "evaluation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def train_real_data(model: nn.Module, data: RealEEGSegments, training: TrainingConfig,
                    split_config: GroupSplitConfig, *, output_dir: Path,
                    augmentation: AugmentationConfig | None = None) -> dict[str, object]:
    """Train with recording/patient groups while leaving the test partition untouched.

    This is intended for the training phase.  It saves the checkpoint with the
    lowest validation loss and records split provenance, but deliberately does
    not calculate test-set predictions or metrics.
    """
    split = split_real_eeg(data, split_config)
    set_reproducible_seed(training.seed)
    dataset = EEGSegmentDataset(data.segments, data.labels)
    original_train_labels = data.labels[split.train_indices]
    train_data = augment_training_segments(
        data.segments[split.train_indices],
        data.labels[split.train_indices],
        data.sampling_frequency,
        augmentation,
    )
    train_labels = train_data.labels
    class_counts = np.bincount(train_labels, minlength=2)
    if np.any(class_counts == 0):
        raise ValueError("the training partition must contain both seizure and non-seizure segments")
    class_weights = len(train_labels) / (len(class_counts) * class_counts.astype(np.float32))
    generator = torch.Generator().manual_seed(training.seed)
    train_dataset = EEGSegmentDataset(train_data.segments, train_data.labels)
    loaders = [
        DataLoader(train_dataset, batch_size=training.batch_size, shuffle=True, generator=generator),
        DataLoader(Subset(dataset, split.validation_indices.tolist()), batch_size=training.batch_size, shuffle=False),
    ]
    device = torch.device(training.device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=training.learning_rate, weight_decay=training.weight_decay)
    criterion = nn.CrossEntropyLoss(weight=torch.as_tensor(class_weights, dtype=torch.float32, device=device))
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = training.checkpoint_path or output_dir / "cnn_bilstm_chbmit_best.pt"
    history: list[dict[str, float | int]] = []
    best_validation_loss, best_epoch, epochs_without_improvement = float("inf"), 0, 0
    for epoch in range(1, training.epochs + 1):
        train_loss, train_target, train_pred = _run_epoch(model, loaders[0], criterion, device, optimizer)
        validation_loss, validation_target, validation_pred = _run_epoch(model, loaders[1], criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        **{f"train_{key}": value for key, value in classification_metrics(train_target, train_pred)[0].items()},
                        "validation_loss": validation_loss,
                        **{f"validation_{key}": value for key, value in classification_metrics(validation_target, validation_pred)[0].items()}})
        if validation_loss < best_validation_loss:
            best_validation_loss, best_epoch, epochs_without_improvement = validation_loss, epoch, 0
            save_checkpoint(checkpoint_path, model, optimizer, training, epoch)
        else:
            epochs_without_improvement += 1
            if training.early_stopping_patience is not None and epochs_without_improvement >= training.early_stopping_patience:
                break
    report: dict[str, object] = {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "class_weights": class_weights.tolist(),
        "split": {
            "group_by": split_config.group_by,
            "train_groups": list(split.train_groups), "validation_groups": list(split.validation_groups),
            "held_out_test_groups": list(split.test_groups),
            "train_segment_count_before_augmentation": len(split.train_indices),
            "train_segment_count": len(train_dataset),
            "augmented_training_segment_count": train_data.augmented_count,
            "validation_segment_count": len(split.validation_indices),
            "held_out_test_segment_count": len(split.test_indices),
            "train_label_counts_before_augmentation": {
                str(label): int(count)
                for label, count in enumerate(np.bincount(original_train_labels, minlength=2))
            },
            "train_label_counts": {str(label): int(count) for label, count in enumerate(class_counts)},
        },
        "augmentation": {
            "enabled": bool((augmentation or AugmentationConfig()).enabled),
            "method": "EEG-specific transformed minority-window oversampling; conventional SMOTE is not applied to raw tensors",
            "augmented_training_segment_count": train_data.augmented_count,
            "validation_and_held_out_test_augmented": False,
        },
        "checkpoint_path": str(checkpoint_path),
    }
    (output_dir / "training_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
