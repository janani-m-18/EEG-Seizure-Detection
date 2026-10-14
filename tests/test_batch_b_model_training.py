"""Batch B model, training, checkpoint, and evaluation behavior."""

from pathlib import Path

import numpy as np
import pytest
import torch

import ml.batch_b as batch_b
from ml.batch_b import (BATCH_B_MODEL_CONFIG, BatchBCompatibilityError, BatchBTrainingConfig, _binary_metrics,
                        build_batch_b_model, evaluate_batch_b, load_batch_b_checkpoint, train_batch_b)
from ml.chbmit import RealEEGSegments
from ml.evaluation import GroupSplit, GroupSplitConfig, split_real_eeg
from ml.model import BatchBModelConfig


def _small_config() -> BatchBModelConfig:
    return BatchBModelConfig(input_channels=2, cnn_channels=(2, 4, 8), first_bilstm_hidden_size=4,
                             second_bilstm_hidden_size=3, dense_features=3, dropout=0)


def _data() -> RealEEGSegments:
    labels = np.tile(np.array([0, 0, 1]), 4)
    return RealEEGSegments(
        np.linspace(-1, 1, 12 * 2 * 64, dtype=np.float32).reshape(12, 2, 64), labels,
        tuple(f"p{index // 3}" for index in range(12)), tuple(f"r{index // 3}" for index in range(12)),
        ("A", "B"), 256.0, tuple(float(index) for index in range(12)),
    )


def test_batch_b_model_has_the_required_default_architecture_and_binary_probabilities() -> None:
    model = build_batch_b_model()
    assert BATCH_B_MODEL_CONFIG.cnn_channels == (64, 128, 256)
    assert model.cnn_temporal_shape == (128, 256)
    assert model.first_bilstm.hidden_size == 128 and model.first_bilstm.bidirectional
    assert model.second_bilstm.hidden_size == 64 and model.second_bilstm.bidirectional
    assert model.dense.in_features == 128 and model.dense.out_features == 64
    assert model.output.out_features == 1
    with torch.no_grad():
        cnn = model.cnn(torch.randn(1, 23, 1024))
        features = model.forward_features(torch.randn(1, 23, 1024))
        probabilities = model(torch.randn(1, 23, 1024))
    assert cnn.shape == (1, 128, 256)
    assert features.shape == (1, 128, 128)
    assert probabilities.shape == (1,) and bool(torch.all((probabilities >= 0) & (probabilities <= 1)))


def test_batch_b_defaults_match_bce_adam_training_methodology() -> None:
    config = BatchBTrainingConfig()
    assert config.learning_rate == 0.001
    assert config.batch_size == 32
    assert config.epochs == 50
    assert config.threshold == 0.5


def test_binary_metrics_include_counts_auc_and_undefined_absent_class_values() -> None:
    metrics, matrix = _binary_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.6, 0.7, 0.3]), 0.5)
    assert matrix.tolist() == [[1, 1], [1, 1]]
    assert metrics["accuracy"] == metrics["sensitivity"] == metrics["specificity"] == metrics["precision"] == metrics["f1"] == 0.5
    assert metrics["auc_roc"] == 0.75
    assert metrics["confusion_counts"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}

    undefined, undefined_matrix = _binary_metrics(np.zeros(3, dtype=np.int64), np.array([0.1, 0.6, 0.2]), 0.5)
    assert undefined_matrix.tolist() == [[2, 1], [0, 0]]
    assert undefined["sensitivity"] is None and undefined["f1"] is None and undefined["auc_roc"] is None
    assert "no seizure labels" in str(undefined["undefined_metrics_reason"])


def test_batch_b_training_uses_adam_bce_training_only_augmentation_and_versioned_checkpoints(tmp_path: Path) -> None:
    model = build_batch_b_model(_small_config())
    report = train_batch_b(
        model, _data(), BatchBTrainingConfig(epochs=1, batch_size=2, seed=1),
        GroupSplitConfig(group_by="recording", seed=1), output_dir=tmp_path,
    )
    assert report["training"]["batch_size"] == 2
    assert report["augmentation"]["validation_and_held_out_test_augmented"] is False
    assert report["split"]["train_segment_count"] > report["split"]["train_segment_count_before_augmentation"]
    checkpoint_path = Path(report["best_checkpoint_path"])
    restored = build_batch_b_model(_small_config())
    assert load_batch_b_checkpoint(checkpoint_path, restored)["epoch"] == 1

    incompatible = torch.load(checkpoint_path, weights_only=False)
    incompatible["preprocessing_id"] = "wrong-preprocessing"
    bad_path = tmp_path / "wrong.pt"
    torch.save(incompatible, bad_path)
    with pytest.raises(BatchBCompatibilityError, match="preprocessing"):
        load_batch_b_checkpoint(bad_path, restored)


def test_early_stopping_monitors_validation_loss(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[bool] = []

    def fake_epoch(model, loader, criterion, device, optimizer=None):
        calls.append(optimizer is not None)
        return 1.0, np.array([0, 1]), np.array([0.2, 0.8])

    monkeypatch.setattr(batch_b, "_run_epoch", fake_epoch)
    report = train_batch_b(
        build_batch_b_model(_small_config()), _data(), BatchBTrainingConfig(epochs=5, batch_size=2, early_stopping_patience=1),
        GroupSplitConfig(group_by="recording", seed=1), output_dir=tmp_path,
    )
    assert len(report["history"]) == 2
    assert calls == [True, False, True, False]


def test_held_out_evaluation_saves_actual_prediction_provenance_and_keeps_absent_seizure_metrics_undefined(tmp_path: Path) -> None:
    data = _data()
    split = GroupSplit(np.array([0, 1, 2, 3, 4, 5]), np.array([6, 7, 8]), np.array([9, 10, 11]), ("r0", "r1"), ("r2",), ("r3",))
    # The chosen held-out recording has one positive label, then replace it with all-negative data to exercise the edge case.
    all_negative = RealEEGSegments(data.segments, np.where(np.asarray(data.recording_ids) == "r3", 0, data.labels),
                                   data.patient_ids, data.recording_ids, data.channel_names, data.sampling_frequency,
                                   data.segment_start_seconds)
    report = evaluate_batch_b(build_batch_b_model(_small_config()), all_negative, split, BatchBTrainingConfig(batch_size=2), output_dir=tmp_path)
    assert (tmp_path / "held_out_confusion_matrix.npy").exists()
    assert (tmp_path / "held_out_segment_predictions.npz").exists()
    assert report["segment_level"]["metrics"]["sensitivity"] is None
    assert report["segment_level"]["metrics"]["auc_roc"] is None
