"""Batch A tests proving augmentation is confined to the training partition."""

import numpy as np

from ml.augmentation import AugmentationConfig, augment_training_segments
from ml.chbmit import RealEEGSegments
from ml.evaluation import GroupSplitConfig, train_real_data
from ml.model import BiLSTMConfig, CNNBiLSTMClassifier, CNNConfig
from ml.training import TrainingConfig


def test_eeg_specific_oversampling_only_derives_minority_training_windows() -> None:
    values = np.arange(6 * 2 * 16, dtype=np.float32).reshape(6, 2, 16)
    labels = np.array([0, 0, 0, 0, 1, 1])
    augmented = augment_training_segments(
        values, labels, 16.0,
        AugmentationConfig(enabled=True, time_shift_seconds=0.5, seed=7),
    )
    assert augmented.segments.shape == (8, 2, 16)
    assert np.array_equal(augmented.segments[:6], values)
    assert np.array_equal(augmented.labels, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
    assert augmented.augmented_count == 2
    assert np.all(augmented.labels[augmented.is_augmented] == 1)
    assert np.all(np.isin(augmented.source_indices[augmented.is_augmented], [4, 5]))


def test_train_real_data_augments_only_the_already_split_training_records(tmp_path) -> None:
    data = RealEEGSegments(
        np.linspace(-1, 1, 24 * 2 * 8, dtype=np.float32).reshape(24, 2, 8),
        np.tile(np.array([0, 0, 0, 0, 1, 1]), 4),
        tuple(f"p{index // 6}" for index in range(24)),
        tuple(f"r{index // 6}" for index in range(24)),
        ("A", "B"),
        10.0,
    )
    model = CNNBiLSTMClassifier(CNNConfig(2, (2,), kernel_size=3, dropout=0), BiLSTMConfig(2, hidden_size=2, dropout=0), 2)
    report = train_real_data(
        model, data, TrainingConfig(epochs=1, batch_size=3, seed=1),
        GroupSplitConfig(group_by="recording", seed=1), output_dir=tmp_path,
        augmentation=AugmentationConfig(enabled=True, seed=1),
    )
    split = report["split"]
    assert not set(split["train_groups"]) & set(split["validation_groups"])
    assert not set(split["train_groups"]) & set(split["held_out_test_groups"])
    assert split["train_segment_count"] > split["train_segment_count_before_augmentation"]
    assert split["validation_segment_count"] == 6
    assert split["held_out_test_segment_count"] == 6
    assert report["augmentation"]["validation_and_held_out_test_augmented"] is False
