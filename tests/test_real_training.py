"""Structural tests for leakage-safe real-data training without test evaluation."""

import numpy as np

from ml.chbmit import RealEEGSegments
from ml.evaluation import GroupSplitConfig, train_real_data
from ml.model import BiLSTMConfig, CNNBiLSTMClassifier, CNNConfig
from ml.training import TrainingConfig


def test_real_training_preserves_the_held_out_recordings_and_saves_best_checkpoint(tmp_path) -> None:
    data = RealEEGSegments(
        np.linspace(-1, 1, 24 * 2 * 8, dtype=np.float32).reshape(24, 2, 8),
        np.array([0, 1] * 12),
        tuple(f"p{i // 6}" for i in range(24)), tuple(f"r{i // 6}" for i in range(24)), ("A", "B"), 10.0,
    )
    model = CNNBiLSTMClassifier(CNNConfig(2, (2,), kernel_size=3, dropout=0), BiLSTMConfig(2, hidden_size=2, dropout=0), 2)
    report = train_real_data(model, data, TrainingConfig(epochs=2, batch_size=3, seed=1, early_stopping_patience=1),
                             GroupSplitConfig(group_by="recording", seed=1), output_dir=tmp_path)
    groups = report["split"]
    assert not set(groups["train_groups"]) & set(groups["validation_groups"])
    assert not set(groups["train_groups"]) & set(groups["held_out_test_groups"])
    assert (tmp_path / "cnn_bilstm_chbmit_best.pt").exists()
    assert (tmp_path / "training_metrics.json").exists()
