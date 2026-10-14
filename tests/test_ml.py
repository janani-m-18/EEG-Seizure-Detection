"""CPU-only tests for Steps 9-11; fixtures are synthetic and deterministic."""

import numpy as np
import pytest
import torch

from ml.model import BiLSTMConfig, CNNBiLSTMClassifier, CNNConfig, EEGCNNFeatureExtractor, TemporalBiLSTM
from ml.training import EEGSegmentDataset, TrainingConfig, load_checkpoint, train_model


def _model() -> CNNBiLSTMClassifier:
    cnn = CNNConfig(input_channels=2, feature_channels=(4, 8), kernel_size=3, dropout=0)
    return CNNBiLSTMClassifier(cnn, BiLSTMConfig(input_features=8, hidden_size=6, dropout=0), num_classes=2)


def test_cnn_forward_pass_has_sequence_features() -> None:
    output = EEGCNNFeatureExtractor(CNNConfig(2, (4, 8), kernel_size=3))(torch.randn(3, 2, 32))
    assert output.shape == (3, 8, 8)


def test_bilstm_forward_pass_summarizes_sequence() -> None:
    output = TemporalBiLSTM(BiLSTMConfig(8, hidden_size=5))(torch.randn(3, 8, 8))
    assert output.shape == (3, 10)


def test_combined_model_forward_pass() -> None:
    assert _model()(torch.randn(4, 2, 32)).shape == (4, 2)


def test_training_checkpoint_and_load(tmp_path) -> None:
    data = np.linspace(-1, 1, 16 * 2 * 32, dtype=np.float32).reshape(16, 2, 32)
    dataset = EEGSegmentDataset(data, np.arange(16) % 2)
    path = tmp_path / "checkpoint.pt"
    model = _model()
    history = train_model(model, dataset, TrainingConfig(epochs=1, batch_size=4, seed=7, checkpoint_path=path))
    assert len(history) == 1 and path.exists()
    restored = _model()
    checkpoint = load_checkpoint(path, restored)
    assert checkpoint["epoch"] == 1
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(torch.from_numpy(data[:2])), restored(torch.from_numpy(data[:2])))


def test_invalid_input_handling() -> None:
    with pytest.raises(ValueError, match="shape"):
        EEGCNNFeatureExtractor(CNNConfig(2))(torch.randn(2, 32))
    with pytest.raises(ValueError, match="labels"):
        EEGSegmentDataset(np.zeros((2, 2, 32)), [0])
    with pytest.raises(ValueError, match="at least two"):
        train_model(_model(), EEGSegmentDataset(np.zeros((1, 2, 32)), [0]), TrainingConfig())
