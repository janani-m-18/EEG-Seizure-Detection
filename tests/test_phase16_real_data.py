"""Opt-in Phase 16 checks using a real CHB-MIT EDF without model evaluation."""

from pathlib import Path

import pytest
import torch

from ml.phase16_training import prepare_phase16_data
from ml.model import BiLSTMConfig, CNNBiLSTMClassifier, CNNConfig


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    value = __import__("os").environ.get("CHBMIT_DATASET_ROOT")
    if not value:
        pytest.skip("Phase 16 real-data checks are opt-in via CHBMIT_DATASET_ROOT")
    root = Path(value)
    if not (root / "chb01").is_dir():
        pytest.skip("Phase 16 real-data checks need the local chb01 directory")
    return root


def test_real_data_has_fixed_model_input_and_supports_one_training_step(dataset_root: Path) -> None:
    data = prepare_phase16_data(dataset_root)
    assert data.segments.shape == (1560, 23, 1024)
    assert data.labels.sum() == 3
    model = CNNBiLSTMClassifier(CNNConfig(23, (4, 8), kernel_size=7, dropout=0), BiLSTMConfig(8, hidden_size=8, dropout=0), 2)
    inputs = torch.from_numpy(data.segments[:2])
    targets = torch.from_numpy(data.labels[:2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    logits = model(inputs)
    loss = torch.nn.CrossEntropyLoss()(logits, targets)
    loss.backward()
    optimizer.step()
    assert logits.shape == (2, 2)
    assert torch.isfinite(loss)
