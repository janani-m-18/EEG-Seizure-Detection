"""Integrity checks for the guarded final CHB-MIT experiment entry point."""

from pathlib import Path

import ml.final_training as final_training
import torch

from ml.batch_b import (BATCH_B_FINAL_CHECKPOINT_TYPE, BatchBTrainingConfig, build_batch_b_model,
                        load_batch_b_checkpoint, save_batch_b_checkpoint)
from ml.model import BatchBModelConfig


def test_final_run_fails_closed_when_fewer_than_three_compatible_patients(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(final_training, "discover_and_verify_final_dataset", lambda _root: {
        "recordings": [
            {"patient_id": "chb01", "exclusion_reason": None, "seizure_interval_count": 1},
            {"patient_id": "chb01", "exclusion_reason": None, "seizure_interval_count": 0},
        ],
        "excluded_recordings": [],
    })
    result = final_training.run_final_experiment(Path("dataset"), output_dir=tmp_path)
    assert result["status"] == "blocked_before_training"
    assert result["checkpoint_path"] is None
    assert (tmp_path / "dataset_summary.json").is_file()
    assert (tmp_path / "exclusion_report.json").is_file()


def test_final_checkpoint_type_loads_with_batch_b_inference_contract(tmp_path: Path) -> None:
    config = BatchBModelConfig(input_channels=2, cnn_channels=(2, 4, 8), first_bilstm_hidden_size=4,
                               second_bilstm_hidden_size=3, dense_features=3, dropout=0)
    model = build_batch_b_model(config)
    checkpoint = tmp_path / final_training.FINAL_CHECKPOINT_NAME
    save_batch_b_checkpoint(checkpoint, model, torch.optim.Adam(model.parameters()), BatchBTrainingConfig(epochs=1), 1,
                            checkpoint_type=BATCH_B_FINAL_CHECKPOINT_TYPE)
    restored = build_batch_b_model(config)
    metadata = load_batch_b_checkpoint(checkpoint, restored)
    assert metadata["checkpoint_type"] == BATCH_B_FINAL_CHECKPOINT_TYPE
    with torch.no_grad():
        probabilities = restored(torch.zeros((2, 2, 64)))
    assert probabilities.shape == (2,) and bool(torch.all((probabilities >= 0) & (probabilities <= 1)))
