"""Checkpoint, inference, and Integrated Gradients tests without fabricated predictions."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ml.explainability import explain_edf_segment, explain_segment, save_attribution_visualization
from ml.inference import EEGInferenceError, EEGInferenceService
from ml.phase16_training import build_phase16_model
from ml.training import TrainingConfig, save_checkpoint


@pytest.fixture()
def checkpoint_path(tmp_path: Path) -> Path:
    model = build_phase16_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "development.pt"
    save_checkpoint(path, model, optimizer, TrainingConfig(epochs=1), epoch=1)
    return path


def test_checkpoint_inference_and_integrated_gradients_are_model_derived(checkpoint_path: Path) -> None:
    service = EEGInferenceService(checkpoint_path)
    segment = np.linspace(-1, 1, 23 * 1024, dtype=np.float32).reshape(23, 1024)
    prediction, probabilities = service.predict_segments(segment[None, ...])
    explanation = explain_segment(service, segment, sampling_frequency=256.0, segment_start_seconds=0.0,
                                  segment_stop_seconds=4.0, integration_steps=4)
    changed = explain_segment(service, segment + 0.1, sampling_frequency=256.0, segment_start_seconds=0.0,
                              segment_stop_seconds=4.0, integration_steps=4)
    assert explanation.prediction == int(prediction[0])
    assert np.isclose(probabilities[0].sum(), 1.0)
    assert explanation.attribution.shape == segment.shape
    assert np.isfinite(explanation.attribution).all()
    assert not np.allclose(explanation.attribution, changed.attribution)


def test_inference_rejects_missing_checkpoint_and_invalid_segment_shapes(tmp_path: Path, checkpoint_path: Path) -> None:
    with pytest.raises(EEGInferenceError, match="does not exist"):
        EEGInferenceService(tmp_path / "missing.pt")
    with pytest.raises(EEGInferenceError, match="Expected segments"):
        EEGInferenceService(checkpoint_path).predict_segments(np.zeros((1, 22, 1024), dtype=np.float32))
    invalid_edf = tmp_path / "invalid.edf"
    invalid_edf.write_text("not an EDF file", encoding="utf-8")
    with pytest.raises(EEGInferenceError, match="Could not process"):
        EEGInferenceService(checkpoint_path).predict_edf(invalid_edf)


@pytest.fixture(scope="module")
def real_data_root() -> Path:
    value = __import__("os").environ.get("CHBMIT_DATASET_ROOT")
    if not value:
        pytest.skip("Real Phase 18/19 checks are opt-in via CHBMIT_DATASET_ROOT")
    root = Path(value)
    if not (root / "chb01").is_dir():
        pytest.skip("Real Phase 18/19 checks need the local chb01 directory")
    return root


def test_real_edf_inference_and_xai_visualization(real_data_root: Path, tmp_path: Path) -> None:
    checkpoint = Path(r"D:\EEG-Major-Project\models\phase16_chb01_development\cnn_bilstm_chbmit_best.pt")
    if not checkpoint.is_file():
        pytest.skip("Phase 16 development checkpoint is unavailable")
    service = EEGInferenceService(checkpoint)
    edf = real_data_root / "chb01" / "chb01_03.edf"
    result = service.predict_edf(edf)
    seizure_window = explain_edf_segment(service, edf, 100, integration_steps=8)  # 3000-3004 seconds, annotated seizure.
    non_seizure_window = explain_edf_segment(service, edf, 0, integration_steps=8)
    segmented = service.prepare_edf(edf)
    image = save_attribution_visualization(segmented.segments[100], seizure_window, tmp_path / "real_xai.png")
    assert len(result.segments) == 120
    assert all(np.isclose(item.non_seizure_probability + item.seizure_probability, 1.0) for item in result.segments)
    assert seizure_window.attribution.shape == (23, 1024)
    assert non_seizure_window.attribution.shape == (23, 1024)
    assert np.isfinite(seizure_window.attribution).all() and image.is_file()
