"""Opt-in integration test for the actual Phase 17 held-out recordings."""

from pathlib import Path

import pytest

from ml.phase17_evaluation import evaluate_phase17


def test_real_held_out_evaluation_writes_undefined_positive_metrics(tmp_path: Path) -> None:
    value = __import__("os").environ.get("CHBMIT_DATASET_ROOT")
    checkpoint = Path(r"D:\EEG-Major-Project\models\phase16_chb01_development\cnn_bilstm_chbmit_best.pt")
    if not value or not checkpoint.is_file():
        pytest.skip("Real Phase 17 check needs CHBMIT_DATASET_ROOT and the Phase 16 checkpoint")
    report = evaluate_phase17(Path(value), checkpoint, tmp_path)
    assert report["recording_count"] == 3 and report["class_distribution"]["seizure"] == 0
    assert report["metrics"]["per_class"]["seizure"]["f1"] is None
    assert (tmp_path / "confusion_matrix.npy").is_file()
    assert (tmp_path / "classification_report.json").is_file()
