"""Synthetic structural tests for Steps 12-14; they do not evaluate CHB-MIT performance."""

from pathlib import Path

import numpy as np
import pytest

from backend.app.processing.types import ProcessedEEG, SegmentMetadata, SegmentedEEG
from ml.chbmit import (CHBMITConfig, CHBMITDatasetError, CHBMITRecording, CHBMITSegmentBuilder,
                       RealEEGSegments, SeizureInterval, discover_chbmit_recordings, parse_summary_annotations)
from ml.evaluation import GroupSplitConfig, classification_metrics, split_real_eeg
from ml.model import BiLSTMConfig, CNNBiLSTMClassifier, CNNConfig
from ml.training import TrainingConfig
from ml.evaluation import train_and_evaluate_real_data


def test_discovery_and_summary_parser(tmp_path: Path) -> None:
    patient = tmp_path / "chb01"; patient.mkdir()
    (patient / "chb01_01.edf").touch()
    (patient / "chb01-summary.txt").write_text(
        "File Name: chb01_01.edf\nSeizure Start Time: 12 seconds\nSeizure End Time: 15.5 seconds\n", encoding="utf-8")
    assert parse_summary_annotations(patient / "chb01-summary.txt")["chb01_01.edf"] == (SeizureInterval(12, 15.5),)
    assert discover_chbmit_recordings(CHBMITConfig(tmp_path))[0].has_seizure


def test_missing_dataset_fails_clearly(tmp_path: Path) -> None:
    with pytest.raises(CHBMITDatasetError, match="unavailable"):
        discover_chbmit_recordings(CHBMITConfig(tmp_path / "missing"))


class _PipelineFixture:
    def process(self, request):
        metadata = (SegmentMetadata(0, 0, 10, 0.0, 1.0), SegmentMetadata(1, 10, 20, 1.0, 2.0))
        segmented = SegmentedEEG(request.input_reference, np.ones((2, 2, 10)), ("A", "B"), 10.0, metadata)
        return ProcessedEEG(request.input_reference, segmented, None, None)  # type: ignore[arg-type]


def test_builder_labels_windows_and_keeps_provenance(tmp_path: Path) -> None:
    recording = CHBMITRecording("chb01", "chb01_01", tmp_path / "chb01_01.edf", (SeizureInterval(0.5, 1.5),))
    data = CHBMITSegmentBuilder(_PipelineFixture()).build([recording])
    assert data.labels.tolist() == [1, 1]
    assert data.patient_ids == ("chb01", "chb01")


def test_patient_split_has_no_leakage_and_metrics_are_real_counts() -> None:
    data = RealEEGSegments(np.zeros((12, 2, 8), dtype=np.float32), np.arange(12) % 2,
                           tuple(f"p{i // 3}" for i in range(12)), tuple(f"r{i // 3}" for i in range(12)), ("A", "B"), 10.0)
    split = split_real_eeg(data, GroupSplitConfig(seed=3))
    groups = [set(split.train_groups), set(split.validation_groups), set(split.test_groups)]
    assert not groups[0] & groups[1] and not groups[0] & groups[2] and not groups[1] & groups[2]
    metrics, matrix = classification_metrics(np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1]))
    assert matrix.tolist() == [[1, 1], [0, 2]] and metrics["accuracy"] == 0.75


def test_patient_split_requires_three_groups() -> None:
    data = RealEEGSegments(np.zeros((4, 2, 8)), np.array([0, 1, 0, 1]), ("a", "a", "b", "b"),
                           ("r1", "r1", "r2", "r2"), ("A", "B"), 1.0)
    with pytest.raises(ValueError, match="three"):
        split_real_eeg(data, GroupSplitConfig())


def test_real_training_writes_held_out_report(tmp_path: Path) -> None:
    data = RealEEGSegments(np.linspace(-1, 1, 16 * 2 * 8, dtype=np.float32).reshape(16, 2, 8), np.arange(16) % 2,
                           tuple(f"p{i // 4}" for i in range(16)), tuple(f"r{i // 4}" for i in range(16)), ("A", "B"), 10.0)
    model = CNNBiLSTMClassifier(CNNConfig(2, (2, 2), kernel_size=3, dropout=0), BiLSTMConfig(2, hidden_size=2, dropout=0), 2)
    report = train_and_evaluate_real_data(model, data, TrainingConfig(epochs=1, batch_size=2, seed=1), GroupSplitConfig(seed=1), output_dir=tmp_path)
    assert (tmp_path / "cnn_bilstm_chbmit.pt").exists() and (tmp_path / "confusion_matrix.npy").exists()
    assert (tmp_path / "evaluation_metrics.json").exists() and "f1_macro" in report["test"]
