"""Phase 20 API tests, including one genuine EDF upload through the real model."""

from pathlib import Path

import mne
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_analyze_requires_an_uploaded_file() -> None:
    response = client.post("/api/eeg/analyze")
    assert response.status_code == 422


def test_analyze_rejects_non_edf_and_empty_uploads() -> None:
    invalid = client.post("/api/eeg/analyze", files={"file": ("recording.csv", b"one,two", "text/csv")})
    empty = client.post("/api/eeg/analyze", files={"file": ("recording.edf", b"", "application/octet-stream")})
    assert invalid.status_code == 400 and invalid.json()["detail"] == "Please upload an EDF file."
    assert empty.status_code == 400 and empty.json()["detail"] == "Uploaded EDF file is empty."


def test_analyze_rejects_corrupted_edf() -> None:
    response = client.post("/api/eeg/analyze", files={"file": ("corrupted.edf", b"not an EDF", "application/octet-stream")})
    assert response.status_code == 422
    assert response.json()["detail"] == "Unable to read and process the EEG recording."


def test_analyze_reports_incompatible_sampling_rate_and_channels(tmp_path: Path) -> None:
    wrong_rate = tmp_path / "wrong-rate.edf"
    raw = mne.io.RawArray(np.zeros((23, 512)), mne.create_info([f"C{index}" for index in range(23)], 128, "eeg"), verbose="ERROR")
    mne.export.export_raw(wrong_rate, raw, overwrite=True, verbose="ERROR")
    with wrong_rate.open("rb") as edf:
        rate_response = client.post("/api/eeg/analyze", files={"file": (wrong_rate.name, edf, "application/octet-stream")})
    wrong_channels = tmp_path / "wrong-channels.edf"
    raw = mne.io.RawArray(np.zeros((22, 1024)), mne.create_info([f"C{index}" for index in range(22)], 256, "eeg"), verbose="ERROR")
    mne.export.export_raw(wrong_channels, raw, overwrite=True, verbose="ERROR")
    with wrong_channels.open("rb") as edf:
        channel_response = client.post("/api/eeg/analyze", files={"file": (wrong_channels.name, edf, "application/octet-stream")})
    assert rate_response.status_code == 422 and "sampling rate" in rate_response.json()["detail"]
    assert channel_response.status_code == 422 and "channel configuration" in channel_response.json()["detail"]


def test_real_edf_upload_returns_model_generated_analysis() -> None:
    path = Path(r"D:\EEG-Datasets\CHB-MIT\physionet.org\files\chbmit\1.0.0\chb01\chb01_03.edf")
    if not path.is_file():
        pytest.skip("The local real CHB-MIT development EDF is unavailable")
    with path.open("rb") as edf:
        response = client.post("/api/eeg/analyze", files={"file": (path.name, edf, "application/octet-stream")})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["recording"]["duration_seconds"] == 3600.0
    assert payload["recording"]["segment_count"] == 1_799
    assert payload["prediction"]["label"] in {"seizure", "non_seizure"}
    assert len(payload["segments"]) == 1_799
    assert len(payload["waveform"]["samples"]) == 23
    assert payload["explainability"]["method"] == "Grad-CAM"
    assert len(payload["explainability"]["gradcam_temporal_map"]) == 1_024
    assert len(payload["explainability"]["raw_gradcam_temporal_map"]) == 128
    assert payload["explainability"]["target_class"] == payload["explainability"]["predicted_class"]


def test_stream_window_real_time_endpoint() -> None:
    # 23 channels x 1024 samples dummy window
    window = [[0.0] * 1024 for _ in range(23)]
    response = client.post(
        "/api/eeg/stream_window",
        json={
            "window_samples": window,
            "current_run": 0,
            "timestamp_seconds": 12.0,
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert "seizure_probability" in payload
    assert "is_positive" in payload
    assert "current_run" in payload
    assert payload["timestamp_seconds"] == 12.0
    assert payload["monitor_state"] in {"MONITORING", "SUSPECT ACTIVITY", "SEIZURE EVENT DETECTED"}

