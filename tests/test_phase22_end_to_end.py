"""Final real-flow verification for the upload API, model, XAI, and temporary-file cleanup."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes import eeg_analysis
from backend.app.core.config import settings
from backend.app.main import app


REAL_EDF = Path(r"D:\EEG-Datasets\CHB-MIT\physionet.org\files\chbmit\1.0.0\chb01\chb01_03.edf")


def test_real_upload_runs_full_analysis_and_removes_temporary_edf(monkeypatch: pytest.MonkeyPatch) -> None:
    if not REAL_EDF.is_file():
        pytest.skip("The local real CHB-MIT development EDF is unavailable")
    created_paths: list[Path] = []
    original = eeg_analysis.NamedTemporaryFile

    def tracked_temporary_file(*args, **kwargs):
        handle = original(*args, **kwargs)
        created_paths.append(Path(handle.name))
        return handle

    monkeypatch.setattr(eeg_analysis, "NamedTemporaryFile", tracked_temporary_file)
    with REAL_EDF.open("rb") as edf:
        response = TestClient(app).post(
            "/api/eeg/analyze", files={"file": (REAL_EDF.name, edf, "application/octet-stream")}
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["status"] == "success"
    assert payload["recording"] == {
        "duration_seconds": 3600.0,
        "sampling_rate": 256.0,
        "channels": payload["recording"]["channels"],
        "segment_count": 1_799,
    }
    assert len(payload["recording"]["channels"]) == 23
    assert payload["prediction"]["label"] in {"seizure", "non_seizure"}
    assert len(payload["segments"]) == 1_799
    assert len(payload["waveform"]["samples"]) == 23
    assert payload["explainability"]["method"] == "Grad-CAM"
    assert len(payload["explainability"]["gradcam_temporal_map"]) == 1_024
    assert payload["explainability"]["important_temporal_regions"]
    assert created_paths and all(not path.exists() for path in created_paths)


def test_oversized_upload_and_missing_checkpoint_return_safe_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    client = TestClient(app)
    monkeypatch.setattr(settings, "analysis_max_upload_bytes", 1)
    oversized = client.post("/api/eeg/analyze", files={"file": ("large.edf", b"12", "application/octet-stream")})
    assert oversized.status_code == 413
    monkeypatch.setattr(settings, "analysis_max_upload_bytes", 200 * 1024 * 1024)
    monkeypatch.setattr(settings, "model_checkpoint_path", tmp_path / "missing.pt")
    eeg_analysis._inference_service.cache_clear()
    unavailable = client.post("/api/eeg/analyze", files={"file": ("recording.edf", b"not-an-edf", "application/octet-stream")})
    assert unavailable.status_code == 503
    assert "unavailable" in unavailable.json()["detail"]
    eeg_analysis._inference_service.cache_clear()
