from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_eeg_input_metadata_is_accepted_without_processing() -> None:
    response = client.post(
        "/api/eeg-input/validate",
        json={
            "filename": "recording.EDF",
            "content_type": "application/octet-stream",
            "file_size_bytes": 1024,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "accepted": True,
        "metadata": {
            "filename": "recording.EDF",
            "content_type": "application/octet-stream",
            "file_size_bytes": 1024,
        },
        "processing_status": "not_started",
    }


def test_eeg_input_metadata_rejects_unsupported_file_type() -> None:
    response = client.post("/api/eeg-input/validate", json={"filename": "recording.csv"})

    assert response.status_code == 422
    assert "unsupported EEG file type" in response.text


def test_eeg_input_metadata_rejects_non_positive_declared_size() -> None:
    response = client.post(
        "/api/eeg-input/validate",
        json={"filename": "recording.edf", "file_size_bytes": 0},
    )

    assert response.status_code == 422
