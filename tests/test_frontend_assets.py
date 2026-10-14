"""Basic checks for the dependency-free Phase 21 frontend assets."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.app.main import app


def test_frontend_page_is_served_with_upload_and_results_surfaces() -> None:
    response = TestClient(app).get("/")
    assert response.status_code == 200
    assert 'id="edf-file"' in response.text
    assert 'id="analyze-button"' in response.text
    assert 'id="waveform-chart"' in response.text
    assert 'id="attribution-chart"' in response.text


def test_frontend_uses_the_actual_analysis_endpoint() -> None:
    script = (Path(__file__).parents[1] / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "/api/eeg/analyze" in script
    assert "new FormData" in script
    assert "EEG_API_BASE_URL" in script
    assert "gradcam_temporal_map" in script
    assert "Grad-CAM model attribution" in script


def test_frontend_labels_gradcam_as_model_attribution() -> None:
    page = (Path(__file__).parents[1] / "frontend" / "index.html").read_text(encoding="utf-8")
    assert "Grad-CAM Explanation" in page
    assert "model attribution" in page.lower()
