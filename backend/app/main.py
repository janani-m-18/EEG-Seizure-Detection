"""FastAPI application entry point for the EEG Analysis API."""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from backend.app.api.routes.eeg_analysis import router as eeg_analysis_router
from backend.app.api.routes.eeg_input import router as eeg_input_router
from backend.app.api.routes.health import router as health_router
from backend.app.core.config import settings


app = FastAPI(title=settings.app_name)
app.include_router(health_router, prefix="/api")
app.include_router(eeg_input_router, prefix="/api")
app.include_router(eeg_analysis_router, prefix="/api")
app.mount("/", StaticFiles(directory=Path(__file__).resolve().parents[2] / "frontend", html=True), name="frontend")
