"""Metadata-only route for the future EEG input boundary."""

from fastapi import APIRouter

from backend.app.schemas.eeg_input import EEGInputMetadata, EEGInputValidationResult


router = APIRouter(prefix="/eeg-input", tags=["eeg-input"])


@router.post("/validate", response_model=EEGInputValidationResult)
def validate_eeg_input(metadata: EEGInputMetadata) -> EEGInputValidationResult:
    """Validate EEG file metadata without accepting or processing file contents."""
    return EEGInputValidationResult(metadata=metadata)
