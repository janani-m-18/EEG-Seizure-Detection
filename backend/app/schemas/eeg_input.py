"""Metadata models for the future EEG input boundary.

These models describe an incoming EEG file without accepting or inspecting its
contents. Signal reading and processing belong to a later pipeline phase.
"""

from pathlib import PurePath

from pydantic import BaseModel, Field, field_validator


SUPPORTED_EEG_FILE_EXTENSIONS = frozenset({".bdf", ".edf", ".vhdr"})


class EEGInputMetadata(BaseModel):
    """Declared metadata for a future raw EEG file submission."""

    filename: str = Field(..., min_length=1, description="Name of the EEG file.")
    content_type: str | None = Field(
        default=None,
        description="Optional declared MIME type; it is not used to read the file.",
    )
    file_size_bytes: int | None = Field(
        default=None,
        ge=1,
        description="Optional declared file size in bytes.",
    )

    @field_validator("filename")
    @classmethod
    def validate_supported_filename(cls, value: str) -> str:
        """Require a filename with one of the supported EEG file extensions."""
        filename = value.strip()
        if not filename:
            raise ValueError("filename must not be blank")

        extension = PurePath(filename).suffix.lower()
        if extension not in SUPPORTED_EEG_FILE_EXTENSIONS:
            supported_extensions = ", ".join(sorted(SUPPORTED_EEG_FILE_EXTENSIONS))
            raise ValueError(
                f"unsupported EEG file type; supported extensions: {supported_extensions}"
            )
        return filename


class EEGInputValidationResult(BaseModel):
    """Result of validating metadata at the future EEG input boundary."""

    accepted: bool = True
    metadata: EEGInputMetadata
    processing_status: str = "not_started"
