"""EEG loading, preprocessing, and segmentation pipeline (Steps 6–8)."""

from backend.app.processing.interfaces import EEGProcessingPipeline
from backend.app.processing.types import (
    EEGProcessingRequest, FilteredEEG, LoadedEEG, NoiseHandledEEG, NormalizedEEG,
    PreprocessingConfig, ProcessedEEG, SegmentationConfig, SegmentedEEG, SegmentMetadata,
)

__all__ = [
    "EEGProcessingPipeline", "EEGProcessingRequest", "LoadedEEG", "FilteredEEG",
    "NoiseHandledEEG", "NormalizedEEG", "PreprocessingConfig", "SegmentMetadata",
    "SegmentedEEG", "SegmentationConfig", "ProcessedEEG",
]
