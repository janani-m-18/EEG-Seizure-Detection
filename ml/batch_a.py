"""Batch A CHB-MIT preparation configuration.

This module prepares data for a future retraining batch.  It does not alter the
existing Phase 16 checkpoint or its legacy inference preprocessing contract.
"""

from __future__ import annotations

from pathlib import Path

from backend.app.processing import EEGProcessingPipeline, PreprocessingConfig, SegmentationConfig
from ml.augmentation import AugmentationConfig
from ml.chbmit import CHBMITConfig, CHBMITDatasetError, CHBMITRecording, CHBMITSegmentBuilder, RealEEGSegments, discover_chbmit_recordings


BATCH_A_PREPROCESSING = PreprocessingConfig(
    l_freq=0.5,
    h_freq=40.0,
    filter_order=4,
    filter_method="iir_butterworth",
    notch_frequency=50.0,
    common_average_reference=True,
    peak_to_peak_rejection_volts=500e-6,
    artifact_channel_policy="preserve",
    normalization="zscore",
)
BATCH_A_SEGMENTATION = SegmentationConfig(window_seconds=4.0, stride_seconds=2.0)
BATCH_A_AUGMENTATION = AugmentationConfig(
    enabled=False,
    time_shift_seconds=0.5,
    amplitude_scale_range=(0.8, 1.2),
    noise_snr_db_range=(15.0, 25.0),
    seed=42,
)


def batch_a_pipeline(*, required_channel_names: tuple[str, ...] | None = None) -> EEGProcessingPipeline:
    """Return the proposal-aligned 0.5--40 Hz/CAR/4 s/2 s pipeline."""
    return EEGProcessingPipeline(BATCH_A_PREPROCESSING, BATCH_A_SEGMENTATION,
                                 required_channel_names=required_channel_names)


def prepare_batch_a_recording(dataset_root: Path, recording_id: str) -> RealEEGSegments:
    """Prepare one discovered CHB-MIT recording with real summary-derived labels."""
    recordings = discover_chbmit_recordings(CHBMITConfig(dataset_root))
    selected = tuple(item for item in recordings if item.recording_id == recording_id)
    if len(selected) != 1:
        raise CHBMITDatasetError(f"Expected exactly one discovered recording named {recording_id}")
    return CHBMITSegmentBuilder(batch_a_pipeline()).build(selected)


def batch_a_recordings(dataset_root: Path) -> tuple[CHBMITRecording, ...]:
    """Discover the locally available recordings without downloading any data."""
    return discover_chbmit_recordings(CHBMITConfig(dataset_root))
