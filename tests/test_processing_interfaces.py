"""Synthetic-signal tests for the real Steps 6–8 processing pipeline."""

import mne
import numpy as np
import pytest

from backend.app.processing import (
    EEGProcessingPipeline,
    EEGProcessingRequest,
    PreprocessingConfig,
    SegmentationConfig,
)
from backend.app.processing.loader import EEGLoadError, load_eeg
from backend.app.processing.preprocessing import handle_noise, normalize_eeg
from backend.app.processing.types import FilteredEEG, NormalizedEEG
from backend.app.processing.segmentation import segment_eeg


def _raw(data: np.ndarray, sampling_frequency: float = 100.0) -> mne.io.RawArray:
    info = mne.create_info(["F3", "F4"], sampling_frequency, ch_types="eeg")
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_loader_rejects_missing_and_unsupported_files(tmp_path) -> None:
    missing = EEGProcessingRequest("missing", tmp_path / "missing.edf")
    with pytest.raises(EEGLoadError, match="does not exist"):
        load_eeg(missing)

    unsupported = tmp_path / "recording.csv"
    unsupported.write_text("not eeg")
    with pytest.raises(EEGLoadError, match="Unsupported EEG file type"):
        load_eeg(EEGProcessingRequest("unsupported", unsupported))


def test_pipeline_processes_synthetic_edf_and_preserves_segment_shape(tmp_path) -> None:
    # MNE writes a small synthetic EDF fixture; no external dataset is required.
    raw = _raw(np.vstack([np.sin(np.linspace(0, 20, 1_000)), np.cos(np.linspace(0, 20, 1_000))]) * 20e-6)
    path = tmp_path / "synthetic.edf"
    mne.export.export_raw(path, raw, overwrite=True, verbose="ERROR")
    pipeline = EEGProcessingPipeline(
        PreprocessingConfig(h_freq=40.0, notch_frequency=None),
        SegmentationConfig(window_seconds=2.0, stride_seconds=1.0),
    )

    processed = pipeline.process(EEGProcessingRequest("synthetic-001", path, label="non_seizure"))

    assert processed.segmented.segments.shape == (9, 2, 200)
    assert len(processed.segmented.metadata) == 9
    assert processed.segmented.metadata[0].start_sample == 0
    assert processed.segmented.metadata[-1].stop_sample == 1_000
    assert processed.segmented.metadata[0].label == "non_seizure"
    assert np.allclose(processed.segmented.segments.mean(axis=(0, 2)), 0, atol=0.1)


def test_noise_handling_flags_channel_over_threshold() -> None:
    data = np.vstack([np.ones(500) * 2e-6, np.linspace(-400e-6, 400e-6, 500)])
    filtered = FilteredEEG("quality", _raw(data))
    handled = handle_noise(
        filtered,
        PreprocessingConfig(h_freq=40, notch_frequency=None, common_average_reference=False,
                            artifact_channel_policy="drop"),
    )

    assert handled.rejected_channel_names == ("F4",)
    assert handled.raw.ch_names == ["F3"]


def test_normalization_and_segmentation_boundaries() -> None:
    normalized = normalize_eeg(
        handle_noise(FilteredEEG("signal", _raw(np.vstack([np.arange(550), np.arange(550) * 2]) * 1e-6)),
                     PreprocessingConfig(h_freq=40, notch_frequency=None, peak_to_peak_rejection_volts=None)),
        PreprocessingConfig(h_freq=40, notch_frequency=None, peak_to_peak_rejection_volts=None),
    )
    segments = segment_eeg(normalized, SegmentationConfig(2.0, 2.0, include_partial_window=True), label=1)

    assert segments.segments.shape == (3, 2, 200)
    assert segments.metadata[2].start_sample == 400
    assert segments.metadata[2].stop_sample == 550
    assert np.all(segments.segments[2, :, 150:] == 0)
    assert all(item.label == 1 for item in segments.metadata)


def test_segmentation_rejects_invalid_data_and_too_short_recording() -> None:
    invalid = NormalizedEEG("bad", np.empty((0, 10)), (), 100.0)
    with pytest.raises(ValueError, match="non-empty 2D"):
        segment_eeg(invalid, SegmentationConfig())

    short = NormalizedEEG("short", np.ones((2, 50)), ("F3", "F4"), 100.0)
    with pytest.raises(ValueError, match="shorter"):
        segment_eeg(short, SegmentationConfig(window_seconds=1.0))


def test_configuration_validation_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="l_freq"):
        PreprocessingConfig(l_freq=0)
    with pytest.raises(ValueError, match="lower"):
        PreprocessingConfig(l_freq=40, h_freq=40)
    with pytest.raises(ValueError, match="window_seconds"):
        SegmentationConfig(window_seconds=0)
