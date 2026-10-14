"""Batch A unit tests for proposal-aligned EEG preparation."""

from pathlib import Path

import mne
import numpy as np

from backend.app.processing.preprocessing import filter_eeg, handle_noise, normalize_eeg
from backend.app.processing.segmentation import segment_eeg
from backend.app.processing.types import FilteredEEG, LoadedEEG, NormalizedEEG, PreprocessingConfig, SegmentationConfig
from ml.batch_a import BATCH_A_AUGMENTATION, BATCH_A_PREPROCESSING, BATCH_A_SEGMENTATION


def _raw(data: np.ndarray, sampling_frequency: float = 256.0) -> mne.io.RawArray:
    return mne.io.RawArray(data, mne.create_info([f"C{index}" for index in range(data.shape[0])], sampling_frequency, "eeg"), verbose="ERROR")


def test_batch_a_configuration_matches_the_declared_methodology() -> None:
    assert BATCH_A_PREPROCESSING.l_freq == 0.5
    assert BATCH_A_PREPROCESSING.h_freq == 40.0
    assert BATCH_A_PREPROCESSING.filter_method == "iir_butterworth"
    assert BATCH_A_PREPROCESSING.filter_order == 4
    assert BATCH_A_PREPROCESSING.notch_frequency == 50.0
    assert BATCH_A_PREPROCESSING.common_average_reference is True
    assert BATCH_A_PREPROCESSING.normalization == "zscore"
    assert BATCH_A_SEGMENTATION.window_seconds == 4.0
    assert BATCH_A_SEGMENTATION.effective_stride_seconds == 2.0
    assert BATCH_A_AUGMENTATION.time_shift_seconds == 0.5


def test_notch_frequency_is_configurable_without_applying_both_mains_frequencies() -> None:
    sixty_hz = PreprocessingConfig(notch_frequency=60.0)
    dual_frequency_review = PreprocessingConfig(notch_frequency=(50.0, 60.0))
    assert sixty_hz.notch_frequency == 60.0
    assert dual_frequency_review.notch_frequency == (50.0, 60.0)


def test_common_average_reference_and_channelwise_zscore_keep_all_channels() -> None:
    raw = _raw(np.array([[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [5.0, 6.0, 7.0, 8.0]]))
    filtered = filter_eeg(
        LoadedEEG("car", raw, Path("car.edf")),
        PreprocessingConfig(l_freq=None, h_freq=None, notch_frequency=None, peak_to_peak_rejection_volts=None),
    )
    assert np.allclose(filtered.raw.get_data().mean(axis=0), 0.0)

    normalized = normalize_eeg(
        handle_noise(FilteredEEG("zscore", _raw(np.array([[2.0, 4.0, 6.0, 8.0], [10.0, 30.0, 50.0, 70.0]]))),
                     PreprocessingConfig(peak_to_peak_rejection_volts=None)),
        PreprocessingConfig(peak_to_peak_rejection_volts=None),
    )
    assert normalized.data.shape[0] == 2
    assert np.allclose(normalized.data.mean(axis=1), 0.0)
    assert np.allclose(normalized.data.std(axis=1), 1.0)


def test_batch_a_preserves_flagged_channels_and_uses_1024_sample_half_overlap_windows() -> None:
    raw = _raw(np.vstack([np.zeros(2_048), np.linspace(-400e-6, 400e-6, 2_048)]))
    handled = handle_noise(FilteredEEG("layout", raw), BATCH_A_PREPROCESSING)
    assert handled.rejected_channel_names == ("C1",)
    assert handled.raw.ch_names == ["C0", "C1"]

    normalized = NormalizedEEG("windows", np.zeros((23, 2_048)), tuple(f"C{index}" for index in range(23)), 256.0)
    segmented = segment_eeg(normalized, BATCH_A_SEGMENTATION)
    assert segmented.segments.shape == (3, 23, 1_024)
    assert [item.start_sample for item in segmented.metadata] == [0, 512, 1_024]


def test_per_recording_zscore_does_not_share_statistics_between_records() -> None:
    policy = PreprocessingConfig(peak_to_peak_rejection_volts=None)
    first = normalize_eeg(handle_noise(FilteredEEG("first", _raw(np.array([[0.0, 2.0, 4.0, 6.0]]))), policy), policy)
    second = normalize_eeg(handle_noise(FilteredEEG("second", _raw(np.array([[100.0, 102.0, 104.0, 106.0]]))), policy), policy)
    assert np.allclose(first.data.mean(axis=1), 0.0)
    assert np.allclose(second.data.mean(axis=1), 0.0)
    assert np.allclose(first.data, second.data)
