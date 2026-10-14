"""Configurable, non-clinical EEG preprocessing helpers."""

import mne
import numpy as np

from backend.app.processing.types import (
    FilteredEEG,
    LoadedEEG,
    NoiseHandledEEG,
    NormalizedEEG,
    PreprocessingConfig,
)


def filter_eeg(loaded: LoadedEEG, config: PreprocessingConfig) -> FilteredEEG:
    """Apply configured band-pass, line filtering, and common-average reference."""
    raw = loaded.raw.copy().pick("eeg")
    nyquist = raw.info["sfreq"] / 2
    if config.h_freq is not None and config.h_freq >= nyquist:
        raise ValueError("h_freq must be below the recording Nyquist frequency")
    if config.l_freq is None and config.h_freq is None:
        pass
    elif config.filter_method == "iir_butterworth":
        raw.filter(
            l_freq=config.l_freq,
            h_freq=config.h_freq,
            method="iir",
            iir_params={"order": config.filter_order, "ftype": "butter"},
            verbose="ERROR",
        )
    else:
        raw.filter(l_freq=config.l_freq, h_freq=config.h_freq, method="fir", verbose="ERROR")
    if config.notch_frequency is not None:
        requested = (config.notch_frequency,) if isinstance(config.notch_frequency, (int, float)) else config.notch_frequency
        viable = tuple(frequency for frequency in requested if frequency < nyquist)
        if viable:
            raw.notch_filter(freqs=viable, verbose="ERROR")
    if config.common_average_reference:
        raw.set_eeg_reference(ref_channels="average", projection=False, verbose="ERROR")
    return FilteredEEG(input_reference=loaded.input_reference, raw=raw)


def handle_noise(filtered: FilteredEEG, config: PreprocessingConfig) -> NoiseHandledEEG:
    """Flag high-amplitude channels without changing the default fixed layout.

    Peak-to-peak screening is quality control, not an artifact-correction or
    clinical-review method.  Batch A preserves flagged channels so the model
    input montage remains stable.  ``artifact_channel_policy='drop'`` remains
    available only for legacy callers that deliberately accept layout changes.
    """
    raw = filtered.raw.copy()
    if config.peak_to_peak_rejection_volts is None:
        return NoiseHandledEEG(filtered.input_reference, raw)

    data = raw.get_data()
    peak_to_peak = np.ptp(data, axis=1)
    bads = tuple(
        channel for channel, amplitude in zip(raw.ch_names, peak_to_peak, strict=True)
        if amplitude > config.peak_to_peak_rejection_volts
    )
    if len(bads) == len(raw.ch_names) and config.artifact_channel_policy == "drop":
        raise ValueError("All EEG channels exceeded the peak-to-peak rejection threshold")
    raw.info["bads"] = list(bads)
    if bads and config.artifact_channel_policy == "drop":
        raw.drop_channels(list(bads))
    return NoiseHandledEEG(filtered.input_reference, raw, bads)


def normalize_eeg(noise_handled: NoiseHandledEEG, config: PreprocessingConfig) -> NormalizedEEG:
    """Return channel-wise z-scored EEG samples, or unscaled data if requested."""
    data = noise_handled.raw.get_data().astype(np.float64, copy=True)
    if config.normalization == "zscore":
        means = data.mean(axis=1, keepdims=True)
        standard_deviations = data.std(axis=1, keepdims=True)
        standard_deviations[standard_deviations == 0] = 1.0
        data = (data - means) / standard_deviations
    return NormalizedEEG(
        input_reference=noise_handled.input_reference,
        data=data,
        channel_names=tuple(noise_handled.raw.ch_names),
        sampling_frequency=float(noise_handled.raw.info["sfreq"]),
        rejected_channel_names=noise_handled.rejected_channel_names,
    )
