"""Fixed-window segmentation for normalized EEG arrays."""

import numpy as np

from backend.app.processing.types import NormalizedEEG, SegmentMetadata, SegmentationConfig, SegmentedEEG


def segment_eeg(
    normalized: NormalizedEEG,
    config: SegmentationConfig,
    label: str | int | None = None,
) -> SegmentedEEG:
    """Split ``(channels, samples)`` EEG data into fixed-length windows."""
    if normalized.data.ndim != 2 or normalized.data.shape[0] == 0 or normalized.data.shape[1] == 0:
        raise ValueError("normalized EEG data must be a non-empty 2D array")
    if len(normalized.channel_names) != normalized.data.shape[0]:
        raise ValueError("channel_names must match the number of EEG channels")

    window_samples = round(config.window_seconds * normalized.sampling_frequency)
    stride_samples = round(config.effective_stride_seconds * normalized.sampling_frequency)
    if window_samples < 1 or stride_samples < 1:
        raise ValueError("window and stride must resolve to at least one sample")

    total_samples = normalized.data.shape[1]
    starts = list(range(0, total_samples - window_samples + 1, stride_samples))
    if config.include_partial_window and (not starts or starts[-1] + window_samples < total_samples):
        starts.append(starts[-1] + stride_samples if starts else 0)
    if not starts:
        raise ValueError("recording is shorter than one full segmentation window")

    windows: list[np.ndarray] = []
    metadata: list[SegmentMetadata] = []
    for index, start in enumerate(starts):
        stop = min(start + window_samples, total_samples)
        window = normalized.data[:, start:stop]
        if window.shape[1] < window_samples:
            window = np.pad(window, ((0, 0), (0, window_samples - window.shape[1])))
        windows.append(window)
        metadata.append(SegmentMetadata(index, start, stop, start / normalized.sampling_frequency,
                                        stop / normalized.sampling_frequency, label))

    return SegmentedEEG(
        input_reference=normalized.input_reference,
        segments=np.stack(windows),
        channel_names=normalized.channel_names,
        sampling_frequency=normalized.sampling_frequency,
        metadata=tuple(metadata),
        rejected_channel_names=normalized.rejected_channel_names,
    )
