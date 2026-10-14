"""Orchestration interface for EEG loading, preprocessing, and segmentation."""

from backend.app.processing.loader import load_eeg, select_required_channels
from backend.app.processing.preprocessing import filter_eeg, handle_noise, normalize_eeg
from backend.app.processing.segmentation import segment_eeg
from backend.app.processing.types import (
    EEGProcessingRequest, FilteredEEG, LoadedEEG, NoiseHandledEEG,
    NormalizedEEG, PreprocessingConfig, ProcessedEEG, SegmentationConfig, SegmentedEEG,
)


class EEGProcessingPipeline:
    """Offline Steps 6–8 EEG pipeline; it performs no ML or clinical inference."""

    def __init__(self, preprocessing: PreprocessingConfig | None = None,
                 segmentation: SegmentationConfig | None = None,
                 *, required_channel_names: tuple[str, ...] | None = None) -> None:
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.segmentation = segmentation or SegmentationConfig()
        self.required_channel_names = required_channel_names

    def load(self, request: EEGProcessingRequest) -> LoadedEEG:
        loaded = load_eeg(request)
        return (select_required_channels(loaded, self.required_channel_names)
                if self.required_channel_names is not None else loaded)

    def filter(self, loaded: LoadedEEG) -> FilteredEEG:
        return filter_eeg(loaded, self.preprocessing)

    def handle_noise(self, filtered: FilteredEEG) -> NoiseHandledEEG:
        return handle_noise(filtered, self.preprocessing)

    def normalize(self, noise_handled: NoiseHandledEEG) -> NormalizedEEG:
        return normalize_eeg(noise_handled, self.preprocessing)

    def segment(self, normalized: NormalizedEEG, label: str | int | None = None) -> SegmentedEEG:
        return segment_eeg(normalized, self.segmentation, label)

    def finalize(self, segmented: SegmentedEEG) -> ProcessedEEG:
        return ProcessedEEG(segmented.input_reference, segmented, self.preprocessing, self.segmentation)

    def process(self, request: EEGProcessingRequest) -> ProcessedEEG:
        loaded = self.load(request)
        filtered = self.filter(loaded)
        noise_handled = self.handle_noise(filtered)
        normalized = self.normalize(noise_handled)
        return self.finalize(self.segment(normalized, request.label))
