"""EEG file loading for the supported clinical-research file formats."""

from pathlib import Path

import mne

from backend.app.processing.types import EEGProcessingRequest, LoadedEEG


SUPPORTED_EEG_SUFFIXES = frozenset({".edf", ".bdf", ".vhdr"})


class EEGLoadError(ValueError):
    """Raised when an EEG file cannot be validated or read."""


def load_eeg(request: EEGProcessingRequest) -> LoadedEEG:
    """Load EDF, BDF, or BrainVision EEG into a preloaded MNE ``Raw`` object."""
    path = request.path
    if not path.exists():
        raise EEGLoadError(f"EEG file does not exist: {path}")
    if not path.is_file():
        raise EEGLoadError(f"EEG input must be a file: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EEG_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_EEG_SUFFIXES))
        raise EEGLoadError(f"Unsupported EEG file type '{suffix}'; supported: {supported}")

    try:
        if suffix == ".edf":
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        elif suffix == ".bdf":
            raw = mne.io.read_raw_bdf(path, preload=True, verbose="ERROR")
        else:
            raw = mne.io.read_raw_brainvision(path, preload=True, verbose="ERROR")
    except Exception as error:
        raise EEGLoadError(f"Could not read EEG file '{path.name}': {error}") from error

    eeg_channel_count = len(mne.pick_types(raw.info, eeg=True, exclude=[]))
    if eeg_channel_count == 0:
        raise EEGLoadError(f"EEG file '{path.name}' contains no EEG channels")

    return LoadedEEG(
        input_reference=request.input_reference,
        raw=raw,
        source_path=path,
    )


def select_required_channels(loaded: LoadedEEG, required_channel_names: tuple[str, ...]) -> LoadedEEG:
    """Select a required channel layout in an explicit, stable order.

    Extra channels are intentionally ignored.  A missing required channel is a
    hard error: callers must never synthesize EEG channels merely to fit a
    model's input layout.
    """
    if not required_channel_names:
        raise ValueError("required_channel_names must not be empty")
    if len(set(required_channel_names)) != len(required_channel_names):
        raise ValueError("required_channel_names must be unique")
    missing = tuple(name for name in required_channel_names if name not in loaded.raw.ch_names)
    if missing:
        raise EEGLoadError(
            f"EEG file '{loaded.source_path.name}' is missing required channels: {', '.join(missing)}"
        )
    raw = loaded.raw.copy()
    raw.pick(list(required_channel_names))
    raw.reorder_channels(list(required_channel_names))
    return LoadedEEG(loaded.input_reference, raw, loaded.source_path)
