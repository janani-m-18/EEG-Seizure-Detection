"""Opt-in integration checks for the locally unpacked CHB-MIT development data.

Set ``CHBMIT_DATASET_ROOT`` to the corpus directory that contains ``chb01``
(not to the ``chb01`` folder itself) to run these checks.  They neither
download data nor train a model.
"""

from pathlib import Path

import mne
import pytest

from backend.app.processing import (
    EEGProcessingPipeline,
    EEGProcessingRequest,
    PreprocessingConfig,
    SegmentationConfig,
)
from ml.chbmit import CHBMITConfig, discover_chbmit_recordings


EXPECTED_FILES = tuple(f"chb01_{number:02d}.edf" for number in range(1, 14))
EXPECTED_CHANNELS = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1",
)


@pytest.fixture(scope="module")
def development_root() -> Path:
    try:
        root = CHBMITConfig.from_environment().dataset_root
    except Exception as error:
        pytest.skip(f"CHB-MIT development-data check is opt-in: {error}")
    if not (root / "chb01").is_dir():
        pytest.skip(f"CHB-MIT development-data check needs {root / 'chb01'}")
    return root


def test_chb01_development_files_metadata_annotations_and_pipeline(development_root: Path) -> None:
    recordings = discover_chbmit_recordings(CHBMITConfig(development_root))
    assert tuple(item.path.name for item in recordings) == EXPECTED_FILES

    selected = next(item for item in recordings if item.recording_id == "chb01_03")
    assert [(item.start_seconds, item.end_seconds) for item in selected.seizure_intervals] == [(2996.0, 3036.0)]

    raw = mne.io.read_raw_edf(selected.path, preload=False, verbose="ERROR")
    assert raw.info["sfreq"] == 256.0
    assert raw.n_times == 921_600
    assert raw.n_times / raw.info["sfreq"] == 3_600.0
    assert tuple(raw.ch_names) == EXPECTED_CHANNELS
    assert len(raw.annotations) == 0  # Labels come from chb01-summary.txt, not EDF annotations.

    processed = EEGProcessingPipeline(PreprocessingConfig(), SegmentationConfig()).process(
        EEGProcessingRequest(selected.recording_id, selected.path)
    )
    assert processed.segmented.sampling_frequency == 256.0
    assert processed.segmented.segments.shape == (1_799, 23, 1_024)
    assert processed.segmented.channel_names == EXPECTED_CHANNELS
    assert processed.segmented.metadata[1].start_sample == 512
