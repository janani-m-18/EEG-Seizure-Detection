"""Opt-in real Batch A CHB01 smoke check; it never downloads or trains."""

from pathlib import Path

import pytest

from ml.batch_a import batch_a_recordings, prepare_batch_a_recording


@pytest.fixture(scope="module")
def dataset_root() -> Path:
    value = __import__("os").environ.get("CHBMIT_DATASET_ROOT")
    if not value:
        pytest.skip("Batch A real-data check is opt-in via CHBMIT_DATASET_ROOT")
    root = Path(value)
    if not (root / "chb01").is_dir():
        pytest.skip("Batch A real-data check needs the local chb01 directory")
    return root


def test_batch_a_real_chb01_03_has_fixed_layout_overlap_and_summary_labels(dataset_root: Path) -> None:
    recordings = batch_a_recordings(dataset_root)
    assert tuple(item.path.name for item in recordings) == tuple(f"chb01_{number:02d}.edf" for number in range(1, 14))
    intervals = {item.recording_id: [(interval.start_seconds, interval.end_seconds) for interval in item.seizure_intervals] for item in recordings}
    assert intervals["chb01_03"] == [(2996.0, 3036.0)]
    assert intervals["chb01_04"] == [(1467.0, 1494.0)]

    data = prepare_batch_a_recording(dataset_root, "chb01_03")
    assert data.sampling_frequency == 256.0
    assert data.segments.shape == (1_799, 23, 1_024)
    assert data.labels.sum() == 21
    assert (data.labels == 0).sum() == 1_778
