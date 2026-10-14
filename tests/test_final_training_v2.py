"""V2 canonical-layout, split, lazy-loading, and checkpoint integrity checks."""

from pathlib import Path

import mne
import numpy as np
import pytest
import torch

from backend.app.processing.loader import EEGLoadError, select_required_channels
from backend.app.processing.types import LoadedEEG
from ml.augmentation import AugmentationConfig
from ml.batch_b import (BATCH_B_FINAL_V2_CHECKPOINT_TYPE, BatchBTrainingConfig, build_batch_b_model,
                        load_batch_b_checkpoint, save_batch_b_checkpoint)
from ml.final_training_v2 import (CachedRecording, DiskBackedEEGWindowDataset, PARTITION_BY_PATIENT,
                                  PHASE16_CHANNEL_NAMES, TRAIN_PATIENTS, V2Recording, _assert_fixed_split)
from ml.model import BatchBModelConfig


def _raw(channel_names: tuple[str, ...]) -> mne.io.RawArray:
    return mne.io.RawArray(np.zeros((len(channel_names), 1_024)),
                            mne.create_info(list(channel_names), 256.0, "eeg"), verbose="ERROR")


def test_canonical_channel_selection_orders_model_input_and_rejects_missing_channels(tmp_path: Path) -> None:
    reversed_channels = tuple(reversed(PHASE16_CHANNEL_NAMES)) + ("EXTRA",)
    selected = select_required_channels(LoadedEEG("test", _raw(reversed_channels), tmp_path / "input.edf"),
                                        PHASE16_CHANNEL_NAMES)
    assert tuple(selected.raw.ch_names) == PHASE16_CHANNEL_NAMES
    missing = PHASE16_CHANNEL_NAMES[1:]
    with pytest.raises(EEGLoadError, match="missing required channels"):
        select_required_channels(LoadedEEG("missing", _raw(missing), tmp_path / "missing.edf"), PHASE16_CHANNEL_NAMES)


def test_v2_fixed_patient_split_is_exact_and_has_no_overlap(tmp_path: Path) -> None:
    records = tuple(V2Recording(patient, f"{patient}_01", tmp_path / f"{patient}.edf", (), 256.0, 1_024,
                                PHASE16_CHANNEL_NAMES, (), (), None, None)
                    for patient in PARTITION_BY_PATIENT)
    _assert_fixed_split(records)
    assert not (set(TRAIN_PATIENTS) & set(PARTITION_BY_PATIENT) - set(TRAIN_PATIENTS))


def test_disk_backed_dataset_augments_training_only(tmp_path: Path) -> None:
    cache = tmp_path / "record.npy"
    np.save(cache, np.linspace(-1, 1, 23 * 1_024, dtype=np.float32).reshape(23, 1_024))
    records = (CachedRecording("chb04", "chb04_01", "source", str(cache), 1_024, 1, 1, 0, True, "train"),)
    index_path = tmp_path / "index.npy"
    index = np.empty(1, dtype=[("record_index", np.uint16), ("start_sample", np.int64), ("label", np.uint8)])
    index[0] = (0, 0, 1)
    np.save(index_path, index)
    policy = AugmentationConfig(enabled=True, seed=1)
    train = DiskBackedEEGWindowDataset(index_path, records, np.array([0]), training=True, minority_label=1,
                                        augmentation=policy, augmentation_probability=1.0, seed=1)
    validation = DiskBackedEEGWindowDataset(index_path, records, np.array([0]), training=False, minority_label=1,
                                             augmentation=policy, augmentation_probability=1.0, seed=1)
    augmented, _ = train[0]
    untouched, _ = validation[0]
    assert augmented.shape == untouched.shape == (23, 1_024)
    assert not torch.allclose(augmented, untouched)


def test_v2_checkpoint_keeps_experiment_metadata_and_loads(tmp_path: Path) -> None:
    config = BatchBModelConfig(input_channels=2, cnn_channels=(2, 4, 8), first_bilstm_hidden_size=4,
                               second_bilstm_hidden_size=3, dense_features=3, dropout=0)
    model = build_batch_b_model(config)
    path = tmp_path / "final_cnn_bilstm_best.pt"
    save_batch_b_checkpoint(path, model, torch.optim.Adam(model.parameters()), BatchBTrainingConfig(epochs=1), 1,
                            checkpoint_type=BATCH_B_FINAL_V2_CHECKPOINT_TYPE,
                            experiment_metadata={"canonical_channel_order": list(PHASE16_CHANNEL_NAMES), "seed": 1})
    metadata = load_batch_b_checkpoint(path, build_batch_b_model(config))
    assert metadata["checkpoint_type"] == BATCH_B_FINAL_V2_CHECKPOINT_TYPE
    assert metadata["experiment_metadata"]["canonical_channel_order"] == list(PHASE16_CHANNEL_NAMES)
