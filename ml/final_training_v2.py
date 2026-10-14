"""Disk-backed, patient-aware final CHB-MIT V2 training workflow.

V2 keeps the completed Batch-A preprocessing and Batch-B CNN--BiLSTM intact.
It verifies every local EDF, selects the canonical 23-channel montage before
preprocessing, stores only continuous per-recording normalized signals on disk,
and materializes 4-second windows lazily per batch.  It deliberately has no
per-recording window cap.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Sequence

import mne
import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, Sampler

from backend.app.processing import EEGProcessingRequest
from ml.augmentation import AugmentationConfig, augment_eeg_segment
from ml.batch_a import BATCH_A_AUGMENTATION, BATCH_A_PREPROCESSING, BATCH_A_SEGMENTATION, batch_a_pipeline
from ml.batch_b import (BATCH_B_FINAL_V2_CHECKPOINT_TYPE, BATCH_B_MODEL_CONFIG, BATCH_B_MODEL_ID,
                        BATCH_B_PREPROCESSING_ID, BatchBTrainingConfig, _binary_metrics, build_batch_b_model,
                        load_batch_b_checkpoint, save_batch_b_checkpoint)
from ml.chbmit import CHBMITConfig, CHBMITRecording, SeizureInterval, discover_chbmit_recordings
from ml.phase16_training import PHASE16_CHANNEL_NAMES
from ml.training import set_reproducible_seed


V2_OUTPUT_DIRECTORY = Path("models/final_chbmit_v2")
V2_CHECKPOINT_NAME = "final_cnn_bilstm_best.pt"
V2_CACHE_DIRECTORY_NAME = "preprocessed_recordings"
V2_INDEX_NAME = "window_index.npy"
WINDOW_SAMPLES = 1024
STRIDE_SAMPLES = 512
SAMPLING_FREQUENCY = 256.0

TRAIN_PATIENTS = (
    "chb04", "chb05", "chb07", "chb08", "chb09", "chb11", "chb14", "chb16", "chb17", "chb18",
    "chb20", "chb21", "chb22", "chb23",
)
VALIDATION_PATIENTS = ("chb12", "chb15", "chb24")
TEST_PATIENTS = ("chb01", "chb02", "chb03", "chb06", "chb10", "chb13", "chb19")
PARTITION_BY_PATIENT = {
    **{item: "train" for item in TRAIN_PATIENTS},
    **{item: "validation" for item in VALIDATION_PATIENTS},
    **{item: "test" for item in TEST_PATIENTS},
}
WINDOW_INDEX_DTYPE = np.dtype([("record_index", np.uint16), ("start_sample", np.int64), ("label", np.uint8)])


class FinalTrainingV2Error(RuntimeError):
    """A V2 preflight, cache, split, or training integrity check failed."""


@dataclass(frozen=True, slots=True)
class V2Recording:
    patient_id: str
    recording_id: str
    path: Path
    seizure_intervals: tuple[SeizureInterval, ...]
    sampling_frequency: float | None
    sample_count: int | None
    source_channel_names: tuple[str, ...]
    missing_canonical_channels: tuple[str, ...]
    additional_channel_names: tuple[str, ...]
    exclusion_category: str | None
    exclusion_reason: str | None

    @property
    def usable(self) -> bool:
        return self.exclusion_category is None

    @property
    def has_seizure(self) -> bool:
        return bool(self.seizure_intervals)


@dataclass(frozen=True, slots=True)
class CachedRecording:
    patient_id: str
    recording_id: str
    source_path: str
    cache_path: str
    sample_count: int
    window_count: int
    seizure_window_count: int
    non_seizure_window_count: int
    has_seizure_annotation: bool
    partition: str


@dataclass(frozen=True, slots=True)
class V2TrainingSettings:
    seed: int = 1
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    early_stopping_patience: int = 5
    threshold: float = 0.5
    device: str = "cpu"
    # Transform minority windows in-place during training.  This preserves all
    # real windows exactly once per epoch and avoids materializing an enormous
    # oversampled tensor on disk.
    train_minority_augmentation_probability: float = 1.0


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _window_labels(sample_count: int, intervals: Sequence[SeizureInterval]) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-window starts and real summary-derived binary labels."""
    if sample_count < WINDOW_SAMPLES:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.uint8)
    starts = np.arange(0, sample_count - WINDOW_SAMPLES + 1, STRIDE_SAMPLES, dtype=np.int64)
    labels = np.zeros(len(starts), dtype=np.uint8)
    for interval in intervals:
        start = starts / SAMPLING_FREQUENCY
        stop = (starts + WINDOW_SAMPLES) / SAMPLING_FREQUENCY
        labels[(np.minimum(stop, interval.end_seconds) - np.maximum(start, interval.start_seconds)) > 0.0] = 1
    return starts, labels


def inspect_v2_recordings(dataset_root: Path) -> tuple[V2Recording, ...]:
    """Inspect every EDF header and classify canonical-channel compatibility."""
    inspected: list[V2Recording] = []
    for recording in discover_chbmit_recordings(CHBMITConfig(dataset_root)):
        try:
            raw = mne.io.read_raw_edf(recording.path, preload=False, verbose="ERROR")
            names = tuple(raw.ch_names)
            missing = tuple(name for name in PHASE16_CHANNEL_NAMES if name not in names)
            additional = tuple(name for name in names if name not in PHASE16_CHANNEL_NAMES)
            sfreq = float(raw.info["sfreq"])
            category = reason = None
            if sfreq != SAMPLING_FREQUENCY:
                category = "sampling_rate_mismatch"
                reason = f"expected {SAMPLING_FREQUENCY:g} Hz, found {sfreq:g} Hz"
            elif missing:
                category = "missing_canonical_channels"
                reason = f"missing canonical channels: {', '.join(missing)}"
            inspected.append(V2Recording(recording.patient_id, recording.recording_id, recording.path,
                                         recording.seizure_intervals, sfreq, int(raw.n_times), names, missing,
                                         additional, category, reason))
        except Exception as error:
            inspected.append(V2Recording(recording.patient_id, recording.recording_id, recording.path,
                                         recording.seizure_intervals, None, None, (), (), (), "unreadable_edf",
                                         f"could not read EDF header: {error}"))
    return tuple(inspected)


def _partition_counts(records: Sequence[CachedRecording]) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for partition in ("train", "validation", "test"):
        selected = [item for item in records if item.partition == partition]
        result[partition] = {
            "patient_count": len({item.patient_id for item in selected}),
            "recording_count": len(selected),
            "seizure_recording_count": sum(item.has_seizure_annotation for item in selected),
            "non_seizure_recording_count": sum(not item.has_seizure_annotation for item in selected),
            "window_count": sum(item.window_count for item in selected),
            "seizure_window_count": sum(item.seizure_window_count for item in selected),
            "non_seizure_window_count": sum(item.non_seizure_window_count for item in selected),
        }
    return result


def _write_compatibility_report(records: Sequence[V2Recording], path: Path) -> None:
    categories: dict[str, list[V2Recording]] = {}
    for item in records:
        if item.usable:
            continue
        categories.setdefault(item.exclusion_category or "other", []).append(item)
    additional = [item for item in records if item.usable and item.additional_channel_names]
    lines = [
        "# Final CHB-MIT V2 dataset compatibility",
        "",
        "The canonical model input is the established 23-channel order. Extra channels are safely excluded before Batch-A preprocessing; missing canonical channels are never synthesized.",
        "",
        f"- EDFs inspected: {len(records)}",
        f"- Canonical-23 usable EDFs: {sum(item.usable for item in records)}",
        f"- Excluded EDFs: {sum(not item.usable for item in records)}",
        f"- Safe extra-channel mappings: {len(additional)}",
        "",
        "## Safe additional-channel conversion",
        "",
        f"{len(additional)} recordings contain all canonical channels plus additional channels. The pipeline deterministically selects the canonical channels in model order before filtering/CAR; no data channels are fabricated.",
        "",
        "## Exclusions",
        "",
    ]
    for category, items in sorted(categories.items()):
        lines.extend((f"### {category}", "", f"Count: {len(items)}", "", "Safe conversion: no. " +
                      ("One or more canonical model channels are absent." if category == "missing_canonical_channels" else "The EDF cannot satisfy the required input contract."),
                      "", "Representative recordings:"))
        for item in items[:10]:
            lines.append(f"- `{item.patient_id}/{item.recording_id}` — {item.exclusion_reason}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _assert_fixed_split(records: Sequence[V2Recording]) -> None:
    patient_ids = {item.patient_id for item in records if item.usable}
    declared = set(PARTITION_BY_PATIENT)
    if patient_ids != declared:
        raise FinalTrainingV2Error(f"Fixed split patient coverage mismatch; usable={sorted(patient_ids)}, declared={sorted(declared)}")
    groups = [set(TRAIN_PATIENTS), set(VALIDATION_PATIENTS), set(TEST_PATIENTS)]
    if any(first & second for index, first in enumerate(groups) for second in groups[index + 1:]):
        raise FinalTrainingV2Error("Fixed patient partitions overlap")


def preflight_v2(dataset_root: Path, *, output_dir: Path = V2_OUTPUT_DIRECTORY) -> dict[str, Any]:
    """Perform the full header/split/disk-capacity audit without preprocessing or training."""
    records = inspect_v2_recordings(dataset_root)
    _assert_fixed_split(records)
    usable = [item for item in records if item.usable]
    excluded = [item for item in records if not item.usable]
    expected_cache_bytes = sum((item.sample_count or 0) * len(PHASE16_CHANNEL_NAMES) * np.dtype(np.float32).itemsize for item in usable)
    free_bytes = shutil.disk_usage(output_dir.resolve().anchor or output_dir).free
    if expected_cache_bytes + 10 * 2**30 > free_bytes:
        raise FinalTrainingV2Error(
            f"Disk-backed cache needs {expected_cache_bytes / 2**30:.2f} GiB plus safety margin; only {free_bytes / 2**30:.2f} GiB is free"
        )
    layouts = Counter((len(item.source_channel_names), len(item.missing_canonical_channels)) for item in records)
    summary = {
        "dataset_root": str(dataset_root), "patient_count": len({item.patient_id for item in records}),
        "edf_count": len(records), "seizure_recording_count": sum(item.has_seizure for item in records),
        "non_seizure_recording_count": sum(not item.has_seizure for item in records),
        "usable_canonical_23_recording_count": len(usable), "excluded_recording_count": len(excluded),
        "sampling_rates_hz": sorted({item.sampling_frequency for item in records if item.sampling_frequency is not None}),
        "channel_layout_counts": {f"source_channels={count};missing_canonical={missing}": total
                                  for (count, missing), total in sorted(layouts.items())},
        "expected_continuous_float32_cache_gib": round(expected_cache_bytes / 2**30, 2),
        "available_disk_gib_before_cache": round(free_bytes / 2**30, 2),
        "canonical_channel_order": list(PHASE16_CHANNEL_NAMES),
        "records": [asdict(item) | {"path": str(item.path)} for item in records],
    }
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "exclusion_report.json", {"excluded_recordings": [asdict(item) | {"path": str(item.path)} for item in excluded]})
    _write_compatibility_report(records, Path("docs/final_training_v2_dataset_compatibility.md"))
    return summary


def _cached_recording_path(cache_directory: Path, record: V2Recording) -> Path:
    return cache_directory / f"{record.patient_id}__{record.recording_id}.npy"


def build_v2_cache(dataset_root: Path, *, output_dir: Path = V2_OUTPUT_DIRECTORY) -> dict[str, Any]:
    """Process every usable recording once and build a disk-backed window index.

    Cache files hold normalized continuous `(23, samples)` float32 signals,
    not overlapping windows.  Windows are later sliced lazily, avoiding the
    roughly doubled storage that a materialized overlap tensor would require.
    """
    summary_path = output_dir / "dataset_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else preflight_v2(dataset_root, output_dir=output_dir)
    by_path = {Path(item["path"]): item for item in summary["records"]}
    recordings = inspect_v2_recordings(dataset_root)
    usable = [item for item in recordings if item.usable]
    if len(usable) != 652:
        raise FinalTrainingV2Error(f"Expected 652 usable canonical recordings, found {len(usable)}")
    cache_directory = output_dir / V2_CACHE_DIRECTORY_NAME
    cache_directory.mkdir(parents=True, exist_ok=True)
    pipeline = batch_a_pipeline(required_channel_names=PHASE16_CHANNEL_NAMES)
    cached: list[CachedRecording] = []
    total_windows = sum(len(_window_labels(item.sample_count or 0, item.seizure_intervals)[0]) for item in usable)
    index_path = output_dir / V2_INDEX_NAME
    index_temp = output_dir / f"{V2_INDEX_NAME}.tmp"
    index = np.lib.format.open_memmap(index_temp, mode="w+", dtype=WINDOW_INDEX_DTYPE, shape=(total_windows,))
    cursor = 0
    started = perf_counter()
    for record_index, item in enumerate(usable):
        if item.sample_count is None or item.sampling_frequency != SAMPLING_FREQUENCY:
            raise FinalTrainingV2Error(f"Invalid V2 cache metadata for {item.recording_id}")
        cache_path = _cached_recording_path(cache_directory, item)
        cache_valid = False
        if cache_path.is_file():
            existing = np.load(cache_path, mmap_mode="r")
            cache_valid = existing.shape == (len(PHASE16_CHANNEL_NAMES), item.sample_count) and existing.dtype == np.float32
        if not cache_valid:
            prepared = pipeline.load(EEGProcessingRequest(item.recording_id, item.path))
            filtered = pipeline.filter(prepared)
            noise_handled = pipeline.handle_noise(filtered)
            normalized = pipeline.normalize(noise_handled)
            if normalized.channel_names != PHASE16_CHANNEL_NAMES or normalized.sampling_frequency != SAMPLING_FREQUENCY:
                raise FinalTrainingV2Error(f"Canonical Batch-A compatibility failed for {item.recording_id}")
            temporary = cache_path.with_name(cache_path.name + ".tmp.npy")
            np.save(temporary, normalized.data.astype(np.float32, copy=False))
            temporary.replace(cache_path)
        starts, labels = _window_labels(item.sample_count, item.seizure_intervals)
        stop = cursor + len(starts)
        index["record_index"][cursor:stop] = record_index
        index["start_sample"][cursor:stop] = starts
        index["label"][cursor:stop] = labels
        partition = PARTITION_BY_PATIENT[item.patient_id]
        cached.append(CachedRecording(item.patient_id, item.recording_id, str(item.path), str(cache_path), item.sample_count,
                                     len(starts), int(labels.sum()), int(len(labels) - labels.sum()), item.has_seizure, partition))
        cursor = stop
        if (record_index + 1) % 10 == 0 or record_index + 1 == len(usable):
            progress = {"status": "running", "completed_recordings": record_index + 1,
                        "total_recordings": len(usable), "elapsed_seconds": perf_counter() - started,
                        "last_recording_id": item.recording_id}
            _write_json(output_dir / "cache_progress.json", progress)
            print(json.dumps(progress), flush=True)
    if cursor != total_windows:
        raise FinalTrainingV2Error("Window-index count does not match cache plan")
    del index
    index_temp.replace(index_path)
    counts = _partition_counts(cached)
    if counts["test"]["seizure_window_count"] == 0 or counts["test"]["non_seizure_window_count"] == 0:
        raise FinalTrainingV2Error("Held-out test lacks one of the required window classes")
    if sum(item["recording_count"] for item in counts.values()) != len(usable):
        raise FinalTrainingV2Error("Not all usable records are accounted for by the fixed partitions")
    manifest = {"cache_directory": str(cache_directory), "window_index_path": str(index_path),
                "recordings": [asdict(item) for item in cached], "partition_counts": counts,
                "cache_representation": "per-recording continuous float32 normalized signals; windows sliced lazily", 
                "all_usable_recordings_accounted_for": True}
    _write_json(output_dir / "cache_manifest.json", manifest)
    _write_json(output_dir / "cache_progress.json", {"status": "complete", "completed_recordings": len(usable),
                                                        "total_recordings": len(usable), "elapsed_seconds": perf_counter() - started})
    _write_json(output_dir / "split_metadata.json", {
        "split_type": "fixed patient-aware", "train_patient_ids": list(TRAIN_PATIENTS),
        "validation_patient_ids": list(VALIDATION_PATIENTS), "held_out_test_patient_ids": list(TEST_PATIENTS),
        "partition_counts": counts, "leakage_checks": {"patient_overlap": False, "recording_overlap": False,
                                                           "window_overlap": False, "all_652_usable_accounted_for": True,
                                                           "held_out_has_seizure_and_non_seizure_windows": True},
    })
    return manifest


class DiskBackedEEGWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Slice one fixed window from a bounded LRU of disk-backed recordings."""

    def __init__(self, index_path: Path, records: Sequence[CachedRecording], rows: np.ndarray, *,
                 training: bool = False, minority_label: int | None = None,
                 augmentation: AugmentationConfig | None = None, augmentation_probability: float = 0.0,
                 seed: int = 1, cache_size: int = 8) -> None:
        self.index = np.load(index_path, mmap_mode="r")
        self.records = tuple(records)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.training = training
        self.minority_label = minority_label
        self.augmentation = augmentation
        self.augmentation_probability = augmentation_probability
        self.seed = seed
        self.epoch = 0
        self.cache_size = cache_size
        self._signals: OrderedDict[int, np.ndarray] = OrderedDict()

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def _signal(self, record_index: int) -> np.ndarray:
        if record_index in self._signals:
            data = self._signals.pop(record_index)
            self._signals[record_index] = data
            return data
        data = np.load(self.records[record_index].cache_path, mmap_mode="r")
        self._signals[record_index] = data
        if len(self._signals) > self.cache_size:
            self._signals.popitem(last=False)
        return data

    def __getitem__(self, position: int) -> tuple[Tensor, Tensor]:
        row = self.index[int(self.rows[int(position)])]
        record_index, start, label = int(row["record_index"]), int(row["start_sample"]), int(row["label"])
        values = np.asarray(self._signal(record_index)[:, start:start + WINDOW_SAMPLES], dtype=np.float32).copy()
        if values.shape != (len(PHASE16_CHANNEL_NAMES), WINDOW_SAMPLES):
            raise FinalTrainingV2Error("Cached window does not satisfy the fixed model input shape")
        if (self.training and self.augmentation is not None and self.augmentation.enabled and label == self.minority_label
                and self.augmentation_probability > 0):
            rng = np.random.default_rng(self.seed + 1_000_003 * self.epoch + int(self.rows[int(position)]))
            if float(rng.random()) < self.augmentation_probability:
                values = augment_eeg_segment(values, SAMPLING_FREQUENCY, self.augmentation, rng)
        return torch.from_numpy(values), torch.tensor(label, dtype=torch.long)


class RecordingBatchSampler(Sampler[list[int]]):
    """Yield complete-recording-local batches in deterministic shuffled order."""

    def __init__(self, dataset: DiskBackedEEGWindowDataset, *, batch_size: int, seed: int, shuffle: bool) -> None:
        self.dataset, self.batch_size, self.seed, self.shuffle, self.epoch = dataset, batch_size, seed, shuffle, 0
        record_ids = np.asarray(dataset.index[dataset.rows]["record_index"], dtype=np.uint16)
        self.batches = [positions[index:index + batch_size].tolist()
                        for record in np.unique(record_ids)
                        for positions in (np.flatnonzero(record_ids == record),)
                        for index in range(0, len(positions), batch_size)]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[list[int]]:
        order = np.arange(len(self.batches))
        if self.shuffle:
            np.random.default_rng(self.seed + self.epoch).shuffle(order)
        for item in order:
            yield self.batches[int(item)]

    def __len__(self) -> int:
        return len(self.batches)


def _partition_rows(index_path: Path, records: Sequence[CachedRecording]) -> dict[str, np.ndarray]:
    index = np.load(index_path, mmap_mode="r")
    record_partitions = np.asarray([item.partition for item in records], dtype=object)
    row_partitions = record_partitions[np.asarray(index["record_index"], dtype=np.int64)]
    return {name: np.flatnonzero(row_partitions == name).astype(np.int64) for name in ("train", "validation", "test")}


def _run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
               optimizer: Adam | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    model.train(optimizer is not None)
    losses: list[float] = []
    labels: list[np.ndarray] = []
    scores: list[np.ndarray] = []
    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for inputs, targets in loader:
            inputs, targets_float = inputs.to(device), targets.to(device, dtype=torch.float32)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model.forward_logits(inputs)
            loss = criterion(logits, targets_float)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            losses.append(float(loss.item()) * len(targets_float))
            labels.append(targets_float.detach().cpu().numpy().astype(np.int64))
            scores.append(torch.sigmoid(logits).detach().cpu().numpy())
    return sum(losses) / len(loader.dataset), np.concatenate(labels), np.concatenate(scores)


def _evaluation_report(model: nn.Module, dataset: DiskBackedEEGWindowDataset, sampler: RecordingBatchSampler,
                       criterion: nn.Module, settings: V2TrainingSettings, records: Sequence[CachedRecording],
                       *, output_dir: Path) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0)
    loss, labels, scores = _run_epoch(model, loader, criterion, torch.device(settings.device))
    metrics, matrix = _binary_metrics(labels, scores, settings.threshold)
    chosen = dataset.index[dataset.rows]
    record_indexes = np.asarray(chosen["record_index"], dtype=np.int64)
    start_seconds = np.asarray(chosen["start_sample"], dtype=np.float64) / SAMPLING_FREQUENCY
    recording_ids = np.asarray([records[item].recording_id for item in record_indexes])
    patient_ids = np.asarray([records[item].patient_id for item in record_indexes])
    predictions = (scores >= settings.threshold).astype(np.int64)
    unique_records = tuple(dict.fromkeys(recording_ids.tolist()))
    record_labels = np.asarray([int(np.any(labels[recording_ids == item] == 1)) for item in unique_records])
    record_scores = np.asarray([float(scores[recording_ids == item].max()) for item in unique_records])
    record_metrics, record_matrix = _binary_metrics(record_labels, record_scores, settings.threshold)
    np.save(output_dir / "confusion_matrix.npy", matrix)
    np.savez(output_dir / "predictions.npz", probabilities=scores, predictions=predictions, ground_truth=labels,
             recording_ids=recording_ids, patient_ids=patient_ids, segment_start_seconds=start_seconds)
    return {
        "threshold": settings.threshold, "loss": loss, "held_out_recording_ids": list(unique_records),
        "held_out_patient_ids": list(dict.fromkeys(patient_ids.tolist())),
        "segment_level": {"metrics": metrics, "confusion_matrix": matrix.tolist(), "segment_count": len(labels)},
        "recording_level": {"aggregation_rule": "maximum segment seizure probability per recording",
                              "metrics": record_metrics, "confusion_matrix": record_matrix.tolist(),
                              "recording_count": len(unique_records)},
        "predictions_path": str(output_dir / "predictions.npz"), "confusion_matrix_path": str(output_dir / "confusion_matrix.npy"),
    }


def run_final_v2_training(dataset_root: Path, *, output_dir: Path = V2_OUTPUT_DIRECTORY) -> dict[str, Any]:
    """Train and evaluate V2 only after a complete cache/partition preflight."""
    manifest_path = output_dir / "cache_manifest.json"
    if not manifest_path.is_file():
        raise FinalTrainingV2Error("Run build_v2_cache successfully before launching V2 training")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = tuple(CachedRecording(**item) for item in manifest["recordings"])
    index_path = Path(manifest["window_index_path"])
    if len(records) != 652 or not manifest.get("all_usable_recordings_accounted_for"):
        raise FinalTrainingV2Error("V2 cache does not account for all 652 usable recordings")
    counts = manifest["partition_counts"]
    if any(counts[name][key] == 0 for name in ("train", "validation", "test") for key in ("seizure_window_count", "non_seizure_window_count")):
        raise FinalTrainingV2Error("Each V2 partition must contain seizure and non-seizure windows")
    settings = V2TrainingSettings()
    rows = _partition_rows(index_path, records)
    index = np.load(index_path, mmap_mode="r")
    train_labels = np.asarray(index[rows["train"]]["label"], dtype=np.int64)
    class_counts = np.bincount(train_labels, minlength=2)
    if np.any(class_counts == 0):
        raise FinalTrainingV2Error("Training partition lacks a class")
    minority = int(np.argmin(class_counts))
    augmentation = AugmentationConfig(enabled=True, time_shift_seconds=BATCH_A_AUGMENTATION.time_shift_seconds,
                                      amplitude_scale_range=BATCH_A_AUGMENTATION.amplitude_scale_range,
                                      noise_snr_db_range=BATCH_A_AUGMENTATION.noise_snr_db_range, seed=settings.seed)
    train_data = DiskBackedEEGWindowDataset(index_path, records, rows["train"], training=True, minority_label=minority,
                                             augmentation=augmentation,
                                             augmentation_probability=settings.train_minority_augmentation_probability,
                                             seed=settings.seed)
    validation_data = DiskBackedEEGWindowDataset(index_path, records, rows["validation"])
    test_data = DiskBackedEEGWindowDataset(index_path, records, rows["test"])
    train_sampler = RecordingBatchSampler(train_data, batch_size=settings.batch_size, seed=settings.seed, shuffle=True)
    validation_sampler = RecordingBatchSampler(validation_data, batch_size=settings.batch_size, seed=settings.seed, shuffle=False)
    test_sampler = RecordingBatchSampler(test_data, batch_size=settings.batch_size, seed=settings.seed, shuffle=False)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_sampler=validation_sampler, num_workers=0)
    set_reproducible_seed(settings.seed)
    model = build_batch_b_model().to(settings.device)
    optimizer = Adam(model.parameters(), lr=settings.learning_rate)
    pos_weight = torch.tensor(float(class_counts[0] / class_counts[1]), dtype=torch.float32, device=settings.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    checkpoint_metadata = {
        "experiment": "final_chbmit_v2", "model_id": BATCH_B_MODEL_ID, "preprocessing_id": BATCH_B_PREPROCESSING_ID,
        "canonical_channel_order": list(PHASE16_CHANNEL_NAMES), "sampling_rate_hz": SAMPLING_FREQUENCY,
        "window_samples": WINDOW_SAMPLES, "stride_samples": STRIDE_SAMPLES,
        "patient_ids": {"train": list(TRAIN_PATIENTS), "validation": list(VALIDATION_PATIENTS), "test": list(TEST_PATIENTS)},
        "recording_ids": {name: [item.recording_id for item in records if item.partition == name]
                          for name in ("train", "validation", "test")},
        "training_configuration": asdict(settings), "augmentation_configuration": asdict(augmentation),
        "class_counts_train": {"non_seizure": int(class_counts[0]), "seizure": int(class_counts[1])},
        "class_weighting": {"BCEWithLogitsLoss_pos_weight": float(pos_weight.item())}, "random_seed": settings.seed,
    }
    _write_json(output_dir / "training_config.json", checkpoint_metadata)
    history: list[dict[str, Any]] = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    started = perf_counter()
    _write_json(output_dir / "training_progress.json", {"status": "running", "epochs_completed": 0,
                                                          "maximum_epochs": settings.epochs})
    for epoch in range(1, settings.epochs + 1):
        train_data.set_epoch(epoch); train_sampler.set_epoch(epoch)
        train_loss, train_labels_epoch, train_scores = _run_epoch(model, train_loader, criterion, torch.device(settings.device), optimizer)
        validation_loss, validation_labels, validation_scores = _run_epoch(model, validation_loader, criterion, torch.device(settings.device))
        train_metrics, _ = _binary_metrics(train_labels_epoch, train_scores, settings.threshold)
        validation_metrics, _ = _binary_metrics(validation_labels, validation_scores, settings.threshold)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss,
                        "train_metrics": train_metrics, "validation_metrics": validation_metrics})
        _write_json(output_dir / "training_progress.json", {"status": "running", "epochs_completed": epoch,
                                                              "maximum_epochs": settings.epochs, "best_epoch": best_epoch,
                                                              "best_validation_loss": best_loss,
                                                              "last_validation_loss": validation_loss,
                                                              "elapsed_seconds": perf_counter() - started})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            save_batch_b_checkpoint(output_dir / V2_CHECKPOINT_NAME, model, optimizer,
                                    BatchBTrainingConfig(seed=settings.seed, epochs=settings.epochs,
                                                         batch_size=settings.batch_size, learning_rate=settings.learning_rate,
                                                         early_stopping_patience=settings.early_stopping_patience,
                                                         threshold=settings.threshold, device=settings.device), epoch,
                                    checkpoint_type=BATCH_B_FINAL_V2_CHECKPOINT_TYPE, experiment_metadata=checkpoint_metadata)
        else:
            stale += 1
            if stale >= settings.early_stopping_patience:
                break
    elapsed_seconds = perf_counter() - started
    _write_json(output_dir / "training_metrics.json", {"history": history, "best_epoch": best_epoch,
                                                         "best_validation_loss": best_loss, "duration_seconds": elapsed_seconds,
                                                         "epochs_completed": len(history)})
    _write_json(output_dir / "training_progress.json", {"status": "evaluating_held_out_test",
                                                          "epochs_completed": len(history), "best_epoch": best_epoch,
                                                          "elapsed_seconds": elapsed_seconds})
    _write_json(output_dir / "validation_metrics.json", history[best_epoch - 1]["validation_metrics"])
    restored = build_batch_b_model()
    checkpoint = load_batch_b_checkpoint(output_dir / V2_CHECKPOINT_NAME, restored, device=settings.device)
    evaluation = _evaluation_report(restored, test_data, test_sampler, criterion, settings, records, output_dir=output_dir)
    _write_json(output_dir / "held_out_metrics.json", evaluation)
    _write_json(output_dir / "model_metadata.json", {"checkpoint_type": checkpoint["checkpoint_type"],
                                                        "model_config": asdict(BATCH_B_MODEL_CONFIG),
                                                        "checkpoint_path": str(output_dir / V2_CHECKPOINT_NAME),
                                                        "input_shape": ["batch", 23, 1024]})
    _write_json(output_dir / "preprocessing_metadata.json", {"preprocessing": asdict(BATCH_A_PREPROCESSING),
                                                                "segmentation": asdict(BATCH_A_SEGMENTATION),
                                                                "canonical_channel_order": list(PHASE16_CHANNEL_NAMES)})
    _write_json(output_dir / "training_progress.json", {"status": "complete", "epochs_completed": len(history),
                                                          "best_epoch": best_epoch, "elapsed_seconds": elapsed_seconds})
    return {"epochs_completed": len(history), "best_epoch": best_epoch, "duration_seconds": elapsed_seconds,
            "checkpoint_path": str(output_dir / V2_CHECKPOINT_NAME), "held_out_evaluation": evaluation}


def main() -> None:
    value = os.environ.get("CHBMIT_DATASET_ROOT")
    if not value:
        raise FinalTrainingV2Error("CHBMIT_DATASET_ROOT must be configured")
    print(json.dumps(preflight_v2(Path(value)), indent=2))


if __name__ == "__main__":
    main()
