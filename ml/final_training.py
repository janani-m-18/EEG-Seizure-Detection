"""Auditable full-corpus CHB-MIT final-training entry point.

The runner deliberately validates the available corpus and its group-level split
before building windows or training.  It therefore cannot silently train a
subset, leak a recording across partitions, or fabricate a final checkpoint.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
from dotenv import load_dotenv

from ml.batch_a import batch_a_pipeline
from ml.batch_b import (BATCH_B_AUGMENTATION, BATCH_B_FINAL_CHECKPOINT_TYPE, BATCH_B_MODEL_CONFIG, BATCH_B_MODEL_ID,
                        BATCH_B_PREPROCESSING_ID, BatchBTrainingConfig, build_batch_b_model,
                        evaluate_batch_b, load_batch_b_checkpoint, train_batch_b)
from ml.chbmit import CHBMITConfig, CHBMITRecording, discover_chbmit_recordings
from ml.chbmit import CHBMITSegmentBuilder, RealEEGSegments
from ml.evaluation import GroupSplitConfig, split_real_eeg
from ml.phase16_training import PHASE16_CHANNEL_NAMES
from ml.training import set_reproducible_seed


# Match the documented project configuration path used by the API.
load_dotenv()


FINAL_OUTPUT_DIRECTORY = Path("models/final_chbmit")
FINAL_CHECKPOINT_NAME = "final_cnn_bilstm_best.pt"


@dataclass(frozen=True, slots=True)
class RecordingInspection:
    patient_id: str
    recording_id: str
    path: str
    readable: bool
    sampling_frequency: float | None
    channel_names: tuple[str, ...]
    channel_count: int | None
    seizure_interval_count: int
    exclusion_reason: str | None

    @property
    def compatible(self) -> bool:
        return self.exclusion_reason is None


class FinalTrainingDataError(RuntimeError):
    """The local corpus cannot support the requested leakage-safe experiment."""


def _inspect_recording(recording: CHBMITRecording) -> RecordingInspection:
    """Read EDF header only; signal samples are not loaded during verification."""
    try:
        raw = mne.io.read_raw_edf(recording.path, preload=False, verbose="ERROR")
        names = tuple(raw.ch_names)
        sfreq = float(raw.info["sfreq"])
        reason = None
        if sfreq != 256.0:
            reason = f"expected 256 Hz Batch-A/Batch-B input, found {sfreq:g} Hz"
        elif names != PHASE16_CHANNEL_NAMES:
            reason = "channel layout is incompatible with the fixed 23-channel Batch-A/Batch-B contract"
        return RecordingInspection(recording.patient_id, recording.recording_id, str(recording.path), True,
                                   sfreq, names, len(names), len(recording.seizure_intervals), reason)
    except Exception as error:
        return RecordingInspection(recording.patient_id, recording.recording_id, str(recording.path), False,
                                   None, (), None, len(recording.seizure_intervals), f"EDF unreadable: {error}")


def discover_and_verify_final_dataset(dataset_root: Path) -> dict[str, Any]:
    """Discover every local EDF and record header/annotation compatibility."""
    recordings = discover_chbmit_recordings(CHBMITConfig(dataset_root))
    inspections = tuple(_inspect_recording(item) for item in recordings)
    patients = sorted({item.patient_id for item in recordings})
    compatible = [item for item in inspections if item.compatible]
    return {
        "dataset_root": str(dataset_root),
        "patient_count": len(patients),
        "patient_ids": patients,
        "edf_count": len(recordings),
        "recordings_per_patient": {patient: sum(item.patient_id == patient for item in recordings) for patient in patients},
        "seizure_recording_count": sum(item.seizure_interval_count > 0 for item in inspections),
        "sampling_rates_hz": sorted({item.sampling_frequency for item in inspections if item.sampling_frequency is not None}),
        "channel_counts": sorted({item.channel_count for item in inspections if item.channel_count is not None}),
        "compatible_recording_count": len(compatible),
        "excluded_recordings": [asdict(item) for item in inspections if not item.compatible],
        "recordings": [asdict(item) for item in inspections],
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _bounded_seizure_recordings(dataset_root: Path, summary: dict[str, Any]) -> tuple[CHBMITRecording, ...]:
    """Pick one annotated, compatible recording per patient in sorted order.

    This selection is deterministic and entirely annotation/header based.  It is
    deliberately opt-in because it is a resource-bounded experiment, not a
    claim that the full local corpus has been used.
    """
    compatible_ids = {item["recording_id"] for item in summary["recordings"] if item["exclusion_reason"] is None}
    candidates = [item for item in discover_chbmit_recordings(CHBMITConfig(dataset_root))
                  if item.recording_id in compatible_ids and item.has_seizure]
    selected: list[CHBMITRecording] = []
    for patient in sorted({item.patient_id for item in candidates}):
        selected.append(next(item for item in candidates if item.patient_id == patient))
    if len(selected) < 3:
        raise FinalTrainingDataError("The bounded experiment requires seizure recordings from at least three compatible patients")
    return tuple(selected)


def _build_bounded_data(recordings: tuple[CHBMITRecording, ...], *, max_seizure_windows_per_recording: int,
                        non_seizure_windows_per_recording: int) -> tuple[RealEEGSegments, list[dict[str, Any]]]:
    """Run the established pipeline one recording at a time, then retain a declared cap."""
    builder = CHBMITSegmentBuilder(batch_a_pipeline())
    segments: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    patients: list[str] = []
    recording_ids: list[str] = []
    starts: list[float] = []
    selection: list[dict[str, Any]] = []
    for recording in recordings:
        prepared = builder.build((recording,))
        positive = np.flatnonzero(prepared.labels == 1)[:max_seizure_windows_per_recording]
        negative = np.flatnonzero(prepared.labels == 0)[:non_seizure_windows_per_recording]
        if not len(positive) or not len(negative):
            raise FinalTrainingDataError(f"{recording.recording_id} does not contain both annotation-derived classes after Batch-A preprocessing")
        indices = np.sort(np.concatenate((negative, positive)))
        segments.append(prepared.segments[indices].copy())
        labels.append(prepared.labels[indices].copy())
        patients.extend(np.asarray(prepared.patient_ids, dtype=object)[indices].tolist())
        recording_ids.extend(np.asarray(prepared.recording_ids, dtype=object)[indices].tolist())
        if prepared.segment_start_seconds is None:
            raise FinalTrainingDataError("Batch-A segment provenance is required for final training")
        starts.extend(np.asarray(prepared.segment_start_seconds)[indices].astype(float).tolist())
        selection.append({
            "patient_id": recording.patient_id, "recording_id": recording.recording_id, "path": str(recording.path),
            "full_window_count": int(len(prepared.labels)), "available_seizure_windows": int((prepared.labels == 1).sum()),
            "available_non_seizure_windows": int((prepared.labels == 0).sum()), "retained_seizure_windows": int(len(positive)),
            "retained_non_seizure_windows": int(len(negative)),
        })
    data = RealEEGSegments(np.concatenate(segments), np.concatenate(labels), tuple(patients), tuple(recording_ids),
                           PHASE16_CHANNEL_NAMES, 256.0, tuple(starts))
    return data, selection


def _partition_counts(data: RealEEGSegments, indices: np.ndarray) -> dict[str, int]:
    labels = data.labels[indices]
    return {"segments": int(len(indices)), "seizure_windows": int((labels == 1).sum()),
            "non_seizure_windows": int((labels == 0).sum()),
            "patients": len(set(np.asarray(data.patient_ids, dtype=object)[indices])),
            "recordings": len(set(np.asarray(data.recording_ids, dtype=object)[indices]))}


def run_bounded_final_experiment(
    dataset_root: Path | None = None, *, output_dir: Path = FINAL_OUTPUT_DIRECTORY,
    max_seizure_windows_per_recording: int = 16, non_seizure_windows_per_recording: int = 4,
) -> dict[str, Any]:
    """Execute the explicitly bounded, deterministic CPU final-training fallback.

    The corpus is first fully verified.  One seizure recording per compatible
    seizure patient is then processed with Batch A, retaining only the declared
    number of real windows after preprocessing.  Validation and held-out groups
    are patient-level, and no retained recording is shared across partitions.
    """
    if max_seizure_windows_per_recording < 1 or non_seizure_windows_per_recording < 1:
        raise ValueError("bounded window caps must be positive")
    root = dataset_root or CHBMITConfig.from_environment().dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = discover_and_verify_final_dataset(root)
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "exclusion_report.json", {"excluded_recordings": summary["excluded_recordings"]})
    recordings = _bounded_seizure_recordings(root, summary)
    data, selected = _build_bounded_data(recordings, max_seizure_windows_per_recording=max_seizure_windows_per_recording,
                                          non_seizure_windows_per_recording=non_seizure_windows_per_recording)
    split_config = GroupSplitConfig(group_by="patient", validation_fraction=0.15, test_fraction=0.30, seed=1)
    split = split_real_eeg(data, split_config)
    partition_summary = {"train": _partition_counts(data, split.train_indices),
                         "validation": _partition_counts(data, split.validation_indices),
                         "held_out_test": _partition_counts(data, split.test_indices)}
    if any(partition["seizure_windows"] == 0 or partition["non_seizure_windows"] == 0 for partition in partition_summary.values()):
        raise FinalTrainingDataError("Deterministic patient split lacks both classes in at least one partition")
    partitions = [set(split.train_groups), set(split.validation_groups), set(split.test_groups)]
    if partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2]:
        raise FinalTrainingDataError("Patient leakage detected before training")
    split_metadata = {
        "split_type": "patient-aware deterministic split", "seed": split_config.seed,
        "target_fraction": {"development": 0.70, "validation_within_development": 0.15, "held_out_test": 0.30},
        "train_patient_ids": list(split.train_groups), "validation_patient_ids": list(split.validation_groups),
        "held_out_test_patient_ids": list(split.test_groups), "partition_counts": partition_summary,
        "leakage_checks": {"patient_overlap": False, "recording_overlap": False,
                           "window_overlap": False, "held_out_has_seizure_support": True},
    }
    _write_json(output_dir / "split_metadata.json", split_metadata)
    _write_json(output_dir / "bounded_selection.json", {
        "selection_policy": "one first sorted compatible seizure recording per patient; retain first annotation-derived windows after Batch-A preprocessing",
        "max_seizure_windows_per_recording": max_seizure_windows_per_recording,
        "non_seizure_windows_per_recording": non_seizure_windows_per_recording,
        "selected_recordings": selected,
        "selected_recording_count": len(selected), "selected_patient_count": len(set(item["patient_id"] for item in selected)),
        "retained_segment_count": int(len(data.labels)),
    })
    training = BatchBTrainingConfig(epochs=50, batch_size=32, learning_rate=1e-3, early_stopping_patience=5, seed=1)
    _write_json(output_dir / "final_configuration.json", {
        "mode": "bounded deterministic fallback", "training": asdict(training), "model": asdict(BATCH_B_MODEL_CONFIG),
        "preprocessing_id": BATCH_B_PREPROCESSING_ID, "augmentation": asdict(BATCH_B_AUGMENTATION),
        "used_all_locally_available_data": False,
        "bounded_reason": "full compatible window tensors cannot be materialized by the existing Batch-B in-memory trainer on the available CPU-only system",
    })
    # Model parameters must be seeded before construction, not merely before
    # DataLoader shuffling inside ``train_batch_b``.
    set_reproducible_seed(training.seed)
    model = build_batch_b_model()
    training_report = train_batch_b(model, data, training, split_config, output_dir=output_dir,
                                    best_checkpoint_name=FINAL_CHECKPOINT_NAME, final_checkpoint_name="final_cnn_bilstm_last.pt",
                                    checkpoint_type=BATCH_B_FINAL_CHECKPOINT_TYPE)
    _write_json(output_dir / "training_history.json", training_report["history"])
    best_model = build_batch_b_model()
    checkpoint = load_batch_b_checkpoint(output_dir / FINAL_CHECKPOINT_NAME, best_model)
    evaluation = evaluate_batch_b(best_model, data, split, training, output_dir=output_dir)
    _write_json(output_dir / "evaluation_metrics.json", evaluation)
    _write_json(output_dir / "model_metadata.json", {
        "checkpoint_type": checkpoint["checkpoint_type"], "model_id": BATCH_B_MODEL_ID,
        "preprocessing_id": BATCH_B_PREPROCESSING_ID, "checkpoint_path": str(output_dir / FINAL_CHECKPOINT_NAME),
        "input_shape": ["batch", 23, 1024], "gradcam_compatible": True,
    })
    result = {"status": "completed_bounded_training", "used_all_locally_available_data": False,
              "selected_recordings": len(selected), "selected_patients": len(set(item["patient_id"] for item in selected)),
              "retained_windows": int(len(data.labels)), "training_epochs_completed": len(training_report["history"]),
              "best_epoch": training_report["best_epoch"], "checkpoint_path": str(output_dir / FINAL_CHECKPOINT_NAME),
              "held_out_test_has_seizure_support": True, "evaluation": evaluation}
    _write_json(output_dir / "final_training_status.json", result)
    return result


def run_final_experiment(dataset_root: Path | None = None, *, output_dir: Path = FINAL_OUTPUT_DIRECTORY) -> dict[str, Any]:
    """Verify the corpus then fail closed until a valid patient-aware split exists.

    Window construction/training is intentionally not entered when fewer than
    three compatible patients are present: a train/validation/test partition
    would otherwise necessarily leak patients or recordings.
    """
    root = dataset_root or CHBMITConfig.from_environment().dataset_root
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = discover_and_verify_final_dataset(root)
    _write_json(output_dir / "dataset_summary.json", summary)
    _write_json(output_dir / "exclusion_report.json", {"excluded_recordings": summary["excluded_recordings"]})
    compatible_patients = {item["patient_id"] for item in summary["recordings"] if item["exclusion_reason"] is None}
    seizure_compatible_patients = {item["patient_id"] for item in summary["recordings"]
                                   if item["exclusion_reason"] is None and item["seizure_interval_count"] > 0}
    if len(compatible_patients) < 3:
        result = {
            "status": "blocked_before_training",
            "reason": "At least three compatible patients are required for leakage-safe train/validation/held-out test splitting; no checkpoint was created.",
            "dataset_summary_path": str(output_dir / "dataset_summary.json"),
            "compatible_patient_count": len(compatible_patients),
            "compatible_seizure_patient_count": len(seizure_compatible_patients),
            "used_all_locally_available_data": False,
            "checkpoint_path": None,
        }
        _write_json(output_dir / "final_training_status.json", result)
        return result
    result = {
        "status": "blocked_before_training",
        "reason": "The verified full corpus requires a streaming, recording-by-recording trainer; the existing Batch-B trainer materializes all windows in RAM. No bounded subset was selected and no checkpoint was created, so this run does not claim final training.",
        "dataset_summary_path": str(output_dir / "dataset_summary.json"),
        "compatible_patient_count": len(compatible_patients),
        "compatible_seizure_patient_count": len(seizure_compatible_patients),
        "used_all_locally_available_data": False,
        "checkpoint_path": None,
    }
    _write_json(output_dir / "final_training_status.json", result)
    return result


def main() -> None:
    root = Path(os.environ["CHBMIT_DATASET_ROOT"]) if os.environ.get("CHBMIT_DATASET_ROOT") else None
    print(json.dumps(run_final_experiment(root), indent=2))


if __name__ == "__main__":
    main()
