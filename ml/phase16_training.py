"""Reproducible Phase 16 training configuration for the local CHB01 subset.

This module trains only on real EDF-derived windows.  It intentionally leaves
the recording-level test partition untouched for the later evaluation phase.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from backend.app.processing import EEGProcessingPipeline, PreprocessingConfig, SegmentationConfig
from ml.chbmit import CHBMITConfig, CHBMITDatasetError, CHBMITSegmentBuilder, RealEEGSegments, discover_chbmit_recordings
from ml.evaluation import GroupSplitConfig, train_real_data
from ml.model import BiLSTMConfig, CNNBiLSTMClassifier, CNNConfig
from ml.training import TrainingConfig, load_checkpoint


EXPECTED_CHB01_FILES = tuple(f"chb01_{number:02d}.edf" for number in range(1, 14))
PHASE16_CHANNEL_NAMES = (
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2", "FP2-F8", "F8-T8", "T8-P8-0", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9", "FT9-FT10", "FT10-T8", "T8-P8-1",
)
# This explicitly preserves the preprocessing that generated the existing
# development checkpoint.  Batch A introduces a separate 0.5--40 Hz IIR/CAR
# preparation policy; applying it to this checkpoint without retraining would
# create a train/inference mismatch.
PHASE16_PREPROCESSING = PreprocessingConfig(
    l_freq=0.5,
    h_freq=70.0,
    filter_method="fir",
    notch_frequency=50.0,
    common_average_reference=False,
    peak_to_peak_rejection_volts=None,
    artifact_channel_policy="preserve",
    normalization="zscore",
)
PHASE16_SEGMENTATION = SegmentationConfig(window_seconds=4.0, stride_seconds=30.0)
PHASE16_CNN = CNNConfig(input_channels=23, feature_channels=(4, 8), kernel_size=7, dropout=0.1)
PHASE16_BILSTM = BiLSTMConfig(input_features=8, hidden_size=8, dropout=0.1)


def phase16_pipeline() -> EEGProcessingPipeline:
    """Return the fixed-layout processing pipeline used for the development checkpoint."""
    return EEGProcessingPipeline(PHASE16_PREPROCESSING, PHASE16_SEGMENTATION)


def build_phase16_model() -> CNNBiLSTMClassifier:
    """Construct the CNN-BiLSTM architecture stored in the Phase 16 checkpoint."""
    return CNNBiLSTMClassifier(PHASE16_CNN, PHASE16_BILSTM, num_classes=2)


def prepare_phase16_data(dataset_root: Path) -> RealEEGSegments:
    """Build fixed-layout four-second real windows from the 13 local EDF files.

    Per-recording peak-to-peak channel removal is disabled because it produced
    different channel layouts on this fixed CHB01 montage.  All 23 MNE-loaded
    EEG channels are retained in their source order.  Thirty-second window
    starts keep this CPU development run bounded while every window remains
    four seconds long and is labeled from the supplied summary intervals.
    """
    recordings = discover_chbmit_recordings(CHBMITConfig(dataset_root))
    names = tuple(item.path.name for item in recordings)
    if names != EXPECTED_CHB01_FILES:
        raise CHBMITDatasetError(f"Phase 16 requires exactly the local CHB01 development files: {EXPECTED_CHB01_FILES}")
    return CHBMITSegmentBuilder(phase16_pipeline()).build(recordings)


def run_phase16_training(dataset_root: Path, output_dir: Path) -> dict[str, object]:
    """Prepare the real development data, train, and verify the best checkpoint."""
    data = prepare_phase16_data(dataset_root)
    if data.segments.shape[1] != PHASE16_CNN.input_channels:
        raise CHBMITDatasetError("Prepared channel count does not match the Phase 16 checkpoint architecture")
    model = build_phase16_model()
    training = TrainingConfig(seed=42, epochs=3, batch_size=16, learning_rate=1e-3,
                              weight_decay=1e-4, device="cpu", early_stopping_patience=2)
    split = GroupSplitConfig(group_by="recording", validation_fraction=0.2, test_fraction=0.2, seed=42)
    report = train_real_data(model, data, training, split, output_dir=output_dir)
    checkpoint_path = Path(str(report["checkpoint_path"]))
    restored = build_phase16_model()
    checkpoint = load_checkpoint(checkpoint_path, restored, device="cpu")
    labels, counts = np.unique(data.labels, return_counts=True)
    report.update({
        "data": {
            "recording_count": len(EXPECTED_CHB01_FILES), "segment_count": len(data.segments),
            "label_counts": {str(label): int(count) for label, count in zip(labels, counts, strict=True)},
            "segment_shape": list(data.segments.shape), "sampling_frequency": data.sampling_frequency,
            "channel_names": list(data.channel_names),
            "channel_policy": "retain the fixed 23-channel CHB01 MNE layout; disable per-recording peak-to-peak channel removal",
            "segmentation_policy": "4-second windows starting every 30 seconds for a bounded CPU development run",
        },
        "model": {"cnn": asdict(PHASE16_CNN), "bilstm": asdict(PHASE16_BILSTM), "num_classes": 2},
        "training": asdict(training),
        "checkpoint_load": {"loaded": True, "epoch": checkpoint["epoch"]},
    })
    (output_dir / "phase16_training_report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report
