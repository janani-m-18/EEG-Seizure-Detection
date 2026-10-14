# Batch A — EEG preprocessing and data-preparation alignment

## Scope

Batch A aligns the reusable CHB-MIT preparation pipeline with the project
proposal. It does **not** retrain or alter the CNN, BiLSTM, checkpoint,
FastAPI API, frontend, explainability implementation, dataset scope, or
patient split. The current checkpoint remains a Phase 16 development model.

## Previous behavior

The generic processing defaults used a 0.5–70 Hz MNE FIR band-pass, a 50 Hz
notch when viable, channel-wise Z-score normalization, and 4-second windows
with a default non-overlapping stride. Peak-to-peak quality control could drop
channels on a per-recording basis. Phase 16 avoided that issue by disabling
peak-to-peak removal, and deliberately used 4-second windows with 30-second
starts to keep its CPU data-preparation run bounded.

## Batch A policy

The proposal-aligned default configuration is defined in
`ml/batch_a.py` and uses:

| Stage | Setting |
| --- | --- |
| Band-pass | 0.5–40.0 Hz, fourth-order IIR Butterworth |
| Power-line interference | Configurable `notch_frequency`; 50 Hz is the default and 60 Hz may be selected for an appropriate mains environment. Both are never selected by default. |
| Re-reference | Common Average Reference (CAR) across retained EEG channels |
| Normalization | Channel-wise Z-score, independently per recording |
| Quality screening | 500 µV peak-to-peak threshold; channels are flagged but retained in their source order |
| Window | 4 seconds = 1,024 samples at the CHB01 256 Hz rate |
| Stride | 2 seconds = 512 samples, giving 50% overlap |

The `PreprocessingConfig` and `SegmentationConfig` defaults reflect this policy,
while the named Batch A constants make the intended configuration explicit.

## Channel layout and artifact handling

CHB01’s fixed 23-channel, MNE-normalized bipolar layout is preserved. A
peak-to-peak exception is recorded in `rejected_channel_names`, but the default
policy does not remove that channel. This prevents a recording-specific channel
count or ordering mismatch at model input.

ICA-based ocular removal is **not enabled**. The local bipolar CHB-MIT EDFs do
not provide dedicated EOG channels or a validated, deterministic component
selection rule. Electrode interpolation is likewise not enabled because reliable
sensor locations/montage information are unavailable for this bipolar layout.
The 40 Hz high cutoff attenuates higher-frequency content but is not a complete
muscle-artifact detector or correction method. These are technical limitations,
not claims of artifact removal.

## Labels and leakage prevention

Windows retain the existing CHB-MIT summary-file annotation policy: a window is
positive when it has non-zero overlap with a recorded seizure interval. No EDF
annotations are fabricated or inferred. For example, the locally verified
intervals are 2996–3036 s for `chb01_03` and 1467–1494 s for `chb01_04`.

Z-score statistics are computed from each recording’s own channels, never from
a pooled train/validation/test array. Recording-aware splitting remains in
`ml.evaluation.split_real_eeg`; a whole recording belongs to exactly one
partition.

## Training-only augmentation

`ml.augmentation` provides optional, deterministic minority-window
oversampling after the recording-aware split. It is disabled by default. When
enabled, only the training partition receives zero-padded time shifts up to
±0.5 s, amplitude scaling in [0.8, 1.2], and additive noise at a randomly
selected 15–25 dB SNR. Validation and held-out test windows are neither copied
nor transformed.

Conventional SMOTE is intentionally **not** run on raw multichannel temporal
EEG tensors: vector interpolation can create implausible waveform morphology
and alter time relationships. The implemented approach uses transformed copies
of real minority-class training windows instead, and records the number of
derived windows. It must not be described as SMOTE.

## Checkpoint compatibility

The existing Phase 16 checkpoint was prepared with the legacy fixed-layout
policy: 0.5–70 Hz FIR band-pass, 50 Hz notch, no CAR, per-recording Z-score,
no peak-to-peak channel dropping, and 30-second window starts. Its explicitly
pinned configuration remains in `ml.phase16_training.PHASE16_PREPROCESSING`
and `PHASE16_SEGMENTATION`, including the API inference path. Applying the
Batch A preprocessing to that old checkpoint without retraining would be a
train/inference mismatch, so it is deliberately not done here.

## Verification

The real, local Batch A smoke test discovers exactly `chb01_01.edf` through
`chb01_13.edf` and processes `chb01_03.edf` without downloading data. The
result is 1,799 windows shaped `(23, 1024)` at 256 Hz; 21 overlap the real
2996–3036 s seizure interval and 1,778 do not. This verifies data preparation
only; it is not model training or a performance measurement.

## Limitations

This remains a one-patient CHB01 development setting. The data are severely
imbalanced, and neither artifact rejection nor augmentation constitutes
clinical validation. No performance improvement is asserted. A future retrain
and evaluation would be required before using the Batch A pipeline with a new
checkpoint.

**Batch B and Batch C have not been implemented.**
