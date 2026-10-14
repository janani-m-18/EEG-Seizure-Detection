# Dataset, Methodology, and Model Record

## Dataset and labels

Only local CHB01 development recordings were used: 13 complete EDFs with 23
channels sampled at 256 Hz. This is a one-patient development subset, not a
percentage of the full CHB-MIT corpus. Seizure intervals are parsed from the
provided `chb01-summary.txt`; no labels are fabricated.

The bounded development preparation produced 1,560 four-second windows: 3
seizure and 1,557 non-seizure. This is an extremely imbalanced distribution.

## Preprocessing and segmentation

The pipeline uses MNE EDF loading, EEG-only channel handling, 0.5–70 Hz
band-pass filtering, a 50 Hz notch where viable, channel-wise z-score
normalization, and four-second windows with 30-second starts. A fixed
23-channel source layout is used across recordings. Per-recording
peak-to-peak removal was disabled because it created inconsistent model input
layouts in Phase 15.

## CNN–BiLSTM

Inputs are `(batch, 23, 1024)`. Two temporal CNN blocks use widths 4 and 8,
kernel size 7, pooling factor 2, and dropout 0.1. Their feature sequence feeds
an 8-unit, single-layer bidirectional LSTM. A linear head produces two logits.

## Training and evaluation methodology

Training used CPU PyTorch, AdamW, class-weighted cross-entropy, seed 42, batch
size 16, learning rate 0.001, weight decay 0.0001, and up to three epochs with
best-validation checkpointing. Splits were recording-aware: complete
recordings were assigned to train, validation, or held-out groups so windows
from one recording cannot leak across groups.

The Phase 17 held-out partition had 360 non-seizure windows and no seizure
windows. Its confusion matrix was `[[360, 0], [0, 0]]`. Positive-class and
macro metrics are undefined because no positive class has support, so this is
not a meaningful seizure-detection test.

## Inference and explainability

Inference validates the checkpoint, sampling rate, channel order, and tensor
shape before calling the existing model. Integrated Gradients uses a zero
baseline and actual predicted-class logit gradients. Attribution maps indicate
which normalized model-input samples affected the development checkpoint
prediction; they are not EEG biomarkers or clinical explanations.
