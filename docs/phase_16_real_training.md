# Phase 16: Real CNN-BiLSTM Training

This is a single-patient CHB01 development experiment, not a patient-independent
evaluation.  It used only the 13 locally available complete EDF recordings and
did not resume or extend the dataset download.

## Data preparation and channel policy

Real EDF files were loaded with the existing MNE pipeline and labeled from
`chb01-summary.txt`.  Four-second windows were started every 30 seconds to
keep the CPU development run bounded.  This produced 1,560 windows of shape
`(23, 1024)` at 256 Hz: 3 seizure windows and 1,557 non-seizure windows.

The Phase 15 default 500 microvolt peak-to-peak rule was not used for training
because it dropped a different set of channels per recording (and left only
two for `chb01_03`).  The Phase 16 policy retains the fixed 23-channel MNE
layout, in source order, for every recording; all other existing preprocessing
steps remain in use.  This is a reproducibility decision for this development
subset, not a clinical recommendation.

## Training run

The existing CNN-BiLSTM was configured with a 23-channel input, CNN widths
`(4, 8)`, kernel size 7, and an 8-unit bidirectional LSTM.  Training used CPU,
AdamW, learning rate 0.001, weight decay 0.0001, batch size 16, seed 42, and
three epochs.  A class-weighted cross-entropy loss used training weights
approximately `[0.5012, 210.0]`, calculated from 838 non-seizure and 2 seizure
training windows.

The deterministic recording-level split was seven train recordings
(`chb01_02`, `03`, `05`, `06`, `07`, `09`, `12`), three validation recordings
(`04`, `10`, `11`), and three held-out recordings (`01`, `08`, `13`).  No
windows from a recording appear in more than one partition.  The held-out
recordings were not evaluated in Phase 16.

Best validation loss occurred at epoch 3: 0.2021956824.  The epoch-3 training
loss/accuracy/macro-F1 were 0.1404981105 / 0.9976190476 / 0.4994040524; the
corresponding validation values were 0.2021956824 / 0.9972222222 /
0.4993045897.  These values are strongly affected by the extreme class
imbalance: the seizure-class recall was zero in the recorded train and
validation predictions.  The high accuracy must not be treated as meaningful
seizure-detection performance.

The best checkpoint was saved and reloaded successfully from:

`D:\EEG-Major-Project\models\phase16_chb01_development\cnn_bilstm_chbmit_best.pt`

The machine-readable training report is in the same directory as
`phase16_training_report.json`; it includes the actual history, split, model,
data preparation, and checkpoint-load metadata.

Phase 17 held-out evaluation was not implemented.
