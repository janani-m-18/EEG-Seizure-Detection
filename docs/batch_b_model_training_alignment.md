# Batch B — CNN–BiLSTM model, training, and evaluation alignment

## Scope and compatibility

Batch B adds a separate development CNN–BiLSTM model for the Batch A preprocessing contract. It does not replace the Phase 16 checkpoint, its two-class-logit architecture, or the existing API/inference path. New artifacts are in `models/batch_b_chb01/`; checkpoint metadata carries model and preprocessing identifiers, and a mismatch is rejected on load.

## Previous model

The Phase 16 development checkpoint used a compact CNN with 4 then 8 feature channels, one 8-unit BiLSTM, two output logits, 30-second window starts, and the legacy 0.5–70 Hz FIR/no-CAR preprocessing policy. It is preserved for backward compatibility, not silently reused here.

## Batch B architecture

Batch B accepts the Batch A tensor `(batch, 23, 1024)` at 256 Hz.

| Layer | Configuration | Output for one batch |
| --- | --- | --- |
| CNN stage 1 | Conv1D 64, kernel 7, same padding, ReLU, max-pool 2 | `(batch, 64, 512)` |
| CNN stage 2 | Conv1D 128, kernel 7, same padding, ReLU, max-pool 2 | `(batch, 128, 256)` |
| CNN stage 3 | Conv1D 256, kernel 7, same padding, ReLU, max-pool 2 | `(batch, 256, 128)` |
| CNN → LSTM | Transpose to time-major features | `(batch, 128, 256)` |
| BiLSTM 1 | 128 units per direction | `(batch, 128, 256)` |
| BiLSTM 2 | 64 units per direction | `(batch, 128, 128)` |
| Dense | 128 to 64, ReLU, dropout | `(batch, 64)` |
| Output | 64 to 1, sigmoid | `(batch,)` seizure probability |

The CNN keeps 128 temporal steps for the two BiLSTMs; it does not collapse time before temporal modelling. A probability at or above the fixed, configurable 0.5 threshold is seizure (`1`); otherwise it is non-seizure (`0`). The threshold was not optimized on held-out data.

## Training policy

The supported default configuration is binary cross-entropy with logits, Adam, learning rate 0.001, batch size 32, maximum 50 epochs, seed 1, and validation-loss early stopping with patience 5. `BCEWithLogitsLoss` is used for numeric stability while the public model output remains a sigmoid probability.

No class weight is used in Batch B. Instead, only after a recording-aware split, the minority seizure windows in the training partition may receive the Batch A EEG-specific augmentation policy: time shifts up to ±0.5 s, amplitude scaling in [0.8, 1.2], and 15–25 dB SNR noise. Conventional SMOTE is not applied to raw multichannel EEG tensors. Validation and held-out test windows are never augmented.

## Recording-aware split and bounded CPU data set

All 13 local CHB01 EDFs were actually processed with Batch A preprocessing and the 4-second/2-second overlap policy. This yielded 23,387 windows: 23,351 non-seizure and 36 seizure. To make CPU development training feasible, the model retained the first 16 real non-seizure windows per recording plus every real seizure window. This is an explicitly bounded tensor of 244 windows (208 non-seizure, 36 seizure), not a claim that all 23,387 windows were used for gradient updates.

| Partition | Recording IDs | Retained windows |
| --- | --- | --- |
| Train | chb01_06, chb01_09, chb01_01, chb01_03, chb01_12, chb01_07, chb01_04 | 148 before augmentation: 112 non-seizure, 36 seizure |
| Validation | chb01_10, chb01_13, chb01_05 | 48 non-seizure, 0 seizure |
| Held-out test | chb01_02, chb01_11, chb01_08 | 48 non-seizure, 0 seizure |

Training-only augmentation created 76 seizure-derived copies, giving the train loader 224 windows: 112 non-seizure and 112 seizure. No recording appears in more than one partition.

## Real CPU development run

The checkpoint was trained for a deliberately bounded **one epoch**; this is below the supported 50-epoch maximum because the proposal-aligned model is much larger and the run is CPU-only. It completed at epoch 1 and the best checkpoint was reloaded successfully.

The genuine one-epoch training loss was 0.6175483465 and validation loss was 0.6704458992. Training-set metrics on the augmented training loader were: accuracy 0.6875, sensitivity 0.4196428571, specificity 0.9553571429, precision 0.9038461538, F1 0.5731707317, and AUC-ROC 0.8801817602. These are not held-out seizure-detection results and must not be interpreted as clinical or generalizable performance.

The validation set contains no seizure windows. Its accuracy was 0.5625 and specificity 0.5625 (27 TN, 21 FP); seizure sensitivity and F1 were undefined, and AUC-ROC was undefined. This validation limitation means the one-epoch loss is not evidence of seizure-detection quality.

## Held-out evaluation

Evaluation saves segment probabilities, labels, predictions, recording IDs, and segment start times in `held_out_segment_predictions.npz`, and saves the segment confusion matrix separately. Segment-level and recording-level results are intentionally reported separately.

The held-out segment set has 48 non-seizure and 0 seizure windows. Its genuine confusion matrix (rows actual, columns predicted; non-seizure then seizure) is `[[36, 12], [0, 0]]`. Accuracy and specificity are both 0.75; seizure sensitivity, seizure F1, and AUC-ROC are **undefined**, because both classes are required for those measures. Accuracy here is not seizure-detection performance.

For separate recording-level aggregation, the maximum segment seizure probability is used for each recording at the same 0.5 threshold. The three held-out recordings are all non-seizure; their matrix is `[[0, 3], [0, 0]]`. The resulting recording-level accuracy and specificity are 0.0, while seizure sensitivity/F1/AUC remain undefined. Recording-level output must not be confused with the 48-window segment-level result.

## Artifacts

The real development run produced `batch_b_best.pt`, `batch_b_final.pt`, `training_report.json`, `held_out_evaluation_report.json`, `held_out_confusion_matrix.npy`, `held_out_segment_predictions.npz`, and `batch_b_development_report.json` under `models/batch_b_chb01/`.

## Limitations

This is a single-patient CHB01 development experiment with a severe class imbalance. The held-out and validation partitions used by this fixed split have no seizure windows, so they cannot establish seizure sensitivity, F1, or AUC. The one-epoch bounded CPU run, augmented training metrics, and non-seizure-only held-out accuracy are not clinical validation, patient-independent evaluation, or evidence of real-world seizure detection performance.

**Batch B does not implement Grad-CAM, API changes, frontend changes, or Batch C.**
