# Final CHB-MIT training experiment report

## Status and scope

The final experiment completed as an explicitly **bounded deterministic CPU fallback**, not as full-corpus training. It created a new final checkpoint at `models/final_chbmit/final_cnn_bilstm_best.pt`; existing development checkpoints were not overwritten.

The full locally available corpus was header-verified, but it was not fully used for model fitting. The existing Batch-B trainer materializes all tensors in memory, which is unsuitable for the 42.14-GB corpus on the available CPU-only, 16-GB system. No data reduction was silent: the retained recording and window caps are saved in `models/final_chbmit/bounded_selection.json`.

## Dataset discovery and compatibility

For this run, `CHBMIT_DATASET_ROOT` was supplied through the documented environment-variable mechanism as `D:\EEG-Datasets\CHB-MIT\physionet.org\files\chbmit\1.0.0`. No data was downloaded.

- Patients found: 24 (`chb01` through `chb24`)
- EDF recordings found and header-verified: 683 (42.14 GB)
- Sampling rate: 256 Hz for all discovered EDFs
- Observed channel counts: 22, 23, 24, 25, 28, 29, 31, and 38
- Seizure recordings from supplied summary annotations: 141
- EDFs compatible with the existing fixed 23-channel Batch-A/Batch-B contract: 273 across 13 patients
- Compatible seizure recordings: 60 across 11 patients
- Excluded recordings: 410, all documented in `models/final_chbmit/exclusion_report.json`; they required channel-layout handling outside the existing fixed 23-channel contract. No annotations were fabricated.

Thus, 100% of locally available EDFs were discovered and header-verified, but **100% were not used for training**. The bounded run processed 11 selected compatible seizure recordings (one deterministic first-sorted recording per compatible seizure patient), representing 37,313 Batch-A windows before the retention cap. It retained 220 genuine windows: the first 16 seizure and first 4 non-seizure windows per selected recording after Batch-A preprocessing.

## Preprocessing, split, and augmentation

The unchanged Batch-A pipeline was used: configured notch filtering, 0.5–40 Hz fourth-order Butterworth filtering, common-average reference, channel-wise Z-score normalization, 4-second windows, and 2-second stride. Inputs remained `(batch, 23, 1024)`.

The split was deterministic and patient-aware (seed 1), with no patient, recording, or window leakage:

| Partition | Patients | Recordings | Segments | Seizure / non-seizure |
| --- | ---: | ---: | ---: | ---: |
| Train | 6 | 6 | 120 | 96 / 24 |
| Validation | 2 | 2 | 40 | 32 / 8 |
| Held-out test | 3 | 3 | 60 | 48 / 12 |

The held-out patients were `chb06`, `chb08`, and `chb24`; their held-out recordings were `chb06_01`, `chb08_02`, and `chb24_01`. The held-out set contains seizure samples. Training-only EEG augmentation balanced the training partition with 72 derived transformed windows; validation and test data were not augmented.

## Model and training

The unchanged Batch-B architecture was used: CNN filters 64→128→256, bidirectional LSTMs with 128 then 64 units per direction, 64-unit dense layer, and binary sigmoid output. Training used BCEWithLogitsLoss, Adam (learning rate 0.001), batch size 32, maximum 50 epochs, deterministic seed 1, and validation-loss early stopping (patience 5).

Training ran for 13 epochs; the best validation checkpoint was epoch 8 with validation loss 0.139829. At that epoch, validation accuracy was 0.9250, sensitivity 0.9688, specificity 0.7500, precision 0.9394, F1 0.9538, and AUC-ROC 0.9766. Training history and final configuration are saved under `models/final_chbmit/`.

## Untouched held-out results

### Segment-level performance

- Accuracy: 0.5667
- Sensitivity / recall: 0.5208
- Specificity: 0.7500
- Precision: 0.8929
- F1: 0.6579
- AUC-ROC: 0.6580
- Support: 48 seizure, 12 non-seizure segments
- Confusion matrix (rows actual `[non-seizure, seizure]`, columns predicted): `[[9, 3], [23, 25]]`
- Counts: TN 9, FP 3, FN 23, TP 25

### Recording-level performance

Recording aggregation used the maximum segment seizure probability per recording. All three selected held-out recordings were seizure recordings, so recording-level specificity and AUC-ROC are undefined because there was no non-seizure recording support. The resulting matrix was `[[0, 0], [0, 3]]` (TN 0, FP 0, FN 0, TP 3); recording-level sensitivity was 1.0. This small, single-class recording-level result must not be compared with the segment-level result or generalized.

## Artifacts and integration

- Final checkpoint: `models/final_chbmit/final_cnn_bilstm_best.pt`
- Last-epoch checkpoint: `models/final_chbmit/final_cnn_bilstm_last.pt`
- Dataset, exclusion, split, selection, configuration, history, metrics, confusion matrix, and segment prediction/provenance files: `models/final_chbmit/`
- The checkpoint is tagged `batch_b_final_chbmit` and preserves the Batch-A preprocessing compatibility identifier. Batch-C Grad-CAM uses the same actual final CNN feature maps and gradients when this checkpoint is selected.
- The API checkpoint setting now prefers this final checkpoint only when it exists; `EEG_MODEL_CHECKPOINT` remains an explicit override and the previous working Batch-B checkpoint remains the fallback.

A live Grad-CAM check was run with this final checkpoint on the real selected-corpus file `chb01_03.edf` (segment 100): it produced a finite 1,024-sample temporal map and 23 channel contributions from the final CNN feature layer.

## Verification and limitations

The complete test suite passed: **54 passed, 5 skipped**, with one existing FastAPI/TestClient deprecation warning. The skipped checks are the pre-existing opt-in real-data checks, left unconfigured to avoid pointing their CHB01-only fixture at the full corpus.

This result is limited by the small, intentionally bounded training set, strict fixed-channel exclusions, one selected seizure recording per patient, the severe segment class imbalance in the retained data, CPU-only training, and a held-out recording-level set with no non-seizure recordings. It is not a full-corpus performance estimate and is not a clinical or diagnostic claim.
