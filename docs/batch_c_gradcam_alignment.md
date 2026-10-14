# Batch C — Grad-CAM, API, and frontend alignment

## Scope

Batch C makes Grad-CAM the primary explainability method for the separate Batch
B CHB01 development checkpoint. Batch A preprocessing and segmentation, Batch
B architecture/training/evaluation, the legacy Phase 16 inference service, and
Integrated Gradients are preserved. No data were downloaded, retrained, or
added.

## Why Grad-CAM

Grad-CAM uses gradients of a selected model-class score with respect to an
actual CNN feature map. It provides a temporal activation map showing which
parts of a model input most positively contributed to the selected model class.
It is computed from the Batch B model at explanation time; no attribution is
stored, randomised, or fabricated.

## CNN target and computation

The target is the final CNN stage's post-activation map: the `ReLU` after the
final 256-channel `Conv1d` and BatchNorm, before the third max-pool. It has 256
temporal positions. For the selected binary class score, Batch C:

1. captures that forward feature map and its back-propagated gradient;
2. globally averages gradients over the temporal axis to obtain one weight per
   feature channel;
3. forms the weighted feature-map sum and applies ReLU;
4. safely normalizes the real map; and
5. linearly interpolates the 256 positions to the original 1,024 samples.

The selected class defaults to the model prediction. In the single-logit binary
model, seizure uses the logit and non-seizure uses its negative (the
corresponding log-odds). An explicit target is supported for testing/debugging.

## Temporal and channel output

The 1,024-sample Grad-CAM map is aligned with the actual four-second segment
timeline at 256 Hz. Important regions use the 90th-percentile activation
threshold, group points separated by no more than 30 ms, discard groups shorter
than 50 ms, and return at most five highest-peak regions. Returned times are
segment-relative positions translated to the recording timeline.

Standard 1D Grad-CAM is temporal; it does not directly prove raw electrode
importance. The optional `channel_contributions` are therefore explicitly
labelled as normalized mean absolute input gradients of the selected class
score. They are supplementary model-input sensitivity values, not clinical
channel importance.

## API and frontend

`POST /api/eeg/analyze` keeps the multipart `file` interface and temporary-file
cleanup. It now loads only `models/batch_b_chb01/batch_b_best.pt`, checks the
Batch A-compatible metadata, validates 23 channels and 256 Hz, and applies the
Batch A 0.5–40 Hz/CAR/Z-score/4-second/2-second pipeline. It selects the
highest seizure-probability segment from real Batch B inference and returns a
primary `explainability` object with `method: "Grad-CAM"`.

The frontend retains upload, error/loading behavior, metadata, waveform, and
channel selection. Its primary explanation panel is now **Grad-CAM
Explanation**: an aligned waveform/temporal-attribution SVG, actual grouped
regions, and clearly labelled gradient-derived channel contributions. It states
that this is model attribution, not diagnosis or confirmed seizure
localization.

## Integrated Gradients

The existing Phase 16 Integrated Gradients implementation remains available as
a legacy secondary method. It is not used internally by Batch C Grad-CAM, and
the Batch C API/frontend no longer presents it as the official primary method.

## Real data verification

With the Batch B checkpoint and Batch A preprocessing:

| EDF / annotated seizure-overlap segment | Actual model prediction | Seizure probability | Grad-CAM target |
| --- | --- | --- | --- |
| `chb01_03`, segment 2994–2998 s (overlaps 2996–3036 s annotation) | non-seizure | 0.4050114155 | non-seizure |
| `chb01_04`, segment 1464–1468 s (overlaps 1467–1494 s annotation) | seizure | 0.6068702936 | seizure |

For `chb01_03`, the model's non-seizure prediction remains unchanged even
though the segment overlaps a seizure annotation; Grad-CAM explains that model
prediction rather than correcting it. Both real explanations contained the
actual 256-step CNN map, 1,024-sample mapped timeline, finite normalized values,
and two grouped temporal regions.

## Limitations

Grad-CAM provides model attribution and should not be interpreted as a clinical
diagnosis or confirmed seizure localization. The current CHB01 development
model has limited seizure-detection capability and Grad-CAM explains the
model's prediction regardless of whether that prediction is clinically correct.
The model remains a single-patient, severely imbalanced development model; its
validation and held-out partition limitations remain unchanged.

**Batch C does not alter Batch A preprocessing, Batch B architecture/training,
or the dataset scope.**
