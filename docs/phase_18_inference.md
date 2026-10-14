# Phase 18: Real EDF Inference

`ml.inference.EEGInferenceService` loads the actual Phase 16 checkpoint and
reuses the exact Phase 16 preprocessing/segmentation policy: 256 Hz EDF input,
the fixed 23-channel CHB01 MNE layout in source order, and four-second windows
with 30-second starts. It validates a missing/corrupt file, checkpoint,
sampling frequency, channel order, finite values, and model tensor shape.

`predict_edf()` returns one real softmax probability pair and class prediction
per window. At the recording level, its aggregation rule is **seizure if any
window is predicted seizure**; the largest seizure probability is reported as
additional context. No endpoint or FastAPI integration is included.

For the real `chb01_03.edf` development example, inference generated 120
window predictions. The annotated 3000--3004 second window received
non-seizure probability 0.8420460820 and seizure probability 0.1579538584,
and was predicted non-seizure. This is a genuine model result and illustrates
the known poor seizure recall of this development checkpoint; it is not a
claim that the annotation is wrong or that the system detects seizures.
