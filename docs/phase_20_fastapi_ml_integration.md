# Phase 20: FastAPI ML Integration

`POST /api/eeg/analyze` accepts one EDF file as `multipart/form-data` with a
`file` field. The upload is written to an operating-system temporary file,
processed by `EEGInferenceService`, and removed in a `finally` block. It is not
stored with the project or exposed in the response.

The endpoint uses `EEG_MODEL_CHECKPOINT`, `ANALYSIS_MAX_UPLOAD_BYTES`, and
`ANALYSIS_XAI_STEPS` from application configuration. It returns real recording
metadata, per-segment model probabilities, recording aggregation information,
a representative model-input waveform, and a real Integrated Gradients result.
The representative segment is the window with the highest seizure probability;
the recording remains seizure-positive only if any window predicts seizure.

Errors are returned without model internals or stack traces: invalid/empty file
uploads return 400, oversized uploads 413, incompatible sampling/channel
layouts or unreadable EDFs 422, and an unavailable configured checkpoint 503.

The real `chb01_03.edf` upload returned 120 segments and a non-seizure model
result. This is the known output of the development checkpoint, not a claim
about the annotated seizure interval or clinical performance.
