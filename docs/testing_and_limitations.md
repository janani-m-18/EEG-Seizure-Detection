# Testing, Limitations, and Submission Readiness

## Final verification

The final Phase 23 run completed with **38 passed tests**. Coverage includes
Python imports, health routing, EEG processing, real-data preparation,
checkpoint loading, inference, Integrated Gradients, no-positive-class
evaluation behavior, API errors, temporary upload deletion, frontend serving,
and frontend JavaScript syntax.

Phase 22 also verified the rendered browser path using the real `chb01_03.edf`:
frontend upload, FastAPI temporary file, MNE processing, segmentation,
CNN–BiLSTM probabilities, Integrated Gradients, JSON response, waveform
rendering, and attribution rendering. The observed result was non-seizure with
a 19.03% representative seizure probability.

## Robustness handling

The API has tests for missing file field, invalid extension, empty upload,
corrupted EDF, incompatible sampling rate, incompatible channel layout,
oversized upload, and missing checkpoint. It returns client-safe messages and
does not expose traceback text. The real upload integration test records the
temporary filename and confirms it no longer exists when the response ends.

## Known limitations

- All development recordings belong to CHB01; there is no patient-independent
  experiment.
- Only three prepared seizure windows are available against 1,557 non-seizure
  windows.
- Phase 16 recorded seizure recall of zero despite high class-imbalanced
  accuracy.
- The Phase 17 held-out set has no seizure windows, so its 1.0 accuracy cannot
  establish detection performance.
- Checkpoint, waveform, probability, and Integrated Gradients values are real
  development outputs, but they are not clinical evidence.

## Repository hygiene

`.env`, virtual environments, caches, temporary pytest directories, frontend
build folders, local datasets, and model checkpoint extensions are ignored. No
additional dataset was downloaded and no model was retrained during final
verification. The persistent model outputs are the existing development
checkpoint and its measured reports.
