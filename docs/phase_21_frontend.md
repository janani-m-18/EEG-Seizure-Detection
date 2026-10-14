# Phase 21: EEG Seizure Analysis Interface

The dependency-free frontend is served by the FastAPI application at `/` and
uses the same-origin API by default. An optional `window.EEG_API_BASE_URL`
allows a configured API base URL without hard-coding localhost in the client.

It accepts EDF files through a browse control or drag-and-drop, validates file
type and non-empty size before enabling analysis, and displays idle, processing,
completed, and error states without fabricated progress percentages.

On a real success response, the page shows the model label and probability,
recording duration/rate/channel/segment information, a selectable real EEG
channel waveform, and an Integrated Gradients panel based on backend-returned
attribution values. It explicitly labels attribution as a model explanation
and includes a development-model limitation notice. No prediction, waveform,
or explanation data is hard-coded in the client.
