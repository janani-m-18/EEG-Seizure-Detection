# Phase 19: CNN-BiLSTM Explainability

The project uses a direct PyTorch implementation of **Integrated Gradients**
for the existing CNN-BiLSTM. It integrates the gradient of the actual
predicted-class logit along a straight path from an all-zero baseline to the
real EEG window. Since Phase 16 applies channel-wise z-score normalization,
the zero baseline represents each channel's normalized mean. This is
deterministic and requires no external XAI framework.

The structured explanation contains the prediction and real softmax
probabilities, a signed `(23, 1024)` attribution array, channel names,
sampling/time axes, mean absolute temporal and channel importance, and the
five highest 95th-percentile attribution regions. Those regions are summaries
of calculated attributions, not independently validated EEG biomarkers.

The real demonstration uses `chb01_03.edf`, window 100 (3000--3004 seconds),
which overlaps the documented seizure interval. The development model predicted
that window as non-seizure, so the result is appropriately described as
**model attribution for the development checkpoint**, not as a seizure
explanation. The attribution JSON and actual waveform/attribution plot are
saved in `models/phase16_chb01_development/phase19_xai/`.

The visualization shows the real normalized mean EEG waveform, temporal mean
absolute Integrated Gradients attribution with highlighted high-attribution
regions, and a channel-by-time absolute attribution heatmap. It is a reusable
artifact for a future frontend but is not a frontend implementation.
