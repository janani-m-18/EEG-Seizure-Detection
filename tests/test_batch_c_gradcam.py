"""Batch C Grad-CAM checks using the real Batch B checkpoint, never fake maps."""

from pathlib import Path

import numpy as np
import pytest

from backend.app.processing.types import SegmentMetadata, SegmentedEEG
from ml.batch_b_inference import BatchBInferenceService
from ml.gradcam import explain_batch_b_segment, final_cnn_conv_layer, final_cnn_feature_layer


BATCH_B_CHECKPOINT = Path(r"D:\EEG-Major-Project\models\batch_b_chb01\batch_b_best.pt")
CHB_ROOT = Path(r"D:\EEG-Datasets\CHB-MIT\physionet.org\files\chbmit\1.0.0")


@pytest.fixture(scope="module")
def service() -> BatchBInferenceService:
    if not BATCH_B_CHECKPOINT.is_file():
        pytest.skip("Batch B checkpoint is unavailable")
    return BatchBInferenceService(BATCH_B_CHECKPOINT)


def _synthetic_segmented() -> SegmentedEEG:
    values = np.linspace(-1, 1, 23 * 1024, dtype=np.float32).reshape(1, 23, 1024)
    return SegmentedEEG("synthetic", values, tuple(f"C{index}" for index in range(23)), 256.0,
                        (SegmentMetadata(0, 0, 1024, 12.0, 16.0),))


def test_gradcam_uses_the_final_batch_b_convolutional_stage_and_maps_to_input_timeline(service: BatchBInferenceService) -> None:
    assert final_cnn_conv_layer(service.model).out_channels == 256
    assert final_cnn_feature_layer(service.model) is service.model.cnn.network[12]
    segmented = _synthetic_segmented()
    explanation = service.explain_segment(segmented, 0)
    assert explanation.target_class == explanation.predicted_class
    assert explanation.raw_temporal_map.shape == (256,)
    assert explanation.temporal_map.shape == (1024,)
    assert np.isfinite(explanation.temporal_map).all()
    assert 0.0 <= explanation.temporal_map.min() <= explanation.temporal_map.max() <= 1.0
    assert np.count_nonzero(explanation.temporal_map) > 0
    assert explanation.important_temporal_regions
    assert np.isclose(explanation.channel_contributions.sum(), 1.0)


def test_gradcam_explicit_target_is_respected_and_is_derived_from_model_gradients(service: BatchBInferenceService) -> None:
    segmented = _synthetic_segmented()
    first = service.explain_segment(segmented, 0, target_class=1)
    assert first.target_class == 1
    assert first.predicted_class in (0, 1)
    assert np.isfinite(first.temporal_map).all()
    assert np.isfinite(first.channel_contributions).all()
    assert np.count_nonzero(first.channel_contributions) > 0


@pytest.mark.parametrize(("recording", "annotation_start"), (("chb01_03", 2996.0), ("chb01_04", 1467.0)))
def test_real_chb01_segments_produce_actual_gradcam_for_model_prediction(
    service: BatchBInferenceService, recording: str, annotation_start: float,
) -> None:
    path = CHB_ROOT / "chb01" / f"{recording}.edf"
    if not path.is_file():
        pytest.skip("Local real CHB01 EDF is unavailable")
    prepared = service.prepare_edf(path)
    index = next(item.index for item in prepared.metadata if item.start_seconds <= annotation_start < item.stop_seconds)
    explanation = service.explain_segment(prepared, index)
    prediction, probability = service.predict_segments(prepared.segments[index:index + 1])
    assert explanation.predicted_class == int(prediction[0])
    assert np.isclose(explanation.seizure_probability, probability[0])
    assert len(explanation.temporal_map) == 1024
    assert np.isfinite(explanation.temporal_map).all()
    assert all(item.start_seconds >= prepared.metadata[index].start_seconds for item in explanation.important_temporal_regions)
