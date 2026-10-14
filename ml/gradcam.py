"""Grad-CAM attribution for the Kaggle-trained CNN-BiLSTM model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from ml.kaggle_model import KaggleCNNBiLSTM


@dataclass(frozen=True, slots=True)
class GradCAMRegion:
    """A grouped high-attribution interval on the recording timeline."""

    start_seconds: float
    end_seconds: float
    activation_score: float

    def as_dict(self) -> dict[str, float]:
        return {
            "start_seconds": self.start_seconds,
            "end_seconds": self.end_seconds,
            "activation_score": self.activation_score,
        }


@dataclass(frozen=True, slots=True)
class GradCAMExplanation:
    """Structured, model-derived Grad-CAM output for one EEG segment."""

    target_class: int
    predicted_class: int
    seizure_probability: float
    non_seizure_probability: float
    segment_index: int
    segment_start_seconds: float
    segment_end_seconds: float
    channel_names: tuple[str, ...]
    sampling_frequency: float
    temporal_map: np.ndarray
    raw_temporal_map: np.ndarray
    important_temporal_regions: tuple[GradCAMRegion, ...]
    channel_contributions: np.ndarray
    checkpoint_path: Path

    def as_dict(self) -> dict[str, object]:
        regions = [
            item.as_dict()
            for item in self.important_temporal_regions
        ]

        return {
            "method": "Grad-CAM",
            "target_class": (
                "seizure"
                if self.target_class
                else "non_seizure"
            ),
            "predicted_class": (
                "seizure"
                if self.predicted_class
                else "non_seizure"
            ),
            "seizure_probability": self.seizure_probability,
            "non_seizure_probability": self.non_seizure_probability,
            "segment_index": self.segment_index,
            "segment_start_seconds": self.segment_start_seconds,
            "segment_end_seconds": self.segment_end_seconds,
            "channels": list(self.channel_names),
            "channel_names": list(self.channel_names),
            "sampling_frequency": self.sampling_frequency,
            "gradcam_temporal_map": self.temporal_map.tolist(),
            "raw_gradcam_temporal_map": self.raw_temporal_map.tolist(),
            "important_temporal_regions": regions,
            "channel_contributions": (
                self.channel_contributions.tolist()
            ),
            "channel_contribution_method": (
                "normalized mean absolute target-score input gradient"
            ),
            "cnn_target_layer": (
                "final 256-filter Conv2d layer "
                "(model.cnn[9])"
            ),
            "temporal_mapping": (
                "final convolutional feature map interpolated "
                "from 128 temporal positions to the original "
                "1024 samples"
            ),
            "model_id": (
                "batch-b-cnn64-128-256-bilstm128-64-dense64-binary-v1"
            ),
            "checkpoint_path": str(self.checkpoint_path),

            # Compatibility aliases for the frontend.
            "temporal_importance": self.temporal_map.tolist(),
            "channel_importance": (
                self.channel_contributions.tolist()
            ),
            "important_regions": [
                {
                    "start_seconds": item.start_seconds,
                    "stop_seconds": item.end_seconds,
                    "peak_importance": item.activation_score,
                }
                for item in self.important_temporal_regions
            ],
        }


def _normalise(values: Tensor) -> Tensor:
    """Min-max normalize values across the final dimension."""

    minimum = values.amin(
        dim=-1,
        keepdim=True,
    )

    maximum = values.amax(
        dim=-1,
        keepdim=True,
    )

    denominator = (
        maximum - minimum
    ).clamp_min(
        torch.finfo(values.dtype).eps
    )

    normalised = (
        values - minimum
    ) / denominator

    return torch.where(
        maximum > minimum,
        normalised,
        torch.zeros_like(values),
    )


def _regions(
    temporal_map: np.ndarray,
    *,
    sampling_frequency: float,
    segment_start_seconds: float,
    max_regions: int = 5,
    threshold_quantile: float = 0.90,
    minimum_duration_seconds: float = 0.05,
    merge_gap_seconds: float = 0.03,
) -> tuple[GradCAMRegion, ...]:
    """Group contiguous high-activation samples into intervals."""

    if (
        temporal_map.ndim != 1
        or len(temporal_map) == 0
        or not np.isfinite(temporal_map).all()
    ):
        return ()

    threshold = float(
        np.quantile(
            temporal_map,
            threshold_quantile,
        )
    )

    active = np.flatnonzero(
        temporal_map >= threshold
    )

    if (
        not len(active)
        or float(temporal_map.max()) == 0
    ):
        return ()

    gap_samples = max(
        1,
        round(
            merge_gap_seconds
            * sampling_frequency
        ),
    )

    minimum_samples = max(
        1,
        round(
            minimum_duration_seconds
            * sampling_frequency
        ),
    )

    grouped: list[tuple[int, int]] = []

    start = previous = int(active[0])

    for item in active[1:]:
        current = int(item)

        if current - previous > gap_samples + 1:
            if (
                previous - start + 1
                >= minimum_samples
            ):
                grouped.append(
                    (
                        start,
                        previous + 1,
                    )
                )

            start = current

        previous = current

    if (
        previous - start + 1
        >= minimum_samples
    ):
        grouped.append(
            (
                start,
                previous + 1,
            )
        )

    regions = [
        GradCAMRegion(
            segment_start_seconds
            + start / sampling_frequency,
            segment_start_seconds
            + stop / sampling_frequency,
            float(
                temporal_map[start:stop].max()
            ),
        )
        for start, stop in grouped
    ]

    return tuple(
        sorted(
            regions,
            key=lambda item: item.activation_score,
            reverse=True,
        )[:max_regions]
    )


def final_cnn_conv_layer(
    model: KaggleCNNBiLSTM,
) -> nn.Conv2d:
    """Return the final 256-filter Conv2d layer."""

    layer = model.cnn[9]

    if not isinstance(layer, nn.Conv2d):
        raise ValueError(
            "Kaggle CNN-BiLSTM model.cnn[9] "
            "is not the expected final Conv2d layer"
        )

    if layer.out_channels != 256:
        raise ValueError(
            "Expected the final CNN layer to have "
            "256 output channels"
        )

    return layer


def final_cnn_feature_layer(
    model: KaggleCNNBiLSTM,
) -> nn.Module:
    """
    Return the post-activation feature map of the
    final CNN convolutional stage.

    The Kaggle CNN structure is:

        9  -> Conv2d(128, 256)
        10 -> BatchNorm2d(256)
        11 -> ReLU
        12 -> Dropout
    """

    layer = model.cnn[11]

    if not isinstance(layer, nn.ReLU):
        raise ValueError(
            "Kaggle CNN-BiLSTM model.cnn[11] "
            "is not the expected final ReLU feature layer"
        )

    return layer


def explain_batch_b_segment(
    model: KaggleCNNBiLSTM,
    segment: np.ndarray,
    *,
    channel_names: tuple[str, ...],
    sampling_frequency: float,
    segment_index: int,
    segment_start_seconds: float,
    segment_end_seconds: float,
    checkpoint_path: Path,
    target_class: int | None = None,
    device: str = "cpu",
) -> GradCAMExplanation:
    """
    Compute model-derived Grad-CAM for one EEG segment.

    The Kaggle model produces one binary seizure logit.

    For seizure:
        target score = logit

    For non-seizure:
        target score = -logit
    """

    values = np.asarray(
        segment,
        dtype=np.float32,
    )

    expected = (
        len(channel_names),
        1024,
    )

    if (
        values.shape != expected
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            f"Expected one finite EEG segment "
            f"shaped {expected}"
        )

    if (
        len(channel_names) != 23
        or sampling_frequency <= 0
    ):
        raise ValueError(
            "Expected 23 channel names and "
            "a positive sampling frequency"
        )

    # Final convolutional feature map.
    target_layer = final_cnn_feature_layer(
        model
    )

    activations: Tensor | None = None
    gradients: Tensor | None = None

    def capture_activation(
        _module: nn.Module,
        _inputs: tuple[Tensor, ...],
        output: Tensor,
    ) -> None:
        nonlocal activations

        activations = output

        output.register_hook(
            capture_gradient
        )

    def capture_gradient(
        gradient: Tensor,
    ) -> None:
        nonlocal gradients

        gradients = gradient

    model_device = torch.device(device)

    model.to(model_device)

    # Normal inference mode.
    model.eval()

    # cuDNN can reject RNN backward in eval mode.
    # Put only the LSTMs in train mode for gradient
    # computation. They are single-layer LSTMs, so this
    # does not introduce active recurrent dropout.
    model.bilstm1.train()
    model.bilstm2.train()

    tensor = (
        torch.from_numpy(
            values[None, ...]
        )
        .to(model_device)
        .requires_grad_(True)
    )

    handle = target_layer.register_forward_hook(
        capture_activation
    )

    selected_target = 0
    predicted = 0
    seizure_probability = 0.0

    temporal_map = np.zeros(
        1024,
        dtype=np.float32,
    )

    raw_values = np.zeros(
        128,
        dtype=np.float32,
    )

    channel_contributions = np.zeros(
        23,
        dtype=np.float32,
    )

    try:
        model.zero_grad(
            set_to_none=True
        )

        # Get raw binary classification logit.
        logit = model.forward_logits(
            tensor
        )

        seizure_probability = float(
            torch.sigmoid(
                logit
            )
            .detach()
            .cpu()
            .item()
        )

        predicted = int(
            seizure_probability >= 0.5
        )

        selected_target = (
            predicted
            if target_class is None
            else int(target_class)
        )

        if selected_target not in (0, 1):
            raise ValueError(
                "target_class must be "
                "0, 1, or None"
            )

        # Binary model:
        #
        # seizure     -> +logit
        # non-seizure -> -logit
        score = (
            logit[0]
            if selected_target == 1
            else -logit[0]
        )

        score.backward()

        if (
            activations is None
            or gradients is None
        ):
            raise RuntimeError(
                "Grad-CAM hooks did not capture "
                "the activation and gradient"
            )

        # Expected:
        #
        # [1, 256, 23, 128]
        #
        # batch = 1
        # CNN filters = 256
        # EEG channels = 23
        # temporal positions = 128
        if activations.ndim != 4:
            raise RuntimeError(
                "Unexpected CNN activation shape: "
                f"{tuple(activations.shape)}"
            )

        if gradients.shape != activations.shape:
            raise RuntimeError(
                "Grad-CAM activation and gradient "
                "shapes do not match"
            )

        # Grad-CAM weights:
        # average gradient across EEG channels
        # and temporal positions.
        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True,
        )

        # Weighted feature map.
        cam = torch.relu(
            (
                weights
                * activations
            ).sum(dim=1)
        )

        # Result:
        # [1, 23, 128]
        cam = _normalise(
            cam
        )

        # Preserve channel-specific Grad-CAM.
        channel_map = cam

        # Average across the 23 EEG channels
        # to obtain the temporal importance map.
        temporal = channel_map.mean(
            dim=1
        )

        temporal = _normalise(
            temporal
        )

        # CNN produces 128 temporal positions.
        # Map them back to the original 1024 samples.
        mapped = functional.interpolate(
            temporal.unsqueeze(1),
            size=1024,
            mode="linear",
            align_corners=False,
        ).squeeze(1)

        temporal_map = (
            mapped
            .detach()
            .cpu()
            .numpy()[0]
        )

        raw_values = (
            temporal[0]
            .detach()
            .cpu()
            .numpy()
        )

        # Supplementary channel contribution:
        # mean absolute input gradient per channel.
        input_gradient = tensor.grad

        if input_gradient is None:
            raise RuntimeError(
                "Input gradient was not captured"
            )

        channel_values = (
            input_gradient
            .detach()
            .abs()
            .mean(dim=2)[0]
        )

        contribution_sum = (
            channel_values.sum()
        )

        if contribution_sum > 0:
            channel_contributions = (
                channel_values
                / contribution_sum
            ).cpu().numpy()
        else:
            channel_contributions = (
                torch.zeros_like(
                    channel_values
                )
                .cpu()
                .numpy()
            )

    finally:
        handle.remove()

        # Restore normal inference mode.
        model.eval()

    return GradCAMExplanation(
        target_class=selected_target,
        predicted_class=predicted,
        seizure_probability=seizure_probability,
        non_seizure_probability=(
            1.0 - seizure_probability
        ),
        segment_index=segment_index,
        segment_start_seconds=(
            segment_start_seconds
        ),
        segment_end_seconds=(
            segment_end_seconds
        ),
        channel_names=channel_names,
        sampling_frequency=sampling_frequency,
        temporal_map=temporal_map,
        raw_temporal_map=raw_values,
        important_temporal_regions=_regions(
            temporal_map,
            sampling_frequency=sampling_frequency,
            segment_start_seconds=(
                segment_start_seconds
            ),
        ),
        channel_contributions=(
            channel_contributions
        ),
        checkpoint_path=checkpoint_path,
    )