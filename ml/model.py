"""Configurable CNN + BiLSTM classifier for segmented EEG tensors.

Inputs use ``(batch, channels, samples)``.  The CNN maps each sample window to
a shorter sequence of feature vectors, and the BiLSTM learns relationships
across that feature sequence.  This module deliberately makes no clinical or
performance claims; data preparation and evaluation remain dataset-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class CNNConfig:
    input_channels: int
    feature_channels: tuple[int, ...] = (32, 64)
    kernel_size: int = 7
    pool_size: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_channels < 1 or not self.feature_channels or any(width < 1 for width in self.feature_channels):
            raise ValueError("input_channels and feature_channels must be positive")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd number")
        if self.pool_size < 1 or not 0 <= self.dropout < 1:
            raise ValueError("pool_size must be positive and dropout must be in [0, 1)")


class EEGCNNFeatureExtractor(nn.Module):
    """Extract a temporal feature sequence from ``(batch, channels, samples)`` EEG."""

    def __init__(self, config: CNNConfig) -> None:
        super().__init__()
        self.config = config
        layers: list[nn.Module] = []
        in_channels = config.input_channels
        for out_channels in config.feature_channels:
            layers.extend((
                nn.Conv1d(in_channels, out_channels, config.kernel_size, padding=config.kernel_size // 2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(config.pool_size),
                nn.Dropout(config.dropout),
            ))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.output_features = in_channels

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("CNN inputs must have shape (batch, channels, samples)")
        if inputs.shape[1] != self.config.input_channels:
            raise ValueError(f"expected {self.config.input_channels} EEG channels, got {inputs.shape[1]}")
        if inputs.shape[2] < self.config.pool_size ** len(self.config.feature_channels):
            raise ValueError("sample length is too short for the configured CNN pooling")
        # LSTM convention: (batch, sequence length, features).
        return self.network(inputs).transpose(1, 2)


@dataclass(frozen=True, slots=True)
class BiLSTMConfig:
    input_features: int
    hidden_size: int = 64
    num_layers: int = 1
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.input_features < 1 or self.hidden_size < 1 or self.num_layers < 1:
            raise ValueError("BiLSTM dimensions and num_layers must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


class TemporalBiLSTM(nn.Module):
    """Summarise CNN feature sequences using the last bidirectional states."""

    def __init__(self, config: BiLSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_features,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
        )
        self.output_features = config.hidden_size * 2

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError("BiLSTM inputs must have shape (batch, sequence, features)")
        if inputs.shape[2] != self.config.input_features:
            raise ValueError(f"expected {self.config.input_features} CNN features, got {inputs.shape[2]}")
        if inputs.shape[1] < 1:
            raise ValueError("BiLSTM requires at least one sequence step")
        _, (hidden, _) = self.lstm(inputs)
        return torch.cat((hidden[-2], hidden[-1]), dim=1)


class CNNBiLSTMClassifier(nn.Module):
    """End-to-end classification head, separated from training and inference concerns."""

    def __init__(self, cnn_config: CNNConfig, bilstm_config: BiLSTMConfig, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least two")
        self.cnn = EEGCNNFeatureExtractor(cnn_config)
        if bilstm_config.input_features != self.cnn.output_features:
            raise ValueError("bilstm_config.input_features must equal CNN output features")
        self.bilstm = TemporalBiLSTM(bilstm_config)
        self.classifier = nn.Linear(self.bilstm.output_features, num_classes)
        self.num_classes = num_classes

    def forward(self, inputs: Tensor) -> Tensor:
        return self.classifier(self.bilstm(self.cnn(inputs)))


@dataclass(frozen=True, slots=True)
class BatchBModelConfig:
    """Architecture contract for the proposal-aligned Batch B binary model."""

    input_channels: int = 23
    cnn_channels: tuple[int, int, int] = (64, 128, 256)
    kernel_size: int = 7
    pool_size: int = 2
    dropout: float = 0.1
    first_bilstm_hidden_size: int = 128
    second_bilstm_hidden_size: int = 64
    dense_features: int = 64

    def __post_init__(self) -> None:
        if self.input_channels < 1 or any(width < 1 for width in self.cnn_channels):
            raise ValueError("input_channels and cnn_channels must be positive")
        if self.kernel_size < 1 or self.kernel_size % 2 == 0 or self.pool_size < 1:
            raise ValueError("kernel_size must be positive and odd; pool_size must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if min(self.first_bilstm_hidden_size, self.second_bilstm_hidden_size, self.dense_features) < 1:
            raise ValueError("BiLSTM and dense dimensions must be positive")


class BatchBCNNBiLSTMClassifier(nn.Module):
    """Proposal-aligned CNN--two-stage-BiLSTM classifier with sigmoid output.

    For a Batch A input of ``(batch, 23, 1024)``, three pool-by-two CNN stages
    yield ``(batch, 128, 256)`` in LSTM convention.  The first BiLSTM keeps a
    256-dimensional bidirectional sequence; the second yields 128 features per
    time step, and its final bidirectional state feeds the 64-unit dense layer.
    ``forward`` returns seizure probabilities, while ``forward_logits`` exists
    for numerically stable binary cross-entropy training.
    """

    def __init__(self, config: BatchBModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or BatchBModelConfig()
        self.cnn = EEGCNNFeatureExtractor(CNNConfig(
            input_channels=self.config.input_channels,
            feature_channels=self.config.cnn_channels,
            kernel_size=self.config.kernel_size,
            pool_size=self.config.pool_size,
            dropout=self.config.dropout,
        ))
        self.first_bilstm = nn.LSTM(
            input_size=self.cnn.output_features,
            hidden_size=self.config.first_bilstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.second_bilstm = nn.LSTM(
            input_size=self.config.first_bilstm_hidden_size * 2,
            hidden_size=self.config.second_bilstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.dense = nn.Linear(self.config.second_bilstm_hidden_size * 2, self.config.dense_features)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(self.config.dropout)
        self.output = nn.Linear(self.config.dense_features, 1)

    @property
    def cnn_temporal_shape(self) -> tuple[int, int]:
        """Return ``(sequence_steps, features)`` for the 1,024-sample Batch A window."""
        return 1024 // (self.config.pool_size ** len(self.config.cnn_channels)), self.cnn.output_features

    def forward_features(self, inputs: Tensor) -> Tensor:
        """Return the final 128-feature bidirectional temporal sequence."""
        cnn_sequence = self.cnn(inputs)
        first_sequence, _ = self.first_bilstm(cnn_sequence)
        second_sequence, _ = self.second_bilstm(first_sequence)
        return second_sequence

    def forward_logits(self, inputs: Tensor) -> Tensor:
        """Return one unthresholded seizure logit for each EEG window."""
        cnn_sequence = self.cnn(inputs)
        first_sequence, _ = self.first_bilstm(cnn_sequence)
        _, (hidden, _) = self.second_bilstm(first_sequence)
        final_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.output(self.dropout(self.activation(self.dense(final_state)))).squeeze(1)

    def forward(self, inputs: Tensor) -> Tensor:
        """Return one seizure probability in [0, 1] per EEG window."""
        return torch.sigmoid(self.forward_logits(inputs))
