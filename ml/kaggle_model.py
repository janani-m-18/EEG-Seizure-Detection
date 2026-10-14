from __future__ import annotations

import torch
from torch import Tensor, nn


class KaggleCNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM model matching the architecture used by the Kaggle checkpoint.

    Input:
        (batch, 23, 1024)

    Output:
        seizure probability for each EEG window
    """

    def __init__(self) -> None:
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 4)),

            nn.Conv2d(64, 128, kernel_size=(1, 5), padding=(0, 2)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=(1, 3), padding=(0, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

        # 23 EEG channels remain as the spatial dimension.
        # After pooling: 1024 -> 256 -> 128
        self.bilstm1 = nn.LSTM(
            input_size=256 * 23,
            hidden_size=128,
            batch_first=True,
            bidirectional=True,
        )

        self.bilstm2 = nn.LSTM(
            input_size=128 * 2,
            hidden_size=64,
            batch_first=True,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1),
        )

    def forward_logits(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 3:
            raise ValueError(
                "Expected input shape (batch, 23, 1024)"
            )

        if inputs.shape[1:] != (23, 1024):
            raise ValueError(
                f"Expected input shape (batch, 23, 1024), "
                f"got {tuple(inputs.shape)}"
            )

        # (B, 23, 1024) -> (B, 1, 23, 1024)
        x = inputs.unsqueeze(1)

        # CNN output: (B, 256, 23, 128)
        x = self.cnn(x)

        batch_size, channels, eeg_channels, time_steps = x.shape

        # IMPORTANT:
        # Time becomes the LSTM sequence dimension.
        # Features = CNN channels × EEG channels.
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(
            batch_size,
            time_steps,
            channels * eeg_channels,
        )

        x, _ = self.bilstm1(x)
        x, _ = self.bilstm2(x)

        # Last temporal step
        x = x[:, -1, :]

        return self.classifier(x).squeeze(1)

    def forward(self, inputs: Tensor) -> Tensor:
        return torch.sigmoid(self.forward_logits(inputs))