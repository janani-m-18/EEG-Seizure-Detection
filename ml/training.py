"""Deterministic dataset conversion, training, validation, and checkpoint helpers."""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split


class EEGSegmentDataset(Dataset[tuple[Tensor, Tensor]]):
    """Convert homogeneous segmented EEG arrays into validated PyTorch tensors."""

    def __init__(self, segments: np.ndarray | Tensor, labels: Sequence[int] | np.ndarray | Tensor) -> None:
        data = torch.as_tensor(segments, dtype=torch.float32)
        targets = torch.as_tensor(labels, dtype=torch.long)
        if data.ndim != 3 or data.shape[0] == 0:
            raise ValueError("segments must be a non-empty array of shape (items, channels, samples)")
        if targets.ndim != 1 or len(targets) != len(data):
            raise ValueError("labels must be one-dimensional and match the number of segments")
        if torch.any(targets < 0):
            raise ValueError("labels must be non-negative class indices")
        self.data, self.targets = data.contiguous(), targets.contiguous()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.data[index], self.targets[index]


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seed: int = 42
    epochs: int = 5
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    validation_fraction: float = 0.2
    device: str = "cpu"
    checkpoint_path: Path | None = None
    early_stopping_patience: int | None = 3

    def __post_init__(self) -> None:
        if self.epochs < 1 or self.batch_size < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("epochs, batch_size, and learning_rate must be positive; weight_decay cannot be negative")
        if not 0 < self.validation_fraction < 1:
            raise ValueError("validation_fraction must be between zero and one")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive or None")


def set_reproducible_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _metrics(logits: Tensor, targets: Tensor, total_loss: float) -> dict[str, float]:
    return {"loss": total_loss / len(targets), "accuracy": (logits.argmax(dim=1) == targets).float().mean().item()}


def _evaluate(model: nn.Module, loader: DataLoader[tuple[Tensor, Tensor]], criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss, all_logits, all_targets = 0.0, [], []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            total_loss += criterion(logits, targets).item() * len(targets)
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
    return _metrics(torch.cat(all_logits), torch.cat(all_targets), total_loss)


def save_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer, config: TrainingConfig, epoch: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                "training_config": asdict(config), "epoch": epoch}, path)


def load_checkpoint(path: Path, model: nn.Module, optimizer: torch.optim.Optimizer | None = None, *, device: str = "cpu") -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def train_model(model: nn.Module, dataset: EEGSegmentDataset, config: TrainingConfig) -> list[dict[str, float]]:
    """Train with a seeded split and return epoch-level train/validation metrics."""
    if len(dataset) < 2:
        raise ValueError("at least two segments are required for training and validation")
    validation_size = max(1, round(len(dataset) * config.validation_fraction))
    if validation_size >= len(dataset):
        validation_size = len(dataset) - 1
    set_reproducible_seed(config.seed)
    train_set, validation_set = random_split(dataset, [len(dataset) - validation_size, validation_size], generator=torch.Generator().manual_seed(config.seed))
    generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, generator=generator)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size, shuffle=False)
    device = torch.device(config.device)
    model.to(device)
    criterion, optimizer = nn.CrossEntropyLoss(), AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss, all_logits, all_targets = 0.0, [], []
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
            all_logits.append(logits.detach().cpu())
            all_targets.append(targets.detach().cpu())
        train_metrics = _metrics(torch.cat(all_logits), torch.cat(all_targets), total_loss)
        validation_metrics = _evaluate(model, validation_loader, criterion, device)
        history.append({"epoch": float(epoch), **{f"train_{key}": value for key, value in train_metrics.items()}, **{f"validation_{key}": value for key, value in validation_metrics.items()}})
        if config.checkpoint_path is not None:
            save_checkpoint(config.checkpoint_path, model, optimizer, config, epoch)
    return history
