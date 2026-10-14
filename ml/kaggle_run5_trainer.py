"""Kaggle Tesla T4 GPU Training Script for CNN-BiLSTM Seizure Detection (Run 5).

This script provides a self-contained, reproducible training workflow
for execution on Kaggle notebooks or any CUDA GPU environment.

Dataset expected format (compact .npy):
  - X_train.npy, y_train.npy, metadata_train.csv
  - X_val.npy, y_val.npy, metadata_val.csv
  - X_test.npy, y_test.npy, metadata_test.csv
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ml.kaggle_model import KaggleCNNBiLSTM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train KaggleCNNBiLSTM on CHB-MIT (Run 5)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/kaggle/input/datasets/jananibm/chb-mit",
        help="Path to folder containing X_train.npy, y_train.npy, etc.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save the trained model checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(y_true: np.ndarray, y_pred_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0

    return {
        "accuracy": round(accuracy, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] Using {device} for training.")
    if device.type == "cuda":
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    # Load Compact Arrays
    print(f"[Data] Loading dataset from {data_dir}...")
    x_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    y_train = np.load(data_dir / "y_train.npy")
    x_val = np.load(data_dir / "X_val.npy", mmap_mode="r")
    y_val = np.load(data_dir / "y_val.npy")
    x_test = np.load(data_dir / "X_test.npy", mmap_mode="r")
    y_test = np.load(data_dir / "y_test.npy")

    print(f"  Train: X={x_train.shape}, y={y_train.shape} (Positives: {np.sum(y_train == 1)})")
    print(f"  Val:   X={x_val.shape}, y={y_val.shape} (Positives: {np.sum(y_val == 1)})")
    print(f"  Test:  X={x_test.shape}, y={y_test.shape} (Positives: {np.sum(y_test == 1)})")

    train_dataset = TensorDataset(
        torch.from_numpy(np.array(x_train, dtype=np.float32)),
        torch.from_numpy(y_train.astype(np.float32)),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(np.array(x_val, dtype=np.float32)),
        torch.from_numpy(y_val.astype(np.float32)),
    )
    test_dataset = TensorDataset(
        torch.from_numpy(np.array(x_test, dtype=np.float32)),
        torch.from_numpy(y_test.astype(np.float32)),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    # Initialize Architecture
    model = KaggleCNNBiLSTM().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] KaggleCNNBiLSTM initialized with {total_params:,} trainable parameters.")

    # Class-weighted Binary Cross-Entropy
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    pos_weight = torch.tensor([neg_count / max(1, pos_count)], dtype=torch.float32).to(device)
    print(f"[Loss] BCEWithLogitsLoss pos_weight = {pos_weight.item():.3f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_val_f1 = -1.0
    best_checkpoint_path = output_dir / "run5_best.pt"
    history: list[dict[str, float]] = []

    print("\n--- Starting Training ---")
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        train_preds, train_targets = [], []

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model.forward_logits(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            train_preds.extend(probs)
            train_targets.extend(batch_y.cpu().numpy())

        train_loss = total_loss / len(train_dataset)
        train_metrics = compute_metrics(np.array(train_targets), np.array(train_preds))

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                logits = model.forward_logits(batch_x)
                loss = criterion(logits, batch_y)

                val_loss += loss.item() * len(batch_y)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend(probs)
                val_targets.extend(batch_y.cpu().numpy())

        val_loss = val_loss / len(val_dataset)
        val_metrics = compute_metrics(np.array(val_targets), np.array(val_preds))
        scheduler.step(val_metrics["f1"])

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f}, F1: {train_metrics['f1']:.4f}, Sens: {train_metrics['sensitivity']:.4f}, Spec: {train_metrics['specificity']:.4f} | "
            f"Val Loss: {val_loss:.4f}, F1: {val_metrics['f1']:.4f}, Sens: {val_metrics['sensitivity']:.4f}, Spec: {val_metrics['specificity']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_metrics["f1"],
            "val_loss": val_loss,
            "val_f1": val_metrics["f1"],
            "val_sensitivity": val_metrics["sensitivity"],
            "val_specificity": val_metrics["specificity"],
        })

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            print(f"  -> New best validation F1 ({best_val_f1:.4f}). Saving checkpoint to {best_checkpoint_path}...")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_metrics": val_metrics,
                    "architecture": "KaggleCNNBiLSTM",
                },
                best_checkpoint_path,
            )

    elapsed = time.time() - start_time
    print(f"\n[Finished] Training complete in {elapsed / 60:.2f} minutes.")

    # Evaluate Best Checkpoint on Held-Out Test Set
    print(f"\n[Test] Evaluating best checkpoint on held-out test set (7 patients)...")
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds, test_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model.forward_logits(batch_x)
            probs = torch.sigmoid(logits).cpu().numpy()
            test_preds.extend(probs)
            test_targets.extend(batch_y.numpy())

    test_metrics = compute_metrics(np.array(test_targets), np.array(test_preds), threshold=0.5)
    print("\n================ Held-Out Test Results ================")
    print(f"  Test Accuracy:    {test_metrics['accuracy']:.4f}")
    print(f"  Test Sensitivity: {test_metrics['sensitivity']:.4f} (Recall)")
    print(f"  Test Specificity: {test_metrics['specificity']:.4f}")
    print(f"  Test Precision:   {test_metrics['precision']:.4f}")
    print(f"  Test F1 Score:    {test_metrics['f1']:.4f}")
    print(f"  Confusion Matrix: TP={test_metrics['tp']}, TN={test_metrics['tn']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}")
    print("=======================================================\n")

    # Save summary report
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": checkpoint["epoch"],
                "best_val_f1": best_val_f1,
                "test_metrics": test_metrics,
                "history": history,
            },
            f,
            indent=2,
        )
    print(f"[Done] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
