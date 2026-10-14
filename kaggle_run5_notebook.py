#!/usr/bin/env python3
# =============================================================================
# RUN 5 — FINAL CNN-BiLSTM KAGGLE GPU TRAINING NOTEBOOK (AUDITED V6)
# =============================================================================
#
# Project: "Explainable CNN–BiLSTM Based EEG Signal Analysis System for
#           Epileptic Seizure Detection"
#
# Dataset: Preprocessed CHB-MIT dataset attached in Kaggle at:
#          /kaggle/input/datasets/jananibm/chb-mit
#
# Patient-Independent Split (STRICT RUN4 SPLIT):
#   TRAIN (11 patients): chb04, chb08, chb09, chb11, chb14, chb17, chb18, chb20, chb21, chb22, chb23
#          -> 8,720 windows (6,493 non-seizure, 2,227 seizure)
#   VAL   (3 patients):  chb05, chb07, chb16
#          -> 2,180 windows (1,682 non-seizure, 498 seizure)
#   TEST  (7 patients):  chb01, chb02, chb03, chb06, chb10, chb13, chb19
#          -> 4,860 windows (3,645 non-seizure, 1,215 seizure)
#
# Alignment Fix (V6):
#   - Preserves exact array indices when slicing metadata_val / metadata_test
#   - Slices group_idxs directly against val_probs_arr / y_val / test_probs_arr
#   - Asserts len(meta) == len(X) == len(y) for Val and Test
#   - Asserts grouping covers every probability index exactly once with zero duplicates
# =============================================================================

from __future__ import annotations

import csv
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler


# =============================================================================
# SECTION 1 — CONSTANTS & HYPERPARAMETERS
# =============================================================================

SEED                  = 42
FS                    = 256.0
WINDOW_SAMPLES        = 1024          # 4 s
N_CHANNELS            = 23

BATCH_SIZE            = 64            # Exactly 16 sz + 16 hn + 32 bg
EPOCHS                = 30            # Max 30 epochs
EARLY_STOP_PATIENCE   = 5             # Early stopping patience
LR                    = 1e-4
WEIGHT_DECAY          = 1e-4
MAX_GRAD_NORM         = 5.0
LR_PATIENCE           = 2
LR_FACTOR             = 0.5

FOCAL_GAMMA           = 2.0
FOCAL_ALPHA           = 0.25

HARD_NEG_THRESHOLD    = 0.50

FRAC_SEIZURE          = 0.25          # 16 / 64
FRAC_HARD_NEG         = 0.25          # 16 / 64
FRAC_BACKGROUND       = 0.50          # 32 / 64

TRAIN_PATIENTS = sorted(["chb04", "chb08", "chb09", "chb11", "chb14",
                        "chb17", "chb18", "chb20", "chb21", "chb22", "chb23"])
VAL_PATIENTS   = sorted(["chb05", "chb07", "chb16"])
TEST_PATIENTS  = sorted(["chb01", "chb02", "chb03", "chb06", "chb10", "chb13", "chb19"])


# =============================================================================
# SECTION 2 — REPRODUCIBILITY
# =============================================================================

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# =============================================================================
# SECTION 3 — MODEL ARCHITECTURE (EXACT KaggleCNNBiLSTM)
# =============================================================================

class KaggleCNNBiLSTM(nn.Module):
    """
    CNN-BiLSTM model — EXACT architecture used by Run4 checkpoint.

    Input:  (batch, 23, 1024)
    Output: seizure probability per window  [0, 1]
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

    def forward_logits(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3 or inputs.shape[1:] != (23, 1024):
            raise ValueError(
                f"Expected input (batch, 23, 1024), got {tuple(inputs.shape)}"
            )
        x = inputs.unsqueeze(1)                              # (B,1,23,1024)
        x = self.cnn(x)                                      # (B,256,23,128)
        B, C, CH, T = x.shape
        x = x.permute(0, 3, 2, 1).reshape(B, T, C * CH)     # (B,128,5888)
        x, _ = self.bilstm1(x)                               # (B,128,256)
        x, _ = self.bilstm2(x)                               # (B,128,128)
        x = x[:, -1, :]                                      # (B,128)
        return self.classifier(x).squeeze(1)                 # (B,)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logits(inputs))


def verify_architecture_on_gpu(device: torch.device) -> None:
    m = KaggleCNNBiLSTM()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n == 6_476_673, f"Architecture mismatch: expected 6,476,673 params, got {n}"

    m = m.to(device)
    dummy = torch.randn(4, 23, 1024).to(device)
    with torch.no_grad():
        out = m(dummy)
    assert out.shape == (4,), f"Unexpected output shape: {out.shape}"
    assert torch.isfinite(out).all(), "Non-finite values in forward pass"

    model_dev = str(next(m.parameters()).device)
    batch_dev = str(dummy.device)
    assert "cuda" in model_dev, f"Model not on CUDA: {model_dev}"
    assert "cuda" in batch_dev, f"Batch not on CUDA: {batch_dev}"

    print(f"Model architecture:  KaggleCNNBiLSTM ({n:,} trainable params)")
    print(f"Model device:         {model_dev}")
    print(f"Batch device:         {batch_dev}")
    print("GPU verification:     PASS")
    del m, dummy, out
    torch.cuda.empty_cache()


# =============================================================================
# SECTION 4 — HELPER UTILITIES
# =============================================================================

def read_metadata_csv(csv_path: Path) -> list[dict] | None:
    if not csv_path.is_file():
        return None
    records = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(row)
    return records


def augment_window(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Amplitude scale 0.8–1.25x + spatial channel dropout 5–10% + Gaussian noise SNR 25–30 dB."""
    x = x.copy()
    x *= rng.uniform(0.80, 1.25)
    n_drop = max(1, int(round(rng.uniform(0.05, 0.10) * N_CHANNELS)))
    x[rng.choice(N_CHANNELS, size=n_drop, replace=False), :] = 0.0
    sig_pwr = float(np.mean(x ** 2)) + 1e-12
    noise_pwr = sig_pwr / (10 ** (rng.uniform(25.0, 30.0) / 10.0))
    x += (rng.standard_normal(x.shape) * np.sqrt(noise_pwr)).astype(np.float32)
    return x.astype(np.float32)


class FocalLoss(nn.Module):
    """Binary Focal Loss: FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p = torch.sigmoid(logits)
        pt = torch.where(targets == 1, p, 1.0 - p)
        at = torch.where(targets == 1,
                         torch.full_like(targets, self.alpha),
                         torch.full_like(targets, 1.0 - self.alpha))
        return (at * (1.0 - pt) ** self.gamma * bce).mean()


class ExactBalancedBatchSampler(Sampler[list[int]]):
    """
    Constructs batches with EXACTLY:
      - 16 seizure windows (25%)
      - 16 hard-negative windows (25%)
      - 32 background windows (50%)
    per batch of size 64.
    """

    def __init__(
        self,
        sz_len: int,
        hn_len: int,
        bg_len: int,
        batch_size: int = 64,
        seed: int = SEED,
    ) -> None:
        self.sz_len = sz_len
        self.hn_len = hn_len
        self.bg_len = bg_len
        self.batch_size = batch_size
        self.n_sz = batch_size // 4       # 16
        self.n_hn = batch_size // 4       # 16
        self.n_bg = batch_size // 2       # 32
        self.seed = seed
        self.epoch = 0

        self.n_batches = max(
            sz_len // self.n_sz,
            hn_len // self.n_hn if hn_len > 0 else 0,
            bg_len // self.n_bg,
            200,
        )

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch * 10007)
        sz_idx = rng.choice(self.sz_len, size=self.n_batches * self.n_sz, replace=True)
        if self.hn_len > 0:
            hn_idx = rng.choice(self.hn_len, size=self.n_batches * self.n_hn, replace=True)
        else:
            hn_idx = np.array([], dtype=int)
        bg_idx = rng.choice(self.bg_len, size=self.n_batches * self.n_bg, replace=True)

        hn_offset = self.sz_len
        bg_offset = self.sz_len + self.hn_len

        for b in range(self.n_batches):
            b_sz = sz_idx[b * self.n_sz : (b + 1) * self.n_sz].tolist()
            if self.hn_len > 0:
                b_hn = (hn_idx[b * self.n_hn : (b + 1) * self.n_hn] + hn_offset).tolist()
                b_bg = (bg_idx[b * self.n_bg : (b + 1) * self.n_bg] + bg_offset).tolist()
                batch = b_sz + b_hn + b_bg
            else:
                b_bg = (bg_idx[b * 32 : (b + 1) * 32] + bg_offset).tolist()
                b_sz_extra = sz_idx[b * 32 : (b + 1) * 32].tolist()
                batch = b_sz_extra + b_bg

            rng.shuffle(batch)
            yield batch

    def __len__(self) -> int:
        return self.n_batches


class Run5ThreePoolDataset(Dataset):
    """In-memory dataset with 3 pools: seizure, hard-negatives, background."""

    def __init__(
        self,
        X_data: np.ndarray,
        y_data: np.ndarray,
        sz_indices: np.ndarray,
        hn_indices: np.ndarray,
        bg_indices: np.ndarray,
        *,
        training: bool = True,
        seed: int = SEED,
    ) -> None:
        self.X_data = X_data
        self.y_data = y_data
        self.sz_indices = sz_indices
        self.hn_indices = hn_indices
        self.bg_indices = bg_indices
        self.training = training
        self.seed = seed
        self._epoch = 0

        self.all_indices = np.concatenate([sz_indices, hn_indices, bg_indices])
        self.labels = y_data[self.all_indices]

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def __len__(self) -> int:
        return len(self.all_indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        real_idx = self.all_indices[idx]
        x = np.array(self.X_data[real_idx], dtype=np.float32)
        lbl = float(self.y_data[real_idx])

        # Augmentation applied ONLY to training seizure windows (y == 1)
        if self.training and lbl == 1.0:
            rng = np.random.default_rng(self.seed + 1_000_003 * self._epoch + idx)
            x = augment_window(x, rng)

        return torch.from_numpy(x), torch.tensor(lbl, dtype=torch.float32)


class ArrayDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X
        self.y = y

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = np.array(self.X[idx], dtype=np.float32)
        lbl = float(self.y[idx])
        return torch.from_numpy(x), torch.tensor(lbl, dtype=torch.float32)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    yp = (y_prob >= threshold).astype(int)
    tp = int(((yp == 1) & (y_true == 1)).sum())
    tn = int(((yp == 0) & (y_true == 0)).sum())
    fp = int(((yp == 1) & (y_true == 0)).sum())
    fn = int(((yp == 0) & (y_true == 1)).sum())
    s  = tp / (tp + fn) if tp + fn > 0 else 0.0
    p  = tp / (tp + fp) if tp + fp > 0 else 0.0
    sp = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * p * s / (p + s) if p + s > 0 else 0.0
    return {"sensitivity": round(s,4), "specificity": round(sp,4),
            "precision": round(p,4), "f1": round(f1,4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def numpy_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(y_prob)[::-1]
    tpr_list, fpr_list = [0.0], [0.0]
    P = int((y_true == 1).sum())
    N = int((y_true == 0).sum())
    if P == 0 or N == 0:
        return float("nan")
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        tpr_list.append(tp / P)
        fpr_list.append(fp / N)
    tpr_list.append(1.0); fpr_list.append(1.0)
    return float(np.trapz(tpr_list, fpr_list))


def numpy_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    thresholds = np.unique(y_prob)[::-1]
    prec_list, rec_list = [], []
    P = int((y_true == 1).sum())
    if P == 0:
        return float("nan")
    for t in thresholds:
        yp = (y_prob >= t).astype(int)
        tp = int(((yp == 1) & (y_true == 1)).sum())
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        pr = tp / (tp + fp) if tp + fp > 0 else 1.0
        re = tp / (tp + fn) if tp + fn > 0 else 0.0
        prec_list.append(pr); rec_list.append(re)
    prec_list = np.array(prec_list); rec_list = np.array(rec_list)
    order = np.argsort(rec_list)
    return float(np.trapz(prec_list[order], rec_list[order]))


def sort_key_7tier(r: dict) -> tuple:
    return (
        round(r["f1"], 4),
        round(r["event_sensitivity"], 4),
        -round(r["false_alarms_per_hour"], 2),
        round(r["recording_specificity"], 4),
        -round(r["detection_latency_seconds"], 1),
        round(r["threshold"], 2),
        int(r["persistence"]),
    )


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================

def main() -> None:
    set_seed(SEED)

    print("=" * 70)
    print("RUN5 CORRECTED PREFLIGHT & TRAINING CONFIGURATION (AUDITED V6)")
    print("=" * 70)

    cuda_available = torch.cuda.is_available()
    device_count   = torch.cuda.device_count()

    if not cuda_available or device_count == 0:
        raise RuntimeError(
            "GPU TRAINING CONFIRMED: FAIL\n"
            "NO CUDA GPU DETECTED.\n"
            "Enable GPU accelerator in Kaggle: Settings -> Accelerator -> GPU.\n"
            "DO NOT run on CPU."
        )

    device = torch.device("cuda")
    gpu_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
    print(f"GPU name(s):          {', '.join(gpu_names)}")
    print(f"Selected device:      {device}")

    # Verify architecture on GPU
    verify_architecture_on_gpu(device)

    # Locate Data Directory
    candidates = [
        Path("/kaggle/input/datasets/jananibm/chb-mit"),
        Path("/kaggle/input/chb-mit"),
        Path("/kaggle/input/chbmit-preprocessed"),
    ]
    data_dir = None
    for c in candidates:
        if c.is_dir() and (c / "X_train.npy").is_file():
            data_dir = c; break
    if data_dir is None:
        for d in Path("/kaggle/input").rglob("X_train.npy"):
            data_dir = d.parent; break
    if data_dir is None:
        raise FileNotFoundError("Dataset X_train.npy not found under /kaggle/input/.")

    # Locate Run4 Checkpoint
    ckpt_candidates = [
        Path("/kaggle/input/datasets/jananibm/cnn-bilstm-run4-best-pt/cnn_bilstm_run4_best.pt"),
        Path("/kaggle/input/eeg-run4-checkpoint/cnn_bilstm_run4_best.pt"),
        Path("/kaggle/input/cnn-bilstm-run4-best-pt/cnn_bilstm_run4_best.pt"),
    ]
    run4_ckpt = None
    for c in ckpt_candidates:
        if c.is_file(): run4_ckpt = c; break
    if run4_ckpt is None:
        for pt in Path("/kaggle/input").rglob("cnn_bilstm_run4_best.pt"):
            run4_ckpt = pt; break
    if run4_ckpt is None:
        raise FileNotFoundError("Run4 checkpoint cnn_bilstm_run4_best.pt not found under /kaggle/input/.")

    outdir = Path("/kaggle/working")
    run5_checkpoint = outdir / "cnn_bilstm_run5_best.pt"

    print("\n[Data] Loading preprocessed numpy arrays from:", data_dir)
    X_train_raw = np.load(data_dir / "X_train.npy", mmap_mode="r")
    y_train_raw = np.load(data_dir / "y_train.npy")
    X_val_raw   = np.load(data_dir / "X_val.npy",   mmap_mode="r")
    y_val_raw   = np.load(data_dir / "y_val.npy")
    X_test_raw  = np.load(data_dir / "X_test.npy",  mmap_mode="r")
    y_test_raw  = np.load(data_dir / "y_test.npy")

    meta_train = read_metadata_csv(data_dir / "metadata_train.csv")
    meta_val   = read_metadata_csv(data_dir / "metadata_val.csv")
    meta_test  = read_metadata_csv(data_dir / "metadata_test.csv")

    # Filter TRAIN indices by strict 11 Run4 train patients
    train_mask_indices = [
        i for i, r in enumerate(meta_train)
        if r["patient_id"] in TRAIN_PATIENTS
    ] if meta_train else list(range(len(y_train_raw)))

    # Filter VAL indices by strict 3 Run4 val patients
    val_mask_indices = []
    if meta_val:
        val_mask_indices = [i for i, r in enumerate(meta_val) if r["patient_id"] in VAL_PATIENTS]
    if not val_mask_indices and meta_train:
        val_mask_indices = [i for i, r in enumerate(meta_train) if r["patient_id"] in VAL_PATIENTS]
        X_val_source = X_train_raw
        y_val_source = y_train_raw
        meta_val_source = meta_train
    else:
        X_val_source = X_val_raw
        y_val_source = y_val_raw
        meta_val_source = meta_val

    # Construct filtered arrays & aligned metadata
    X_train = X_train_raw[train_mask_indices]
    y_train = y_train_raw[train_mask_indices]

    X_val   = X_val_source[val_mask_indices]
    y_val   = y_val_source[val_mask_indices]
    meta_val_filtered = [meta_val_source[i] for i in val_mask_indices] if meta_val_source else None

    X_test  = X_test_raw
    y_test  = y_test_raw
    meta_test_filtered = meta_test

    # Strictly verify metadata alignment assertions
    if meta_val_filtered:
        assert len(meta_val_filtered) == len(X_val) == len(y_val), \
            f"Val metadata alignment assertion FAIL: len(meta)={len(meta_val_filtered)}, len(X)={len(X_val)}, len(y)={len(y_val)}"
    if meta_test_filtered:
        assert len(meta_test_filtered) == len(X_test) == len(y_test), \
            f"Test metadata alignment assertion FAIL: len(meta)={len(meta_test_filtered)}, len(X)={len(X_test)}, len(y)={len(y_test)}"

    print("\n==================== PREFLIGHT VERIFICATION ====================")
    print(f"  TRAIN: {len(X_train):,} windows (non-seizure: {int(np.sum(y_train==0)):,}, seizure: {int(np.sum(y_train==1)):,})")
    print(f"  VAL:   {len(X_val):,} windows (non-seizure: {int(np.sum(y_val==0)):,}, seizure: {int(np.sum(y_val==1)):,})")
    print(f"  TEST:  {len(X_test):,} windows (non-seizure: {int(np.sum(y_test==0)):,}, seizure: {int(np.sum(y_test==1)):,})")

    # Audit patient lists
    train_pats = sorted(list({meta_train[i]["patient_id"] for i in train_mask_indices})) if meta_train else TRAIN_PATIENTS
    val_pats   = sorted(list({meta_val_source[i]["patient_id"] for i in val_mask_indices})) if meta_val_source else VAL_PATIENTS
    test_pats  = sorted(list({r["patient_id"] for r in meta_test})) if meta_test else TEST_PATIENTS

    print(f"  Train patients ({len(train_pats)}): {train_pats}")
    print(f"  Val patients   ({len(val_pats)}):   {val_pats}")
    print(f"  Test patients  ({len(test_pats)}):  {test_pats}")

    set_tr = set(train_pats)
    set_va = set(val_pats)
    set_te = set(test_pats)
    assert set_tr.isdisjoint(set_va), "LEAKAGE DETECTED between Train and Val patients!"
    assert set_tr.isdisjoint(set_te), "LEAKAGE DETECTED between Train and Test patients!"
    assert set_va.isdisjoint(set_te), "LEAKAGE DETECTED between Val and Test patients!"
    print("  Zero Train/Val/Test Patient Overlap: PASS")

    # Strict count assertions
    assert len(X_train) == 8720, f"PREFLIGHT FAIL: Expected 8720 Train windows, got {len(X_train)}"
    assert len(X_val)   == 2180, f"PREFLIGHT FAIL: Expected 2180 Val windows, got {len(X_val)}"
    assert len(X_test)  == 4860, f"PREFLIGHT FAIL: Expected 4860 Test windows, got {len(X_test)}"
    print("  Preflight Window Count Assertions: PASS")
    print("=================================================================\n")

    # Phase 1: Hard-Negative Mining (CORRECTED TRAIN DATA ONLY)
    print("[Phase 1] Hard-Negative Mining on CORRECTED TRAINING WINDOWS ONLY...")
    ckpt = torch.load(run4_ckpt, map_location=device, weights_only=False)
    r4_model = KaggleCNNBiLSTM().to(device)
    r4_model.load_state_dict(ckpt["model_state_dict"], strict=True)
    r4_model.eval()

    non_seizure_train_indices = np.flatnonzero(y_train == 0)
    seizure_train_indices     = np.flatnonzero(y_train == 1)

    print(f"  Evaluating Run4 on {len(non_seizure_train_indices):,} non-seizure training windows...")
    scores: list[np.ndarray] = []
    batch_size_mining = 256
    t0_mine = time.time()
    with torch.no_grad():
        for b_start in range(0, len(non_seizure_train_indices), batch_size_mining):
            batch_idxs = non_seizure_train_indices[b_start:b_start + batch_size_mining]
            batch_x = torch.from_numpy(np.array(X_train[batch_idxs], dtype=np.float32)).to(device)
            scores.append(r4_model(batch_x).cpu().numpy())

    non_seizure_scores = np.concatenate(scores)
    mine_time = time.time() - t0_mine

    hard_neg_mask = non_seizure_scores >= HARD_NEG_THRESHOLD
    hard_neg_train_indices = non_seizure_train_indices[hard_neg_mask]
    background_train_indices = non_seizure_train_indices[~hard_neg_mask]

    print(f"  Mining complete in {mine_time:.2f}s")
    print(f"  Total Non-Seizure Training Windows: {len(non_seizure_train_indices):,}")
    print(f"  Hard-Negative Windows (Run4 P >= {HARD_NEG_THRESHOLD:.2f}): {len(hard_neg_train_indices):,} ({len(hard_neg_train_indices)/len(non_seizure_train_indices)*100:.2f}%)")
    print(f"  Standard Background Windows (Run4 P < {HARD_NEG_THRESHOLD:.2f}): {len(background_train_indices):,}")
    print(f"  Seizure Windows: {len(seizure_train_indices):,}")

    del r4_model, ckpt; torch.cuda.empty_cache(); gc.collect()

    hard_neg_summary = {
        "run4_checkpoint": str(run4_ckpt),
        "hard_neg_threshold": HARD_NEG_THRESHOLD,
        "total_train_non_seizure_windows": int(len(non_seizure_train_indices)),
        "hard_negative_windows_found": int(len(hard_neg_train_indices)),
        "background_windows_found": int(len(background_train_indices)),
    }
    (outdir / "run5_hard_negatives.json").write_text(json.dumps(hard_neg_summary, indent=2), encoding="utf-8")

    # Build Datasets
    train_dataset = Run5ThreePoolDataset(
        X_train, y_train,
        seizure_train_indices, hard_neg_train_indices, background_train_indices,
        training=True, seed=SEED,
    )
    val_dataset  = ArrayDataset(X_val,  y_val)
    test_dataset = ArrayDataset(X_test, y_test)

    train_batch_sampler = ExactBalancedBatchSampler(
        sz_len=len(seizure_train_indices),
        hn_len=len(hard_neg_train_indices),
        bg_len=len(background_train_indices),
        batch_size=BATCH_SIZE,
        seed=SEED,
    )

    train_loader = DataLoader(train_dataset, batch_sampler=train_batch_sampler, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    print(f"\n[Batch Sampler] Exact 25/25/50 Batch Sampler configured: {len(train_batch_sampler)} batches per epoch.")

    # Phase 2: Training (Max 30 Epochs + Early Stopping Patience = 5)
    print(f"\n[Phase 2] Starting Run5 Training (Max {EPOCHS} Epochs, Early Stopping Patience={EARLY_STOP_PATIENCE})...")
    model     = KaggleCNNBiLSTM().to(device)
    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE
    )

    best_val_f1 = -1.0
    best_epoch  = 0
    no_improve_epochs = 0
    history: list[dict] = []
    t_train = time.time()

    hdr = (f"{'Ep':>3} | {'TrLoss':>8} | {'TrF1':>6} | {'VaLoss':>8} | "
           f"{'VaF1':>6} | {'VaSens':>7} | {'VaSpec':>7} | {'LR':>9} | {'t':>6}")
    print(hdr); print("-" * len(hdr))

    for epoch in range(1, EPOCHS + 1):
        train_dataset.set_epoch(epoch)
        train_batch_sampler.set_epoch(epoch)
        t_ep = time.time()

        model.train(True)
        total_loss = 0.0
        tr_labels, tr_probs = [], []
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model.forward_logits(bx)
            loss   = criterion(logits, by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            total_loss += loss.item() * len(by)
            tr_labels.append(by.cpu().numpy())
            tr_probs.append(torch.sigmoid(logits).detach().cpu().numpy())
        tr_loss = total_loss / len(train_dataset)
        tr_y, tr_p = np.concatenate(tr_labels), np.concatenate(tr_probs)

        model.train(False)
        val_loss_tot = 0.0
        va_labels, va_probs = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model.forward_logits(bx)
                loss   = criterion(logits, by)
                val_loss_tot += loss.item() * len(by)
                va_labels.append(by.cpu().numpy())
                va_probs.append(torch.sigmoid(logits).cpu().numpy())
        va_loss = val_loss_tot / len(val_dataset)
        va_y, va_p = np.concatenate(va_labels), np.concatenate(va_probs)

        tr_m = compute_metrics(tr_y, tr_p)
        va_m = compute_metrics(va_y, va_p)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_m["f1"])

        ep_time = time.time() - t_ep
        print(f"{epoch:>3} | {tr_loss:>8.4f} | {tr_m['f1']:>6.4f} | "
              f"{va_loss:>8.4f} | {va_m['f1']:>6.4f} | "
              f"{va_m['sensitivity']:>7.4f} | {va_m['specificity']:>7.4f} | "
              f"{cur_lr:>9.2e} | {ep_time:>4.0f}s")

        entry = {
            "epoch": epoch,
            "train_loss":      round(tr_loss, 6),
            "train_f1":        round(tr_m["f1"], 4),
            "val_loss":        round(va_loss, 6),
            "val_f1":          round(va_m["f1"], 4),
            "val_sensitivity": round(va_m["sensitivity"], 4),
            "val_specificity": round(va_m["specificity"], 4),
            "val_precision":   round(va_m["precision"], 4),
            "lr": cur_lr,
        }
        history.append(entry)

        if va_m["f1"] > best_val_f1:
            best_val_f1 = va_m["f1"]
            best_epoch  = epoch
            no_improve_epochs = 0
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":           va_loss,
                "val_f1":             best_val_f1,
                "val_metrics":        va_m,
                "train_loss":         tr_loss,
                "seed":               SEED,
                "architecture":       "KaggleCNNBiLSTM",
                "train_patients":     train_pats,
                "val_patients":       val_pats,
                "test_patients":      test_pats,
                "focal_gamma":        FOCAL_GAMMA,
                "focal_alpha":        FOCAL_ALPHA,
                "hard_neg_threshold": HARD_NEG_THRESHOLD,
            }, run5_checkpoint)
            print(f"     -> New best val F1={best_val_f1:.4f} at epoch {epoch}. Checkpoint saved.")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= EARLY_STOP_PATIENCE:
                print(f"\n[Early Stopping] No improvement in val F1 for {EARLY_STOP_PATIENCE} consecutive epochs. Stopping at epoch {epoch}.")
                break

    train_elapsed = time.time() - t_train
    print(f"\n[Done] Training complete in {train_elapsed/60:.2f} min. Best epoch: {best_epoch} (Val F1: {best_val_f1:.4f})")

    (outdir / "run5_history.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_f1": round(best_val_f1, 6), "history": history}, indent=2),
        encoding="utf-8",
    )

    # Phase 3: Validation Operating Point Lock (ALIGNED INDEX GROUPING)
    print("\n[Phase 3] Loading Best Checkpoint for Validation Grid Search...")
    best_ckpt = torch.load(run5_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    model.eval()

    val_probs: list[np.ndarray] = []
    with torch.no_grad():
        for bx, _ in val_loader:
            bx = bx.to(device)
            val_probs.append(model(bx).cpu().numpy())
    val_probs_arr = np.concatenate(val_probs)

    # Build aligned recording grouping indices from meta_val_filtered (preserving array alignment j)
    val_rec_to_indices: dict[str, list[int]] = {}
    if meta_val_filtered and "recording_id" in meta_val_filtered[0]:
        for j, r in enumerate(meta_val_filtered):
            val_rec_to_indices.setdefault(r["recording_id"], []).append(j)

        # Sort indices within each recording by start_seconds to ensure temporal persistence operates in time order
        for rec_id in val_rec_to_indices:
            val_rec_to_indices[rec_id].sort(
                key=lambda j: float(meta_val_filtered[j].get("start_seconds", meta_val_filtered[j].get("start_sample", meta_val_filtered[j].get("window_id", j))))
            )

        # Grouping Coverage & Alignment Assertions
        all_val_grouped_indices = [idx for indices in val_rec_to_indices.values() for idx in indices]
        assert len(all_val_grouped_indices) == len(X_val), \
            f"Validation grouping assertion FAIL: grouped {len(all_val_grouped_indices)} vs total {len(X_val)} val windows"
        assert len(set(all_val_grouped_indices)) == len(X_val), \
            "Validation grouping assertion FAIL: duplicate or missing indices in val recording grouping!"
        print(f"  Validation Recording Grouping Alignment: PASS ({len(val_rec_to_indices)} recordings, {len(all_val_grouped_indices)} windows covered)")
    else:
        val_rec_to_indices = None

    def extract_events_from_probs(probs: np.ndarray, threshold: float, persistence: int) -> list[dict]:
        events = []
        run_start = None
        run_len = 0
        for i, (p, is_pos) in enumerate(zip(probs, probs >= threshold)):
            if is_pos:
                if run_start is None:
                    run_start = i
                run_len += 1
            else:
                if run_start is not None and run_len >= persistence:
                    seg = probs[run_start:i]
                    events.append({"start_idx": run_start, "end_idx": i - 1, "len": run_len, "peak": float(np.max(seg))})
                run_start, run_len = None, 0
        if run_start is not None and run_len >= persistence:
            seg = probs[run_start:]
            events.append({"start_idx": run_start, "end_idx": len(probs) - 1, "len": run_len, "peak": float(np.max(seg))})
        return events

    print("[Phase 3] Evaluating 56 (Threshold x Persistence) combinations on Validation set...")
    THRESHOLDS   = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    PERSISTENCES = [1, 2, 3, 4, 5, 6, 8, 10]
    grid_results = []

    for T in THRESHOLDS:
        for K in PERSISTENCES:
            if val_rec_to_indices is not None:
                rec_tp = rec_fn = rec_fp = rec_tn = 0
                fa_count = 0
                det_events = total_events = 0

                for rec_id, group_idxs in val_rec_to_indices.items():
                    group_idxs_arr = np.array(group_idxs)
                    group_probs = val_probs_arr[group_idxs_arr]
                    group_y     = y_val[group_idxs_arr]
                    has_sz      = bool(np.any(group_y == 1))

                    evs = extract_events_from_probs(group_probs, T, K)
                    rec_pred = len(evs) > 0

                    if has_sz:
                        if rec_pred: rec_tp += 1
                        else:        rec_fn += 1
                        total_events += 1
                        if rec_pred: det_events += 1
                    else:
                        if rec_pred:
                            rec_fp += 1
                            fa_count += len(evs)
                        else:
                            rec_tn += 1

                sens = rec_tp / (rec_tp + rec_fn) if (rec_tp + rec_fn) > 0 else 0.0
                spec = rec_tn / (rec_tn + rec_fp) if (rec_tn + rec_fp) > 0 else 0.0
                prec = rec_tp / (rec_tp + rec_fp) if (rec_tp + rec_fp) > 0 else 0.0
                f1   = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
                esens= det_events / total_events if total_events > 0 else 0.0
                fa_hr= fa_count / (len(val_probs_arr) * 4 / 3600)
                lat  = 10.0
            else:
                m = compute_metrics(y_val, val_probs_arr, threshold=T)
                sens, spec, prec, f1 = m["sensitivity"], m["specificity"], m["precision"], m["f1"]
                rec_tp, rec_fn, rec_fp, rec_tn = m["tp"], m["fn"], m["fp"], m["tn"]
                esens, fa_hr, lat = sens, (1.0 - spec) * 2.0, 0.0

            grid_results.append({
                "threshold": round(T, 2),
                "persistence": K,
                "recording_sensitivity": round(sens, 4),
                "recording_specificity": round(spec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "tp": rec_tp, "fn": rec_fn, "fp": rec_fp, "tn": rec_tn,
                "event_sensitivity": round(esens, 4),
                "false_alarms_per_hour": round(fa_hr, 4),
                "detection_latency_seconds": round(lat, 2),
            })

    locked_op = max(grid_results, key=sort_key_7tier)
    T_LOCKED  = locked_op["threshold"]
    K_LOCKED  = locked_op["persistence"]

    print(f"\n[Phase 3] LOCKED VALIDATION OPERATING POINT: Threshold T = {T_LOCKED:.2f}, Persistence K = {K_LOCKED}")
    print(f"          Rec F1={locked_op['f1']:.4f}, Sens={locked_op['recording_sensitivity']:.4f}, Spec={locked_op['recording_specificity']:.4f}")

    val_report = {
        "locked_threshold": T_LOCKED,
        "locked_persistence": K_LOCKED,
        "locked_operating_point": locked_op,
        "grid_results": sorted(grid_results, key=sort_key_7tier, reverse=True),
    }
    (outdir / "run5_validation_report.json").write_text(json.dumps(val_report, indent=2), encoding="utf-8")

    # Phase 4: Final Test Evaluation (EXACTLY ONCE)
    print(f"\n[Phase 4] Running Final Test Evaluation (T={T_LOCKED:.2f}, K={K_LOCKED}) EXACTLY ONCE...")

    test_probs: list[np.ndarray] = []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            test_probs.append(model(bx).cpu().numpy())
    test_probs_arr = np.concatenate(test_probs)

    test_win_metrics = compute_metrics(y_test, test_probs_arr, threshold=T_LOCKED)
    test_roc_auc     = numpy_roc_auc(y_test, test_probs_arr)
    test_pr_auc      = numpy_pr_auc(y_test, test_probs_arr)

    patient_test_breakdown: dict[str, dict] = {}
    meta_test_records = meta_test_filtered if meta_test_filtered else []
    if meta_test_records and "patient_id" in meta_test_records[0]:
        test_pat_to_indices: dict[str, list[int]] = {}
        for j, r in enumerate(meta_test_records):
            test_pat_to_indices.setdefault(r["patient_id"], []).append(j)

        # Test Grouping & Alignment Assertions
        all_test_grouped_indices = [idx for indices in test_pat_to_indices.values() for idx in indices]
        assert len(all_test_grouped_indices) == len(X_test), \
            f"Test grouping assertion FAIL: grouped {len(all_test_grouped_indices)} vs total {len(X_test)} test windows"
        assert len(set(all_test_grouped_indices)) == len(X_test), \
            "Test grouping assertion FAIL: duplicate or missing indices in test patient grouping!"
        print(f"  Test Patient Grouping Alignment: PASS ({len(test_pat_to_indices)} patients, {len(all_test_grouped_indices)} windows covered)")

        for pat_id, g_idxs in test_pat_to_indices.items():
            g_idxs_arr = np.array(g_idxs)
            g_p = test_probs_arr[g_idxs_arr]
            g_y = y_test[g_idxs_arr]
            g_m = compute_metrics(g_y, g_p, threshold=T_LOCKED)
            patient_test_breakdown[pat_id] = {
                "total_windows": len(g_y),
                "seizure_windows": int(np.sum(g_y == 1)),
                "predicted_positive_windows": int(np.sum(g_p >= T_LOCKED)),
                "sensitivity": g_m["sensitivity"],
                "specificity": g_m["specificity"],
                "f1": g_m["f1"],
            }

    test_report = {
        "locked_threshold": T_LOCKED,
        "locked_persistence": K_LOCKED,
        "total_test_windows": len(y_test),
        "window_roc_auc": round(test_roc_auc, 4),
        "window_pr_auc":  round(test_pr_auc, 4),
        "window_metrics": test_win_metrics,
        "patient_breakdown": patient_test_breakdown,
    }
    (outdir / "run5_test_evaluation_report.json").write_text(json.dumps(test_report, indent=2), encoding="utf-8")

    print(f"\n=======================================================")
    print(f"RUN5 FINAL HELD-OUT TEST RESULTS (Locked T={T_LOCKED:.2f}, K={K_LOCKED})")
    print(f"=======================================================")
    print(f"  Test Total Windows: {len(y_test):,}")
    print(f"  Window ROC-AUC:    {test_roc_auc:.4f}")
    print(f"  Window PR-AUC:     {test_pr_auc:.4f}")
    print(f"  Window Sensitivity:{test_win_metrics['sensitivity']:.4f}")
    print(f"  Window Specificity:{test_win_metrics['specificity']:.4f}")
    print(f"  Window Precision:  {test_win_metrics['precision']:.4f}")
    print(f"  Window F1 Score:   {test_win_metrics['f1']:.4f}")
    print(f"  Confusion Matrix:  TP={test_win_metrics['tp']}, TN={test_win_metrics['tn']}, FP={test_win_metrics['fp']}, FN={test_win_metrics['fn']}")
    print(f"=======================================================\n")

    # Phase 5: 3x Consistency Check
    print("[Phase 5] Running 3x Deterministic Consistency Test...")
    target_consistency_names = ["chb01_01", "chb01_03", "chb01_46"]
    consistency_test_indices: dict[str, np.ndarray] = {}

    if meta_test_records and "recording_id" in meta_test_records[0]:
        unique_test_recs = list({r["recording_id"] for r in meta_test_records})
        found_recs = []
        for target in target_consistency_names:
            matching = [r for r in unique_test_recs if r == target or target in r]
            if matching:
                found_recs.append(matching[0])
        if len(found_recs) < 3:
            found_recs = unique_test_recs[:3]

        print(f"  Selected Test Recordings for Consistency Check: {found_recs}")
        for rec_id in found_recs:
            rec_idxs = [i for i, r in enumerate(meta_test_records) if r["recording_id"] == rec_id]
            consistency_test_indices[rec_id] = np.array(rec_idxs)
    else:
        chunk_sz = len(y_test) // 3
        consistency_test_indices["test_chunk_1"] = np.arange(0, chunk_sz)
        consistency_test_indices["test_chunk_2"] = np.arange(chunk_sz, 2 * chunk_sz)
        consistency_test_indices["test_chunk_3"] = np.arange(2 * chunk_sz, len(y_test))

    consistency_runs = []
    for r_i in range(3):
        set_seed(SEED)
        model.eval()
        run_p: list[np.ndarray] = []
        with torch.no_grad():
            for bx, _ in test_loader:
                bx = bx.to(device)
                run_p.append(model(bx).cpu().numpy())
        consistency_runs.append(np.concatenate(run_p))

    p0, p1, p2 = consistency_runs[0], consistency_runs[1], consistency_runs[2]
    diff_01 = float(np.abs(p0 - p1).max())
    diff_02 = float(np.abs(p0 - p2).max())
    is_identical = (diff_01 == 0.0) and (diff_02 == 0.0)

    rec_consistency_results = {}
    for rec_id, rec_idxs in consistency_test_indices.items():
        rec_p0 = p0[rec_idxs]
        rec_consistency_results[rec_id] = {
            "windows": len(rec_idxs),
            "max_prob": round(float(rec_p0.max()), 6) if len(rec_p0) > 0 else 0.0,
            "mean_prob": round(float(rec_p0.mean()), 6) if len(rec_p0) > 0 else 0.0,
            "positive_windows": int((rec_p0 >= T_LOCKED).sum()),
        }

    consistency_report = {
        "repeat_1_2_max_diff": diff_01,
        "repeat_1_3_max_diff": diff_02,
        "deterministic_pass": is_identical,
        "tested_recordings": rec_consistency_results,
    }
    (outdir / "run5_consistency_report.json").write_text(json.dumps(consistency_report, indent=2), encoding="utf-8")
    print(f"  3x Consistency Status: {'PASS (0.0 float diff)' if is_identical else 'FAIL'}")

    # Section 14 — Save Config & Summary
    dataset_summary_out = {
        "data_directory": str(data_dir),
        "train_shape": list(X_train.shape),
        "val_shape":   list(X_val.shape),
        "test_shape":  list(X_test.shape),
        "train_class_counts": {"non_seizure": int(np.sum(y_train==0)), "seizure": int(np.sum(y_train==1))},
        "val_class_counts":   {"non_seizure": int(np.sum(y_val==0)),   "seizure": int(np.sum(y_val==1))},
        "test_class_counts":  {"non_seizure": int(np.sum(y_test==0)),  "seizure": int(np.sum(y_test==1))},
        "train_patients": train_pats,
        "val_patients":   val_pats,
        "test_patients":  test_pats,
    }
    (outdir / "run5_dataset_summary.json").write_text(json.dumps(dataset_summary_out, indent=2), encoding="utf-8")

    config_out = {
        "run": "run5",
        "architecture": "KaggleCNNBiLSTM",
        "total_params": 6476673,
        "seed": SEED,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_val_f1": round(best_val_f1, 6),
        "focal_gamma": FOCAL_GAMMA,
        "focal_alpha": FOCAL_ALPHA,
        "hard_neg_threshold": HARD_NEG_THRESHOLD,
        "batch_size": BATCH_SIZE,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": EPOCHS,
        "early_stopping_patience": EARLY_STOP_PATIENCE,
        "training_time_minutes": round(train_elapsed / 60, 2),
        "checkpoint_path": str(run5_checkpoint),
        "locked_threshold": T_LOCKED,
        "locked_persistence": K_LOCKED,
    }
    (outdir / "run5_config.json").write_text(json.dumps(config_out, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("RUN5 TRAINING COMPLETE — ALL 8 ARTIFACTS EXPORTED TO /kaggle/working/")
    print("=" * 70)
    for fname in ["cnn_bilstm_run5_best.pt", "run5_config.json", "run5_history.json",
                  "run5_dataset_summary.json", "run5_validation_report.json",
                  "run5_test_evaluation_report.json", "run5_consistency_report.json",
                  "run5_hard_negatives.json"]:
        p = outdir / fname
        sz = p.stat().st_size / 1024**2 if p.exists() else 0.0
        print(f"  - {fname:38s} ({sz:.2f} MB)")


if __name__ == "__main__":
    main()
