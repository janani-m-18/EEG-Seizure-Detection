"""Run 5 — Final CNN-BiLSTM Training Script.

Pipeline:
    Phase 0: Preprocess each train+val EDF with batch_a_pipeline → local
             .npy cache (same preprocessing as Run4/inference: 0.5-40 Hz IIR
             Butterworth, 50 Hz notch, CAR, per-recording z-score, 256 Hz,
             23 canonical channels).

    Phase 1: Hard-negative mining on 253 non-seizure TRAIN recordings using
             Run-4 checkpoint.  Windows scoring >= 0.50 become hard-negatives.
             TRAIN patients only — val/test NEVER touched.

    Phase 2: Dataset construction
             - Dense seizure windows (0.5 s stride over seizure intervals)
             - Background windows from non-seizure regions (2 s stride)
             - Hard-negative windows from Phase 1
             - WeightedRandomSampler: 25% seizure / 25% hard-neg / 50% background

    Phase 3: Training
             - KaggleCNNBiLSTM (UNCHANGED architecture, 6,476,673 params)
             - Focal Loss (gamma=2.0, alpha=0.25)
             - AdamW (lr=1e-4, wd=1e-4)
             - ReduceLROnPlateau(mode=max, patience=2, factor=0.5)
             - Gradient clipping 5.0, max 15 epochs, seed 42
             - Checkpoint saved when val F1 improves

Patient split (locked to Run4):
    TRAIN: chb04, chb08, chb09, chb11, chb14, chb17, chb18, chb20, chb21, chb22, chb23
    VAL:   chb05, chb07, chb16
    TEST:  chb01, chb02, chb03, chb06, chb10, chb13, chb19  (NEVER USED)

Output:
    models/cnn_bilstm_run5_best.pt   — best-epoch checkpoint
    models/run5_config.json          — training metadata
    models/run5_preproc_cache/       — per-recording .npy signal cache
    models/run5_hard_negatives.npz   — hard-negative window cache
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mne
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from backend.app.processing import EEGProcessingPipeline, EEGProcessingRequest
from ml.batch_a import BATCH_A_PREPROCESSING, BATCH_A_SEGMENTATION, batch_a_pipeline
from ml.chbmit import CHBMITConfig, SeizureInterval, discover_chbmit_recordings
from ml.kaggle_model import KaggleCNNBiLSTM
from ml.phase16_training import PHASE16_CHANNEL_NAMES

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 42
FS = 256.0                    # sampling frequency
WINDOW_SAMPLES = 1024         # 4 s
STRIDE_SAMPLES = 512          # 2 s  (standard background/val stride)
DENSE_STRIDE_SAMPLES = 128    # 0.5 s  (dense seizure-interval stride)

TRAIN_PATIENTS = frozenset({
    "chb04", "chb08", "chb09", "chb11", "chb14",
    "chb17", "chb18", "chb20", "chb21", "chb22", "chb23",
})
VAL_PATIENTS = frozenset({"chb05", "chb07", "chb16"})
TEST_PATIENTS = frozenset({
    "chb01", "chb02", "chb03", "chb06", "chb10", "chb13", "chb19",
})

DATASET_ROOT = Path(r"D:\EEG-Datasets\CHB-MIT\physionet.org\files\chbmit\1.0.0")
PREPROC_CACHE_DIR = Path("models/run5_preproc_cache")
RUN4_CHECKPOINT = Path("models/cnn_bilstm_run4_best.pt")
RUN5_CHECKPOINT = Path("models/cnn_bilstm_run5_best.pt")
HARD_NEG_CACHE = Path("models/run5_hard_negatives.npz")
RUN5_CONFIG_OUT = Path("models/run5_config.json")

BATCH_SIZE = 64
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 5.0
LR_PATIENCE = 2
LR_FACTOR = 0.5

FOCAL_GAMMA = 2.0
FOCAL_ALPHA = 0.25

HARD_NEG_THRESHOLD = 0.50

FRAC_SEIZURE = 0.25
FRAC_HARD_NEG = 0.25
FRAC_BACKGROUND = 0.50

N_CHANNELS = len(PHASE16_CHANNEL_NAMES)   # 23


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------
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
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(
            targets == 1,
            torch.full_like(targets, self.alpha),
            torch.full_like(targets, 1.0 - self.alpha),
        )
        return (alpha_t * (1.0 - pt) ** self.gamma * bce).mean()


# ---------------------------------------------------------------------------
# Augmentation (seizure windows only, training only)
# ---------------------------------------------------------------------------
def augment_window(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Amplitude scale, channel dropout, Gaussian noise."""
    x = x.copy()
    x *= rng.uniform(0.80, 1.25)
    n_drop = max(1, int(round(rng.uniform(0.05, 0.10) * N_CHANNELS)))
    x[rng.choice(N_CHANNELS, size=n_drop, replace=False), :] = 0.0
    signal_power = float(np.mean(x ** 2)) + 1e-12
    noise_power = signal_power / (10 ** (rng.uniform(25.0, 30.0) / 10.0))
    x += (rng.standard_normal(x.shape) * np.sqrt(noise_power)).astype(np.float32)
    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Label helper
# ---------------------------------------------------------------------------
def _overlaps_seizure(
    w_start_s: float,
    w_end_s: float,
    intervals: list[tuple[float, float]],
) -> bool:
    return any(
        min(w_end_s, e) - max(w_start_s, s) > 0.0
        for s, e in intervals
    )


# ---------------------------------------------------------------------------
# Phase 0 — Preprocessing cache (preprocess every train+val EDF once)
# ---------------------------------------------------------------------------
def _cache_path(patient_id: str, recording_id: str) -> Path:
    return PREPROC_CACHE_DIR / f"{patient_id}__{recording_id}.npy"


def preprocess_and_cache(recordings: list[tuple], pipeline: EEGProcessingPipeline) -> None:
    """Preprocess EDFs that are not yet cached.  recordings = [(patient_id, rec_id, path, intervals), ...]"""
    todo = [r for r in recordings if not _cache_path(r[0], r[1]).is_file()]
    print(f"[Preproc] {len(recordings) - len(todo)} already cached, {len(todo)} to process.")
    PREPROC_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for i, (patient_id, rec_id, edf_path, _) in enumerate(todo):
        t0 = time.time()
        try:
            req = EEGProcessingRequest(rec_id, edf_path)
            loaded = pipeline.load(req)
            filtered = pipeline.filter(loaded)
            noise_handled = pipeline.handle_noise(filtered)
            normalized = pipeline.normalize(noise_handled)
            signal = normalized.data.astype(np.float32)  # (23, samples)
            if signal.shape[0] != N_CHANNELS:
                print(f"  [SKIP] {rec_id}: channel mismatch {signal.shape[0]} != {N_CHANNELS}")
                continue
            tmp = _cache_path(patient_id, rec_id).with_suffix(".tmp.npy")
            np.save(tmp, signal)
            tmp.replace(_cache_path(patient_id, rec_id))
            print(f"  [{i+1}/{len(todo)}] {rec_id}: {signal.shape} in {time.time()-t0:.1f}s")
        except Exception as exc:
            print(f"  [FAIL] {rec_id}: {exc}")


# ---------------------------------------------------------------------------
# Phase 1 — Hard-negative mining
# ---------------------------------------------------------------------------
def mine_hard_negatives(
    nonseizure_train: list[tuple],
    device: torch.device,
) -> np.ndarray:
    """Return (M, 23, 1024) hard-negative windows from non-seizure TRAIN recs."""
    if HARD_NEG_CACHE.exists():
        data = np.load(HARD_NEG_CACHE)
        hns = data["hard_negatives"]
        print(f"[Mining] Loaded {len(hns):,} hard-negative windows from cache.")
        return hns

    print("[Mining] Building hard-negative cache from Run-4 inference...")
    ckpt = torch.load(RUN4_CHECKPOINT, map_location=device, weights_only=False)
    model = KaggleCNNBiLSTM().to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    print(f"[Mining] Run-4 checkpoint loaded (epoch {ckpt['epoch']})")

    all_hns: list[np.ndarray] = []
    t_start = time.time()

    for idx, (patient_id, rec_id, _, _) in enumerate(nonseizure_train):
        assert patient_id in TRAIN_PATIENTS, f"Safety: non-train patient {patient_id} in mining!"
        cp = _cache_path(patient_id, rec_id)
        if not cp.is_file():
            continue
        signal = np.load(cp, mmap_mode="r")
        n_samples = signal.shape[1]
        windows = []
        for start in range(0, n_samples - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            w = signal[:, start:start + WINDOW_SAMPLES].astype(np.float32)
            if w.shape == (N_CHANNELS, WINDOW_SAMPLES):
                windows.append(w)
        if not windows:
            continue
        windows_arr = np.stack(windows)
        scores_list: list[np.ndarray] = []
        with torch.no_grad():
            for b in range(0, len(windows_arr), 128):
                batch = torch.from_numpy(windows_arr[b:b + 128]).to(device)
                scores_list.append(model(batch).cpu().numpy())
        scores = np.concatenate(scores_list)
        mask = scores >= HARD_NEG_THRESHOLD
        if mask.sum() > 0:
            all_hns.append(windows_arr[mask].copy())
        if (idx + 1) % 30 == 0 or idx + 1 == len(nonseizure_train):
            total = sum(len(a) for a in all_hns)
            print(f"  [{idx+1}/{len(nonseizure_train)}] {rec_id} | "
                  f"HN+={int(mask.sum())} | total={total:,} | {time.time()-t_start:.0f}s")

    if all_hns:
        hard_negs = np.concatenate(all_hns, axis=0)
    else:
        hard_negs = np.empty((0, N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)

    HARD_NEG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(HARD_NEG_CACHE, hard_negatives=hard_negs)
    print(f"[Mining] {len(hard_negs):,} hard-negative windows saved → {HARD_NEG_CACHE}")
    print(f"[Mining] Phase 1 complete in {(time.time()-t_start)/60:.1f} min")
    return hard_negs


# ---------------------------------------------------------------------------
# Phase 2 — Window extraction
# ---------------------------------------------------------------------------
def extract_windows(
    patient_id: str,
    rec_id: str,
    intervals: list[tuple[float, float]],
    *,
    dense: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (windows, labels) from cached signal.

    dense=True: use DENSE_STRIDE_SAMPLES for seizure windows (training only).
    Returns arrays shaped (N, 23, 1024) and (N,) int8.
    """
    cp = _cache_path(patient_id, rec_id)
    if not cp.is_file():
        return (np.empty((0, N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32),
                np.empty((0,), dtype=np.int8))
    signal = np.load(cp, mmap_mode="r")
    n_samples = signal.shape[1]

    # For seizure recordings with dense=True, we do TWO passes:
    #   1. Dense stride over seizure regions → positive windows
    #   2. Standard stride over non-seizure regions → negative windows
    # For non-seizure recordings: standard stride only.

    windows: list[np.ndarray] = []
    labels: list[int] = []

    if dense and intervals:
        # Dense stride → all windows in seizure region
        seizure_stride = DENSE_STRIDE_SAMPLES
        # Standard stride → windows outside seizure regions
        bg_stride = STRIDE_SAMPLES

        for start in range(0, n_samples - WINDOW_SAMPLES + 1, seizure_stride):
            w_s = start / FS
            w_e = (start + WINDOW_SAMPLES) / FS
            if _overlaps_seizure(w_s, w_e, intervals):
                w = signal[:, start:start + WINDOW_SAMPLES].astype(np.float32)
                if w.shape == (N_CHANNELS, WINDOW_SAMPLES):
                    windows.append(w)
                    labels.append(1)

        # Background windows (standard stride, non-overlapping)
        for start in range(0, n_samples - WINDOW_SAMPLES + 1, bg_stride):
            w_s = start / FS
            w_e = (start + WINDOW_SAMPLES) / FS
            if not _overlaps_seizure(w_s, w_e, intervals):
                w = signal[:, start:start + WINDOW_SAMPLES].astype(np.float32)
                if w.shape == (N_CHANNELS, WINDOW_SAMPLES):
                    windows.append(w)
                    labels.append(0)
    else:
        # Standard 2s stride for everything (val / non-seizure train recs)
        for start in range(0, n_samples - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            w_s = start / FS
            w_e = (start + WINDOW_SAMPLES) / FS
            lbl = 1 if _overlaps_seizure(w_s, w_e, intervals) else 0
            w = signal[:, start:start + WINDOW_SAMPLES].astype(np.float32)
            if w.shape == (N_CHANNELS, WINDOW_SAMPLES):
                windows.append(w)
                labels.append(lbl)

    if not windows:
        return (np.empty((0, N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32),
                np.empty((0,), dtype=np.int8))
    return np.stack(windows), np.array(labels, dtype=np.int8)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class Run5Dataset(Dataset):
    """In-memory dataset with three pools: seizure, hard-neg, background."""

    def __init__(
        self,
        seizure_windows: np.ndarray,
        hard_neg_windows: np.ndarray,
        background_windows: np.ndarray,
        *,
        training: bool = False,
        seed: int = SEED,
    ) -> None:
        self.training = training
        self.seed = seed
        self._epoch = 0

        self.X = np.concatenate([seizure_windows, hard_neg_windows, background_windows], axis=0)
        self.y = np.concatenate([
            np.ones(len(seizure_windows), dtype=np.float32),
            np.zeros(len(hard_neg_windows), dtype=np.float32),
            np.zeros(len(background_windows), dtype=np.float32),
        ])
        # Pool IDs: 0=seizure, 1=hard_neg, 2=background
        self.pool = np.concatenate([
            np.zeros(len(seizure_windows), dtype=np.int8),
            np.ones(len(hard_neg_windows), dtype=np.int8),
            np.full(len(background_windows), 2, dtype=np.int8),
        ])

        print(f"[Dataset] Seizure={len(seizure_windows):,}  "
              f"HardNeg={len(hard_neg_windows):,}  "
              f"Background={len(background_windows):,}  "
              f"Total={len(self.X):,}")

    def set_epoch(self, e: int) -> None:
        self._epoch = e

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.X[idx].copy()
        label = float(self.y[idx])
        if self.training and label == 1.0:
            rng = np.random.default_rng(self.seed + 1_000_003 * self._epoch + idx)
            x = augment_window(x, rng)
        return torch.from_numpy(x), torch.tensor(label, dtype=torch.float32)

    def sampler_weights(self) -> torch.Tensor:
        fracs = np.array([FRAC_SEIZURE, FRAC_HARD_NEG, FRAC_BACKGROUND])
        counts = np.bincount(self.pool.astype(np.int64), minlength=3)
        w = np.zeros(len(self.y), dtype=np.float64)
        for pid in range(3):
            if counts[pid] > 0:
                w[self.pool == pid] = fracs[pid] / counts[pid]
        return torch.from_numpy(w.astype(np.float32))


class ValDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.X[idx].copy()), torch.tensor(float(self.y[idx]))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    s = tp / (tp + fn) if tp + fn > 0 else 0.0
    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    sp = tn / (tn + fp) if tn + fp > 0 else 0.0
    f1 = 2 * p * s / (p + s) if p + s > 0 else 0.0
    return {"sensitivity": round(s, 4), "specificity": round(sp, 4),
            "precision": round(p, 4), "f1": round(f1, 4),
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    labels_list: list[np.ndarray] = []
    probs_list: list[np.ndarray] = []
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            if is_train:
                optimizer.zero_grad()
            logits = model.forward_logits(bx)
            loss = criterion(logits, by)
            if is_train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
            total_loss += loss.item() * len(by)
            labels_list.append(by.cpu().numpy())
            probs_list.append(torch.sigmoid(logits).detach().cpu().numpy())
    y_true = np.concatenate(labels_list)
    y_prob = np.concatenate(probs_list)
    return total_loss / max(1, len(y_true)), y_true, y_prob


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("RUN 5 — FINAL CNN-BiLSTM TRAINING")
    print("=" * 70)
    print(f"Seed={SEED}  Epochs={EPOCHS}  BatchSize={BATCH_SIZE}")
    print(f"LR={LR}  WD={WEIGHT_DECAY}  GradClip={MAX_GRAD_NORM}")
    print(f"Focal: gamma={FOCAL_GAMMA}, alpha={FOCAL_ALPHA}")
    print(f"HardNegThreshold={HARD_NEG_THRESHOLD}")
    print(f"Mix: {int(FRAC_SEIZURE*100)}% seizure / "
          f"{int(FRAC_HARD_NEG*100)}% hard-neg / "
          f"{int(FRAC_BACKGROUND*100)}% background\n")

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(8)
    print(f"Device: {device}\n")

    # -----------------------------------------------------------------------
    # Discover recordings
    # -----------------------------------------------------------------------
    print("[Discover] Scanning CHB-MIT dataset...")
    all_discovered = discover_chbmit_recordings(CHBMITConfig(DATASET_ROOT))
    print(f"[Discover] {len(all_discovered)} EDFs found.\n")

    train_recs: list[tuple] = []  # (patient_id, rec_id, path, [(start_s, end_s)])
    val_recs: list[tuple] = []
    skipped = 0

    for rec in all_discovered:
        pid = rec.patient_id
        intervals = [(iv.start_seconds, iv.end_seconds) for iv in rec.seizure_intervals]
        entry = (pid, rec.recording_id, rec.path, intervals)
        if pid in TRAIN_PATIENTS:
            train_recs.append(entry)
        elif pid in VAL_PATIENTS:
            val_recs.append(entry)
        else:
            skipped += 1   # test patients or unknown — never used

    # Safety: test patients must not appear
    for pid, *_ in train_recs:
        assert pid not in TEST_PATIENTS, f"FATAL: test patient {pid} in train!"
    for pid, *_ in val_recs:
        assert pid not in TEST_PATIENTS, f"FATAL: test patient {pid} in val!"

    train_seizure = [r for r in train_recs if r[3]]
    train_nonseizure = [r for r in train_recs if not r[3]]
    print(f"[Split] Train: {len(train_recs)} recs "
          f"({len(train_seizure)} seizure, {len(train_nonseizure)} non-seizure)")
    print(f"[Split] Val: {len(val_recs)} recs")
    print(f"[Split] Test: {skipped} recs EXCLUDED from all phases\n")

    # -----------------------------------------------------------------------
    # Phase 0 — Preprocess and cache signals
    # -----------------------------------------------------------------------
    print("[Phase 0] Preprocessing and caching train+val signals...")
    pipeline = batch_a_pipeline(required_channel_names=PHASE16_CHANNEL_NAMES)

    t_preproc = time.time()
    preprocess_and_cache(train_recs + val_recs, pipeline)
    preproc_time = time.time() - t_preproc
    print(f"[Phase 0] Preprocessing complete in {preproc_time/60:.1f} min\n")

    # -----------------------------------------------------------------------
    # Phase 1 — Hard-negative mining
    # -----------------------------------------------------------------------
    print("[Phase 1] Hard-Negative Mining...")
    hard_negs = mine_hard_negatives(train_nonseizure, device)
    print()

    # -----------------------------------------------------------------------
    # Phase 2 — Build window arrays
    # -----------------------------------------------------------------------
    print("[Phase 2] Extracting training windows...")
    t_build = time.time()

    seizure_list: list[np.ndarray] = []
    background_list: list[np.ndarray] = []

    for pid, rec_id, _, intervals in train_recs:
        wins, lbls = extract_windows(pid, rec_id, intervals, dense=bool(intervals))
        if len(wins) == 0:
            continue
        sz_mask = lbls == 1
        bg_mask = lbls == 0
        if sz_mask.sum() > 0:
            seizure_list.append(wins[sz_mask])
        if bg_mask.sum() > 0:
            background_list.append(wins[bg_mask])

    seizure_windows = (np.concatenate(seizure_list)
                       if seizure_list
                       else np.empty((0, N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32))
    background_all = (np.concatenate(background_list)
                      if background_list
                      else np.empty((0, N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32))

    # Subsample background: keep at most 8× the seizure count
    n_sz = len(seizure_windows)
    n_hn = len(hard_negs)
    target_bg = max(n_sz * 8, n_hn * 8, 20000)
    rng = np.random.default_rng(SEED)
    if len(background_all) > target_bg:
        idx_bg = rng.choice(len(background_all), size=target_bg, replace=False)
        idx_bg.sort()
        background_windows = background_all[idx_bg]
    else:
        background_windows = background_all

    print(f"[Phase 2] Seizure={n_sz:,}  HardNeg={n_hn:,}  Background={len(background_windows):,}")
    if n_sz == 0:
        raise RuntimeError("No seizure windows extracted — check preprocessing cache and EDF paths!")
    print(f"[Phase 2] Build time: {time.time()-t_build:.0f}s\n")

    print("[Phase 2] Extracting validation windows...")
    val_wins_list: list[np.ndarray] = []
    val_lbls_list: list[np.ndarray] = []
    for pid, rec_id, _, intervals in val_recs:
        wins, lbls = extract_windows(pid, rec_id, intervals, dense=False)
        if len(wins) > 0:
            val_wins_list.append(wins)
            val_lbls_list.append(lbls)
    X_val = np.concatenate(val_wins_list) if val_wins_list else np.empty((0, N_CHANNELS, WINDOW_SAMPLES), dtype=np.float32)
    y_val = np.concatenate(val_lbls_list) if val_lbls_list else np.empty((0,), dtype=np.int8)
    print(f"[Phase 2] Val: {len(X_val):,} windows  pos={int((y_val==1).sum()):,}  neg={int((y_val==0).sum()):,}\n")

    # -----------------------------------------------------------------------
    # Datasets and loaders
    # -----------------------------------------------------------------------
    train_dataset = Run5Dataset(
        seizure_windows, hard_negs, background_windows,
        training=True, seed=SEED,
    )
    val_dataset = ValDataset(X_val, y_val)

    sample_weights = train_dataset.sampler_weights()
    n_epoch_samples = max(len(train_dataset), BATCH_SIZE * 300)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=n_epoch_samples,
        replacement=True,
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )

    # -----------------------------------------------------------------------
    # Phase 3 — Training
    # -----------------------------------------------------------------------
    print("[Phase 3] Training")
    model = KaggleCNNBiLSTM().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] KaggleCNNBiLSTM  params={total_params:,}")
    assert total_params == 6_476_673, f"Architecture changed! Expected 6,476,673 params, got {total_params}"

    criterion = FocalLoss(gamma=FOCAL_GAMMA, alpha=FOCAL_ALPHA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=LR_FACTOR, patience=LR_PATIENCE,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    history: list[dict] = []
    t_train = time.time()

    print(f"\n{'Ep':>4} | {'TrLoss':>8} | {'TrF1':>6} | {'VaLoss':>8} | {'VaF1':>6} | {'VaSens':>7} | {'VaSpec':>7} | {'LR':>10} | {'Time':>7}")
    print("-" * 85)

    for epoch in range(1, EPOCHS + 1):
        train_dataset.set_epoch(epoch)
        t_ep = time.time()

        tr_loss, tr_y, tr_p = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_y, va_p = run_epoch(model, val_loader, criterion, device)

        tr_m = compute_metrics(tr_y, tr_p)
        va_m = compute_metrics(va_y, va_p)
        cur_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(va_m["f1"])

        ep_time = time.time() - t_ep
        print(
            f"{epoch:>4} | {tr_loss:>8.4f} | {tr_m['f1']:>6.4f} | "
            f"{va_loss:>8.4f} | {va_m['f1']:>6.4f} | "
            f"{va_m['sensitivity']:>7.4f} | {va_m['specificity']:>7.4f} | "
            f"{cur_lr:>10.2e} | {ep_time:>5.0f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 6),
            "train_f1": round(tr_m["f1"], 4),
            "val_loss": round(va_loss, 6),
            "val_f1": round(va_m["f1"], 4),
            "val_sensitivity": round(va_m["sensitivity"], 4),
            "val_specificity": round(va_m["specificity"], 4),
            "val_precision": round(va_m["precision"], 4),
            "lr": cur_lr,
        })

        if va_m["f1"] > best_val_f1:
            best_val_f1 = va_m["f1"]
            best_epoch = epoch
            RUN5_CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va_loss,
                    "val_f1": best_val_f1,
                    "val_metrics": va_m,
                    "train_loss": tr_loss,
                    "seed": SEED,
                    "architecture": "KaggleCNNBiLSTM",
                    "train_patients": sorted(TRAIN_PATIENTS),
                    "val_patients": sorted(VAL_PATIENTS),
                    "test_patients": sorted(TEST_PATIENTS),
                    "focal_gamma": FOCAL_GAMMA,
                    "focal_alpha": FOCAL_ALPHA,
                    "hard_neg_threshold": HARD_NEG_THRESHOLD,
                },
                RUN5_CHECKPOINT,
            )
            print(f"       -> New best val F1={best_val_f1:.4f} at epoch {epoch}. Checkpoint saved.")

    train_elapsed = time.time() - t_train
    print(f"\n[Done] Training complete in {train_elapsed/60:.1f} min. Best epoch: {best_epoch}")
    print(f"[Done] Best val F1: {best_val_f1:.4f}")
    print(f"[Done] Checkpoint: {RUN5_CHECKPOINT}")

    # -----------------------------------------------------------------------
    # Save config
    # -----------------------------------------------------------------------
    meta = {
        "run": "run5",
        "architecture": "KaggleCNNBiLSTM",
        "total_params": total_params,
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
        "window_samples": WINDOW_SAMPLES,
        "dense_stride_samples": DENSE_STRIDE_SAMPLES,
        "standard_stride_samples": STRIDE_SAMPLES,
        "train_patients": sorted(TRAIN_PATIENTS),
        "val_patients": sorted(VAL_PATIENTS),
        "test_patients": sorted(TEST_PATIENTS),
        "preprocessing_time_minutes": round(preproc_time / 60, 1),
        "training_time_minutes": round(train_elapsed / 60, 1),
        "checkpoint_path": str(RUN5_CHECKPOINT),
        "history": history,
    }
    RUN5_CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    RUN5_CONFIG_OUT.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[Done] Config saved: {RUN5_CONFIG_OUT}")


if __name__ == "__main__":
    main()
