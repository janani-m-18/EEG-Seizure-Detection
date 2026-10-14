"""Fast, decoupled validation evaluation and grid search.

Architecture:
  Phase 1: Preprocess each validation EDF ONCE and run Run-4 inference ONCE.
           Store window probabilities in val_cache.npz.
  Phase 2: Evaluate all 56 (threshold, persistence) combinations ONLY on
           cached probabilities in memory (CNN does NOT re-run).
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from ml.batch_b_inference import BatchBInferenceService

CACHE_FILE = Path(r"C:\Users\Janani\.gemini\antigravity\brain\0a9b40d0-a485-4dcb-9cdd-39aca0ac107c\scratch\val_cache.npz")
RAW_DATA_ROOT = Path(r"D:\EEG-Datasets\CHB-MIT\physionet.org\files\chbmit\1.0.0")

# All 13 canonical seizure recordings in the validation cohort (chb05, chb07, chb16)
SEIZURE_RECS = [
    ("chb05", "chb05_06.edf"),
    ("chb05", "chb05_13.edf"),
    ("chb05", "chb05_16.edf"),
    ("chb05", "chb05_17.edf"),
    ("chb05", "chb05_22.edf"),
    ("chb07", "chb07_12.edf"),
    ("chb07", "chb07_13.edf"),
    ("chb07", "chb07_19.edf"),
    ("chb16", "chb16_10.edf"),
    ("chb16", "chb16_11.edf"),
    ("chb16", "chb16_14.edf"),
    ("chb16", "chb16_16.edf"),
    ("chb16", "chb16_17.edf"),
]

# Baseline non-seizure recordings from validation patients
NONSEIZURE_RECS = [
    ("chb05", "chb05_01.edf"),
    ("chb05", "chb05_02.edf"),
    ("chb07", "chb07_01.edf"),
    ("chb07", "chb07_02.edf"),
    ("chb16", "chb16_01.edf"),
    ("chb16", "chb16_02.edf"),
]

GT_INTERVALS: dict[str, list[list[int]]] = {
    "chb05_06.edf": [[417, 532]],
    "chb05_13.edf": [[1086, 1196]],
    "chb05_16.edf": [[2317, 2413]],
    "chb05_17.edf": [[2451, 2571]],
    "chb05_22.edf": [[2348, 2465]],
    "chb07_12.edf": [[4920, 5006]],
    "chb07_13.edf": [[3285, 3381]],
    "chb07_19.edf": [[13688, 13831]],
    "chb16_10.edf": [[2290, 2299]],
    "chb16_11.edf": [[1120, 1129]],
    "chb16_14.edf": [[1854, 1868]],
    "chb16_16.edf": [[1214, 1220]],
    "chb16_17.edf": [[227, 236], [1694, 1700], [2162, 2170], [3290, 3298]],
    "chb16_18.edf": [[627, 635], [1909, 1916]],
}


def extract_events_vectorized(
    probs: np.ndarray,
    starts: np.ndarray,
    stops: np.ndarray,
    threshold: float,
    persistence: int,
) -> list[dict[str, float]]:
    """Pure NumPy event extractor — executes in microseconds."""
    pos = probs >= threshold
    events = []
    run = 0
    start_idx = 0
    for i, is_pos in enumerate(pos):
        if is_pos:
            if run == 0:
                start_idx = i
            run += 1
        else:
            if run >= persistence:
                events.append({
                    "start_sec": float(starts[start_idx]),
                    "stop_sec": float(stops[i - 1]),
                    "duration": float(stops[i - 1] - starts[start_idx]),
                    "windows": run,
                    "peak": float(np.max(probs[start_idx:i])),
                    "mean": float(np.mean(probs[start_idx:i])),
                })
            run = 0
    if run >= persistence:
        events.append({
            "start_sec": float(starts[start_idx]),
            "stop_sec": float(stops[-1]),
            "duration": float(stops[-1] - starts[start_idx]),
            "windows": run,
            "peak": float(np.max(probs[start_idx:])),
            "mean": float(np.mean(probs[start_idx:])),
        })
    return events


def main() -> None:
    print("=================================================================")
    print("VALIDATION EVALUATION: PRECOMPUTATION & FAST GRID SEARCH")
    print("=================================================================\n")

    # Step 1: Load existing cache
    cached_data: dict[str, dict[str, object]] = {}
    if CACHE_FILE.exists():
        with np.load(CACHE_FILE, allow_pickle=True) as data:
            for k in data.files:
                cached_data[k] = data[k].item()
        print(f"[Cache] Successfully loaded {len(cached_data)} precomputed recordings.")

    all_targets = SEIZURE_RECS + NONSEIZURE_RECS
    todo = [item for item in all_targets if item[1] not in cached_data]
    print(f"[Target] Total validation recordings needed: {len(all_targets)}")
    print(f"[Queue] Recordings needing inference: {len(todo)}")

    inference_runs_count = 0
    total_inference_time = 0.0

    # Step 2: Infer remaining recordings ONCE
    if todo:
        torch.set_num_threads(6)
        service = BatchBInferenceService(Path("models/cnn_bilstm_run4_best.pt"))
        print(f"\n[Inference] Processing {len(todo)} recordings with Run 4 (single-pass)...")

        for pat, edf_name in todo:
            t0 = time.time()
            raw_path = RAW_DATA_ROOT / pat / edf_name
            print(f"  -> Loading & inferring {edf_name} ({pat})...", end="", flush=True)

            try:
                prep = service.prepare_edf_details(raw_path)
                res = service.predict_prepared_segments(edf_name, prep.segmented)

                probs = np.array([s.seizure_probability for s in res.segments], dtype=np.float32)
                starts = np.array([s.start_seconds for s in res.segments], dtype=np.float32)
                stops = np.array([s.stop_seconds for s in res.segments], dtype=np.float32)
                elapsed = time.time() - t0

                cached_data[edf_name] = {
                    "probs": probs,
                    "starts": starts,
                    "stops": stops,
                    "duration": prep.duration_seconds,
                }
                inference_runs_count += 1
                total_inference_time += elapsed
                print(f" done in {elapsed:.1f}s ({len(probs)} windows)")

                # Save updated cache immediately
                np.savez_compressed(CACHE_FILE, **cached_data)
            except Exception as exc:
                print(f" SKIPPED ({exc})")

    print(f"\n[Status] All {len(cached_data)} validation recordings ready in cache.")

    # Step 3: Fast Grid Search across 56 combinations ONLY on cached arrays
    print("\n--- Evaluating 56 (Threshold x Persistence) Combinations from Memory ---")
    grid_start_time = time.time()

    thresholds = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90]
    persistences = [1, 2, 3, 4, 5, 6, 8, 10]
    grid_results = []

    seizure_names = [item[1] for item in SEIZURE_RECS if item[1] in cached_data]
    nonseizure_names = [item[1] for item in NONSEIZURE_RECS if item[1] in cached_data]

    total_nonseizure_seconds = sum(float(cached_data[k]["duration"]) for k in nonseizure_names)
    total_nonseizure_hours = total_nonseizure_seconds / 3600.0

    for thresh in thresholds:
        for persist in persistences:
            tp, fn, fp, tn = 0, 0, 0, 0
            total_false_events = 0
            detected_events = 0
            total_events = 0
            latencies = []

            # 1. Seizure recordings
            for name in seizure_names:
                d = cached_data[name]
                evs = extract_events_vectorized(d["probs"], d["starts"], d["stops"], thresh, persist)
                if len(evs) > 0:
                    tp += 1
                else:
                    fn += 1

                gt_list = GT_INTERVALS.get(name, [])
                total_events += len(gt_list)

                for gt_start, gt_end in gt_list:
                    hit = False
                    for ev in evs:
                        if not (ev["stop_sec"] < gt_start or ev["start_sec"] > gt_end + 30):
                            hit = True
                            lat = max(0.0, ev["start_sec"] - gt_start)
                            latencies.append(lat)
                            break
                    if hit:
                        detected_events += 1

            # 2. Non-seizure recordings
            for name in nonseizure_names:
                d = cached_data[name]
                evs = extract_events_vectorized(d["probs"], d["starts"], d["stops"], thresh, persist)
                if len(evs) > 0:
                    fp += 1
                    total_false_events += len(evs)
                else:
                    tn += 1

            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = (2 * prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
            event_sens = detected_events / total_events if total_events > 0 else 0.0
            fa_rate = total_false_events / total_nonseizure_hours if total_nonseizure_hours > 0 else 0.0
            mean_lat = float(np.mean(latencies)) if latencies else 0.0

            grid_results.append({
                "threshold": thresh,
                "persistence": persist,
                "recording_sensitivity": round(sens, 4),
                "recording_specificity": round(spec, 4),
                "precision": round(prec, 4),
                "f1": round(f1, 4),
                "tp": tp,
                "fn": fn,
                "fp": fp,
                "tn": tn,
                "false_alarms_per_hour": round(fa_rate, 2),
                "event_sensitivity": round(event_sens, 4),
                "detected_events": detected_events,
                "total_events": total_events,
                "detection_latency_seconds": round(mean_lat, 2),
            })

    grid_calc_time = time.time() - grid_start_time

    # Sort results by F1 score, specificity, then event sensitivity
    grid_results.sort(key=lambda r: (r["f1"], r["recording_specificity"], r["event_sensitivity"]), reverse=True)

    # Save to JSON
    out_path = Path(r"C:\Users\Janani\.gemini\antigravity\brain\0a9b40d0-a485-4dcb-9cdd-39aca0ac107c\scratch\val_grid_results_56.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(grid_results, f, indent=2)

    # Print Report
    print(f"\n[Execution Summary]")
    print(f"  - Number of validation recordings: {len(cached_data)} ({len(seizure_names)} seizure, {len(nonseizure_names)} non-seizure)")
    print(f"  - Number of model inference runs: {inference_runs_count} (all others retrieved from cache)")
    print(f"  - Total inference time: {total_inference_time:.1f}s")
    print(f"  - 56-combination grid-search calculation time: {grid_calc_time:.4f}s ({grid_calc_time * 1000:.1f} ms)")

    print("\n" + "=" * 98)
    print("VALIDATION GRID SEARCH: TOP 15 OPERATING POINTS (Ranked by F1 & Specificity)")
    print("=" * 98)
    print("Thresh | Persist | RecSens | RecSpec | Prec   | F1     | EventSens | FA/hr  | Latency (s) | TP/FP/FN")
    print("-" * 98)
    for r in grid_results[:15]:
        print(
            f" {r['threshold']:.2f}  |   {r['persistence']:2d}    | "
            f"{r['recording_sensitivity']:.4f}  | {r['recording_specificity']:.4f}  | "
            f"{r['precision']:.4f} | {r['f1']:.4f} |   {r['event_sensitivity']:.4f}  | "
            f"{r['false_alarms_per_hour']:5.2f}  |   {r['detection_latency_seconds']:5.1f}s    | "
            f"{r['tp']}/{r['fp']}/{r['fn']}"
        )
    print("=" * 98)

    # Locked Optimal Operating Point
    best = grid_results[0]
    print(f"\n>>> LOCKED VALIDATION OPERATING POINT: Threshold = {best['threshold']:.2f}, Persistence = {best['persistence']} windows <<<")
    print(f"    Sensitivity = {best['recording_sensitivity']:.4f}, Specificity = {best['recording_specificity']:.4f}, F1 = {best['f1']:.4f}, Event Sens = {best['event_sensitivity']:.4f}\n")


if __name__ == "__main__":
    main()
