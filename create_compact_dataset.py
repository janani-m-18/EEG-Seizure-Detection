import json
import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"D:\EEG-Major-Project")

CACHE_DIR = PROJECT / "models" / "final_chbmit_v2" / "preprocessed_recordings"
MANIFEST_PATH = PROJECT / "models" / "final_chbmit_v2" / "cache_manifest.json"
INDEX_PATH = PROJECT / "models" / "final_chbmit_v2" / "window_index.npy"

OUTPUT_DIR = PROJECT / "compact_colab_dataset"


# ============================================================
# SETTINGS
# ============================================================

SEED = 1
NEGATIVE_RATIO = 3

WINDOW_SAMPLES = 1024
CHANNELS = 23

SPLITS = {
    "train": [
        "chb04", "chb05", "chb07", "chb08", "chb09",
        "chb11", "chb14", "chb16", "chb17", "chb18",
        "chb20", "chb21", "chb22", "chb23"
    ],
    "val": [
        "chb12", "chb15", "chb24"
    ],
    "test": [
        "chb01", "chb02", "chb03", "chb06",
        "chb10", "chb13", "chb19"
    ]
}


# ============================================================
# HELPERS
# ============================================================

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_patient(recording_name):
    return recording_name.split("_")[0]


def allocate_proportional(negative_rows_by_recording, target):
    """
    Allocate the requested number of negative windows across
    recordings proportionally to their available negative windows.
    """
    recordings = list(negative_rows_by_recording.keys())

    total_available = sum(
        len(negative_rows_by_recording[r]) for r in recordings
    )

    target = min(target, total_available)

    if target == 0:
        return {}

    exact = {
        r: target * len(negative_rows_by_recording[r]) / total_available
        for r in recordings
    }

    quotas = {
        r: min(
            int(np.floor(exact[r])),
            len(negative_rows_by_recording[r])
        )
        for r in recordings
    }

    remaining = target - sum(quotas.values())

    order = sorted(
        recordings,
        key=lambda r: exact[r] - quotas[r],
        reverse=True
    )

    for r in order:
        if remaining <= 0:
            break

        if quotas[r] < len(negative_rows_by_recording[r]):
            quotas[r] += 1
            remaining -= 1

    return quotas


def select_rows_for_split(split_name, manifest, index):
    """
    Select:
      - ALL seizure windows
      - 3x as many non-seizure windows
    while preserving recording coverage.
    """

    allowed_patients = set(SPLITS[split_name])

    seizure_rows = []
    negative_rows_by_recording = {}

    # window_index columns:
    # [record_index, start_sample, label]
    for row_id, row in enumerate(index):

        record_index = int(row[0])
        label = int(row[2])

        recording = manifest[record_index]["recording"]

        patient = get_patient(recording)

        if patient not in allowed_patients:
            continue

        if label == 1:
            seizure_rows.append(row_id)

        else:
            negative_rows_by_recording.setdefault(
                recording, []
            ).append(row_id)

    target_negatives = len(seizure_rows) * NEGATIVE_RATIO

    quotas = allocate_proportional(
        negative_rows_by_recording,
        target_negatives
    )

    rng = np.random.default_rng(SEED)

    selected_negative_rows = []

    for recording, quota in quotas.items():

        candidates = np.asarray(
            negative_rows_by_recording[recording],
            dtype=np.int64
        )

        if quota >= len(candidates):
            chosen = candidates
        else:
            chosen = rng.choice(
                candidates,
                size=quota,
                replace=False
            )

        selected_negative_rows.extend(chosen.tolist())

    selected_rows = seizure_rows + selected_negative_rows

    rng.shuffle(selected_rows)

    return np.asarray(selected_rows, dtype=np.int64)


def create_split(split_name, selected_rows, manifest, index):

    print()
    print("=" * 70)
    print(f"CREATING {split_name.upper()} SPLIT")
    print("=" * 70)

    seizure_count = int(
        np.sum(index[selected_rows, 2] == 1)
    )

    negative_count = int(
        np.sum(index[selected_rows, 2] == 0)
    )

    print(f"Selected windows : {len(selected_rows):,}")
    print(f"Seizure windows  : {seizure_count:,}")
    print(f"Non-seizure      : {negative_count:,}")

    x_path = OUTPUT_DIR / f"X_{split_name}.npy"
    y_path = OUTPUT_DIR / f"y_{split_name}.npy"
    metadata_path = OUTPUT_DIR / f"metadata_{split_name}.csv"

    # Create disk-backed .npy array.
    X = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(selected_rows),
            CHANNELS,
            WINDOW_SAMPLES
        )
    )

    y = np.lib.format.open_memmap(
        y_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(selected_rows),)
    )

    metadata_rows = []

    cache_handles = {}

    try:

        for output_id, row_id in enumerate(selected_rows):

            record_index = int(index[row_id][0])
            start_sample = int(index[row_id][1])
            label = int(index[row_id][2])

            recording_info = manifest[record_index]

            recording = recording_info["recording"]

            cache_path = recording_info.get("cache_path")

            if cache_path is None:
                cache_path = recording_info.get("path")

            if cache_path is None:
                raise RuntimeError(
                    f"No cache path found for {recording}"
                )

            cache_path = Path(cache_path)

            if not cache_path.is_absolute():
                cache_path = CACHE_DIR / cache_path

            if recording not in cache_handles:

                print(
                    f"\rReading {split_name}: "
                    f"{output_id + 1:,}/{len(selected_rows):,} "
                    f"({recording})",
                    end="",
                    flush=True
                )

                cache_handles[recording] = np.load(
                    cache_path,
                    mmap_mode="r"
                )

            signal = cache_handles[recording]

            window = signal[
                :CHANNELS,
                start_sample:start_sample + WINDOW_SAMPLES
            ]

            if window.shape != (CHANNELS, WINDOW_SAMPLES):
                raise RuntimeError(
                    f"Bad window shape for {recording}: "
                    f"{window.shape}"
                )

            X[output_id] = window
            y[output_id] = label

            metadata_rows.append([
                output_id,
                recording,
                get_patient(recording),
                start_sample,
                start_sample / 256.0,
                label
            ])

    finally:

        X.flush()
        y.flush()

    print()

    with open(
        metadata_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "window_id",
            "recording",
            "patient",
            "start_sample",
            "start_seconds",
            "label"
        ])

        writer.writerows(metadata_rows)

    del X
    del y

    print(f"Saved: {x_path}")
    print(f"Saved: {y_path}")
    print(f"Saved: {metadata_path}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("COMPACT CHB-MIT DATASET CREATOR")
    print("=" * 70)

    print(f"Project : {PROJECT}")
    print(f"Cache   : {CACHE_DIR}")
    print(f"Output  : {OUTPUT_DIR}")
    print()

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing manifest:\n{MANIFEST_PATH}"
        )

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Missing window index:\n{INDEX_PATH}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    manifest_data = load_json(MANIFEST_PATH)

    # Support either a direct list or a dictionary containing recordings.
    if isinstance(manifest_data, list):
        manifest = manifest_data
    elif "recordings" in manifest_data:
        manifest = manifest_data["recordings"]
    else:
        raise RuntimeError(
            "Could not find recordings in cache_manifest.json"
        )

    index = np.load(
        INDEX_PATH,
        mmap_mode="r"
    )

    print(f"Manifest recordings : {len(manifest):,}")
    print(f"Window index shape   : {index.shape}")
    print()

    summary = {
        "seed": SEED,
        "negative_ratio": NEGATIVE_RATIO,
        "window_samples": WINDOW_SAMPLES,
        "channels": CHANNELS,
        "sampling_rate": 256,
        "splits": {}
    }

    for split_name in ["train", "val", "test"]:

        selected_rows = select_rows_for_split(
            split_name,
            manifest,
            index
        )

        seizure_count = int(
            np.sum(index[selected_rows, 2] == 1)
        )

        negative_count = int(
            np.sum(index[selected_rows, 2] == 0)
        )

        summary["splits"][split_name] = {
            "total_windows": len(selected_rows),
            "seizure_windows": seizure_count,
            "non_seizure_windows": negative_count
        }

        create_split(
            split_name,
            selected_rows,
            manifest,
            index
        )

    summary_path = OUTPUT_DIR / "dataset_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            summary,
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(json.dumps(summary, indent=2))
    print()
    print(f"Output folder:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()