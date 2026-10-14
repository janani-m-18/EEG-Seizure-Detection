import json
import csv
from pathlib import Path

import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT = Path(r"D:\EEG-Major-Project")

MODEL_DIR = PROJECT / "models" / "final_chbmit_v2"

MANIFEST_PATH = MODEL_DIR / "cache_manifest.json"
INDEX_PATH = MODEL_DIR / "window_index.npy"

OUTPUT_DIR = PROJECT / "compact_kaggle_dataset"


# ============================================================
# SETTINGS
# ============================================================

SEED = 1
NEGATIVE_RATIO = 3

CHANNELS = 23
WINDOW_SAMPLES = 1024
SAMPLING_RATE = 256

SPLITS = {
    "train": {
        "chb04", "chb05", "chb07", "chb08",
        "chb09", "chb11", "chb14", "chb16",
        "chb17", "chb18", "chb20", "chb21",
        "chb22", "chb23"
    },

    "val": {
        "chb12", "chb15", "chb24"
    }
}


# ============================================================
# PROPORTIONAL NEGATIVE SAMPLING
# ============================================================

def allocate_quotas(groups, target):

    total = sum(len(v) for v in groups.values())

    if total == 0:
        return {k: 0 for k in groups}

    target = min(target, total)

    exact = {
        k: target * len(v) / total
        for k, v in groups.items()
    }

    quotas = {
        k: min(
            int(np.floor(exact[k])),
            len(groups[k])
        )
        for k in groups
    }

    remaining = target - sum(quotas.values())

    order = sorted(
        groups.keys(),
        key=lambda k: exact[k] - quotas[k],
        reverse=True
    )

    for k in order:

        if remaining <= 0:
            break

        if quotas[k] < len(groups[k]):
            quotas[k] += 1
            remaining -= 1

    return quotas


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FAST COMPACT DATASET CREATOR")
    print("=" * 70)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    with open(
        MANIFEST_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        manifest_data = json.load(f)

    recordings = manifest_data["recordings"]

    # --------------------------------------------------------
    # Load window index
    # --------------------------------------------------------

    index = np.load(
        INDEX_PATH,
        mmap_mode="r"
    )

    print(f"Recordings : {len(recordings):,}")
    print(f"Windows    : {len(index):,}")
    print(f"Index dtype: {index.dtype}")
    print(f"Fields     : {index.dtype.names}")
    print()

    # Exact verified fields
    record_field = "record_index"
    start_field = "start_sample"
    label_field = "label"

    rng = np.random.default_rng(SEED)

    summary = {}

    # ========================================================
    # EACH SPLIT
    # ========================================================

    for split_name, allowed_patients in SPLITS.items():

        print()
        print("=" * 70)
        print(f"BUILDING {split_name.upper()}")
        print("=" * 70)

        seizure_rows = []

        negative_by_recording = {}

        # ----------------------------------------------------
        # Find rows belonging to this patient split
        # ----------------------------------------------------

        print("Scanning window index...")

        for row_id in range(len(index)):

            row = index[row_id]

            record_index = int(
                row[record_field]
            )

            label = int(
                row[label_field]
            )

            recording_info = recordings[
                record_index
            ]

            patient_id = recording_info[
                "patient_id"
            ]

            recording_id = recording_info[
                "recording_id"
            ]

            if patient_id not in allowed_patients:
                continue

            if label == 1:

                seizure_rows.append(
                    row_id
                )

            else:

                negative_by_recording.setdefault(
                    recording_id,
                    []
                ).append(row_id)

        print(
            f"All seizure windows: "
            f"{len(seizure_rows):,}"
        )

        # ----------------------------------------------------
        # Select 3x negative windows
        # ----------------------------------------------------

        target_negative = (
            len(seizure_rows)
            * NEGATIVE_RATIO
        )

        quotas = allocate_quotas(
            negative_by_recording,
            target_negative
        )

        selected_negative = []

        for recording_id, quota in quotas.items():

            candidates = np.asarray(
                negative_by_recording[
                    recording_id
                ],
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

            selected_negative.extend(
                chosen.tolist()
            )

        selected_rows = (
            seizure_rows
            + selected_negative
        )

        rng.shuffle(selected_rows)

        selected_rows = np.asarray(
            selected_rows,
            dtype=np.int64
        )

        seizure_count = int(
            np.sum(
                index[selected_rows][label_field]
                == 1
            )
        )

        negative_count = int(
            np.sum(
                index[selected_rows][label_field]
                == 0
            )
        )

        print()
        print(f"Selected total     : {len(selected_rows):,}")
        print(f"Selected seizure   : {seizure_count:,}")
        print(f"Selected nonseizure: {negative_count:,}")

        # ----------------------------------------------------
        # Output files
        # ----------------------------------------------------

        x_path = (
            OUTPUT_DIR
            / f"X_{split_name}.npy"
        )

        y_path = (
            OUTPUT_DIR
            / f"y_{split_name}.npy"
        )

        metadata_path = (
            OUTPUT_DIR
            / f"metadata_{split_name}.csv"
        )

        # ----------------------------------------------------
        # Create disk-backed output
        # ----------------------------------------------------

        print()
        print(
            f"Creating {x_path.name}..."
        )

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
            shape=(
                len(selected_rows),
            )
        )

        metadata_rows = []

        cache_handles = {}

        # ----------------------------------------------------
        # Extract windows
        # ----------------------------------------------------

        for output_id, row_id in enumerate(
            selected_rows
        ):

            row = index[row_id]

            record_index = int(
                row[record_field]
            )

            start_sample = int(
                row[start_field]
            )

            label = int(
                row[label_field]
            )

            info = recordings[
                record_index
            ]

            patient_id = info[
                "patient_id"
            ]

            recording_id = info[
                "recording_id"
            ]

            cache_path = Path(
                info["cache_path"]
            )

            if not cache_path.is_absolute():

                cache_path = (
                    PROJECT / cache_path
                )

            # Load each recording only once
            if recording_id not in cache_handles:

                print(
                    f"\rExtracting "
                    f"{split_name}: "
                    f"{output_id + 1:,}/"
                    f"{len(selected_rows):,} "
                    f"[{recording_id}]",
                    end="",
                    flush=True
                )

                cache_handles[
                    recording_id
                ] = np.load(
                    cache_path,
                    mmap_mode="r"
                )

            signal = cache_handles[
                recording_id
            ]

            window = signal[
                :CHANNELS,
                start_sample:
                start_sample + WINDOW_SAMPLES
            ]

            if window.shape != (
                CHANNELS,
                WINDOW_SAMPLES
            ):

                raise RuntimeError(
                    f"Wrong window shape "
                    f"{window.shape} "
                    f"for {recording_id}"
                )

            X[output_id] = window
            y[output_id] = label

            metadata_rows.append([
                output_id,
                patient_id,
                recording_id,
                start_sample,
                start_sample / SAMPLING_RATE,
                label
            ])

        print()

        X.flush()
        y.flush()

        del X
        del y

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        with open(
            metadata_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow([
                "window_id",
                "patient_id",
                "recording_id",
                "start_sample",
                "start_seconds",
                "label"
            ])

            writer.writerows(
                metadata_rows
            )

        summary[split_name] = {
            "total_windows": len(selected_rows),
            "seizure_windows": seizure_count,
            "non_seizure_windows": negative_count
        }

        print()
        print("Saved:")
        print(f"  {x_path}")
        print(f"  {y_path}")
        print(f"  {metadata_path}")

    # ========================================================
    # SUMMARY
    # ========================================================

    summary_path = (
        OUTPUT_DIR
        / "dataset_summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {
                "seed": SEED,
                "negative_ratio": NEGATIVE_RATIO,
                "sampling_rate": SAMPLING_RATE,
                "channels": CHANNELS,
                "window_samples": WINDOW_SAMPLES,
                "splits": summary
            },
            f,
            indent=2
        )

    print()
    print("=" * 70)
    print("TRAIN + VALIDATION DATASET COMPLETE")
    print("=" * 70)

    print(
        json.dumps(
            summary,
            indent=2
        )
    )


if __name__ == "__main__":
    main()