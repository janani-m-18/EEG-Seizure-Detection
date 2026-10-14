"""Background-safe V2 training handoff after the complete cache is available."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from ml.final_training_v2 import V2_OUTPUT_DIRECTORY, run_final_v2_training


def main() -> None:
    root = os.environ.get("CHBMIT_DATASET_ROOT")
    if not root:
        raise RuntimeError("CHBMIT_DATASET_ROOT must be configured")
    manifest = V2_OUTPUT_DIRECTORY / "cache_manifest.json"
    print("waiting for full cache", flush=True)
    while not manifest.is_file():
        time.sleep(15)
    summary = json.loads(manifest.read_text(encoding="utf-8"))
    print(json.dumps({"cache_complete_partition_counts": summary["partition_counts"]}, indent=2), flush=True)
    print(json.dumps(run_final_v2_training(Path(root)), indent=2), flush=True)


if __name__ == "__main__":
    main()
