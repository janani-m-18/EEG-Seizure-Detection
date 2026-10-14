"""Phase 17 metric behavior, including the deliberately all-negative held-out case."""

import numpy as np

from ml.phase17_evaluation import binary_metrics_with_support


def test_no_positive_class_keeps_seizure_metrics_undefined_and_saves_actual_counts() -> None:
    metrics, matrix = binary_metrics_with_support(np.zeros(4, dtype=np.int64), np.array([0, 0, 1, 0]))
    assert matrix.tolist() == [[3, 1], [0, 0]]
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["seizure"]["support"] == 0
    assert metrics["per_class"]["seizure"]["recall"] is None
    assert metrics["f1_macro"] is None
    assert "no seizure windows" in metrics["undefined_metrics_reason"]
