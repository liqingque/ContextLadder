import numpy as np
import pandas as pd

from src.evaluation.control_matching import match_controls
from src.evaluation.evaluator import basic_metrics, sample_metrics


def test_perfect_prediction():
    y = np.arange(12, dtype=float).reshape(3, 4)
    metrics = basic_metrics(y, y)
    assert metrics["abs_pcc"] == 1.0
    assert metrics["abs_r2"] == 1.0
    assert metrics["rmse"] == 0.0


def test_constant_prediction_has_undefined_pcc_but_finite_error():
    y_true = np.arange(8, dtype=float).reshape(2, 4)
    y_pred = np.ones_like(y_true)
    metrics = basic_metrics(y_pred, y_true)
    assert metrics["abs_pcc"] is None
    assert metrics["rmse"] is not None


def test_row_order_is_not_silently_repaired():
    y_true = np.arange(12, dtype=float).reshape(3, 4)
    y_pred = y_true[:, [1, 0, 2, 3]]
    metrics = basic_metrics(y_pred, y_true)
    assert metrics["abs_pcc"] < 1.0


def test_sample_metric_shape():
    y = np.arange(12, dtype=float).reshape(3, 4)
    out = sample_metrics(y, y, sample_ids=["a", "b", "c"])
    assert list(out["sample_ID"]) == ["a", "b", "c"]
    assert len(out) == 3


def test_zero_variance_truth_is_explicitly_invalid():
    y_true = np.array([[1.0, 1.0, 1.0]])
    y_pred = np.array([[1.0, 1.0, 1.0]])
    assert basic_metrics(y_pred, y_true)["abs_pcc"] is None


def test_unknown_split_is_reported_not_relabelled():
    from src.evaluation.evaluator import split_metrics
    y = np.arange(8, dtype=float).reshape(2, 4)
    meta = pd.DataFrame({"split_final": ["unknown_label", "unknown_label"]})
    out = split_metrics(y, y, meta)
    assert list(out["split"]) == ["unknown_label"]


def test_duplicate_controls_are_retained_for_mean_matching_and_mismatch_is_visible():
    mapping = {
        "sample_id": "sample_ID", "compound": "compound", "source": "source", "strain": "strain",
        "medium": "medium", "temperature": "temperature", "time": "time", "time_unit": "time_unit",
        "instrument": "instrument", "plate": "plate",
    }
    meta = pd.DataFrame([
        {"sample_ID": "t1", "compound": "Drug", "source": "S", "strain": "A", "medium": "M", "temperature": "30", "time": "15", "time_unit": "min", "instrument": "I", "plate": "P"},
        {"sample_ID": "c1", "compound": "DMSO", "source": "S", "strain": "A", "medium": "M", "temperature": "30", "time": "15", "time_unit": "min", "instrument": "I", "plate": "P"},
        {"sample_ID": "c2", "compound": "DMSO", "source": "S", "strain": "A", "medium": "M", "temperature": "30", "time": "15", "time_unit": "min", "instrument": "I", "plate": "P"},
        {"sample_ID": "t2", "compound": "Drug", "source": "S", "strain": "B", "medium": "M", "temperature": "30", "time": "15", "time_unit": "min", "instrument": "I", "plate": "P"},
    ])
    out = match_controls(meta, np.ones(len(meta), dtype=bool), mapping)
    row_t1 = out[out["sample_ID"] == "t1"].iloc[0]
    row_t2 = out[out["sample_ID"] == "t2"].iloc[0]
    assert bool(row_t1["matched"]) and int(row_t1["n_controls"]) == 2
    assert not bool(row_t2["matched"])
