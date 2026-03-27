import numpy as np

from src.metrics import (
    evaluate_threshold,
    roc_auc,
    roc_curve_points,
    select_threshold_at_fpr,
    tpr_at_fpr,
)


def test_roc_auc_perfect_separation():
    y_true = np.array([1, 1, 0, 0], dtype=np.int32)
    y_score = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    assert roc_auc(y_true, y_score) == 1.0


def test_tpr_at_fpr_handles_threshold_sweep():
    y_true = np.array([1, 0, 1, 0], dtype=np.int32)
    y_score = np.array([0.9, 0.8, 0.3, 0.1], dtype=np.float32)
    assert tpr_at_fpr(y_true, y_score, 0.0) == 0.5
    assert tpr_at_fpr(y_true, y_score, 0.5) == 1.0


def test_roc_curve_points_include_endpoints():
    y_true = np.array([1, 0], dtype=np.int32)
    y_score = np.array([0.7, 0.2], dtype=np.float32)
    points = roc_curve_points(y_true, y_score)
    assert points[0]["threshold"] == float("inf")
    assert points[0]["tpr"] == 0.0
    assert points[-1]["tpr"] == 1.0
    assert points[-1]["fpr"] == 1.0


def test_select_threshold_at_fpr_uses_feasible_validation_point():
    y_true = np.array([1, 1, 0, 0], dtype=np.int32)
    y_score = np.array([0.9, 0.8, 0.7, 0.1], dtype=np.float32)
    selected = select_threshold_at_fpr(y_true, y_score, 0.0)
    assert selected["threshold"] == np.float32(0.8)
    assert selected["tpr"] == 1.0
    assert selected["fpr"] == 0.0


def test_evaluate_threshold_reports_confusion_counts():
    y_true = np.array([1, 1, 0, 0], dtype=np.int32)
    y_score = np.array([0.9, 0.4, 0.3, 0.1], dtype=np.float32)
    stats = evaluate_threshold(y_true, y_score, 0.35)
    assert stats["tp"] == 2
    assert stats["fp"] == 0
    assert stats["tn"] == 2
    assert stats["fn"] == 0
    assert stats["tpr"] == 1.0
    assert stats["fpr"] == 0.0
