from __future__ import annotations

from typing import Dict, List

import numpy as np


def roc_curve_points(y_true: np.ndarray, y_score: np.ndarray) -> List[Dict[str, float]]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    pos = int(np.sum(y_true == 1))
    neg = int(np.sum(y_true == 0))
    if pos == 0 or neg == 0:
        return []

    order = np.argsort(-y_score, kind="mergesort")
    y_true = y_true[order]
    y_score = y_score[order]

    tp = 0
    fp = 0
    idx = 0
    points: List[Dict[str, float]] = [{"threshold": float("inf"), "tpr": 0.0, "fpr": 0.0}]

    while idx < len(y_score):
        score = y_score[idx]
        while idx < len(y_score) and y_score[idx] == score:
            if y_true[idx] == 1:
                tp += 1
            else:
                fp += 1
            idx += 1
        points.append(
            {
                "threshold": float(score),
                "tpr": tp / pos,
                "fpr": fp / neg,
            }
        )

    if points[-1]["tpr"] < 1.0 or points[-1]["fpr"] < 1.0:
        points.append({"threshold": float("-inf"), "tpr": 1.0, "fpr": 1.0})

    return points


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    points = roc_curve_points(y_true, y_score)
    if not points:
        return float("nan")
    auc = 0.0
    for left, right in zip(points[:-1], points[1:]):
        auc += (right["fpr"] - left["fpr"]) * (left["tpr"] + right["tpr"]) * 0.5
    return float(auc)


def tpr_at_fpr(y_true: np.ndarray, y_score: np.ndarray, fpr_target: float) -> float:
    points = roc_curve_points(y_true, y_score)
    if not points:
        return float("nan")
    best = 0.0
    for point in points:
        if point["fpr"] <= fpr_target:
            best = max(best, point["tpr"])
    return float(best)


def select_threshold_at_fpr(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fpr_target: float,
) -> Dict[str, float]:
    points = roc_curve_points(y_true, y_score)
    if not points:
        return {
            "threshold": float("nan"),
            "tpr": float("nan"),
            "fpr": float("nan"),
            "fpr_target": float(fpr_target),
        }

    feasible = [point for point in points if point["fpr"] <= fpr_target]
    if not feasible:
        best = points[0]
    else:
        best = max(feasible, key=lambda point: (point["tpr"], -point["fpr"]))

    return {
        "threshold": float(best["threshold"]),
        "tpr": float(best["tpr"]),
        "fpr": float(best["fpr"]),
        "fpr_target": float(fpr_target),
    }


def evaluate_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)

    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return {
            "threshold": float(threshold),
            "tpr": float("nan"),
            "fpr": float("nan"),
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
        }

    predictions = y_score >= threshold
    tp = int(np.sum(predictions & (y_true == 1)))
    fp = int(np.sum(predictions & (y_true == 0)))
    tn = int(np.sum((~predictions) & (y_true == 0)))
    fn = int(np.sum((~predictions) & (y_true == 1)))

    return {
        "threshold": float(threshold),
        "tpr": tp / positives,
        "fpr": fp / negatives,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def mean_std(values: List[float]) -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}
