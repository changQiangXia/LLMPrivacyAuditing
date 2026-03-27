from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt

from .utils import load_json


def _ensure_out_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_plot(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_train_history(run_dir: Path, out_dir: Path) -> None:
    history_path = run_dir / "train_history.json"
    if not history_path.is_file():
        return
    history = load_json(str(history_path))
    metrics = {
        "loss": [],
        "learning_rate": [],
        "grad_norm": [],
    }
    for item in history:
        step = item.get("step", len(metrics["loss"]) + 1)
        for key in metrics:
            if key in item:
                metrics[key].append((step, item[key]))

    for key, values in metrics.items():
        if not values:
            continue
        steps = [item[0] for item in values]
        scores = [item[1] for item in values]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, scores, marker="o", linewidth=1.5)
        ax.set_title(f"{run_dir.name} | {key}")
        ax.set_xlabel("step")
        ax.set_ylabel(key)
        ax.grid(alpha=0.3)
        _save_plot(fig, out_dir / f"{run_dir.name}_{key}.png")


def plot_exposure(run_dir: Path, out_dir: Path) -> None:
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        return
    results = load_json(str(results_path))
    exposures = [item["approx_exposure"] for item in results.get("per_canary", [])]
    if not exposures:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(exposures, bins=min(20, len(exposures)), color="#1f77b4", alpha=0.85)
    ax.set_title(f"{run_dir.name} | Approx Exposure Distribution")
    ax.set_xlabel("approx_exposure")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    _save_plot(fig, out_dir / f"{run_dir.name}_exposure_hist.png")


def plot_mia(run_dir: Path, out_dir: Path) -> None:
    results_path = run_dir / "results.json"
    if not results_path.is_file():
        return
    results = load_json(str(results_path))
    roc = results.get("roc", {})
    if not roc:
        return
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, points in roc.items():
        if not points:
            continue
        ax.plot(
            [item["fpr"] for item in points],
            [item["tpr"] for item in points],
            linewidth=1.8,
            label=name,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1.0)
    ax.set_title(f"{run_dir.name} | MIA ROC")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.legend()
    ax.grid(alpha=0.3)
    _save_plot(fig, out_dir / f"{run_dir.name}_mia_roc.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_run_dir", default="")
    ap.add_argument("--exposure_run_dir", default="")
    ap.add_argument("--mia_run_dir", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    out_dir = _ensure_out_dir(args.out_dir)
    if args.train_run_dir:
        plot_train_history(Path(args.train_run_dir), out_dir)
    if args.exposure_run_dir:
        plot_exposure(Path(args.exposure_run_dir), out_dir)
    if args.mia_run_dir:
        plot_mia(Path(args.mia_run_dir), out_dir)

    print(f"[OK] plots -> {os.path.abspath(out_dir)}")


if __name__ == "__main__":
    main()
