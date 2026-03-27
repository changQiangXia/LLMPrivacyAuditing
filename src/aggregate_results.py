from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import matplotlib.pyplot as plt
import yaml

from .utils import load_json, save_text

RUN_FIELDS = [
    "label",
    "aggregate_label",
    "category",
    "seed",
    "train_loss",
    "extraction_success_rate",
    "avg_exposure",
    "mia_auc_loss",
    "mia_auc_neighbourhood",
    "notes",
]

GROUP_FIELDS = [
    "aggregate_label",
    "category",
    "n",
    "seeds",
    "train_loss_mean",
    "train_loss_std",
    "extraction_success_rate_mean",
    "extraction_success_rate_std",
    "avg_exposure_mean",
    "avg_exposure_std",
    "mia_auc_loss_mean",
    "mia_auc_loss_std",
    "mia_auc_neighbourhood_mean",
    "mia_auc_neighbourhood_std",
    "notes",
]

METRIC_KEYS = [
    "train_loss",
    "extraction_success_rate",
    "avg_exposure",
    "mia_auc_loss",
    "mia_auc_neighbourhood",
]


def load_registry(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    items = data.get("experiments", [])
    if not isinstance(items, list):
        raise ValueError("registry 中的 experiments 必须为列表")
    return items


def load_metrics(path: str | None) -> Dict:
    if not path:
        return {}
    metrics_path = Path(path) / "metrics.json"
    if not metrics_path.is_file():
        return {}
    return load_json(str(metrics_path))


def _to_float(value) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def _unique_join(values: List[str]) -> str:
    seen = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return ", ".join(seen)


def collect_rows(entries: List[Dict]) -> List[Dict]:
    rows = []
    for item in entries:
        train_metrics = load_metrics(item.get("train_dir", ""))
        attack_metrics = load_metrics(item.get("attack_dir", ""))
        exposure_metrics = load_metrics(item.get("exposure_dir", ""))
        mia_metrics = load_metrics(item.get("mia_dir", ""))

        rows.append(
            {
                "label": item["label"],
                "aggregate_label": item.get("aggregate_label", item["label"]),
                "category": item.get("category", ""),
                "seed": item.get("seed", ""),
                "train_loss": train_metrics.get("train_loss", ""),
                "extraction_success_rate": attack_metrics.get("success_rate", ""),
                "avg_exposure": exposure_metrics.get("avg_exposure", ""),
                "mia_auc_loss": mia_metrics.get("loss_threshold", {}).get("auc", ""),
                "mia_auc_neighbourhood": mia_metrics.get("neighbourhood", {}).get("auc", ""),
                "notes": item.get("notes", ""),
            }
        )
    return rows


def summarise_groups(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, Dict] = {}
    for row in rows:
        key = row["aggregate_label"] or row["label"]
        bucket = grouped.setdefault(
            key,
            {
                "aggregate_label": key,
                "category": row.get("category", ""),
                "seeds": [],
                "notes": [],
                "metrics": {metric: [] for metric in METRIC_KEYS},
            },
        )
        seed = row.get("seed")
        if seed not in ("", None):
            bucket["seeds"].append(str(seed))
        note = row.get("notes", "")
        if note:
            bucket["notes"].append(note)
        if not bucket["category"] and row.get("category"):
            bucket["category"] = row["category"]
        for metric in METRIC_KEYS:
            value = _to_float(row.get(metric))
            if value is not None:
                bucket["metrics"][metric].append(value)

    summary_rows = []
    for key in sorted(grouped):
        bucket = grouped[key]
        seeds = sorted(bucket["seeds"], key=lambda item: int(item))
        summary = {
            "aggregate_label": key,
            "category": bucket["category"],
            "n": max((len(values) for values in bucket["metrics"].values()), default=0),
            "seeds": ",".join(seeds),
            "notes": _unique_join(bucket["notes"]),
        }
        for metric in METRIC_KEYS:
            values = bucket["metrics"][metric]
            metric_mean = mean(values) if values else None
            metric_std = stdev(values) if len(values) >= 2 else 0.0 if values else None
            summary[f"{metric}_mean"] = metric_mean
            summary[f"{metric}_std"] = metric_std
        summary_rows.append(summary)
    return summary_rows


def write_csv(rows: List[Dict], path: str, fieldnames: List[str]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_runs_markdown(rows: List[Dict], path: str) -> None:
    header = [
        "# baseline 逐运行明细",
        "",
        (
            "| label | aggregate_label | category | seed | train_loss | "
            "extraction_success_rate | avg_exposure | mia_auc_loss | "
            "mia_auc_neighbourhood | notes |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines = list(header)
    for row in rows:
        lines.append(
            "| {label} | {aggregate_label} | {category} | {seed} | {train_loss} | "
            "{extraction_success_rate} | {avg_exposure} | {mia_auc_loss} | "
            "{mia_auc_neighbourhood} | {notes} |".format(**row)
        )
    save_text("\n".join(lines) + "\n", path)


def write_group_markdown(rows: List[Dict], path: str) -> None:
    header = [
        "# baseline 汇总",
        "",
        (
            "| aggregate_label | category | n | seeds | train_loss(mean±std) | "
            "extraction_success_rate(mean±std) | avg_exposure(mean±std) | "
            "mia_auc_loss(mean±std) | mia_auc_neighbourhood(mean±std) | notes |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines = list(header)
    for row in rows:
        lines.append(
            "| {aggregate_label} | {category} | {n} | {seeds} | {train_loss} | "
            "{extraction_success_rate} | {avg_exposure} | {mia_auc_loss} | "
            "{mia_auc_neighbourhood} | {notes} |".format(
                aggregate_label=row["aggregate_label"],
                category=row["category"],
                n=row["n"],
                seeds=row["seeds"],
                train_loss=_format_mean_std(row, "train_loss"),
                extraction_success_rate=_format_mean_std(row, "extraction_success_rate"),
                avg_exposure=_format_mean_std(row, "avg_exposure"),
                mia_auc_loss=_format_mean_std(row, "mia_auc_loss"),
                mia_auc_neighbourhood=_format_mean_std(row, "mia_auc_neighbourhood"),
                notes=row["notes"],
            )
        )
    save_text("\n".join(lines) + "\n", path)


def _format_mean_std(row: Dict, prefix: str) -> str:
    mean_value = row.get(f"{prefix}_mean")
    std_value = row.get(f"{prefix}_std")
    if mean_value is None:
        return ""
    if std_value is None:
        return _format_number(mean_value)
    return f"{mean_value:.6f} ± {std_value:.6f}"


def plot_grouped(rows: List[Dict], out_dir: Path) -> None:
    metrics = [
        "extraction_success_rate",
        "avg_exposure",
        "mia_auc_loss",
        "mia_auc_neighbourhood",
    ]
    labels = [row["aggregate_label"] for row in rows]
    for metric_name in metrics:
        means = [row.get(f"{metric_name}_mean") or 0.0 for row in rows]
        errors = [row.get(f"{metric_name}_std") or 0.0 for row in rows]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(labels, means, color="#2a6f97", yerr=errors, capsize=4)
        ax.set_title(metric_name)
        ax.set_ylabel(metric_name)
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"{metric_name}.png", dpi=200)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry_yaml", required=True)
    ap.add_argument("--out_csv", default="artifacts/reports/baseline_summary.csv")
    ap.add_argument("--out_md", default="artifacts/reports/baseline_summary.md")
    ap.add_argument("--runs_csv", default="artifacts/reports/baseline_runs.csv")
    ap.add_argument("--runs_md", default="artifacts/reports/baseline_runs.md")
    ap.add_argument("--plot_dir", default="artifacts/reports/baseline_plots")
    args = ap.parse_args()

    entries = load_registry(args.registry_yaml)
    run_rows = collect_rows(entries)
    group_rows = summarise_groups(run_rows)

    write_csv(run_rows, args.runs_csv, RUN_FIELDS)
    write_csv(group_rows, args.out_csv, GROUP_FIELDS)
    write_runs_markdown(run_rows, args.runs_md)
    write_group_markdown(group_rows, args.out_md)

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_grouped(group_rows, plot_dir)

    print(f"[OK] grouped csv -> {args.out_csv}")
    print(f"[OK] grouped markdown -> {args.out_md}")
    print(f"[OK] run csv -> {args.runs_csv}")
    print(f"[OK] run markdown -> {args.runs_md}")
    print(f"[OK] plots -> {plot_dir}")


if __name__ == "__main__":
    main()
