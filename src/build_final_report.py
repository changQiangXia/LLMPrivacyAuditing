from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import yaml

from .utils import load_json, save_text

RUN_FIELDS = [
    "label",
    "aggregate_label",
    "category",
    "factor",
    "value",
    "seed",
    "train_loss",
    "extraction_success_rate",
    "avg_exposure",
    "mia_auc_loss",
    "mia_auc_neighbourhood",
    "mia_roc_tpr_1e-3_loss",
    "mia_roc_tpr_1e-4_loss",
    "mia_roc_tpr_1e-3_neighbourhood",
    "mia_roc_tpr_1e-4_neighbourhood",
    "mia_val_tpr_1e-3_loss",
    "mia_val_tpr_1e-4_loss",
    "mia_val_tpr_1e-3_neighbourhood",
    "mia_val_tpr_1e-4_neighbourhood",
    "notes",
]

SUMMARY_METRICS = [
    "train_loss",
    "extraction_success_rate",
    "avg_exposure",
    "mia_auc_loss",
    "mia_auc_neighbourhood",
    "mia_roc_tpr_1e-3_loss",
    "mia_roc_tpr_1e-4_loss",
    "mia_roc_tpr_1e-3_neighbourhood",
    "mia_roc_tpr_1e-4_neighbourhood",
    "mia_val_tpr_1e-3_loss",
    "mia_val_tpr_1e-4_loss",
    "mia_val_tpr_1e-3_neighbourhood",
    "mia_val_tpr_1e-4_neighbourhood",
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


def _format_mean_std(row: Dict, prefix: str) -> str:
    mean_value = row.get(f"{prefix}_mean")
    std_value = row.get(f"{prefix}_std")
    if mean_value is None:
        return ""
    if std_value is None:
        return _format_number(mean_value)
    return f"{mean_value:.6f} ± {std_value:.6f}"


def _unique_join(values: Iterable[str]) -> str:
    seen: List[str] = []
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.append(cleaned)
    return ", ".join(seen)


def _nested_get(obj: Dict, *keys: str):
    current = obj
    for key in keys:
        if not isinstance(current, dict):
            return ""
        current = current.get(key, "")
    return current


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
                "factor": item.get("factor", ""),
                "value": item.get("value", ""),
                "seed": item.get("seed", ""),
                "train_loss": train_metrics.get("train_loss", ""),
                "extraction_success_rate": attack_metrics.get("success_rate", ""),
                "avg_exposure": exposure_metrics.get("avg_exposure", ""),
                "mia_auc_loss": _nested_get(mia_metrics, "loss_threshold", "auc"),
                "mia_auc_neighbourhood": _nested_get(mia_metrics, "neighbourhood", "auc"),
                "mia_roc_tpr_1e-3_loss": _nested_get(
                    mia_metrics, "loss_threshold", "roc_tpr_at_1e-3"
                ),
                "mia_roc_tpr_1e-4_loss": _nested_get(
                    mia_metrics, "loss_threshold", "roc_tpr_at_1e-4"
                ),
                "mia_roc_tpr_1e-3_neighbourhood": _nested_get(
                    mia_metrics, "neighbourhood", "roc_tpr_at_1e-3"
                ),
                "mia_roc_tpr_1e-4_neighbourhood": _nested_get(
                    mia_metrics, "neighbourhood", "roc_tpr_at_1e-4"
                ),
                "mia_val_tpr_1e-3_loss": _nested_get(
                    mia_metrics,
                    "validation_selected",
                    "loss_threshold",
                    "target_1e-3",
                    "report_tpr",
                ),
                "mia_val_tpr_1e-4_loss": _nested_get(
                    mia_metrics,
                    "validation_selected",
                    "loss_threshold",
                    "target_1e-4",
                    "report_tpr",
                ),
                "mia_val_tpr_1e-3_neighbourhood": _nested_get(
                    mia_metrics,
                    "validation_selected",
                    "neighbourhood",
                    "target_1e-3",
                    "report_tpr",
                ),
                "mia_val_tpr_1e-4_neighbourhood": _nested_get(
                    mia_metrics,
                    "validation_selected",
                    "neighbourhood",
                    "target_1e-4",
                    "report_tpr",
                ),
                "notes": item.get("notes", ""),
            }
        )
    return rows


def summarise_rows(rows: List[Dict], group_keys: List[str]) -> List[Dict]:
    grouped: Dict[tuple, Dict] = {}
    for row in rows:
        key = tuple(row.get(name, "") for name in group_keys)
        bucket = grouped.setdefault(
            key,
            {
                **{name: row.get(name, "") for name in group_keys},
                "seeds": [],
                "notes": [],
                "metrics": {metric: [] for metric in SUMMARY_METRICS},
            },
        )
        seed = row.get("seed")
        if seed not in ("", None):
            bucket["seeds"].append(str(seed))
        note = row.get("notes", "")
        if note:
            bucket["notes"].append(note)
        for metric in SUMMARY_METRICS:
            value = _to_float(row.get(metric))
            if value is not None:
                bucket["metrics"][metric].append(value)

    summary_rows = []
    for key in sorted(grouped):
        bucket = grouped[key]
        seeds = sorted(bucket["seeds"], key=lambda item: int(item))
        summary = {
            **{name: bucket.get(name, "") for name in group_keys},
            "n": max((len(values) for values in bucket["metrics"].values()), default=0),
            "seeds": ",".join(seeds),
            "notes": _unique_join(bucket["notes"]),
        }
        for metric in SUMMARY_METRICS:
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
    lines = [
        "# 全量实验逐运行明细",
        "",
        (
            "| label | aggregate_label | category | factor | value | seed | train_loss | "
            "extraction_success_rate | avg_exposure | mia_auc_loss | "
            "mia_auc_neighbourhood | mia_val_tpr_1e-3_neighbourhood | notes |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {label} | {aggregate_label} | {category} | {factor} | {value} | {seed} | "
            "{train_loss} | {extraction_success_rate} | {avg_exposure} | {mia_auc_loss} | "
            "{mia_auc_neighbourhood} | {mia_val_tpr_1e-3_neighbourhood} | {notes} |".format(
                **row
            )
        )
    save_text("\n".join(lines) + "\n", path)


def write_group_markdown(
    rows: List[Dict],
    path: str,
    title: str,
    group_key: str,
    include_value: bool = False,
) -> None:
    lines = [
        f"# {title}",
        "",
        (
            f"| {group_key} | "
            + ("value | " if include_value else "")
            + "category | n | seeds | train_loss(mean±std) | "
            "extraction_success_rate(mean±std) | avg_exposure(mean±std) | "
            "mia_auc_loss(mean±std) | mia_auc_neighbourhood(mean±std) | "
            "mia_val_tpr_1e-3_neighbourhood(mean±std) | notes |"
        ),
        "| --- | "
        + ("--- | " if include_value else "")
        + "--- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        parts = [
            row.get(group_key, ""),
            row.get("value", "") if include_value else None,
            row.get("category", ""),
            str(row.get("n", 0)),
            row.get("seeds", ""),
            _format_mean_std(row, "train_loss"),
            _format_mean_std(row, "extraction_success_rate"),
            _format_mean_std(row, "avg_exposure"),
            _format_mean_std(row, "mia_auc_loss"),
            _format_mean_std(row, "mia_auc_neighbourhood"),
            _format_mean_std(row, "mia_val_tpr_1e-3_neighbourhood"),
            row.get("notes", ""),
        ]
        rendered = [str(item) for item in parts if item is not None]
        lines.append("| " + " | ".join(rendered) + " |")
    save_text("\n".join(lines) + "\n", path)


def _metric_series(rows: List[Dict], prefix: str) -> tuple[List[float], List[float]]:
    means = [row.get(f"{prefix}_mean") or 0.0 for row in rows]
    errors = [row.get(f"{prefix}_std") or 0.0 for row in rows]
    return means, errors


def _metric_points(rows: List[Dict], prefix: str) -> tuple[List[float | None], List[float]]:
    means = [row.get(f"{prefix}_mean") for row in rows]
    errors = [row.get(f"{prefix}_std") or 0.0 for row in rows]
    return means, errors


def _overview_sort_key(row: Dict) -> tuple[int, int | str]:
    label = row.get("aggregate_label", "")
    baseline_order = {
        "base_model": 0,
        "lora_no_canary": 1,
        "lora_standard": 2,
        "lora_dedup": 3,
        "decode_safe": 4,
    }
    if label in baseline_order:
        return (0, baseline_order[label])
    return (1, label)


def plot_overview(group_rows: List[Dict], out_dir: Path, title_prefix: str) -> None:
    metrics = [
        ("extraction_success_rate", "Extraction Success Rate"),
        ("avg_exposure", "Average Exposure"),
        ("mia_auc_loss", "MIA AUC Loss"),
        ("mia_auc_neighbourhood", "MIA AUC Neighbourhood"),
    ]
    display_rows = sorted(group_rows, key=_overview_sort_key)
    labels = [row["aggregate_label"] for row in display_rows]
    y_positions = list(range(len(display_rows)))
    baseline_count = sum(1 for row in display_rows if row.get("category") == "baseline")
    height = max(12, len(display_rows) * 0.34 + 2.5)
    fig, axes = plt.subplots(1, 4, figsize=(24, height), sharey=True)
    fig.suptitle(title_prefix, fontsize=16, y=0.995)

    for index, (ax, (metric_name, title)) in enumerate(zip(axes, metrics)):
        means, errors = _metric_points(display_rows, metric_name)
        valid_values = [value for value in means if value is not None]
        missing_positions = [y for y, value in zip(y_positions, means) if value is None]

        if metric_name.startswith("mia_auc"):
            x_min = min(valid_values + [0.5]) - 0.01 if valid_values else 0.49
            x_max = max(valid_values + [0.5]) + 0.01 if valid_values else 0.55
            x_min = max(0.0, x_min)
            ax.axvline(0.5, color="#adb5bd", linewidth=1.0, linestyle="--", zorder=1)
        else:
            x_min = 0.0
            max_value = max(valid_values) if valid_values else 0.0
            x_max = max(0.05 if metric_name == "extraction_success_rate" else 1.0, max_value * 1.18)

        for y, value, error, row in zip(y_positions, means, errors, display_rows):
            if value is None:
                continue
            color = "#2a6f97" if row.get("category") == "baseline" else "#bc4749"
            ax.scatter(value, y, color=color, s=34, zorder=3)
            if error > 0:
                ax.errorbar(
                    value,
                    y,
                    xerr=error,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.1,
                    capsize=3,
                    zorder=2,
                )

        for y in missing_positions:
            ax.text(
                x_min + (x_max - x_min) * 0.03,
                y,
                "N/A",
                va="center",
                ha="left",
                fontsize=8,
                color="#6c757d",
                fontstyle="italic",
            )

        if baseline_count:
            ax.axhline(
                baseline_count - 0.5,
                color="#adb5bd",
                linewidth=1.0,
                linestyle="--",
                zorder=1,
            )

        ax.set_xlim(x_min, x_max)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(y_positions)
        if index == 0:
            ax.set_yticklabels(labels, fontsize=8)
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()

    fig.subplots_adjust(left=0.34, right=0.98, top=0.95, bottom=0.03, wspace=0.18)
    fig.savefig(out_dir / "final_overview.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_tpr_compare(group_rows: List[Dict], out_dir: Path, title_prefix: str) -> None:
    metrics = [
        ("mia_val_tpr_1e-3_loss", "Loss | validation-selected TPR@1e-3"),
        ("mia_val_tpr_1e-4_loss", "Loss | validation-selected TPR@1e-4"),
        ("mia_val_tpr_1e-3_neighbourhood", "Neighbourhood | validation-selected TPR@1e-3"),
        ("mia_val_tpr_1e-4_neighbourhood", "Neighbourhood | validation-selected TPR@1e-4"),
    ]
    display_rows = sorted(group_rows, key=_overview_sort_key)
    labels = [row["aggregate_label"] for row in display_rows]
    y_positions = list(range(len(display_rows)))
    baseline_count = sum(1 for row in display_rows if row.get("category") == "baseline")
    height = max(12, len(display_rows) * 0.34 + 2.5)
    fig, axes = plt.subplots(1, 4, figsize=(24, height), sharey=True)
    fig.suptitle(f"{title_prefix} | Validation-selected TPR@FPR", fontsize=16, y=0.995)

    for index, (ax, (metric_name, title)) in enumerate(zip(axes, metrics)):
        means, errors = _metric_points(display_rows, metric_name)
        valid_values = [value for value in means if value is not None]
        missing_positions = [y for y, value in zip(y_positions, means) if value is None]
        max_value = max(valid_values) if valid_values else 0.0
        x_min = 0.0
        x_max = max(0.0012, max_value * 1.4 if max_value > 0 else 0.0012)

        for y, value, error, row in zip(y_positions, means, errors, display_rows):
            if value is None:
                continue
            color = "#7b9e87" if row.get("category") == "baseline" else "#bc6c25"
            ax.scatter(value, y, color=color, s=34, zorder=3)
            if error > 0:
                ax.errorbar(
                    value,
                    y,
                    xerr=error,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.1,
                    capsize=3,
                    zorder=2,
                )

        for y in missing_positions:
            ax.text(
                x_min + (x_max - x_min) * 0.03,
                y,
                "N/A",
                va="center",
                ha="left",
                fontsize=8,
                color="#6c757d",
                fontstyle="italic",
            )

        if baseline_count:
            ax.axhline(
                baseline_count - 0.5,
                color="#adb5bd",
                linewidth=1.0,
                linestyle="--",
                zorder=1,
            )

        if max_value == 0 and valid_values:
            ax.text(
                0.98,
                0.98,
                "All observed values are 0",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#6c757d",
            )

        ax.set_xlim(x_min, x_max)
        ax.set_title(title, fontsize=11, pad=8)
        ax.grid(axis="x", alpha=0.3)
        ax.tick_params(axis="x", labelsize=9)
        ax.tick_params(axis="y", length=0)
        ax.set_yticks(y_positions)
        if index == 0:
            ax.set_yticklabels(labels, fontsize=8)
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.invert_yaxis()

    fig.subplots_adjust(left=0.34, right=0.98, top=0.95, bottom=0.03, wspace=0.18)
    fig.savefig(out_dir / "mia_tpr_compare.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _sortable_value(value: str):
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def plot_ablation_factors(factor_rows: List[Dict], out_dir: Path, title_prefix: str) -> None:
    factor_dir = out_dir / "ablations"
    factor_dir.mkdir(parents=True, exist_ok=True)
    factors = sorted({row.get("factor", "") for row in factor_rows if row.get("factor", "")})
    for factor in factors:
        rows = [row for row in factor_rows if row.get("factor") == factor]
        rows.sort(key=lambda row: _sortable_value(row.get("value", "")))
        if len(rows) < 2:
            continue

        labels = [str(row.get("value", "")) for row in rows]
        metrics = [
            ("train_loss", "Train Loss"),
            ("extraction_success_rate", "Extraction Success Rate"),
            ("avg_exposure", "Average Exposure"),
            ("mia_auc_neighbourhood", "MIA AUC Neighbourhood"),
        ]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (metric_name, title) in zip(axes.ravel(), metrics):
            means, errors = _metric_series(rows, metric_name)
            x = list(range(len(labels)))
            ax.plot(x, means, marker="o", linewidth=1.8, color="#bc4749")
            if any(error > 0 for error in errors):
                ax.errorbar(x, means, yerr=errors, fmt="none", ecolor="#6c757d", capsize=4)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0)
            ax.set_title(f"{title_prefix} | {factor} | {title}")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(factor_dir / f"{factor}.png", dpi=220)
        plt.close(fig)

        factor_fields = ["factor", "value", "category", "n", "seeds", "notes"] + [
            field
            for metric in SUMMARY_METRICS
            for field in (f"{metric}_mean", f"{metric}_std")
        ]
        write_csv(rows, str(factor_dir / f"{factor}.csv"), factor_fields)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry_yaml", required=True)
    ap.add_argument("--runs_csv", default="artifacts/reports/final_runs.csv")
    ap.add_argument("--runs_md", default="artifacts/reports/final_runs.md")
    ap.add_argument("--summary_csv", default="artifacts/reports/final_summary.csv")
    ap.add_argument("--summary_md", default="artifacts/reports/final_summary.md")
    ap.add_argument("--factor_csv", default="artifacts/reports/final_factor_summary.csv")
    ap.add_argument("--factor_md", default="artifacts/reports/final_factor_summary.md")
    ap.add_argument("--plot_dir", default="artifacts/reports/final_plots")
    ap.add_argument("--title_prefix", default="Final Audit")
    args = ap.parse_args()

    entries = load_registry(args.registry_yaml)
    run_rows = collect_rows(entries)
    group_rows = summarise_rows(run_rows, ["aggregate_label", "category"])
    factor_rows = summarise_rows(
        [row for row in run_rows if row.get("factor")],
        ["factor", "value", "category"],
    )

    summary_fields = ["aggregate_label", "category", "n", "seeds", "notes"] + [
        field
        for metric in SUMMARY_METRICS
        for field in (f"{metric}_mean", f"{metric}_std")
    ]
    factor_fields = ["factor", "value", "category", "n", "seeds", "notes"] + [
        field
        for metric in SUMMARY_METRICS
        for field in (f"{metric}_mean", f"{metric}_std")
    ]

    write_csv(run_rows, args.runs_csv, RUN_FIELDS)
    write_csv(group_rows, args.summary_csv, summary_fields)
    write_csv(factor_rows, args.factor_csv, factor_fields)
    write_runs_markdown(run_rows, args.runs_md)
    write_group_markdown(group_rows, args.summary_md, "全量实验汇总", "aggregate_label")
    write_group_markdown(
        factor_rows,
        args.factor_md,
        "Ablation 因子汇总",
        "factor",
        include_value=True,
    )

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_overview(group_rows, plot_dir, args.title_prefix)
    plot_tpr_compare(group_rows, plot_dir, args.title_prefix)
    plot_ablation_factors(factor_rows, plot_dir, args.title_prefix)

    write_csv(group_rows, str(plot_dir / "group_metrics.csv"), summary_fields)
    write_csv(factor_rows, str(plot_dir / "factor_metrics.csv"), factor_fields)
    write_csv(factor_rows, str(plot_dir / "ablations" / "factor_metrics.csv"), factor_fields)

    print(f"[OK] runs csv -> {args.runs_csv}")
    print(f"[OK] summary csv -> {args.summary_csv}")
    print(f"[OK] factor csv -> {args.factor_csv}")
    print(f"[OK] plots -> {plot_dir}")


if __name__ == "__main__":
    main()
