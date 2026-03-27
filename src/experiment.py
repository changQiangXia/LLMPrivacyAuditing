from __future__ import annotations

import json
import re
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


def slugify(text: str, max_len: int = 48) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "-", text.strip())
    value = re.sub(r"-{2,}", "-", value).strip("-._")
    value = value or "run"
    return value[:max_len]


def model_tag(model_name_or_path: str) -> str:
    text = model_name_or_path.strip().rstrip("/")
    if not text:
        return "model"
    return slugify(Path(text).name if "/" not in text else text.split("/")[-1])


def data_tag(path: str) -> str:
    return slugify(Path(path).stem)


def build_experiment_id(
    stage: str,
    model_name_or_path: str,
    data_path: str,
    seed: int,
    extra_tags: Iterable[str] | None = None,
) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    parts = [
        stamp,
        slugify(stage),
        model_tag(model_name_or_path),
        data_tag(data_path),
        f"seed{seed}",
    ]
    if extra_tags:
        parts.extend(slugify(tag) for tag in extra_tags if tag)
    return "__".join(parts)


def create_run_dir(path: str, allow_existing: bool = False) -> Path:
    run_dir = Path(path)
    if run_dir.exists() and any(run_dir.iterdir()) and not allow_existing:
        raise FileExistsError(f"输出目录非空: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def create_stage_run_dir(
    output_root: str,
    stage: str,
    exp_id: str,
    allow_existing: bool = False,
) -> Path:
    return create_run_dir(str(Path(output_root) / stage / exp_id), allow_existing=allow_existing)


def shell_command_repr(argv: Iterable[str] | None = None) -> str:
    parts = list(sys.argv if argv is None else argv)
    return " ".join(shlex.quote(part) for part in parts)


def save_run_metadata(
    run_dir: str | Path,
    stage: str,
    effective_config: Dict[str, Any],
    extra_metadata: Dict[str, Any] | None = None,
) -> None:
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    metadata = {
        "stage": stage,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shell_command_repr(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    with (path / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    with (path / "effective_config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(effective_config, handle, allow_unicode=True, sort_keys=False)
