from __future__ import annotations

from pathlib import Path
from typing import Dict

from .utils import load_json, read_jsonl


def require_file(path: str, label: str) -> Path:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"{label}不存在: {target}")
    return target


def require_dir(path: str, label: str) -> Path:
    target = Path(path)
    if not target.is_dir():
        raise FileNotFoundError(f"{label}不存在: {target}")
    return target


def ensure_empty_or_missing_dir(path: str) -> Path:
    target = Path(path)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"输出目录非空: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def validate_text_jsonl(path: str) -> Dict[str, int]:
    rows = read_jsonl(path)
    total_rows = len(rows)
    valid_rows = 0
    invalid_rows = 0
    empty_rows = 0

    for row in rows:
        text = row.get("text")
        if not isinstance(text, str):
            invalid_rows += 1
            continue
        if not text.strip():
            empty_rows += 1
            continue
        valid_rows += 1

    return {
        "total_rows": total_rows,
        "valid_rows": valid_rows,
        "invalid_rows": invalid_rows,
        "empty_rows": empty_rows,
    }


def validate_canary_meta(meta: Dict, output_size: int) -> Dict[str, int]:
    canaries = meta.get("canaries", [])
    positions = meta.get("positions", [])
    repeats = int(meta.get("repeats", 0))
    num_canaries = int(meta.get("num_canaries", 0))

    if len(canaries) != num_canaries:
        raise ValueError("canary 元数据中的数量字段与实际条目数不一致")
    if len(positions) != num_canaries * repeats:
        raise ValueError("canary 插入位置数量与 num_canaries * repeats 不一致")

    invalid_positions = [
        item for item in positions if item["pos"] < 0 or item["pos"] >= output_size
    ]
    if invalid_positions:
        raise ValueError("存在越界 canary 插入位置")

    canary_ids = {item["id"] for item in canaries}
    position_ids = {item["canary_id"] for item in positions}
    if canary_ids != position_ids:
        raise ValueError("canary 元数据中的 canary_id 集合不一致")

    return {
        "num_canaries": num_canaries,
        "repeats": repeats,
        "num_positions": len(positions),
        "output_size": output_size,
    }


def validate_canary_meta_file(path: str, output_size: int) -> Dict[str, int]:
    require_file(path, "canary 元数据文件")
    meta = load_json(path)
    return validate_canary_meta(meta, output_size=output_size)
