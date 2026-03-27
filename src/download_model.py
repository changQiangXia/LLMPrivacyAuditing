#!/usr/bin/env python3
"""优先从 ModelScope 下载模型到本地目录。"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from modelscope import snapshot_download

from .utils import save_json, setup_logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--local_dir", default="./models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--revision", default="")
    ap.add_argument("--cache_dir", default="./models/.cache")
    ap.add_argument(
        "--metadata_out", default="", help="default: <local_dir>/download_metadata.json"
    )
    ap.add_argument("--log_file", default="")
    args = ap.parse_args()

    metadata_out = args.metadata_out or os.path.join(args.local_dir, "download_metadata.json")
    log_file = args.log_file or os.path.join(args.local_dir, "download.log")
    logger = setup_logger(log_file)

    logger.info("开始从 ModelScope 下载模型: %s", args.model_id)
    logger.info("目标目录: %s", os.path.abspath(args.local_dir))
    logger.info("缓存目录: %s", os.path.abspath(args.cache_dir))

    try:
        downloaded_dir = snapshot_download(
            model_id=args.model_id,
            revision=args.revision or None,
            cache_dir=args.cache_dir,
            local_dir=args.local_dir,
            local_files_only=False,
        )
    except Exception as exc:
        raise SystemExit(f"[ERROR] ModelScope 下载失败: {exc}") from exc

    abs_dir = os.path.abspath(downloaded_dir)
    if not os.path.isdir(abs_dir):
        raise SystemExit(f"[ERROR] 下载结果目录不存在: {abs_dir}")

    metadata = {
        "source": "ModelScope",
        "model_id": args.model_id,
        "local_dir": abs_dir,
        "cache_dir": os.path.abspath(args.cache_dir),
        "revision": args.revision or "",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    save_json(metadata, metadata_out)

    logger.info("模型下载完成 -> %s", abs_dir)
    logger.info("metadata -> %s", metadata_out)
    logger.info("训练与评估使用路径: --model_name_or_path %s", abs_dir)


if __name__ == "__main__":
    main()
