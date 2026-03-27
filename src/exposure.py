from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

from .canary import make_canaries
from .config_utils import load_section_defaults
from .experiment import build_experiment_id, create_run_dir, create_stage_run_dir, save_run_metadata
from .utils import (
    batched_avg_neg_logprob,
    load_causal_lm,
    load_json,
    save_json,
    set_seed,
    setup_logger,
)
from .validation import require_dir, require_file


def approx_exposure(rank: int, num_ref: int) -> float:
    return math.log2(num_ref) - math.log2(max(rank, 1))


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    defaults = load_section_defaults(known.config, "exposure")

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--model_name_or_path", default=defaults.get("model_name_or_path", ""))
    ap.add_argument("--lora_dir", default=defaults.get("lora_dir", ""))
    ap.add_argument("--canary_meta", default=defaults.get("canary_meta", ""))
    ap.add_argument("--output_dir", default=defaults.get("output_dir", ""))
    ap.add_argument("--output_root", default=defaults.get("output_root", "outputs/experiments"))
    ap.add_argument("--exp_id", default=defaults.get("exp_id", ""))
    ap.add_argument("--num_reference", type=int, default=defaults.get("num_reference", 2000))
    ap.add_argument(
        "--max_canaries",
        type=int,
        default=defaults.get("max_canaries", 0),
        help="0 表示使用全部 canary",
    )
    ap.add_argument("--batch_size", type=int, default=defaults.get("batch_size", 8))
    ap.add_argument("--max_length", type=int, default=defaults.get("max_length", 512))
    ap.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    ap.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=defaults.get("load_in_4bit", False),
    )
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=defaults.get("trust_remote_code", False),
    )
    ap.add_argument("--out_json", default=defaults.get("out_json", ""))
    args = ap.parse_args()
    if not args.model_name_or_path:
        ap.error("缺少 --model_name_or_path，且配置文件中未提供")
    if not args.canary_meta:
        ap.error("缺少 --canary_meta，且配置文件中未提供")
    return args


def determine_run_dir(args) -> Path:
    if args.output_dir:
        return create_run_dir(args.output_dir)
    exp_id = args.exp_id or build_experiment_id(
        stage="exposure",
        model_name_or_path=args.model_name_or_path,
        data_path=args.canary_meta,
        seed=args.seed,
        extra_tags=[f"ref{args.num_reference}", f"c{args.max_canaries or 'all'}"],
    )
    return create_stage_run_dir(args.output_root, "exposure", exp_id)


def main():
    args = parse_args()
    if args.lora_dir:
        require_dir(args.lora_dir, "LoRA 目录")
    require_file(args.canary_meta, "canary 元数据文件")
    run_dir = determine_run_dir(args)
    logger = setup_logger(str(run_dir / "exposure.log"))

    effective_config = {
        "config": args.config,
        "model_name_or_path": args.model_name_or_path,
        "lora_dir": os.path.abspath(args.lora_dir) if args.lora_dir else "",
        "canary_meta": os.path.abspath(args.canary_meta),
        "output_dir": str(run_dir),
        "num_reference": args.num_reference,
        "max_canaries": args.max_canaries,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "load_in_4bit": bool(args.load_in_4bit),
        "trust_remote_code": bool(args.trust_remote_code),
        "definition": "approximate exposure based on full-canary NLL ranking",
    }
    save_run_metadata(run_dir, "exposure", effective_config)

    set_seed(args.seed)
    lm = load_causal_lm(
        args.model_name_or_path,
        lora_dir=args.lora_dir or None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = load_json(args.canary_meta)
    canaries: List[Dict] = meta["canaries"]
    if args.max_canaries > 0:
        canaries = canaries[: args.max_canaries]
    refs = make_canaries(args.num_reference, seed=args.seed + 999)

    logger.info("开始计算参考 canary NLL，数量=%s", len(refs))
    ref_texts = [item["canary"] for item in refs]
    ref_nlls = batched_avg_neg_logprob(
        lm.model,
        lm.tokenizer,
        ref_texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    sorted_ref_nlls = sorted(ref_nlls)

    results = []
    logger.info("开始计算目标 canary exposure，数量=%s", len(canaries))
    for c in tqdm(canaries, desc="exposure"):
        target_nll = batched_avg_neg_logprob(
            lm.model,
            lm.tokenizer,
            [c["canary"]],
            device=device,
            batch_size=1,
            max_length=args.max_length,
        )[0]
        rank = 1 + sum(score < target_nll for score in sorted_ref_nlls)
        exp = approx_exposure(rank, args.num_reference + 1)
        results.append(
            {
                "canary_id": c["id"],
                "target_nll": target_nll,
                "rank": rank,
                "approx_exposure": exp,
            }
        )

    exposures = [item["approx_exposure"] for item in results]
    summary = {
        "metric_name": "approx_exposure",
        "definition": "approximate exposure based on full-canary NLL ranking",
        "avg_exposure": float(np.mean(exposures)) if exposures else float("nan"),
        "std_exposure": float(np.std(exposures)) if exposures else float("nan"),
        "max_exposure": float(np.max(exposures)) if exposures else float("nan"),
        "min_exposure": float(np.min(exposures)) if exposures else float("nan"),
        "num_canaries": len(results),
        "num_reference": args.num_reference,
        "seed": args.seed,
    }
    save_json(summary, str(run_dir / "metrics.json"))
    save_json(
        {
            "summary": summary,
            "per_canary": results,
            "reference_nll_summary": {
                "mean": float(np.mean(ref_nlls)) if ref_nlls else float("nan"),
                "std": float(np.std(ref_nlls)) if ref_nlls else float("nan"),
            },
        },
        str(run_dir / "results.json"),
    )
    logger.info("Avg approx exposure = %.4f", summary["avg_exposure"])
    logger.info("results -> %s", run_dir)
    if args.out_json:
        save_json({"summary": summary, "per_canary": results}, args.out_json)
        logger.info("额外汇总输出 -> %s", args.out_json)


if __name__ == "__main__":
    main()
