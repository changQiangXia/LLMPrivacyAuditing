from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm

from .config_utils import load_section_defaults
from .experiment import build_experiment_id, create_run_dir, create_stage_run_dir, save_run_metadata
from .utils import (
    load_causal_lm,
    load_json,
    redact_pii,
    save_json,
    set_seed,
    setup_logger,
    write_jsonl,
)
from .validation import require_dir, require_file


@torch.no_grad()
def sample_completions(
    model,
    tok,
    prefix: str,
    num_samples: int,
    max_new: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    device: str,
) -> List[str]:
    enc = tok([prefix] * num_samples, return_tensors="pt", padding=True).to(device)
    out = model.generate(
        **enc,
        max_new_tokens=max_new,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tok.eos_token_id,
    )
    return tok.batch_decode(out, skip_special_tokens=True)


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    defaults = load_section_defaults(known.config, "attack")

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--model_name_or_path", default=defaults.get("model_name_or_path", ""))
    ap.add_argument("--lora_dir", default=defaults.get("lora_dir", ""))
    ap.add_argument("--canary_meta", default=defaults.get("canary_meta", ""))
    ap.add_argument("--output_dir", default=defaults.get("output_dir", ""))
    ap.add_argument("--output_root", default=defaults.get("output_root", "outputs/experiments"))
    ap.add_argument("--exp_id", default=defaults.get("exp_id", ""))
    ap.add_argument(
        "--num_samples_per_prefix",
        type=int,
        default=defaults.get("num_samples_per_prefix", 50),
    )
    ap.add_argument(
        "--generation_batch_size",
        type=int,
        default=defaults.get("generation_batch_size", 10),
    )
    ap.add_argument(
        "--max_canaries",
        type=int,
        default=defaults.get("max_canaries", 0),
        help="0 表示使用全部 canary",
    )
    ap.add_argument("--max_new_tokens", type=int, default=defaults.get("max_new_tokens", 64))
    ap.add_argument("--temperature", type=float, default=defaults.get("temperature", 0.9))
    ap.add_argument("--top_p", type=float, default=defaults.get("top_p", 0.95))
    ap.add_argument(
        "--repetition_penalty",
        type=float,
        default=defaults.get("repetition_penalty", 1.0),
    )
    ap.add_argument("--seed", type=int, default=defaults.get("seed", 0))
    ap.add_argument(
        "--load_in_4bit",
        action="store_true",
        default=defaults.get("load_in_4bit", False),
    )
    ap.add_argument(
        "--safe_print",
        action="store_true",
        default=defaults.get("safe_print", False),
        help="仅影响打印输出",
    )
    ap.add_argument(
        "--decode_redact",
        action="store_true",
        default=defaults.get("decode_redact", False),
        help="将生成结果先脱敏，再参与命中评估；用于推理阶段防护 baseline",
    )
    ap.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=defaults.get("trust_remote_code", False),
    )
    ap.add_argument(
        "--out_json",
        default=defaults.get("out_json", ""),
        help="额外写出一份汇总 JSON",
    )
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
        stage="attack_extract",
        model_name_or_path=args.model_name_or_path,
        data_path=args.canary_meta,
        seed=args.seed,
        extra_tags=[f"n{args.num_samples_per_prefix}", f"c{args.max_canaries or 'all'}"],
    )
    return create_stage_run_dir(args.output_root, "attack_extract", exp_id)


def main():
    args = parse_args()
    if args.lora_dir:
        require_dir(args.lora_dir, "LoRA 目录")
    require_file(args.canary_meta, "canary 元数据文件")
    run_dir = determine_run_dir(args)
    logger = setup_logger(str(run_dir / "attack.log"))
    effective_config = {
        "config": args.config,
        "model_name_or_path": args.model_name_or_path,
        "lora_dir": os.path.abspath(args.lora_dir) if args.lora_dir else "",
        "canary_meta": os.path.abspath(args.canary_meta),
        "output_dir": str(run_dir),
        "num_samples_per_prefix": args.num_samples_per_prefix,
        "generation_batch_size": args.generation_batch_size,
        "max_canaries": args.max_canaries,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "load_in_4bit": bool(args.load_in_4bit),
        "trust_remote_code": bool(args.trust_remote_code),
        "safe_print": bool(args.safe_print),
        "decode_redact": bool(args.decode_redact),
    }
    save_run_metadata(run_dir, "attack_extract", effective_config)

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

    hit = 0
    total = 0
    per_canary = []
    per_sample = []

    for c in tqdm(canaries, desc="canaries"):
        prefix = c["prefix"]
        target = c["canary"]
        local_hit = 0

        sample_idx = 0
        while sample_idx < args.num_samples_per_prefix:
            batch_size = min(args.generation_batch_size, args.num_samples_per_prefix - sample_idx)
            generations = sample_completions(
                lm.model,
                lm.tokenizer,
                prefix,
                num_samples=batch_size,
                max_new=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                device=device,
            )
            for gen in generations:
                eval_text = redact_pii(gen) if args.decode_redact else gen
                raw_matched = target in gen
                matched = target in eval_text
                total += 1
                if matched:
                    hit += 1
                    local_hit += 1

                per_sample.append(
                    {
                        "canary_id": c["id"],
                        "sample_idx": sample_idx,
                        "hit": matched,
                        "raw_hit": raw_matched,
                        "generated_text_redacted": (
                            redact_pii(gen) if (raw_matched or matched) else ""
                        ),
                    }
                )
                sample_idx += 1

        per_canary.append(
            {
                "canary_id": c["id"],
                "num_hits": local_hit,
                "num_trials": args.num_samples_per_prefix,
                "hit_rate": local_hit / args.num_samples_per_prefix,
                "prefix": prefix,
            }
        )
        if local_hit > 0:
            show = prefix + " ..."
            if args.safe_print:
                show = redact_pii(show)
            logger.info(
                "命中 canary_id=%s hits=%s/%s prefix=%s",
                c["id"],
                local_hit,
                args.num_samples_per_prefix,
                show,
            )

    rate = hit / total if total else 0.0
    summary = {
        "metric_name": "canary_extraction_success_rate",
        "hit": hit,
        "total": total,
        "success_rate": rate,
        "num_canaries": len(canaries),
        "num_samples_per_prefix": args.num_samples_per_prefix,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "decode_redact": bool(args.decode_redact),
        "seed": args.seed,
    }
    save_json(summary, str(run_dir / "metrics.json"))
    save_json({"summary": summary, "per_canary": per_canary}, str(run_dir / "results.json"))
    write_jsonl(per_sample, str(run_dir / "sample_hits.jsonl"))

    logger.info("Extraction success rate = %s/%s = %.4f", hit, total, rate)
    logger.info("results -> %s", run_dir)
    if args.out_json:
        save_json({"summary": summary, "per_canary": per_canary}, args.out_json)
        logger.info("额外汇总输出 -> %s", args.out_json)


if __name__ == "__main__":
    main()
