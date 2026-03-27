from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import tqdm

from .config_utils import load_section_defaults
from .experiment import (
    build_experiment_id,
    create_run_dir,
    create_stage_run_dir,
    save_run_metadata,
)
from .metrics import (
    evaluate_threshold,
    roc_auc,
    roc_curve_points,
    select_threshold_at_fpr,
    tpr_at_fpr,
)
from .utils import (
    batched_avg_neg_logprob,
    load_causal_lm,
    read_jsonl,
    save_json,
    set_seed,
    setup_logger,
    write_jsonl,
)
from .validation import require_dir, require_file, validate_text_jsonl


def word_dropout(text: str, drop_p: float, rng: random.Random) -> str:
    words = text.split()
    if len(words) <= 4:
        return text
    kept = [word for word in words if rng.random() > drop_p]
    if len(kept) < max(3, int(0.5 * len(words))):
        kept = words[: max(3, len(words) // 2)]
    return " ".join(kept)


def score_neighbourhood(
    model,
    tok,
    text: str,
    k: int,
    drop_p: float,
    rng: random.Random,
    device: str,
    batch_size: int,
    max_length: int,
) -> float:
    base = batched_avg_neg_logprob(
        model, tok, [text], device=device, batch_size=1, max_length=max_length
    )[0]
    perturbations = [word_dropout(text, drop_p, rng) for _ in range(k)]
    neigh_scores = batched_avg_neg_logprob(
        model,
        tok,
        perturbations,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
    )
    return base - float(np.mean(neigh_scores))


def parse_args():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    known, _ = pre.parse_known_args()
    defaults = load_section_defaults(known.config, "mia")

    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--model_name_or_path", default=defaults.get("model_name_or_path", ""))
    ap.add_argument("--lora_dir", default=defaults.get("lora_dir", ""))
    ap.add_argument("--member_jsonl", default=defaults.get("member_jsonl", ""))
    ap.add_argument("--nonmember_jsonl", default=defaults.get("nonmember_jsonl", ""))
    ap.add_argument("--member_val_jsonl", default=defaults.get("member_val_jsonl", ""))
    ap.add_argument("--nonmember_val_jsonl", default=defaults.get("nonmember_val_jsonl", ""))
    ap.add_argument("--output_dir", default=defaults.get("output_dir", ""))
    ap.add_argument("--output_root", default=defaults.get("output_root", "outputs/experiments"))
    ap.add_argument("--exp_id", default=defaults.get("exp_id", ""))
    ap.add_argument("--max_samples", type=int, default=defaults.get("max_samples", 2000))
    ap.add_argument(
        "--max_validation_samples",
        type=int,
        default=defaults.get("max_validation_samples", 500),
    )
    ap.add_argument(
        "--validation_fraction",
        type=float,
        default=defaults.get("validation_fraction", 0.2),
    )
    ap.add_argument("--neigh_k", type=int, default=defaults.get("neigh_k", 5))
    ap.add_argument("--word_drop", type=float, default=defaults.get("word_drop", 0.1))
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
    if not args.member_jsonl:
        ap.error("缺少 --member_jsonl，且配置文件中未提供")
    if not args.nonmember_jsonl:
        ap.error("缺少 --nonmember_jsonl，且配置文件中未提供")
    return args


def determine_run_dir(args) -> Path:
    if args.output_dir:
        return create_run_dir(args.output_dir)
    exp_id = args.exp_id or build_experiment_id(
        stage="mia",
        model_name_or_path=args.model_name_or_path,
        data_path=args.member_jsonl,
        seed=args.seed,
        extra_tags=[f"k{args.neigh_k}", f"d{args.word_drop:g}"],
    )
    return create_stage_run_dir(args.output_root, "mia", exp_id)


def _load_texts(path: str) -> List[str]:
    return [
        row["text"]
        for row in read_jsonl(path)
        if isinstance(row.get("text", ""), str) and row["text"].strip()
    ]


def _text_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _split_validation(
    texts: List[str],
    validation_fraction: float,
    max_validation_samples: int,
) -> tuple[List[str], List[str]]:
    if validation_fraction <= 0 or len(texts) < 2:
        return [], list(texts)

    candidate = int(round(len(texts) * validation_fraction))
    candidate = max(1, candidate)
    candidate = min(candidate, len(texts) - 1)
    if max_validation_samples > 0:
        candidate = min(candidate, max_validation_samples)
    return list(texts[:candidate]), list(texts[candidate:])


def _score_texts(
    model,
    tok,
    texts: List[str],
    rng: random.Random,
    args,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    nlls = batched_avg_neg_logprob(
        model,
        tok,
        texts,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    score_a = -np.asarray(nlls, dtype=np.float32)

    deltas = []
    for text in tqdm(texts, desc="MIA: Neighbourhood"):
        deltas.append(
            score_neighbourhood(
                model,
                tok,
                text,
                args.neigh_k,
                args.word_drop,
                rng,
                device,
                args.batch_size,
                args.max_length,
            )
        )
    score_b = -np.asarray(deltas, dtype=np.float32)
    return score_a, score_b


def main():
    args = parse_args()
    if args.lora_dir:
        require_dir(args.lora_dir, "LoRA 目录")
    require_file(args.member_jsonl, "成员集")
    require_file(args.nonmember_jsonl, "非成员集")
    if args.member_val_jsonl:
        require_file(args.member_val_jsonl, "成员验证集")
    if args.nonmember_val_jsonl:
        require_file(args.nonmember_val_jsonl, "非成员验证集")
    run_dir = determine_run_dir(args)
    logger = setup_logger(str(run_dir / "mia.log"))

    effective_config = {
        "config": args.config,
        "model_name_or_path": args.model_name_or_path,
        "lora_dir": os.path.abspath(args.lora_dir) if args.lora_dir else "",
        "member_jsonl": os.path.abspath(args.member_jsonl),
        "nonmember_jsonl": os.path.abspath(args.nonmember_jsonl),
        "member_val_jsonl": os.path.abspath(args.member_val_jsonl) if args.member_val_jsonl else "",
        "nonmember_val_jsonl": os.path.abspath(args.nonmember_val_jsonl)
        if args.nonmember_val_jsonl
        else "",
        "output_dir": str(run_dir),
        "max_samples": args.max_samples,
        "max_validation_samples": args.max_validation_samples,
        "validation_fraction": args.validation_fraction,
        "neigh_k": args.neigh_k,
        "word_drop": args.word_drop,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "seed": args.seed,
        "load_in_4bit": bool(args.load_in_4bit),
        "trust_remote_code": bool(args.trust_remote_code),
    }
    save_run_metadata(
        run_dir,
        "mia",
        effective_config,
        extra_metadata={
            "member_report": validate_text_jsonl(args.member_jsonl),
            "nonmember_report": validate_text_jsonl(args.nonmember_jsonl),
            "member_validation_report": validate_text_jsonl(args.member_val_jsonl)
            if args.member_val_jsonl
            else None,
            "nonmember_validation_report": validate_text_jsonl(args.nonmember_val_jsonl)
            if args.nonmember_val_jsonl
            else None,
        },
    )

    set_seed(args.seed)
    score_rng = random.Random(args.seed)
    lm = load_causal_lm(
        args.model_name_or_path,
        lora_dir=args.lora_dir or None,
        load_in_4bit=args.load_in_4bit,
        trust_remote_code=args.trust_remote_code,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    member_texts = _load_texts(args.member_jsonl)
    nonmember_texts = _load_texts(args.nonmember_jsonl)
    split_rng = random.Random(args.seed)
    split_rng.shuffle(member_texts)
    split_rng.shuffle(nonmember_texts)

    if args.member_val_jsonl and args.nonmember_val_jsonl:
        member_val_texts = _load_texts(args.member_val_jsonl)
        nonmember_val_texts = _load_texts(args.nonmember_val_jsonl)
        member_texts = member_texts[: args.max_samples]
        nonmember_texts = nonmember_texts[: args.max_samples]
    else:
        member_val_texts, member_texts = _split_validation(
            member_texts,
            validation_fraction=args.validation_fraction,
            max_validation_samples=args.max_validation_samples,
        )
        nonmember_val_texts, nonmember_texts = _split_validation(
            nonmember_texts,
            validation_fraction=args.validation_fraction,
            max_validation_samples=args.max_validation_samples,
        )
        member_texts = member_texts[: args.max_samples]
        nonmember_texts = nonmember_texts[: args.max_samples]

    if args.max_validation_samples > 0:
        member_val_texts = member_val_texts[: args.max_validation_samples]
        nonmember_val_texts = nonmember_val_texts[: args.max_validation_samples]

    texts = member_texts + nonmember_texts
    labels = np.array([1] * len(member_texts) + [0] * len(nonmember_texts), dtype=np.int32)

    logger.info(
        "开始计算报告集 NLL 分数，member=%s nonmember=%s", len(member_texts), len(nonmember_texts)
    )
    score_a, score_b = _score_texts(
        lm.model,
        lm.tokenizer,
        texts,
        score_rng,
        args,
        device,
    )
    auc_a = roc_auc(labels, score_a)
    tpr1_a = tpr_at_fpr(labels, score_a, 1e-3)
    tpr2_a = tpr_at_fpr(labels, score_a, 1e-4)

    auc_b = roc_auc(labels, score_b)
    tpr1_b = tpr_at_fpr(labels, score_b, 1e-3)
    tpr2_b = tpr_at_fpr(labels, score_b, 1e-4)

    validation_summary = {}
    validation_rows = []
    if member_val_texts and nonmember_val_texts:
        val_texts = member_val_texts + nonmember_val_texts
        val_labels = np.array(
            [1] * len(member_val_texts) + [0] * len(nonmember_val_texts),
            dtype=np.int32,
        )
        logger.info(
            "开始计算验证集分数，member=%s nonmember=%s",
            len(member_val_texts),
            len(nonmember_val_texts),
        )
        val_score_a, val_score_b = _score_texts(
            lm.model,
            lm.tokenizer,
            val_texts,
            random.Random(args.seed + 991),
            args,
            device,
        )
        validation_targets = {"1e-3": 1e-3, "1e-4": 1e-4}
        for attack_name, val_scores, report_scores in (
            ("loss_threshold", val_score_a, score_a),
            ("neighbourhood", val_score_b, score_b),
        ):
            validation_summary[attack_name] = {}
            for target_name, target_value in validation_targets.items():
                selected = select_threshold_at_fpr(val_labels, val_scores, target_value)
                report_stats = evaluate_threshold(
                    labels,
                    report_scores,
                    selected["threshold"],
                )
                validation_summary[attack_name][f"target_{target_name}"] = {
                    "target_fpr": target_value,
                    "selected_threshold": selected["threshold"],
                    "validation_tpr": selected["tpr"],
                    "validation_fpr": selected["fpr"],
                    "report_tpr": report_stats["tpr"],
                    "report_fpr": report_stats["fpr"],
                    "report_tp": report_stats["tp"],
                    "report_fp": report_stats["fp"],
                    "report_tn": report_stats["tn"],
                    "report_fn": report_stats["fn"],
                }

        for idx, text in enumerate(val_texts):
            validation_rows.append(
                {
                    "sample_id": _text_id(text),
                    "split": "validation",
                    "label": int(val_labels[idx]),
                    "loss_threshold_score": float(val_score_a[idx]),
                    "neighbourhood_score": float(val_score_b[idx]),
                    "text_length": len(text),
                }
            )

    summary = {
        "report_member_count": len(member_texts),
        "report_nonmember_count": len(nonmember_texts),
        "validation_member_count": len(member_val_texts),
        "validation_nonmember_count": len(nonmember_val_texts),
        "loss_threshold": {
            "auc": auc_a,
            "auc_if_reversed": roc_auc(labels, -score_a),
            "roc_tpr_at_1e-3": tpr1_a,
            "roc_tpr_at_1e-4": tpr2_a,
            "member_score_mean": float(np.mean(score_a[: len(member_texts)])),
            "nonmember_score_mean": float(np.mean(score_a[len(member_texts) :])),
        },
        "neighbourhood": {
            "auc": auc_b,
            "auc_if_reversed": roc_auc(labels, -score_b),
            "roc_tpr_at_1e-3": tpr1_b,
            "roc_tpr_at_1e-4": tpr2_b,
            "member_score_mean": float(np.mean(score_b[: len(member_texts)])),
            "nonmember_score_mean": float(np.mean(score_b[len(member_texts) :])),
        },
        "validation_selected": validation_summary,
        "seed": args.seed,
    }
    sample_rows = []
    for idx, text in enumerate(texts):
        sample_rows.append(
            {
                "sample_id": _text_id(text),
                "split": "report",
                "label": int(labels[idx]),
                "loss_threshold_score": float(score_a[idx]),
                "neighbourhood_score": float(score_b[idx]),
                "text_length": len(text),
            }
        )

    results = {
        "summary": summary,
        "roc": {
            "loss_threshold": roc_curve_points(labels, score_a),
            "neighbourhood": roc_curve_points(labels, score_b),
        },
    }
    save_json(summary, str(run_dir / "metrics.json"))
    save_json(results, str(run_dir / "results.json"))
    write_jsonl(sample_rows + validation_rows, str(run_dir / "sample_scores.jsonl"))

    logger.info(
        "Attack A Loss threshold: AUC=%.4f ROC-TPR@1e-3=%.4f ROC-TPR@1e-4=%.4f",
        auc_a,
        tpr1_a,
        tpr2_a,
    )
    logger.info(
        "Attack B Neighbourhood: AUC=%.4f ROC-TPR@1e-3=%.4f ROC-TPR@1e-4=%.4f",
        auc_b,
        tpr1_b,
        tpr2_b,
    )
    if validation_summary:
        logger.info("已完成基于独立验证集的阈值选择与报告集 TPR/FPR 评估")
    logger.info("results -> %s", run_dir)
    if args.out_json:
        save_json(summary, args.out_json)
        logger.info("额外汇总输出 -> %s", args.out_json)


if __name__ == "__main__":
    main()
