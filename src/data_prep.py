import argparse
import os
import sys
from datetime import datetime, timezone

from .canary import insert_canaries, make_canaries
from .utils import read_jsonl, save_json, set_seed, setup_logger, simple_dedup_texts, write_jsonl
from .validation import validate_canary_meta, validate_text_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--dedup", action="store_true", help="Exact-text dedup before insertion")
    ap.add_argument("--insert_canary", action="store_true")
    ap.add_argument("--num_canaries", type=int, default=200)
    ap.add_argument("--canary_repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--canary_meta_out", default="data/processed/canaries.json")
    ap.add_argument("--manifest_out", default="", help="default: <out_jsonl>.manifest.json")
    ap.add_argument("--validation_out", default="", help="default: <out_jsonl>.validation.json")
    ap.add_argument("--log_file", default="")
    args = ap.parse_args()

    if not os.path.isfile(args.in_jsonl):
        sys.exit(f"[ERROR] 输入文件不存在: {args.in_jsonl}")
    log_file = args.log_file or os.path.splitext(args.out_jsonl)[0] + ".log"
    logger = setup_logger(log_file)
    set_seed(args.seed)

    manifest_out = args.manifest_out or os.path.splitext(args.out_jsonl)[0] + ".manifest.json"
    validation_out = args.validation_out or os.path.splitext(args.out_jsonl)[0] + ".validation.json"

    input_report = validate_text_jsonl(args.in_jsonl)
    rows = read_jsonl(args.in_jsonl)
    texts = [r["text"] for r in rows if "text" in r and isinstance(r["text"], str)]
    if not texts:
        sys.exit(f"[ERROR] 未找到有效 text 字段或文件为空: {args.in_jsonl}")
    input_count = len(texts)

    if args.dedup:
        texts = simple_dedup_texts(texts)
    deduped_count = len(texts)

    meta = None
    if args.insert_canary:
        canaries = make_canaries(args.num_canaries, seed=args.seed)
        texts, meta = insert_canaries(texts, canaries, repeats=args.canary_repeats, seed=args.seed)
        save_json(meta, args.canary_meta_out)
        validate_canary_meta(meta, output_size=len(texts))

    out_rows = [{"text": t} for t in texts]
    write_jsonl(out_rows, args.out_jsonl)

    output_report = validate_text_jsonl(args.out_jsonl)
    validation_report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_report": input_report,
        "output_report": output_report,
        "canary_validation": validate_canary_meta(meta, output_size=len(out_rows))
        if meta
        else None,
    }
    save_json(validation_report, validation_out)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": os.path.abspath(args.in_jsonl),
        "output_file": os.path.abspath(args.out_jsonl),
        "canary_meta_out": os.path.abspath(args.canary_meta_out) if meta else "",
        "seed": args.seed,
        "dedup": bool(args.dedup),
        "insert_canary": bool(args.insert_canary),
        "input_count_before_dedup": input_count,
        "count_after_dedup": deduped_count,
        "output_count": len(out_rows),
        "num_canaries": args.num_canaries if args.insert_canary else 0,
        "canary_repeats": args.canary_repeats if args.insert_canary else 0,
        "validation_file": os.path.abspath(validation_out),
    }
    save_json(manifest, manifest_out)

    logger.info("数据处理完成，输出 %s 行 -> %s", len(out_rows), args.out_jsonl)
    logger.info("manifest -> %s", manifest_out)
    logger.info("validation -> %s", validation_out)
    if meta:
        logger.info("canary 元数据 -> %s", args.canary_meta_out)


if __name__ == "__main__":
    main()
