from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from .utils import save_json, setup_logger, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--report_out", default="")
    args = ap.parse_args()

    report_out = args.report_out or args.out_jsonl.rsplit(".", 1)[0] + ".repair_report.json"
    logger = setup_logger(args.out_jsonl.rsplit(".", 1)[0] + ".repair.log")

    repaired_rows = []
    bad_lines = []
    with open(args.in_jsonl, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                repaired_rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                bad_lines.append(
                    {
                        "line_no": line_no,
                        "error": str(exc),
                        "preview": text[:120],
                    }
                )

    write_jsonl(repaired_rows, args.out_jsonl)
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": args.in_jsonl,
        "output_file": args.out_jsonl,
        "valid_rows": len(repaired_rows),
        "invalid_rows": len(bad_lines),
        "bad_lines": bad_lines,
    }
    save_json(report, report_out)

    logger.info("JSONL 修复完成，valid_rows=%s invalid_rows=%s", len(repaired_rows), len(bad_lines))
    logger.info("输出 -> %s", args.out_jsonl)
    logger.info("报告 -> %s", report_out)


if __name__ == "__main__":
    main()
