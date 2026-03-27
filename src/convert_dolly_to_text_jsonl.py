import argparse
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)

    n = 0
    with open(args.out_jsonl, "w", encoding="utf-8") as w:
        with open(args.in_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                inst = ex.get("instruction", "").strip()
                ctx = ex.get("context", "").strip()
                resp = ex.get("response", "").strip()
                if not inst or not resp:
                    continue
                if ctx:
                    text = f"### Instruction:\n{inst}\n### Context:\n{ctx}\n### Response:\n{resp}"
                else:
                    text = f"### Instruction:\n{inst}\n### Response:\n{resp}"
                w.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                n += 1
    print("[OK] wrote", n, "->", args.out_jsonl)


if __name__ == "__main__":
    main()
