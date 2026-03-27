import argparse
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_nonmember", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nonmember_ratio", type=float, default=0.2)
    args = ap.parse_args()

    rows = []
    with open(args.in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            t = ex.get("text", "")
            if isinstance(t, str) and len(t.strip()) >= 50:
                rows.append({"text": t.strip()})

    rnd = random.Random(args.seed)
    rnd.shuffle(rows)
    n_non = int(len(rows) * args.nonmember_ratio)
    non = rows[:n_non]
    mem = rows[n_non:]

    os.makedirs(os.path.dirname(args.out_train), exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for r in mem:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out_nonmember, "w", encoding="utf-8") as f:
        for r in non:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] total={len(rows)} train={len(mem)} nonmember={len(non)}")


if __name__ == "__main__":
    main()
