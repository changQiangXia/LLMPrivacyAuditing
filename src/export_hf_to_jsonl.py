import argparse
import json
import os
import random
from typing import Callable, Dict, Iterable, List, Optional

from datasets import load_dataset


def write_jsonl(rows: Iterable[Dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def get_text_extractor(name: str) -> Callable[[Dict], Optional[str]]:
    """
    Return a function that maps a dataset example -> text.
    Add more datasets here as needed.
    """
    name = name.lower()

    if "openwebtext" in name:
        # Skylion007/openwebtext has "text"
        return lambda ex: ex.get("text")

    if "ultrachat" in name:
        # HuggingFaceH4/ultrachat_200k structure can vary.
        # Common pattern: conversations list of dicts with 'content'/'value'
        def f(ex: Dict) -> Optional[str]:
            # Try typical keys
            if "messages" in ex and isinstance(ex["messages"], list):
                parts = []
                for m in ex["messages"]:
                    role = m.get("role", "unknown")
                    content = m.get("content") or m.get("value")
                    if content:
                        parts.append(f"{role}: {content}")
                return "\n".join(parts) if parts else None
            if "data" in ex and isinstance(ex["data"], list):
                # some ultrachat variants store alternating turns in "data"
                return "\n".join(str(x) for x in ex["data"] if isinstance(x, str))
            if "conversations" in ex and isinstance(ex["conversations"], list):
                parts = []
                for m in ex["conversations"]:
                    role = m.get("from", "unknown")
                    content = m.get("value")
                    if content:
                        parts.append(f"{role}: {content}")
                return "\n".join(parts) if parts else None
            return None

        return f

    if "dolly" in name:
        # databricks/databricks-dolly-15k fields: instruction, context, response
        def f(ex: Dict) -> Optional[str]:
            inst = ex.get("instruction", "")
            ctx = ex.get("context", "")
            resp = ex.get("response", "")
            if not inst or not resp:
                return None
            if ctx:
                return f"### Instruction:\n{inst}\n### Context:\n{ctx}\n### Response:\n{resp}"
            return f"### Instruction:\n{inst}\n### Response:\n{resp}"

        return f

    # Fallback: try "text" directly
    return lambda ex: ex.get("text")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="e.g., Skylion007/openwebtext")
    ap.add_argument("--split", default="train", help="e.g., train")
    ap.add_argument("--out_train", default="data/raw/train.jsonl")
    ap.add_argument("--out_nonmember", default="data/raw/nonmember.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max_samples", type=int, default=50000, help="cap total samples for quick runs"
    )
    ap.add_argument("--nonmember_ratio", type=float, default=0.2)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    ds = load_dataset(args.dataset, split=args.split)
    extractor = get_text_extractor(args.dataset)

    texts: List[str] = []
    for ex in ds:
        t = extractor(ex)
        if isinstance(t, str):
            t = t.strip()
            if len(t) >= 50:  # filter too-short
                texts.append(t)
        if len(texts) >= args.max_samples:
            break

    rng.shuffle(texts)
    n_non = int(len(texts) * args.nonmember_ratio)
    non = texts[:n_non]
    mem = texts[n_non:]

    write_jsonl([{"text": t} for t in mem], args.out_train)
    write_jsonl([{"text": t} for t in non], args.out_nonmember)

    print(f"[OK] total={len(texts)} member(train)={len(mem)} nonmember={len(non)}")
    print(f"[OK] train -> {args.out_train}")
    print(f"[OK] nonmember -> {args.out_nonmember}")


if __name__ == "__main__":
    main()
