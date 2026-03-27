import argparse

from tqdm import tqdm

from .utils import avg_neg_logprob, load_causal_lm, read_jsonl, set_seed, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    lm = load_causal_lm(
        args.model_name_or_path, lora_dir=args.lora_dir, load_in_4bit=args.load_in_4bit
    )
    device = "cuda" if lm.model.device.type == "cuda" else "cpu"

    rows = read_jsonl(args.in_jsonl)
    out = []
    for r in tqdm(rows, desc="loss"):
        text = r.get("text", "")
        if not isinstance(text, str) or not text.strip():
            continue
        nll = avg_neg_logprob(lm.model, lm.tokenizer, text, device=device)
        out.append({"text": text, "nll": nll})

    write_jsonl(out, args.out_jsonl)
    print(f"[OK] wrote {len(out)} -> {args.out_jsonl}")


if __name__ == "__main__":
    main()
