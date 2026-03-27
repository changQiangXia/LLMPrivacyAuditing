import argparse

import torch

from .utils import load_causal_lm, redact_pii


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", required=True)
    ap.add_argument("--lora_dir", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--safe", action="store_true", help="enable PII redaction")
    ap.add_argument("--load_in_4bit", action="store_true")
    args = ap.parse_args()

    lm = load_causal_lm(
        args.model_name_or_path, lora_dir=args.lora_dir, load_in_4bit=args.load_in_4bit
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    enc = lm.tokenizer(args.prompt, return_tensors="pt").to(device)
    out = lm.model.generate(
        **enc,
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=1.1,  # mild decoding defense
        pad_token_id=lm.tokenizer.eos_token_id,
    )
    text = lm.tokenizer.decode(out[0], skip_special_tokens=True)

    if args.safe:
        text = redact_pii(text)

    print(text)


if __name__ == "__main__":
    main()
