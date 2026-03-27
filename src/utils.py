from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# 国内网络无法直连 huggingface.co 时使用镜像
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import numpy as np
import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

PII_PATTERNS = [
    # email
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    # phone (very rough)
    (re.compile(r"\b(\+?\d[\d -]{7,}\d)\b"), "[REDACTED_PHONE]"),
    # api key-ish (rough)
    (re.compile(r"\b(sk-[A-Za-z0-9]{16,})\b"), "[REDACTED_KEY]"),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"JSONL 解析失败: {path}:{line_no}: {exc}") from exc
    return rows


def _ensure_parent_dir(path: str) -> None:
    parent = Path(path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(rows: Iterable[Dict], path: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_json(obj, path: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: str) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def simple_dedup_texts(texts: List[str]) -> List[str]:
    seen = set()
    out = []
    for t in texts:
        key = t.strip()
        if key and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def redact_pii(text: str) -> str:
    out = text
    for pat, repl in PII_PATTERNS:
        out = pat.sub(repl, out)
    return out


@dataclass
class LoadedModel:
    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer


def setup_logger(log_path: str | None = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("llm_privacy_audit")
    logger.setLevel(level)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path:
        _ensure_parent_dir(log_path)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


def resolve_model_path(model_name_or_path: str) -> str:
    path = Path(model_name_or_path)
    if path.is_dir():
        return str(path.resolve())
    return model_name_or_path


@contextmanager
def local_hub_offline(model_name_or_path: str):
    path = Path(model_name_or_path)
    if not path.is_dir():
        yield
        return

    env_keys = ["HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"]
    backup = {key: os.environ.get(key) for key in env_keys}
    for key in env_keys:
        os.environ[key] = "1"
    try:
        yield
    finally:
        for key, value in backup.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def load_causal_lm(
    model_name_or_path: str,
    lora_dir: Optional[str] = None,
    device: Optional[str] = None,
    load_in_4bit: bool = False,
    trust_remote_code: bool = False,
) -> LoadedModel:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_model_path = resolve_model_path(model_name_or_path)
    resolved_lora_dir = str(Path(lora_dir).resolve()) if lora_dir else None
    local_files_only = os.path.isdir(resolved_model_path)

    with local_hub_offline(resolved_model_path):
        tok = AutoTokenizer.from_pretrained(
            resolved_model_path,
            use_fast=True,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        kwargs = dict(
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        if load_in_4bit and device == "cuda":
            kwargs.update(dict(load_in_4bit=True, device_map="auto"))
        else:
            kwargs.update(dict(device_map=None))

        model = AutoModelForCausalLM.from_pretrained(resolved_model_path, **kwargs)
        if resolved_lora_dir:
            model = PeftModel.from_pretrained(
                model,
                resolved_lora_dir,
                local_files_only=local_files_only,
            )

    model.eval()
    if not load_in_4bit:
        model.to(device)

    return LoadedModel(model=model, tokenizer=tok)


@torch.no_grad()
def batched_avg_neg_logprob(
    model,
    tok,
    texts: List[str],
    device: str,
    batch_size: int = 4,
    max_length: Optional[int] = None,
) -> List[float]:
    scores: List[float] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attn[:, 1:].contiguous()

        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())

        seq_loss = (token_loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1).clamp(min=1)
        scores.extend(float(value) for value in seq_loss.detach().cpu().tolist())
    return scores


@torch.no_grad()
def avg_neg_logprob(model, tok, text: str, device: str) -> float:
    """Average NLL over tokens (excluding first token)."""
    return batched_avg_neg_logprob(model, tok, [text], device=device, batch_size=1)[0]
