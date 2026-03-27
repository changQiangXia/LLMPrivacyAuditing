import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src import data_prep, mia


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_data_prep_to_mia_smoke(tmp_path: Path, monkeypatch):
    train_jsonl = tmp_path / "train.jsonl"
    nonmember_jsonl = tmp_path / "nonmember.jsonl"
    processed_jsonl = tmp_path / "train_with_canary.jsonl"
    canary_meta = tmp_path / "canaries.json"
    mia_out = tmp_path / "mia_run"

    _write_jsonl(train_jsonl, [{"text": "alpha beta gamma"}, {"text": "delta epsilon zeta"}])
    _write_jsonl(nonmember_jsonl, [{"text": "theta iota kappa"}, {"text": "lambda mu nu"}])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "data_prep",
            "--in_jsonl",
            str(train_jsonl),
            "--out_jsonl",
            str(processed_jsonl),
            "--insert_canary",
            "--num_canaries",
            "1",
            "--canary_repeats",
            "1",
            "--canary_meta_out",
            str(canary_meta),
        ],
    )
    data_prep.main()

    def fake_load_causal_lm(*args, **kwargs):
        return SimpleNamespace(model=object(), tokenizer=object())

    def fake_batched_avg_neg_logprob(model, tok, texts, device, batch_size=4, max_length=None):
        return [float(len(text.split())) / 10.0 for text in texts]

    monkeypatch.setattr(mia, "load_causal_lm", fake_load_causal_lm)
    monkeypatch.setattr(mia, "batched_avg_neg_logprob", fake_batched_avg_neg_logprob)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "mia",
            "--model_name_or_path",
            "dummy/model",
            "--lora_dir",
            str(tmp_path),
            "--member_jsonl",
            str(processed_jsonl),
            "--nonmember_jsonl",
            str(nonmember_jsonl),
            "--output_dir",
            str(mia_out),
            "--max_samples",
            "2",
            "--neigh_k",
            "2",
        ],
    )
    mia.main()

    assert (mia_out / "metrics.json").is_file()
    assert (mia_out / "sample_scores.jsonl").is_file()
