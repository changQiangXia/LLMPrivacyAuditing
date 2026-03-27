import json
import sys
from pathlib import Path
from types import SimpleNamespace

from src import mia


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_mia_validation_thresholds_are_written(tmp_path: Path, monkeypatch):
    member_jsonl = tmp_path / "member.jsonl"
    nonmember_jsonl = tmp_path / "nonmember.jsonl"
    output_dir = tmp_path / "mia_run"

    _write_jsonl(
        member_jsonl,
        [{"text": "alpha beta gamma"} for _ in range(6)],
    )
    _write_jsonl(
        nonmember_jsonl,
        [{"text": "theta iota kappa lambda"} for _ in range(6)],
    )

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
            str(member_jsonl),
            "--nonmember_jsonl",
            str(nonmember_jsonl),
            "--output_dir",
            str(output_dir),
            "--validation_fraction",
            "0.5",
            "--max_validation_samples",
            "2",
            "--max_samples",
            "4",
            "--neigh_k",
            "2",
        ],
    )
    mia.main()

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["validation_member_count"] == 2
    assert metrics["validation_nonmember_count"] == 2
    assert "target_1e-3" in metrics["validation_selected"]["loss_threshold"]
    assert "target_1e-4" in metrics["validation_selected"]["neighbourhood"]

    rows = [
        json.loads(line)
        for line in (output_dir / "sample_scores.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert {row["split"] for row in rows} == {"report", "validation"}
