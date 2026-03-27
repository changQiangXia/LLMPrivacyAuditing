import sys
from pathlib import Path

from src import attack_extract, train_lora


def test_attack_config_defaults_are_applied(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "experiment.yaml"
    cfg.write_text(
        "\n".join(
            [
                "attack:",
                "  model_name_or_path: models/local-model",
                "  canary_meta: data/processed/canaries.json",
                "  generation_batch_size: 50",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["attack_extract", "--config", str(cfg)])
    args = attack_extract.parse_args()

    assert args.model_name_or_path == "models/local-model"
    assert args.canary_meta == "data/processed/canaries.json"
    assert args.generation_batch_size == 50


def test_train_config_defaults_are_applied(tmp_path: Path, monkeypatch):
    cfg = tmp_path / "train.yaml"
    cfg.write_text(
        "\n".join(
            [
                "model_name_or_path: models/local-model",
                "train_jsonl: data/processed/train_with_canary.jsonl",
                "batch_size: 4",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["train_lora", "--config", str(cfg)])
    args = train_lora.parse_args()

    assert args.model_name_or_path == "models/local-model"
    assert args.train_jsonl == "data/processed/train_with_canary.jsonl"
    assert args.batch_size == 4
