import json
import subprocess
from pathlib import Path


def _write_jsonl(path: Path, rows):
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def test_data_prep_writes_manifest_and_validation(tmp_path: Path):
    in_jsonl = tmp_path / "train.jsonl"
    out_jsonl = tmp_path / "train_with_canary.jsonl"
    canary_meta = tmp_path / "canaries.json"
    _write_jsonl(
        in_jsonl,
        [
            {"text": "sample text one"},
            {"text": "sample text two"},
            {"text": "sample text two"},
        ],
    )

    subprocess.run(
        [
            "python",
            "-m",
            "src.data_prep",
            "--in_jsonl",
            str(in_jsonl),
            "--out_jsonl",
            str(out_jsonl),
            "--insert_canary",
            "--num_canaries",
            "2",
            "--canary_repeats",
            "2",
            "--dedup",
            "--canary_meta_out",
            str(canary_meta),
        ],
        check=True,
        cwd="/root/autodl-tmp",
    )

    manifest = json.loads(
        (tmp_path / "train_with_canary.manifest.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (tmp_path / "train_with_canary.validation.json").read_text(encoding="utf-8")
    )

    assert manifest["input_count_before_dedup"] == 3
    assert manifest["count_after_dedup"] == 2
    assert manifest["output_count"] == 6
    assert validation["output_report"]["valid_rows"] == 6
