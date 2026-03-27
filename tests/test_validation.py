import json
from pathlib import Path

import pytest

from src.validation import validate_canary_meta, validate_text_jsonl


def test_validate_text_jsonl_counts_rows(tmp_path: Path):
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": "hello"}),
                json.dumps({"text": ""}),
                json.dumps({"bad": "field"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    report = validate_text_jsonl(str(path))
    assert report == {
        "total_rows": 3,
        "valid_rows": 1,
        "invalid_rows": 1,
        "empty_rows": 1,
    }


def test_validate_canary_meta_rejects_out_of_range_positions():
    meta = {
        "num_canaries": 1,
        "repeats": 1,
        "canaries": [{"id": 0, "canary": "x", "prefix": "x"}],
        "positions": [{"canary_id": 0, "pos": 5}],
    }
    with pytest.raises(ValueError):
        validate_canary_meta(meta, output_size=1)
