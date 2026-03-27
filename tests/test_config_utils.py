from pathlib import Path

from src.config_utils import load_section_defaults


def test_load_section_defaults_supports_flat_config(tmp_path: Path):
    path = tmp_path / "flat.yaml"
    path.write_text("epochs: 3\nbatch_size: 4\n", encoding="utf-8")
    cfg = load_section_defaults(str(path), "train")
    assert cfg["epochs"] == 3
    assert cfg["batch_size"] == 4


def test_load_section_defaults_supports_nested_config(tmp_path: Path):
    path = tmp_path / "nested.yaml"
    path.write_text("train:\n  epochs: 5\nattack:\n  top_p: 0.9\n", encoding="utf-8")
    cfg = load_section_defaults(str(path), "train")
    assert cfg == {"epochs": 5}
