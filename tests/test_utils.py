import os

from src.utils import local_hub_offline, resolve_model_path


def test_resolve_model_path_returns_absolute_for_local_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert resolve_model_path(str(model_dir)) == str(model_dir.resolve())


def test_local_hub_offline_sets_env_for_local_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()

    assert os.environ.get("HF_HUB_OFFLINE") is None
    with local_hub_offline(str(model_dir)):
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ.get("HF_HUB_OFFLINE") is None
