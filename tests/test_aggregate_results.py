from src.aggregate_results import collect_rows, summarise_groups


def test_summarise_groups_computes_mean_and_std(tmp_path):
    train0 = tmp_path / "train0"
    train0.mkdir()
    (train0 / "metrics.json").write_text('{"train_loss": 2.0}', encoding="utf-8")

    train1 = tmp_path / "train1"
    train1.mkdir()
    (train1 / "metrics.json").write_text('{"train_loss": 4.0}', encoding="utf-8")

    exposure0 = tmp_path / "exposure0"
    exposure0.mkdir()
    (exposure0 / "metrics.json").write_text('{"avg_exposure": 1.0}', encoding="utf-8")

    exposure1 = tmp_path / "exposure1"
    exposure1.mkdir()
    (exposure1 / "metrics.json").write_text('{"avg_exposure": 3.0}', encoding="utf-8")

    mia0 = tmp_path / "mia0"
    mia0.mkdir()
    (mia0 / "metrics.json").write_text(
        '{"loss_threshold": {"auc": 0.51}, "neighbourhood": {"auc": 0.52}}',
        encoding="utf-8",
    )

    mia1 = tmp_path / "mia1"
    mia1.mkdir()
    (mia1 / "metrics.json").write_text(
        '{"loss_threshold": {"auc": 0.53}, "neighbourhood": {"auc": 0.54}}',
        encoding="utf-8",
    )

    attack0 = tmp_path / "attack0"
    attack0.mkdir()
    (attack0 / "metrics.json").write_text('{"success_rate": 0.0}', encoding="utf-8")

    attack1 = tmp_path / "attack1"
    attack1.mkdir()
    (attack1 / "metrics.json").write_text('{"success_rate": 0.1}', encoding="utf-8")

    rows = collect_rows(
        [
            {
                "label": "standard_seed0",
                "aggregate_label": "lora_standard",
                "category": "baseline",
                "seed": 0,
                "train_dir": str(train0),
                "attack_dir": str(attack0),
                "exposure_dir": str(exposure0),
                "mia_dir": str(mia0),
                "notes": "seed0",
            },
            {
                "label": "standard_seed1",
                "aggregate_label": "lora_standard",
                "category": "baseline",
                "seed": 1,
                "train_dir": str(train1),
                "attack_dir": str(attack1),
                "exposure_dir": str(exposure1),
                "mia_dir": str(mia1),
                "notes": "seed1",
            },
        ]
    )

    summary = summarise_groups(rows)

    assert len(summary) == 1
    group = summary[0]
    assert group["aggregate_label"] == "lora_standard"
    assert group["n"] == 2
    assert group["seeds"] == "0,1"
    assert group["train_loss_mean"] == 3.0
    assert round(group["train_loss_std"], 6) == round(2**0.5, 6)
    assert group["avg_exposure_mean"] == 2.0
    assert group["extraction_success_rate_mean"] == 0.05
