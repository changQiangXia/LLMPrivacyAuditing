from src.canary import insert_canaries, make_canaries


def test_insert_canaries_records_final_positions():
    texts = ["alpha", "beta", "gamma"]
    canaries = make_canaries(3, seed=123)
    new_texts, meta = insert_canaries(texts, canaries, repeats=2, seed=7)

    assert meta["input_size"] == 3
    assert meta["output_size"] == len(new_texts)
    assert len(meta["positions"]) == 6
    assert all(0 <= item["pos"] < len(new_texts) for item in meta["positions"])

    counts = {}
    for item in meta["positions"]:
        counts[item["canary_id"]] = counts.get(item["canary_id"], 0) + 1
    assert sorted(counts.values()) == [2, 2, 2]
