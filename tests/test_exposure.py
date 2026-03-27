from src.exposure import approx_exposure


def test_approx_exposure_matches_log2_rank_formula():
    value = approx_exposure(rank=4, num_ref=1024)
    assert value == 8.0


def test_approx_exposure_decreases_with_larger_rank():
    assert approx_exposure(rank=1, num_ref=1024) > approx_exposure(rank=8, num_ref=1024)
