"""Metrics math on hand-computed vectors."""

import math

from judge_lab import metrics


def test_mae_hand_computed():
    assert abs(metrics.mae([0.5, 0.7], [0.5, 0.9]) - 0.1) < 1e-12


def test_rmse_hand_computed():
    assert abs(metrics.rmse([0.5, 0.7], [0.5, 0.9]) - math.sqrt(0.02)) < 1e-12


def test_mean_residual_sign():
    assert abs(metrics.mean_residual([0.5, 0.7], [0.5, 0.9]) - (-0.1)) < 1e-12
    assert abs(metrics.mean_residual([0.9, 0.9], [0.5, 0.5]) - 0.4) < 1e-12


def test_mean_empty():
    assert metrics.mean([]) == 0.0
    assert metrics.mae([], []) == 0.0


def test_spearman_perfect():
    assert metrics.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_spearman_reversed():
    assert metrics.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_spearman_ties():
    # ties get average ranks; still perfectly monotonic here
    assert abs(metrics.spearman([1, 1, 3], [1, 1, 3]) - 1.0) < 1e-12


def test_spearman_partial():
    r = metrics.spearman([1, 2, 3, 4], [1, 3, 2, 4])
    assert 0.7 < r < 1.0


def test_qwk_perfect():
    assert metrics.qwk([0.1, 0.5, 0.9], [0.1, 0.5, 0.9]) == 1.0


def test_qwk_empty():
    assert metrics.qwk([], []) == 0.0


def test_qwk_bad_below_good():
    good = metrics.qwk([0.1, 0.5, 0.9] * 4, [0.1, 0.5, 0.9] * 4)
    bad = metrics.qwk([0.9, 0.1, 0.5] * 4, [0.1, 0.5, 0.9] * 4)
    assert good > bad


def test_correctness_tolerance():
    assert metrics.correctness([0.95, 0.80], [1.0, 1.0]) == [1.0, 0.0]  # tol 0.1


def test_ece_perfect_calibration():
    assert metrics.ece([0.9, 0.9, 0.2], [1.0, 1.0, 0.0]) < 0.15


def test_ece_worst_calibration():
    # confident and always wrong
    assert metrics.ece([0.99, 0.99], [0.0, 0.0]) > 0.9


def test_ece_empty():
    assert metrics.ece([], []) == 0.0


def test_mce_spots_bad_bin():
    # one bad bin among good ones
    conf = [0.95] * 8 + [0.55]
    corr = [1.0] * 8 + [0.0]
    assert metrics.mce(conf, corr) > metrics.ece(conf, corr)


def test_brier_hand_computed():
    # (0.9-1)^2 + (0.2-0)^2 = 0.01 + 0.04 = 0.05 / 2
    assert abs(metrics.brier([0.9, 0.2], [1.0, 0.0]) - 0.025) < 1e-12


def test_reliability_bins_skip_empty():
    bins = metrics.reliability_bins([0.95, 0.96], [1.0, 1.0], n_bins=10)
    assert len(bins) == 1
    assert bins[0]["n"] == 2
    assert sum(b["n"] for b in bins) == 2


def test_reliability_bin_fields():
    bins = metrics.reliability_bins([0.15], [1.0], n_bins=10)
    b = bins[0]
    assert b["bin"] == "[0.1, 0.2)"
    assert b["accuracy"] == 1.0
    assert b["avg_confidence"] == 0.15
    assert b["gap"] == 0.85
