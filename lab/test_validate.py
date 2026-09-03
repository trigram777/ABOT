#!/usr/bin/env python3
"""Tests for walk-forward splitting and the Deflated Sharpe Ratio."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from lab import validate as V
from lab.session import calendar


@pytest.fixture(scope="module")
def days():
    return calendar()


# --------------------------------------------------------------- folds

def test_folds_are_contiguous_and_ordered(days):
    folds = V.walk_forward(days, n_folds=5)
    assert len(folds) == 5
    for f in folds:
        assert f.train_start <= f.train_end < f.test_start <= f.test_end


def test_train_and_test_never_overlap(days):
    for f in V.walk_forward(days, n_folds=5):
        assert not (set(f.train(days)) & set(f.test(days)))


def test_a_purge_gap_separates_them(days):
    """Nothing holds a position overnight, so leakage through open trades is
    impossible — but the indicator series IS continuous across sessions, and a
    band on the first test day is computed from bars inside training."""
    purge = 25
    for f in V.walk_forward(days, n_folds=4, purge=purge):
        gap = [d for d in days if f.train_end < d < f.test_start]
        assert len(gap) >= purge - 1


def test_anchored_training_grows_and_unanchored_slides(days):
    anch = V.walk_forward(days, n_folds=4, anchored=True)
    slide = V.walk_forward(days, n_folds=4, anchored=False)
    assert all(f.train_start == anch[0].train_start for f in anch)
    assert len(anch[-1].train(days)) > len(anch[0].train(days))
    assert slide[-1].train_start > slide[0].train_start
    sizes = [len(f.train(days)) for f in slide]
    assert max(sizes) - min(sizes) <= 2


def test_the_last_fold_reaches_the_end_of_the_data(days):
    assert V.walk_forward(days, n_folds=5)[-1].test_end == max(days)


def test_an_impossible_split_is_refused(days):
    with pytest.raises(ValueError, match="purge"):
        V.walk_forward(days, n_folds=60, purge=25)
    with pytest.raises(ValueError, match="n_folds"):
        V.walk_forward(days, n_folds=0)


def test_folds_partition_by_date_not_at_random(days):
    """Randomly assigned folds would put 2025-04-08 in train and 2025-04-09 in
    test — a rule fitted on one tariff headline validated on the next day of
    the same event."""
    for f in V.walk_forward(days, n_folds=5):
        tr, te = f.train(days), f.test(days)
        assert max(tr) < min(te)


# ----------------------------------------------------------------- the DSR

def test_the_benchmark_rises_with_the_number_of_trials():
    v = 0.25
    b = [V.expected_max_sharpe(n, v) for n in (2, 10, 100, 1000)]
    assert b == sorted(b)


def test_the_benchmark_rises_with_the_spread_of_trial_sharpes():
    b = [V.expected_max_sharpe(200, v) for v in (0.01, 0.1, 0.5, 1.0)]
    assert b == sorted(b)


def test_a_single_trial_has_no_selection_bias():
    assert V.expected_max_sharpe(1, 0.5) == 0.0


def test_an_edge_survives_a_narrow_search_and_noise_does_not():
    rng = np.random.default_rng(3)
    trials = rng.normal(0.0, 0.15, 200)
    edge = V.deflated_sharpe(rng.normal(0.12, 1.0, 1000), trials, n_trials=200)
    noise = V.deflated_sharpe(rng.normal(0.0, 1.0, 1000), trials, n_trials=200)
    assert edge.dsr > 0.9 and edge.verdict() == "survives"
    assert noise.dsr < 0.9 and noise.verdict() != "survives"


def test_the_same_edge_stops_surviving_a_wide_search():
    """More diverse trials means more chances to get lucky, so the bar rises."""
    rng = np.random.default_rng(4)
    r = rng.normal(0.12, 1.0, 1000)
    narrow = V.deflated_sharpe(r, rng.normal(0.0, 0.15, 200), n_trials=200)
    wide = V.deflated_sharpe(r, rng.normal(0.0, 1.0, 200), n_trials=200)
    assert narrow.dsr > wide.dsr


def test_the_trial_count_can_be_overridden_with_the_honest_total():
    """Under-counting trials is the failure the statistic exists to prevent."""
    rng = np.random.default_rng(5)
    r = rng.normal(0.12, 1.0, 1000)
    trials = rng.normal(0.0, 0.2, 50)
    honest = V.deflated_sharpe(r, trials, n_trials=5000)
    naive = V.deflated_sharpe(r, trials)
    assert honest.n_trials == 5000 and naive.n_trials == 50
    assert honest.dsr < naive.dsr


def test_positive_skew_makes_a_sharpe_more_credible_not_less():
    """A 0DTE long's returns are violently right-skewed, and the textbook
    Sharpe standard error assumes they are not."""
    base = V.probabilistic_sharpe(0.05, 0.0, 500, skew=0.0, kurtosis=3.0)
    skewed = V.probabilistic_sharpe(0.05, 0.0, 500, skew=2.0, kurtosis=3.0)
    fat = V.probabilistic_sharpe(0.05, 0.0, 500, skew=0.0, kurtosis=12.0)
    assert skewed > base > fat


def test_degenerate_inputs_do_not_explode():
    assert V.sharpe(np.array([])) == 0.0
    assert V.sharpe(np.array([1.0])) == 0.0
    d = V.deflated_sharpe(np.array([1.0, 2.0]), [0.1, 0.2])
    assert np.isfinite(d.sharpe)


# ------------------------------------------------- robust selection benchmark

def test_a_few_catastrophic_trials_do_not_set_the_bar():
    """The first real study run here produced trial Sharpes from -0.5 to -10, and
    the sample variance those implied put the benchmark at 6.44 — a bar nothing
    clears, which makes the DSR uninformative rather than strict."""
    bulk = list(np.random.default_rng(9).normal(0.0, 0.2, 96))
    blowups = [-10.4, -8.7, -6.2, -5.1]
    classical = V.expected_max_sharpe(100, V.trial_variance(bulk + blowups,
                                                            robust=False))
    robust = V.expected_max_sharpe(100, V.trial_variance(bulk + blowups))
    clean = V.expected_max_sharpe(100, V.trial_variance(bulk))
    assert classical > 3 * robust
    assert abs(robust - clean) < 0.3


def test_the_trial_count_is_never_reduced_by_the_robust_scale():
    """Every configuration tried still counts toward N — that is the
    multiple-testing term, and trimming it is the dishonesty the statistic
    exists to catch. Only the dispersion is made robust."""
    trials = [0.1] * 90 + [-9.0] * 10
    d = V.deflated_sharpe(np.random.default_rng(2).normal(0.1, 1, 500), trials)
    assert d.n_trials == 100


def test_a_degenerate_spread_falls_back_rather_than_zeroing_the_bar():
    """A zero benchmark would make every result look significant."""
    assert V.trial_variance([0.5, 0.5, 0.5, 0.5, 0.9]) > 0
