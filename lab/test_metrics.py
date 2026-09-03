#!/usr/bin/env python3
"""Tests for scoring — in particular for the thing a Sharpe ratio hides."""

from __future__ import annotations

import numpy as np
import pytest

from lab.metrics import TRADING_DAYS, score


def test_sharpe_is_annualised_on_the_daily_series():
    daily = np.array([1.0, -1.0] * 50)
    s = score(daily, daily)
    assert s.sharpe == pytest.approx(0.0, abs=1e-9)
    up = np.full(100, 1.0) + np.array([0.1, -0.1] * 50)
    s2 = score(up, up)
    assert s2.sharpe == pytest.approx(up.mean() / up.std(ddof=1)
                                      * np.sqrt(TRADING_DAYS))


def test_max_drawdown_is_on_the_cumulative_curve():
    daily = np.array([10.0, -30.0, 5.0, 20.0])
    assert score(daily, daily).max_drawdown == pytest.approx(-30.0)


def test_a_total_carried_by_one_day_is_reported_as_such():
    """The failure this exists to catch. 1,000 losing days and one huge winner
    has a positive total, and every conventional statistic flatters it."""
    daily = np.concatenate([np.full(1000, -100.0), [200_000.0]])
    s = score(daily, daily)
    assert s.total == pytest.approx(100_000.0)
    assert s.total_ex_top1 == pytest.approx(-100_000.0)
    assert s.top1_share == pytest.approx(2.0)      # the day is 2x the total
    assert s.days_to_half == 1


def test_a_broadly_earned_total_looks_different():
    rng = np.random.default_rng(0)
    daily = rng.normal(100.0, 20.0, 1000)
    s = score(daily, daily)
    assert s.total_ex_top1 > 0.95 * s.total
    assert s.top1_share < 0.01
    assert s.days_to_half > 400


def test_days_to_half_counts_only_winning_days():
    daily = np.array([-5.0, -5.0, 10.0, 10.0])
    assert score(daily, daily).days_to_half == 1


def test_empty_and_single_observations_do_not_explode():
    for daily in (np.array([]), np.array([1.0])):
        s = score(daily, daily)
        assert np.isfinite(s.sharpe) and np.isfinite(s.max_drawdown)
