#!/usr/bin/env python3
"""Tests for regime classification, and for the rule/report boundary."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from lab import regimes as R
from lab.regimes import ELEVATED, EXTREME, NORMAL, RegimeConfig


def _frame(rows):
    """rows: (date, open_spot, settle, straddle). Everything else derived."""
    return pl.DataFrame([
        dict(date=d, open_spot=o, settle=c, atm=round(o / 5) * 5.0,
             straddle=st, vix=15.0, excursion=abs(c - o))
        for d, o, c, st in rows])


def _days(n, start=dt.date(2024, 1, 2)):
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += dt.timedelta(days=1)
    return out


# ------------------------------------------------------------------- tiers

def test_a_mispriced_day_is_extreme_even_at_a_small_return():
    """2025-10-10's shape: a 191-point move against a $22.90 straddle. Only
    -2.8% but 8.4x what the market priced."""
    d = _days(1)[0]
    out = R._derive(_frame([(d, 6743.78, 6552.51, 22.90)]))
    assert out["move_ratio"][0] == pytest.approx(8.35, abs=0.01)
    assert out["tier"][0] == EXTREME


def test_a_huge_day_is_extreme_even_when_the_straddle_was_expensive():
    """2025-04-09's shape: +9.99% but the straddle was $157.80, so the
    mispricing is only 3.1x. Magnitude alone must still catch it."""
    d = _days(1)[0]
    out = R._derive(_frame([(d, 4959.0, 5454.6, 157.80)]))
    assert out["move_ratio"][0] < RegimeConfig().extreme_ratio
    assert out["abs_return"][0] > RegimeConfig().extreme_return
    assert out["tier"][0] == EXTREME


def test_the_two_axes_are_an_OR_not_an_AND():
    cfg = RegimeConfig()
    d = _days(2)
    # Neither axis crosses: normal.
    out = R._derive(_frame([(d[0], 5000.0, 5005.0, 30.0),
                            (d[1], 5000.0, 5005.0, 30.0)]))
    assert set(out["tier"].to_list()) == {NORMAL}


def test_an_ordinary_day_is_normal():
    d = _days(1)[0]
    out = R._derive(_frame([(d, 5000.0, 5012.0, 25.0)]))   # ratio 0.48, ret 0.24%
    assert out["tier"][0] == NORMAL


def test_elevated_sits_between():
    d = _days(1)[0]
    out = R._derive(_frame([(d, 5000.0, 5075.0, 28.0)]))   # ratio 2.68, ret 1.5%
    assert out["tier"][0] == ELEVATED


# ---------------------------------------------------------------- episodes

def test_nearby_extremes_are_one_event():
    d = _days(10)
    rows = [(x, 5000.0, 5005.0, 30.0) for x in d]
    rows[2] = (d[2], 5000.0, 5200.0, 30.0)     # extreme
    rows[4] = (d[4], 5000.0, 5200.0, 30.0)     # extreme, 2 sessions later
    out = R._derive(_frame(rows))
    ex = out.filter(pl.col("tier") == EXTREME)
    assert ex.height == 2 and ex["episode"].n_unique() == 1


def test_distant_extremes_are_separate_events():
    d = _days(20)
    rows = [(x, 5000.0, 5005.0, 30.0) for x in d]
    rows[2] = (d[2], 5000.0, 5200.0, 30.0)
    rows[15] = (d[15], 5000.0, 5200.0, 30.0)
    out = R._derive(_frame(rows))
    assert out.filter(pl.col("tier") == EXTREME)["episode"].n_unique() == 2


def test_days_from_extreme_measures_the_neighbourhood():
    d = _days(10)
    rows = [(x, 5000.0, 5005.0, 30.0) for x in d]
    rows[5] = (d[5], 5000.0, 5200.0, 30.0)
    out = R._derive(_frame(rows))
    assert out["days_from_extreme"].to_list() == [5, 4, 3, 2, 1, 0, 1, 2, 3, 4]


# ------------------------------------------------------- the safety boundary

def test_safe_and_reporting_columns_do_not_overlap():
    assert not (set(R.SAFE_FOR_RULES) & set(R.REPORTING_ONLY))


def test_rule_safe_hides_everything_computed_from_the_realised_day():
    """Feeding `move_ratio` to an entry rule would be perfect foresight of the
    exact thing being predicted."""
    safe = R.rule_safe()
    for c in R.REPORTING_ONLY:
        assert c not in safe.columns
    assert "straddle" in safe.columns and "vix" in safe.columns


def test_the_table_carries_both_kinds_and_nothing_stray():
    t = R.table()
    known = set(R.SAFE_FOR_RULES) | set(R.REPORTING_ONLY)
    assert set(t.columns) - known == set()


# --------------------------------------------------------- the real table

@pytest.mark.realdata
def test_the_real_table_covers_the_calendar():
    t = R.table()
    assert 1_800 < t.height <= 1_894
    assert t["tier"].is_in(list(R.TIERS)).all()


@pytest.mark.realdata
def test_the_known_extremes_are_classified_as_such():
    t = R.table()
    for d in ("2025-04-09", "2025-10-10", "2024-12-18", "2018-02-05",
              "2020-03-20"):
        row = t.filter(pl.col("date") == dt.date.fromisoformat(d))
        assert row.height == 1 and row["tier"][0] == EXTREME, d


def test_extremes_are_rare_and_normals_dominate():
    t = R.table()
    share = t.group_by("tier").len()
    frac = {r["tier"]: r["len"] / t.height for r in share.to_dicts()}
    assert frac[EXTREME] < 0.03
    assert frac[NORMAL] > 0.85


def test_prior_vol_is_present_and_uses_prior_sessions_only():
    t = R.table()
    v = t["prior_vol"].to_numpy()
    assert np.isfinite(v).mean() > 0.98
    assert np.nanmedian(v) < 0.02      # daily log-return vol, not annualised


# --------------------------------------------------------------- splitting

@pytest.mark.realdata
def test_split_tiers_sum_to_the_whole():
    trades = pl.DataFrame(dict(
        date=[dt.date(2025, 4, 9), dt.date(2024, 6, 3), dt.date(2024, 6, 4)],
        pnl=[1000.0, -50.0, -25.0], fees=[1.6, 1.6, 1.6]))
    s = R.split(trades)
    by = {r["tier"]: r["total"] for r in s.to_dicts()}
    assert by["all"] == pytest.approx(925.0)
    tiers = sum(by.get(t, 0.0) for t in R.TIERS)
    assert tiers == pytest.approx(by["all"])
    assert by["ex-extreme"] == pytest.approx(-75.0)


def test_attach_does_not_change_the_row_count():
    trades = pl.DataFrame(dict(date=[dt.date(2025, 4, 9)] * 3,
                               pnl=[1.0, 2.0, 3.0], fees=[0.0] * 3))
    assert R.attach(trades).height == 3
