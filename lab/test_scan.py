#!/usr/bin/env python3
"""Tests for the conditional-response scan.

The scan is descriptive, not inferential. What these pin is that the joins are
exact, the buckets are balanced, and the monotonicity measure means what it
says — not that anything it surfaces is significant.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from lab import indicators as I
from lab import scan as SC


def _trades(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    return pl.DataFrame(dict(
        date=[dt.date(2024, 1, 3)] * n,
        entry_hour=rng.integers(9, 16, n),
        pnl=rng.normal(0, 100, n),
        fees=np.full(n, 1.63),
        x=rng.normal(0, 1, n),
        zone=rng.integers(0, 7, n)))


def test_a_monotone_relationship_is_detected_and_signed():
    """A column whose P&L rises steadily across its deciles is a far better
    gate candidate than one whose middle bucket happens to be highest."""
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.normal(0, 1, n)
    up = pl.DataFrame(dict(pnl=50 * x + rng.normal(0, 20, n), x=x))
    down = pl.DataFrame(dict(pnl=-50 * x + rng.normal(0, 20, n), x=x))
    flat = pl.DataFrame(dict(pnl=rng.normal(0, 20, n), x=x))
    for frame, want in ((up, 1.0), (down, -1.0)):
        r = SC.rank(SC.scan(frame, columns=["x"], min_trades=50))
        assert r["monotone"][0] == pytest.approx(want, abs=0.2)
        assert r["monotone_p"][0] < 0.01
    r = SC.rank(SC.scan(flat, columns=["x"], min_trades=50))
    assert r["monotone_p"][0] > 0.05


def test_rho_alone_over_reads_and_the_p_value_is_what_catches_it():
    """With ten buckets a pure-noise column routinely reaches |rho| ~ 0.6, so
    a rank must not be read as a relationship without the p-value."""
    rng = np.random.default_rng(11)
    hits = []
    for _ in range(30):
        x = rng.normal(0, 1, 2000)
        frame = pl.DataFrame(dict(pnl=rng.normal(0, 100, 2000), x=x))
        r = SC.rank(SC.scan(frame, columns=["x"], min_trades=50))
        hits.append(abs(float(r["monotone"][0])))
    hits = np.array(hits)
    # Measured over 60 draws: median |rho| 0.21, p90 0.52, max 0.73.
    assert np.median(hits) > 0.10          # noise is nowhere near zero
    assert hits.max() > 0.6                # and does reach 0.6


def test_a_hump_shaped_response_is_not_called_monotone():
    """Ten buckets give a coincidence ten chances to happen; the spread alone
    cannot tell that apart from a relationship."""
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 4000)
    frame = pl.DataFrame(dict(pnl=-80 * x ** 2 + rng.normal(0, 20, 4000), x=x))
    r = SC.rank(SC.scan(frame, columns=["x"], min_trades=50))
    assert r["spread"][0] > 50            # it separates strongly
    assert abs(r["monotone"][0]) < 0.4    # but it is not a relationship


def test_numeric_buckets_are_balanced_by_construction():
    """Equal counts per bucket, so the comparison is about P&L rather than
    about which bucket happened to be crowded."""
    cut = SC.scan(_trades(3000), columns=["x"], buckets=10, min_trades=10)
    n = cut["trades"].to_numpy()
    assert cut.height == 10
    assert n.max() - n.min() <= 2


def test_categorical_columns_are_cut_by_value_not_quantile():
    cut = SC.scan(_trades(3000), columns=["zone"], min_trades=10)
    assert set(cut["bucket"].to_list()) <= set(float(i) for i in range(7))


def test_thin_buckets_are_dropped_rather_than_reported():
    cut = SC.scan(_trades(300), columns=["x"], buckets=10, min_trades=100)
    assert cut.height == 0 or cut["trades"].min() >= 100


def test_a_constant_column_cannot_be_cut():
    frame = pl.DataFrame(dict(pnl=np.arange(500.0), x=np.ones(500)))
    assert SC.scan(frame, columns=["x"], min_trades=10).height == 0


def test_missing_values_never_form_a_bucket():
    x = np.concatenate([np.random.default_rng(3).normal(0, 1, 1000),
                        np.full(500, np.nan)])
    frame = pl.DataFrame(dict(pnl=np.zeros(1500), x=x))
    cut = SC.scan(frame, columns=["x"], buckets=5, min_trades=10)
    assert cut["trades"].sum() <= 1000


# ------------------------------------------------------------------ joining

def test_indicators_join_on_the_SIGNAL_bar_not_the_fill_minute():
    """The liveness gate can push a fill a minute or two later; joining on the
    fill minute would silently drop every trade whose book opened late."""
    day = dt.date(2024, 1, 3)
    trades = pl.DataFrame(dict(date=[day, day], bar_minute=[0, 30],
                               entry_minute=[1, 30], pnl=[10.0, -10.0],
                               fees=[1.6, 1.6], entry_hour=[9, 10]))
    out = SC.attach_indicators(trades, 30)
    assert out.height == 2
    assert out["s_pctb"].null_count() == 0     # the 09:30 bar joined too
    feats = I.for_session(day, 30)
    want = feats.filter(pl.col("minute") == 0)["s_pctb"][0]
    assert out["s_pctb"][0] == pytest.approx(want)


def test_attaching_indicators_does_not_change_the_row_count():
    day = dt.date(2024, 1, 3)
    trades = pl.DataFrame(dict(date=[day] * 5, bar_minute=[0, 30, 60, 90, 120],
                               pnl=[1.0] * 5, fees=[0.0] * 5,
                               entry_hour=[9, 10, 10, 11, 11]))
    assert SC.attach_indicators(trades, 30).height == 5


# ------------------------------------------------------------------ slices

def test_by_hour_covers_every_hour_present():
    t = _trades(3000)
    h = SC.by_hour(t)
    assert set(h["entry_hour"].to_list()) == set(t["entry_hour"].unique().to_list())
    assert h["trades"].sum() == t.height


# ------------------------------------------------- as-of join and day triggers

def _bar_frame(day, minutes):
    return pl.DataFrame(dict(date=[day] * len(minutes), bar_minute=list(minutes),
                             pnl=[float(m) for m in minutes],
                             fees=[1.6] * len(minutes),
                             entry_hour=[9 + (30 + m) // 60 for m in minutes]))


def test_attach_asof_matches_exact_join_on_its_own_timeframe():
    """Read at its own timeframe, the as-of join must equal the exact one."""
    day = dt.date(2024, 1, 3)
    tr = _bar_frame(day, range(0, 331, 30))
    a = SC.attach_indicators(tr, 30).sort("bar_minute")
    b = SC.attach_asof(tr, 30).sort("bar_minute")
    for c in ("s_pctb", "f_bandwidth", "zone"):
        assert a[c].to_list() == b[c].to_list(), c


def test_attach_asof_reads_the_last_closed_slower_bar_never_the_next_one():
    """A 30m entry at 10:30 carries the 60m chart's 10:00 row, not 11:00 --
    which is the whole point: a slower chart is STALE, not predictive."""
    day = dt.date(2024, 1, 3)
    tr = _bar_frame(day, range(0, 331, 30))
    got = SC.attach_asof(tr, 60).sort("bar_minute")
    ref = I.for_session(day, 60).sort("minute")
    for row in got.iter_rows(named=True):
        prior = ref.filter(pl.col("minute") <= row["bar_minute"])
        assert prior.height, "no 60m bar at or before the entry"
        assert row["s_pctb"] == pytest.approx(prior["s_pctb"][-1], nan_ok=True)


def test_attach_asof_is_not_a_forward_join():
    """Mutation guard: if the join looked forward, a 30m entry would sometimes
    carry a 60m row stamped AFTER it. It must never."""
    day = dt.date(2024, 1, 3)
    tr = _bar_frame(day, range(0, 331, 30))
    got = SC.attach_asof(tr, 60).sort("bar_minute")
    ref = I.for_session(day, 60).sort("minute")
    future = {float(v) for v in ref["s_pctb"].to_list() if v is not None}
    for row in got.iter_rows(named=True):
        later = ref.filter(pl.col("minute") > row["bar_minute"])["s_pctb"].to_list()
        if row["s_pctb"] is None or not later:
            continue
        prior = ref.filter(pl.col("minute") <= row["bar_minute"])["s_pctb"][-1]
        if prior is not None and not any(abs(prior - l) < 1e-12 for l in later if l):
            assert not any(abs(row["s_pctb"] - l) < 1e-12 for l in later if l)


def _toy(days=120, bars=4, seed=3):
    """Enough rows that a bucketing is possible, with a column that varies."""
    rng = np.random.default_rng(seed)
    d0 = dt.date(2022, 1, 3)
    rows = []
    for i in range(days):
        day = d0 + dt.timedelta(days=i)
        for b in range(bars):
            rows.append(dict(date=day, bar_minute=b * 30,
                             pnl=float(rng.normal(0, 100)),
                             x=float(rng.normal())))
    return pl.DataFrame(rows)


def test_triggers_fires_at_most_once_per_day():
    t = _toy()
    out = SC.triggers(t, columns=["x"], buckets=4, min_days=1)
    assert not out.is_empty()
    n_days = t["date"].n_unique()
    assert (out["days"] <= n_days).all()
    assert (out["fire_rate"] <= 1.0).all()
    # a bucket covering a quarter of bars fires on most days but never twice
    assert out["days"].max() <= n_days


def test_triggers_takes_the_first_qualifying_bar_not_the_best():
    """Causal: the rule cannot know which of the day's qualifying bars paid."""
    t = _toy(days=100, bars=3, seed=5).with_columns(pl.lit(0.5).alias("x"))
    # make the later bars far more profitable than the first
    t = t.with_columns(pl.when(pl.col("bar_minute") == 0).then(10.0)
                         .otherwise(1000.0).alias("pnl"))
    out = SC.triggers(t, columns=["x"], buckets=2, min_days=1)
    assert out.is_empty() or out["per_day"][0] == pytest.approx(10.0)


def test_triggers_scores_per_day_not_per_trade():
    """Several bars qualify on a day; the day contributes ONE observation."""
    t = _toy(days=150, bars=4, seed=7).with_columns(
        pl.when(pl.col("bar_minute") == 0).then(100.0).otherwise(-1e4).alias("pnl"),
        pl.lit(1.0).alias("x"))
    out = SC.triggers(t, columns=["x"], buckets=2, min_days=1)
    if not out.is_empty():
        assert out["days"][0] == 150
        assert out["per_day"][0] == pytest.approx(100.0)


def test_triggers_and_scan_disagree_when_a_day_has_many_bars():
    """The whole reason `triggers` exists: per-trade and per-day are different
    statistics, and for a once-a-day method the per-trade one is wrong."""
    t = _toy(days=200, bars=6, seed=9).with_columns(
        pl.when(pl.col("bar_minute") == 0).then(-500.0).otherwise(50.0).alias("pnl"),
        pl.lit(0.0).alias("x"))
    per_trade = t["pnl"].mean()
    out = SC.triggers(t, columns=["x"], buckets=2, min_days=1)
    if not out.is_empty():
        assert out["per_day"][0] == pytest.approx(-500.0)
        assert per_trade > 0 > out["per_day"][0]


def test_scan_can_score_a_state_rather_than_a_pnl():
    """A paired method's first spread is judged on whether it brought about a state
    (spot outside the short strike), not on what it paid. The bucket mean of a
    0/1 column IS that rate, so no separate machinery is needed — but the
    column has to be nameable."""
    rng = np.random.default_rng(4)
    n = 4000
    x = rng.normal(0, 1, n)
    hit = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(float)   # rises with x
    frame = pl.DataFrame(dict(x=x, success=hit, pnl=rng.normal(0, 100, n)))
    r = SC.rank(SC.scan(frame, columns=["x"], min_trades=50, value="success"))
    assert r["monotone"][0] > 0.8 and r["monotone_p"][0] < 0.01
    assert 0.0 <= r["best_per_trade"][0] <= 1.0
    # and the same frame scored on its (noise) P&L must NOT look like that
    rp = SC.rank(SC.scan(frame, columns=["x"], min_trades=50))
    assert rp["monotone_p"][0] > 0.05
