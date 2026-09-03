#!/usr/bin/env python3
"""Tests for indicators computed on the traded option's OWN price chart.

The specification asks for entries and exits on both the SPX chart and the option chart,
and for mixing them on one order. The metric vocabulary is shared with the SPX
frame deliberately — two implementations of `zone` or `%b` would let the two
drift, and a gate named `s_pctb` would then mean different things depending on
where it was pointed.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from lab import gates as G
from lab import indicators as I
from lab.exits import ExitPolicy
from lab.fills import MID
from lab.indicators import ChartSpec
from lab.examples import ZoneEntry
from lab.runner import run_session
from lab.session import CALL, PUT, cached

DAY = dt.date(2024, 1, 3)


@pytest.fixture(scope="module")
def sess():
    return cached(DAY)


@pytest.fixture(scope="module")
def contract(sess):
    return sess.contract(CALL, sess.by_price(CALL, 2.00, 30).strike)


# -------------------------------------------------------------------- bars

def test_one_minute_bars_are_one_per_minute(sess, contract):
    b = I.option_bars(sess, contract, ChartSpec(timeframe=1))
    assert b.height <= sess.n_minutes
    assert b["minute"].is_sorted() and b["minute"].n_unique() == b.height


def test_bars_are_aligned_to_the_session_open(sess, contract):
    """Bar k covers [k*tf, (k+1)*tf), so a spot gate and an option gate on the
    same timeframe refer to the same instants."""
    for tf in (1, 5, 15):
        b = I.option_bars(sess, contract, ChartSpec(timeframe=tf))
        assert (np.asarray(b["minute"].to_list()) % tf == 0).all()


def test_a_five_minute_bar_aggregates_its_minutes(sess, contract):
    one = I.option_bars(sess, contract, ChartSpec(timeframe=1))
    five = I.option_bars(sess, contract, ChartSpec(timeframe=5))
    m = 100
    window = one.filter((pl.col("minute") >= m) & (pl.col("minute") < m + 5))
    row = five.filter(pl.col("minute") == m)
    if window.height and row.height:
        assert row["high"][0] == pytest.approx(window["high"].max())
        assert row["low"][0] == pytest.approx(window["low"].min())
        assert row["close"][0] == pytest.approx(window["close"][-1])


def test_the_mid_source_is_the_quote_midpoint(sess, contract):
    b = I.option_bars(sess, contract, ChartSpec(timeframe=1, source="mid"))
    m = int(b["minute"][50])
    assert b["close"][50] == pytest.approx(sess.mid(contract, m))


def test_the_trade_source_is_the_printed_ohlc(sess, contract):
    b = I.option_bars(sess, contract, ChartSpec(timeframe=1, source="trade"))
    m = int(b["minute"][50])
    assert b["close"][50] == pytest.approx(
        sess.value(contract.right, contract.strike, "last", m))


def test_a_contract_not_in_the_chain_gives_an_empty_chart(sess):
    b = I.option_bars(sess, sess.contract(CALL, 99999.0), ChartSpec())
    assert b.height == 0
    assert I.option_features(sess, sess.contract(CALL, 99999.0)).height == 0


# ------------------------------------------------- session-bounded by nature

def test_coarse_timeframes_are_refused_with_the_reason():
    """A 0DTE contract exists for one day, so a 20-period band at 30m would
    need more history than the contract has."""
    for tf in (30, 60):
        with pytest.raises(ValueError, match="session-bounded"):
            ChartSpec(timeframe=tf).validate()


def test_the_slow_band_is_ready_exactly_twenty_bars_in(sess, contract):
    for tf, expected in ((1, 20), (5, 100)):
        f = I.option_features(sess, contract, ChartSpec(timeframe=tf))
        ready = f.filter(pl.col("SM").is_not_null())
        assert int(ready["minute"].min()) == expected


def test_there_is_no_leakage_from_a_previous_session(sess, contract):
    """The SPX series is continuous and has a band from minute 0; an option's
    cannot, and must not pretend to."""
    spx = I.for_session(DAY, 5)
    assert spx.filter(pl.col("SM").is_not_null())["minute"].min() == 0
    opt = I.option_features(sess, contract, ChartSpec(timeframe=5))
    assert opt.filter(pl.col("SM").is_not_null())["minute"].min() > 0


# ------------------------------------------------------ shared vocabulary

def test_the_option_frame_carries_every_metric_the_spx_frame_does(sess, contract):
    f = I.option_features(sess, contract, ChartSpec(timeframe=5))
    assert set(I.METRICS + I.BANDS).issubset(f.columns)


def test_zones_and_pctb_are_computed_the_same_way(sess, contract):
    f = I.option_features(sess, contract, ChartSpec(timeframe=5)).drop_nulls("SM")
    z = f["zone"].to_numpy()
    assert ((z >= 0) & (z <= I.BH)).all()
    pb = f["s_pctb"].to_numpy()
    op, sm, sh, sl = (f[c].to_numpy() for c in ("open", "SM", "SH", "SL"))
    ok = np.isfinite(pb)
    assert np.allclose(pb[ok], ((op - sm) / ((sh - sl) / 2.0))[ok])


def test_option_metrics_never_read_their_own_bar(sess, contract):
    """The same lookahead rule as the SPX chart: bands from closed bars only."""
    f = I.option_features(sess, contract, ChartSpec(
        timeframe=1, bands=I.BandConfig(fast=3, slow=5)))
    src = f["close"].to_numpy()
    for t in (50, 120, 200):
        assert f["SM"][t] == pytest.approx(src[t - 5:t].mean())


# ------------------------------------------------------------- gate routing

def test_a_mixed_gateset_splits_by_chart():
    gs = G.all_of(G.zone_in(I.BL, I.UL), G.pctb("s_pctb", hi=-0.3, chart=G.OPTION))
    assert len(gs.for_chart(G.SPOT).gates) == 1
    assert len(gs.for_chart(G.OPTION).gates) == 1
    assert "opt:" in gs.label()


def test_an_any_gateset_cannot_span_both_charts():
    """The two frames have different rows, so the disjunction is not defined."""
    gs = G.any_of(G.zone_in(I.BL), G.zone_in(I.M, chart=G.OPTION))
    with pytest.raises(ValueError, match="cannot span both charts"):
        gs.for_chart(G.SPOT)


# ------------------------------------------------------- through the runner

def test_an_option_entry_gate_is_applied_after_the_strike_is_known(sess):
    """Until a strike is chosen there is no option to have a chart, so the
    gate cannot live inside the method."""
    spec = ChartSpec(timeframe=5)
    plain = run_session(sess, ZoneEntry(timeframe=5, target=2.0),
                        ExitPolicy(), MID)
    gated = run_session(sess, ZoneEntry(timeframe=5, target=2.0, option_chart=spec,
                                   entry_gate=G.all_of(
                                       G.pctb("s_pctb", hi=-0.5, chart=G.OPTION))),
                        ExitPolicy(), MID)
    assert len(gated.trades) < len(plain.trades)
    assert gated.refused > 0 and gated.reconciles


def test_an_option_exit_gate_closes_positions(sess):
    spec = ChartSpec(timeframe=5)
    pol = ExitPolicy(option_chart=spec,
                     exit_gate=G.all_of(G.pctb("s_pctb", lo=1.0, chart=G.OPTION)))
    run = run_session(sess, ZoneEntry(timeframe=5, target=2.0), pol, MID)
    assert any(t.exit_reason == "gate" for t in run.trades)
    assert run.reconciles


def test_spot_entry_with_an_option_exit_is_expressible(sess):
    """The specification's mixed order: SPX price action triggers the entry, the
    option's own Bollingers trigger the exit."""
    spec = ChartSpec(timeframe=5)
    run = run_session(
        sess,
        ZoneEntry(timeframe=5, target=2.0,
             entry_gate=G.all_of(G.zone_in(I.BL, I.UL, I.L))),
        ExitPolicy(w=3.0, l=0.4, option_chart=spec,
                   exit_gate=G.all_of(G.pctb("s_pctb", lo=1.0, chart=G.OPTION))),
        MID)
    reasons = {t.exit_reason for t in run.trades}
    assert "gate" in reasons and run.reconciles


def test_both_halves_of_a_mixed_exit_gate_must_agree(sess):
    """A mixed `all` set fires only where both charts pass, so it can only fire
    less often than either half alone."""
    spec = ChartSpec(timeframe=5)
    opt = G.pctb("s_pctb", lo=1.0, chart=G.OPTION)
    spot = G.zone_in(I.H, I.UH, I.BH)
    counts = []
    for gs in (G.all_of(opt), G.all_of(spot), G.all_of(opt, spot)):
        run = run_session(sess, ZoneEntry(timeframe=5, target=2.0),
                          ExitPolicy(option_chart=spec, exit_gate=gs), MID)
        counts.append(sum(t.exit_reason == "gate" for t in run.trades))
    assert counts[2] <= min(counts[0], counts[1])
