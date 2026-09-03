#!/usr/bin/env python3
"""Tests for the method protocol and the session runner, on real sessions."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from lab import indicators as I
from lab.exits import COVER, NONE, SHORT, ExitPolicy
from lab.fills import CROSS, MID
from lab.examples import ABOVE_MIDS, BELOW_MIDS, ZoneEntry
from lab.runner import Signal, run_session
from lab.select import live_minute, long_by_price
from lab.session import CALL, PUT, cached

DAY = dt.date(2024, 1, 3)


@pytest.fixture(scope="module")
def sess():
    return cached(DAY)


class OneShot:
    """A method with a single hardcoded signal — the runner under a microscope."""

    name = "ONE"

    def __init__(self, minute=30, right=CALL, target=2.0, qty=1):
        self.args = (minute, right, target, qty)

    def features(self, sess):
        return I.for_session(sess.date, 30)

    def signals(self, sess):
        m, r, t, q = self.args
        m = live_minute(sess, m)
        if m is None:
            return []
        got = long_by_price(sess, r, t, m)
        if got is None:
            return []
        leg, miss = got
        return [Signal(minute=m, legs=(leg._replace_qty(q),), tag=f"ONE {r}",
                       target_miss=miss)]


# ---------------------------------------------------------------- the runner

def test_a_signal_becomes_a_trade_with_its_own_basis(sess):
    run = run_session(sess, OneShot(), ExitPolicy(), CROSS)
    assert len(run.trades) == 1
    t = run.trades[0]
    assert t.entry_minute == 30 and t.entry_price > 0
    assert run.reconciles


def test_trade_pnl_always_sums_to_the_account(sess):
    """The invariant that catches an attribution mistake. Trades own their own
    cash so the rows are meaningful; the broker owns the account."""
    for pol in (ExitPolicy(), ExitPolicy(w=2.0), ExitPolicy(w=2.0, l=0.5),
                ExitPolicy(w=2.0, w_action=COVER),
                ExitPolicy(l=0.5, l_action=SHORT, short_width=3),
                ExitPolicy(l=0.5, l_action=NONE)):
        for model in (MID, CROSS):
            run = run_session(sess, ZoneEntry(timeframe=30), pol, model)
            assert run.reconciles, f"{pol.label()} / {model.label()}"


def test_the_liveness_gate_moves_a_0930_signal_to_the_first_real_book(sess):
    run = run_session(sess, OneShot(minute=0), ExitPolicy(), CROSS)
    assert run.trades[0].entry_minute == sess.first_live()


def test_an_unreachable_target_still_fills_at_a_price_the_chain_showed(sess):
    """A million-dollar target finds the nearest priced strike and reports the
    miss. What must never happen is a fill at a price the book did not carry —
    so the entry is the ASK of the picked strike, not the mid the search
    ranked on."""
    run = run_session(sess, OneShot(minute=30, target=1e6), ExitPolicy(), CROSS)
    pick = sess.by_price(CALL, 1e6, 30)
    t = run.trades[0]
    _, ask = sess.quote(sess.contract(CALL, pick.strike), 30)
    assert t.entry_price == pytest.approx(ask)
    assert t.target_miss == pytest.approx(pick.price - 1e6)


def test_rows_carry_the_entry_hour_and_the_opening_zone(sess):
    run = run_session(sess, ZoneEntry(timeframe=30), ExitPolicy(w=2.0), CROSS)
    rows = run.rows()
    assert rows and all(9 <= r["entry_hour"] <= 15 for r in rows)
    assert all(r["open_zone"] in BELOW_MIDS + ABOVE_MIDS for r in rows)
    assert {r["exit_reason"] for r in rows} <= {"W", "L", "settle"}
    assert all(r["legs"] == 1 for r in rows)


def test_the_frame_is_one_row_per_trade_not_per_session(sess):
    run = run_session(sess, ZoneEntry(timeframe=30), ExitPolicy(), CROSS)
    df = pl.DataFrame(run.rows())
    assert df.height == len(run.trades) > 1
    assert df["date"].n_unique() == 1


# ------------------------------------------------------------------- ZONE

def test_abot_buys_calls_low_and_puts_high(sess):
    """A call when the bar opens beneath both mids, a put when above both,
    and nothing at all when it opens between them."""
    feats = I.for_session(DAY, 30)
    run = run_session(sess, ZoneEntry(timeframe=30), ExitPolicy(), CROSS)
    by_minute = {t.entry_minute: t for t in run.trades}
    for row in feats.iter_rows(named=True):
        m, z = row["minute"], row["zone"]
        if z == I.M:
            assert m not in by_minute
        elif z in BELOW_MIDS and m in by_minute:
            assert by_minute[m].tag.endswith(CALL)
        elif z in ABOVE_MIDS and m in by_minute:
            assert by_minute[m].tag.endswith(PUT)


def test_abot_respects_its_last_entry_time(sess):
    run = run_session(sess, ZoneEntry(timeframe=30, last_entry="12:00"),
                      ExitPolicy(), CROSS)
    assert max(t.entry_minute for t in run.trades) <= sess.minute_of("12:00") + 5


@pytest.mark.parametrize("tf", [15, 30, 60])
def test_abot_runs_on_every_timeframe_the_bible_asks_for(tf):
    run = run_session(cached(DAY), ZoneEntry(timeframe=tf), ExitPolicy(w=2.0, l=0.5),
                      CROSS)
    assert run.reconciles and len(run.trades) >= 1


# ------------------------------------------------------- policy sanity, real

@pytest.mark.realdata
def test_a_stop_cannot_lose_more_than_holding_on_a_single_long(sess):
    """Not a law of markets — a law of this instrument. A long option's worst
    case is total loss, so a stop can only ever cut it short."""
    held = run_session(sess, ZoneEntry(timeframe=30), ExitPolicy(), CROSS)
    stopped = run_session(sess, ZoneEntry(timeframe=30), ExitPolicy(l=0.5), CROSS)
    for a, b in zip(held.trades, stopped.trades):
        assert b.pnl >= a.pnl - 1e-6 or b.exit_reason == "settle"


def test_a_take_profit_never_fires_below_its_level(sess):
    pol = ExitPolicy(w=2.5)
    run = run_session(sess, ZoneEntry(timeframe=15), pol, CROSS)
    for t in run.trades:
        if t.exit_reason == "W":
            assert t.exit_price == pytest.approx(2.5 * t.entry_price)
