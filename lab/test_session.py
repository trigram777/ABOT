#!/usr/bin/env python3
"""Tests for the data layer, against the real dataset.

These are deliberately not fixture tests. What they assert is that the dataset
has the shape the engine assumes — the 09:30 non-book, the tz-correct minute
axis, hygiene actually applied, settlement present — and a fixture cannot fail
when the dataset changes underneath it. They load two real sessions and run in
about a second.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from lab import session as S

DAY = dt.date(2024, 1, 3)
OLD = dt.date(2019, 3, 15)


@pytest.fixture(scope="module")
def sess():
    return S.load(DAY)


# ----------------------------------------------------------------- calendar

@pytest.mark.realdata
def test_calendar_is_the_eligible_intersection():
    every = S.calendar(eligible_only=False)
    elig = S.calendar(eligible_only=True)
    assert len(elig) == 1894 and len(every) == 1895
    # The one difference is the known IBKR spot gap, which is exactly the kind
    # of session a path-dependent rule must not see.
    assert set(every) - set(elig) == {dt.date(2020, 12, 7)}


def test_calendar_windows():
    days = S.calendar(start=dt.date(2024, 1, 1), end=dt.date(2024, 12, 31))
    assert days[0].year == days[-1].year == 2024


# ------------------------------------------------------------------- shape

def test_the_minute_axis_is_rth_inclusive_of_the_close(sess):
    assert sess.n_minutes == 391
    assert sess.clock(0) == "09:30" and sess.clock(390) == "16:00"
    assert sess.minute_of("10:00") == 30


def test_the_first_minute_is_not_a_book(sess):
    """09:30:00 is the pre-rotation snapshot: the option file reports no
    underlying at all. Every rule must gate on this, and the spot series
    cannot tell you — IBKR does print a 09:30 bar."""
    assert not sess.book_reported[0]
    assert sess.book_reported[1] and sess.book_reported[30]
    assert sess.spot_reported[0]


def test_the_close_minute_carries_spot_rather_than_inventing_it(sess):
    assert not sess.spot_reported[390]
    assert sess.spot_open[390] == pytest.approx(sess.spot_close[389])


def test_settlement_comes_from_the_daily_close_not_the_1559_bar(sess):
    assert np.isfinite(sess.settle)
    assert sess.settle != sess.spot_close[389]


def test_opening_vix_is_present_and_plausible(sess):
    assert 5.0 < sess.vix < 90.0


# ---------------------------------------------------------------- hygiene

def test_no_zero_bids_and_no_crossed_books_survive_the_load(sess):
    for right in (S.CALL, S.PUT):
        bid = sess.arrays[(right, "bid")]
        ask = sess.arrays[(right, "ask")]
        live = np.isfinite(bid)
        assert (bid[live] > 0).all()
        both = live & np.isfinite(ask)
        assert (ask[both] >= bid[both]).all()


def test_a_bidless_strike_is_untradeable_rather_than_half_priced(sess):
    """The failure this prevents: a 0.00/8.00 quote whose midpoint of $4.00 is
    a credit no counterparty ever offered."""
    m = sess.minute_of("15:30")
    far = sess.strikes[S.CALL][-1]
    c = sess.contract(S.CALL, far)
    bid, ask = sess.quote(c, m)
    assert bid is None or bid > 0
    if bid is None:
        assert not sess.tradeable(c, m)


# ------------------------------------------------------- lookahead convention

def test_the_quote_is_the_book_at_the_start_of_its_minute(sess):
    """The measurement the whole timing convention rests on: a minute's quote
    tracks that minute's OPEN print far better than its own close, so a rule
    firing at m:00 and filling on bid[m] is reading what it could see."""
    m = sess.minute_of("11:00")
    spot = sess.spot(m)
    errs_open, errs_close = [], []
    for right in (S.CALL, S.PUT):
        ks = sess.strikes[right]
        near = ks[np.abs(ks - spot) < 30]
        for k in near:
            c = sess.contract(right, k)
            mid = sess.mid(c, m)
            o = sess.value(right, k, "last", m - 1)   # prior minute's close
            cl = sess.value(right, k, "last", m)      # this minute's close
            vol = sess.value(right, k, "volume", m)
            if None in (mid, o, cl) or not vol or vol < 20:
                continue
            errs_open.append(abs(mid - o))
            errs_close.append(abs(mid - cl))
    assert len(errs_open) >= 5
    assert np.mean(errs_open) < np.mean(errs_close)


def test_spot_open_is_the_decision_price_and_close_is_lookahead(sess):
    m = sess.minute_of("10:00")
    assert sess.spot(m) == sess.spot_open[m]
    assert sess.spot_open[m] != sess.spot_close[m]


# ----------------------------------------------------------- strike selection

def test_atm_is_the_nearest_listed_strike(sess):
    m = sess.minute_of("10:00")
    k = sess.atm(S.CALL, m)
    assert abs(k - sess.spot(m)) <= 2.5 + 1e-9
    assert k in set(sess.strikes[S.CALL].tolist())


def test_by_price_finds_the_nearest_priced_strike_and_reports_the_miss(sess):
    m = sess.minute_of("10:00")
    pick = sess.by_price(S.CALL, 2.00, m)
    assert pick is not None
    assert pick.miss == pytest.approx(pick.price - 2.00)
    assert pick.target_met is None            # no tolerance supplied, no verdict
    assert sess.by_price(S.CALL, 2.00, m, tol=0.25).target_met is not None


def test_by_price_at_the_money_falls_back_to_distance(sess):
    m = sess.minute_of("10:00")
    pick = sess.by_price(S.CALL, 0.0, m)
    assert pick.strike == sess.atm(S.CALL, m)


def test_by_price_stays_out_of_the_money_by_default(sess):
    m = sess.minute_of("10:00")
    spot = sess.spot(m)
    assert sess.by_price(S.CALL, 1.0, m).strike >= spot
    assert sess.by_price(S.PUT, 1.0, m).strike <= spot


def test_step_walks_outward_on_both_rights(sess):
    m = sess.minute_of("10:00")
    kc, kp = sess.atm(S.CALL, m), sess.atm(S.PUT, m)
    assert sess.step(S.CALL, kc, 2) > kc          # calls go OTM upward
    assert sess.step(S.PUT, kp, 2) < kp           # puts downward
    assert sess.step(S.CALL, kc, -1) < kc
    assert sess.step(S.CALL, sess.strikes[S.CALL][-1], 1) is None


def test_by_delta_lands_near_the_asked_delta(sess):
    m = sess.minute_of("11:00")
    k = sess.by_delta(S.CALL, 0.25, m)
    d = sess.value(S.CALL, k, "delta", m)
    assert d is not None and 0.10 < abs(d) < 0.45


# ----------------------------------------------------------- older sessions

@pytest.mark.realdata
def test_an_early_narrow_session_still_loads():
    s = S.load(OLD)
    assert s.n_minutes == 391 and s.strikes[S.CALL].size >= 20
    assert np.isfinite(s.settle)


# ------------------------------------------------------------------- bars

@pytest.mark.realdata
@pytest.mark.parametrize("tf", [1, 5, 15, 30, 60])
def test_every_timeframe_is_one_continuous_series(tf):
    b = S.bars(tf)
    assert b.height > 15_000
    assert b["timestamp"].is_sorted()
    assert b["date"].n_unique() > 2_300


# ------------------------------------------------------------- liveness gate

def test_first_live_skips_the_pre_rotation_snapshot(sess):
    assert sess.first_live() == 1
    assert sess.entry_minute("09:30") == 1
    assert sess.entry_minute("10:00") == sess.minute_of("10:00")


@pytest.mark.realdata
def test_the_one_session_that_opens_late_is_found_not_assumed():
    """2019-06-14 does not report an option book until 09:41. A rule that
    hardcoded 09:31 would trade it against a snapshot that is not a book."""
    s = S.load(dt.date(2019, 6, 14))
    assert not s.book_reported[1]
    assert s.first_live() == s.minute_of("09:41")


def test_first_live_gives_up_rather_than_searching_all_day(sess):
    dead = S.load(DAY)
    dead.book_reported[:] = False
    assert dead.first_live(window=30) is None
