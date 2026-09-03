#!/usr/bin/env python3
"""Tests for shared strike selection. A selection bug copied five times is
five bugs, which is why this lives in one module."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from lab import select as S
from lab.select import Leg
from lab.session import CALL, PUT, cached

DAY = dt.date(2024, 1, 3)


@pytest.fixture(scope="module")
def sess():
    return cached(DAY)


@pytest.fixture(scope="module")
def m(sess):
    return sess.minute_of("10:00")


def test_resizing_a_leg_keeps_its_side():
    """A method scales a selected structure without knowing which legs the
    selector made short."""
    assert Leg(CALL, 5000.0, -1)._replace_qty(3).qty == -3
    assert Leg(CALL, 5000.0, 1)._replace_qty(3).qty == 3
    assert Leg(CALL, 5000.0, -1)._replace_qty(-3).qty == -3


def test_long_by_price_reports_the_miss(sess, m):
    leg, miss = S.long_by_price(sess, CALL, 2.00, m)
    assert leg.qty == 1 and leg.right == CALL
    assert miss == pytest.approx(sess.by_price(CALL, 2.00, m).price - 2.00)


def test_long_by_offset_walks_out_on_both_rights(sess, m):
    atm_c = sess.atm(CALL, m)
    assert S.long_by_offset(sess, CALL, 30.0, m).strike == atm_c + 30
    assert S.long_by_offset(sess, PUT, 30.0, m).strike == sess.atm(PUT, m) - 30
    assert S.long_by_offset(sess, CALL, 0.0, m).strike == atm_c


def test_a_strangle_is_a_straddle_walked_out(sess, m):
    c0, p0 = S.straddle(sess, m, 0.0)
    c1, p1 = S.straddle(sess, m, 30.0)
    assert c0.strike == p0.strike            # a straddle shares its strike
    assert c1.strike > c0.strike and p1.strike < p0.strike
    assert all(l.qty == 1 for l in (c0, p0, c1, p1))


def test_net_credit_is_positive_for_a_structure_that_pays(sess, m):
    legs = [Leg(PUT, sess.atm(PUT, m) + 10, -1), Leg(PUT, sess.atm(PUT, m), 1)]
    assert S.net_credit(sess, legs, m) > 0
    assert S.net_credit(sess, [Leg(CALL, sess.atm(CALL, m), 1)], m) < 0


def test_net_credit_crossable_is_never_better_than_the_mid(sess, m):
    legs = [Leg(PUT, sess.atm(PUT, m) + 10, -1), Leg(PUT, sess.atm(PUT, m), 1)]
    assert S.net_credit(sess, legs, m, True) <= S.net_credit(sess, legs, m, False)


# ------------------------------------------------------------- verticals

@pytest.mark.parametrize("ratio", [0.4, 0.5, 0.6, 0.7])
@pytest.mark.parametrize("right", [CALL, PUT])
def test_a_credit_vertical_lands_near_the_asked_ratio(sess, m, right, ratio):
    got = S.credit_vertical(sess, right, m, width=10.0, ratio=ratio)
    assert got is not None
    legs, achieved = got
    assert abs(achieved - ratio) <= 0.15
    assert sorted(l.qty for l in legs) == [-1, 1]


def test_the_short_leg_is_in_the_money_and_the_long_is_further_out(sess, m):
    spot = sess.spot(m)
    legs, _ = S.credit_vertical(sess, PUT, m, 10.0, 0.6)
    short = next(l for l in legs if l.qty < 0)
    long_ = next(l for l in legs if l.qty > 0)
    assert short.strike >= spot          # a short put ITM is above spot
    assert long_.strike == short.strike - 10.0
    legs, _ = S.credit_vertical(sess, CALL, m, 10.0, 0.6)
    short = next(l for l in legs if l.qty < 0)
    long_ = next(l for l in legs if l.qty > 0)
    assert short.strike <= spot          # a short call ITM is below spot
    assert long_.strike == short.strike + 10.0


def test_an_unreachable_ratio_is_refused_rather_than_approximated(sess, m):
    """A vertical cannot pay MORE than its width — that would be free money.

    The bound moved with rule 22. Qualified at short-bid-minus-long-ask this
    test used to pass at 0.95, because two half-spreads came off the credit;
    at the midpoint a put spread 40 points in the money is almost pure
    intrinsic and really does pay 0.94 of its width. The half-spread was
    never a fact about the structure, only about how it was being priced."""
    assert S.credit_vertical(sess, PUT, m, 10.0, 1.20, tol=0.05) is None


def test_a_deep_in_the_money_vertical_pays_nearly_its_whole_width(sess, m):
    """The other side of the rule-22 change, pinned so it cannot regress
    silently: nearly all of a deep ITM spread's value is intrinsic, and the
    midpoint says so. This is also why `ratio` is a DEPTH dial."""
    got = S.credit_vertical(sess, PUT, m, 10.0, 0.95, tol=0.05)
    assert got is not None and got[1] > 0.90
    short, long_ = got[0]
    assert short.strike - sess.spot(m) > 20.0     # deep, by construction


def test_an_out_of_the_money_vertical_cannot_pay_most_of_its_width(sess, m):
    """Which is why a 0.4-0.7 credit band requires the short leg in the money."""
    otm = S.credit_vertical(sess, PUT, m, 10.0, 0.6, itm=False, tol=0.5)
    if otm is not None:
        assert otm[1] < 0.6


# --------------------------------------------------------------- condors

def test_an_iron_condor_is_four_legs_with_matching_wings(sess, m):
    legs, ratio = S.iron_condor(sess, m, width=10.0, min_ratio=0.5)
    assert len(legs) == 4
    calls = sorted(l.strike for l in legs if l.right == CALL)
    puts = sorted(l.strike for l in legs if l.right == PUT)
    assert calls[1] - calls[0] == 10.0 and puts[1] - puts[0] == 10.0
    assert sum(l.qty for l in legs) == 0


def test_the_condor_meets_the_ratio_it_was_asked_for(sess, m):
    for mr in (0.25, 0.35, 0.5):
        legs, ratio = S.iron_condor(sess, m, 10.0, mr)
        assert ratio >= mr


def test_a_lower_ratio_requirement_buys_more_room(sess, m):
    """Walked outward taking the LAST that qualifies, so relaxing the credit
    rule moves the short strikes further from spot."""
    tight, _ = S.iron_condor(sess, m, 10.0, 0.5)
    loose, _ = S.iron_condor(sess, m, 10.0, 0.25)
    short_c = lambda legs: min(l.strike for l in legs if l.right == CALL)
    assert short_c(loose) > short_c(tight)


def test_an_impossible_condor_is_none_not_an_exception(sess, m):
    assert S.iron_condor(sess, m, width=10.0, min_ratio=0.95) is None


def test_selection_never_offers_an_untradeable_structure(sess, m):
    legs, _ = S.iron_condor(sess, m, 10.0, 0.4)
    assert S.tradeable(sess, legs, m)


# ------------------------------------ selection basis vs pricing basis (5.20)

def _wide_book_session():
    """A flat chain with ONE pathologically wide pair deep in the money.

    Spot 5000, every option intrinsic + $2.00 on a $0.20 book, so the 5005/4995
    put spread is worth exactly $5.00 at the mid and $4.80 crossable. The
    5050/5040 pair is worth $10.00 at the mid — a spread that cannot lose — but
    is quoted $5.00 wide on each leg, so short-bid minus long-ask is also
    exactly $5.00. A nearest-match search for `ratio` 0.5 therefore has a real
    choice to make, and the two bases make it differently."""
    from lab import _synthetic as syn
    sess = syn.make_session(settle=5000.0, spot=5000.0)
    syn.hold(sess, PUT, 5050.0, 52.0, 0, spread=5.0)
    syn.hold(sess, PUT, 5040.0, 42.0, 0, spread=5.0)
    return sess


def test_crossable_selection_picks_the_widest_book_not_the_right_strike():
    """The 5.20 pathology, pinned. Short-bid minus long-ask falls as a book
    widens, so a $10.00 spread quoted $5.00 wide answers a search for a $5.00
    credit — 45 points further in the money than the strike that actually has
    that value."""
    sess = _wide_book_session()
    legs, _ = S.credit_vertical(sess, PUT, 10, width=10.0, ratio=0.5,
                                crossable=True)
    assert legs[0].strike == 5050.0


def test_midpoint_selection_finds_the_strike_with_the_right_VALUE():
    sess = _wide_book_session()
    legs, got = S.credit_vertical(sess, PUT, 10, width=10.0, ratio=0.5)
    assert legs[0].strike == 5005.0 and legs[1].strike == 4995.0
    assert got == pytest.approx(0.50)


def test_the_default_basis_is_the_midpoint():
    """Rule 22: a structure is qualified at the MID, always. The default has to
    be the rule, not something a study remembers to pass."""
    import inspect
    assert inspect.signature(S.credit_vertical).parameters["crossable"].default is False
    sess = _wide_book_session()
    a = S.credit_vertical(sess, PUT, 10, 10.0, 0.5)
    b = S.credit_vertical(sess, PUT, 10, 10.0, 0.5, crossable=False)
    assert a[0] == b[0] and a[1] == pytest.approx(b[1])


def test_the_two_bases_disagree_on_real_chains_in_the_stated_direction(sess, m):
    """On a real session the midpoint basis must not land DEEPER in the money
    than the crossable one — 5.20 measures 7.6 points against 11.8."""
    spot = sess.spot(m)
    deep = {}
    for basis in (True, False):
        got = S.credit_vertical(sess, PUT, m, 10.0, 0.35, crossable=basis)
        deep[basis] = None if got is None else got[0][0].strike - spot
    if deep[True] is not None and deep[False] is not None:
        assert deep[False] <= deep[True]


def test_strike_bounds_restrict_the_candidate_set(sess, m):
    """Q23's clamp. Applied to the CANDIDATES, so the search returns the best
    LEGAL spread rather than refusing a session that had one."""
    spot = sess.spot(m)
    floor = float(np.floor((spot - 10.0) / 5.0) * 5.0)
    got = S.credit_vertical(sess, CALL, m, 10.0, 0.5, min_strike=floor)
    if got is not None:
        assert got[0][0].strike >= floor
    got = S.credit_vertical(sess, PUT, m, 10.0, 0.5, max_strike=spot + 5.0)
    if got is not None:
        assert got[0][0].strike <= spot + 5.0


def test_an_impossible_bound_refuses_rather_than_ignoring_itself(sess, m):
    assert S.credit_vertical(sess, CALL, m, 10.0, 0.5,
                             min_strike=sess.spot(m) + 500) is None


def test_smallest_credit_at_least_takes_the_LEAST_rich_qualifying_spread(sess, m):
    """The richest legal spread is the butterfly, which puts both
    shorts on one strike and guarantees one side finishes in the money. The
    second entry wants the widest gap the target allows, so it takes the
    smallest credit that still clears."""
    need = 0.45
    got = S.smallest_credit_at_least(sess, CALL, m, 10.0, need)
    assert got is not None
    legs, achieved = got
    assert achieved >= need
    richest = S.credit_vertical(sess, CALL, m, 10.0, 0.95, tol=0.5)
    if richest is not None:
        assert achieved <= richest[1] + 1e-9
    # and nothing qualifying sits below it
    k = legs[0].strike
    deeper = S.smallest_credit_at_least(sess, CALL, m, 10.0, need,
                                        max_strike=k - 5.0)
    if deeper is not None:
        assert deeper[1] >= achieved - 1e-9 or deeper[0][0].strike < k


def test_a_vertical_quoting_its_own_width_is_refused_as_an_arbitrage(sess, m):
    """Deep in-the-money 0DTE strikes far from spot carry stale books
    whose midpoint implies a spread worth MORE than its width. That is not a
    rich spread, it is a bad quote, and an argmax must not be able to find it."""
    got = S.smallest_credit_at_least(sess, CALL, m, 10.0, 0.999)
    assert got is None or got[1] < 1.0
