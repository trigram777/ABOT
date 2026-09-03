#!/usr/bin/env python3
"""Tests for the execution core. Structures appear only as compositions."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from lab.broker import (MULTIPLIER, Broker, leg_commission)
from lab.fills import BUY, CROSS, MID, SELL, FillModel, round_conceding
from lab.session import CALL, PUT, Contract, Session

from lab._synthetic import C, P, STRIKES, make_session


# --------------------------------------------------------------------- prices

def test_tick_concession_never_improves_a_price():
    assert round_conceding(2.53, BUY) == 2.55
    assert round_conceding(2.53, SELL) == 2.50
    assert round_conceding(3.04, BUY) == 3.10
    assert round_conceding(3.04, SELL) == 3.00
    # A price already on the grid must not move.
    assert round_conceding(2.4000000000000004, BUY) == 2.40


def test_edge_is_monotone_and_never_worse_than_crossing():
    bid, ask = 2.95, 3.05
    buys = [FillModel(edge=e).price(bid, ask, BUY) for e in (0, .25, .5, .75, 1)]
    sells = [FillModel(edge=e).price(bid, ask, SELL) for e in (0, .25, .5, .75, 1)]
    assert buys == sorted(buys) and max(buys) <= ask
    assert sells == sorted(sells, reverse=True) and min(sells) >= bid


def test_one_sided_quotes_are_refused_by_default():
    assert MID.price(None, 1.20, BUY) is None
    assert CROSS.price(1.00, None, SELL) is None
    assert FillModel(edge=1.0, allow_one_sided_buy=True).price(None, 1.20, BUY) == 1.20


# ---------------------------------------------------------------- commissions

@pytest.mark.parametrize("symbol,legs,qty,expected", [
    ("XSP", 1, 1, 1.22), ("XSP", 1, 2, 1.74),
    ("XSP", 2, 1, 2.44), ("XSP", 2, 2, 3.48),
    ("SPXW", 2, 1, 3.26), ("SPXW", 4, 1, 6.52),
])
def test_commission_reproduces_the_broker_prints(symbol, legs, qty, expected):
    """The IBKR paper prints of 17-20 Aug 2026. Paper fills are simulated;
    paper commissions are computed with the live schedule."""
    got = legs * leg_commission(symbol, qty, premium=1.50)
    assert got == pytest.approx(expected, abs=0.005)


def test_spxw_premium_band():
    """A leg under $1.00 pays the low third-party rate. The mixed four-leg
    condor that printed $6.43 is three legs over a dollar and one under."""
    high = leg_commission("SPXW", 1, 1.50)
    low = leg_commission("SPXW", 1, 0.90)
    assert 3 * high + low == pytest.approx(6.43, abs=0.005)


def test_multi_lot_prints_reproduce_to_the_cent():
    """The eight live prints of 25 Aug 2026, spanning 1/2/10/12 contracts.

    They retired the XSP bulk-surcharge guard: XSP at 10 contracts pays the
    same formula as at one, so there is no break to refuse to guess at."""
    for symbol, qty, premium, observed in [
            ("XSP", 1, 1.32, 1.22), ("XSP", 1, 1.82, 1.22),
            ("SPXW", 2, 0.60, 2.38), ("SPXW", 2, 0.70, 2.38),
            ("SPXW", 2, 3.50, 2.56), ("XSP", 10, 0.66, 8.70),
            ("SPXW", 10, 3.60, 12.80), ("SPXW", 12, 3.40, 15.36)]:
        assert leg_commission(symbol, qty, premium) == pytest.approx(observed), \
            f"{symbol} x{qty} @ {premium}"


def test_no_multi_lot_price_break_on_either_symbol():
    for symbol, flat in (("SPXW", 1.28), ("XSP", 0.87)):
        rates = [leg_commission(symbol, q, 3.5) / q for q in (2, 5, 10, 12, 20)]
        assert all(r == pytest.approx(flat) for r in rates), rates


def test_one_contract_is_the_worst_size():
    per_one = leg_commission("SPXW", 1, 1.5)
    per_two = leg_commission("SPXW", 2, 1.5) / 2
    per_five = leg_commission("SPXW", 5, 1.5) / 5
    assert per_one == pytest.approx(1.63) and per_two == pytest.approx(1.28)
    assert per_one > per_two
    assert per_two == pytest.approx(per_five, abs=1e-9)   # flat beyond two


# ----------------------------------------------------------------- cash signs

def test_sell_credits_and_buy_debits_and_fees_always_debit():
    b = Broker(make_session(), CROSS)
    sold = b.sell(C(5050), 1, 10, "open")
    assert sold and sold.fills[0].price == pytest.approx(1.90)   # bid side
    assert b.cash == pytest.approx(1.90 * 100 - leg_commission("SPXW", 1, 1.90))
    bought = b.buy(C(5060), 1, 10, "hedge")
    assert bought.fills[0].price == pytest.approx(2.10)          # ask side
    assert b.cash < 1.90 * 100


def test_round_trip_costs_exactly_the_spread_plus_two_commissions():
    s = make_session(spread=0.20)
    b = Broker(s, CROSS)
    b.buy(C(5050), 1, 10)
    b.sell(C(5050), 1, 60)
    assert not b.held()
    comm = 2 * leg_commission("SPXW", 1, 2.0)
    assert b.pnl == pytest.approx(-0.20 * MULTIPLIER - comm)


def test_a_mid_fill_round_trip_costs_only_commission():
    b = Broker(make_session(spread=0.20), MID)
    b.buy(C(5050), 1, 10)
    b.sell(C(5050), 1, 60)
    assert b.pnl == pytest.approx(-2 * leg_commission("SPXW", 1, 2.0))


# ----------------------------------------------------------------- atomicity

def test_one_bad_leg_kills_the_whole_package():
    s = make_session()
    col = s.column(CALL, 5050)
    s.arrays[(CALL, "bid")][10, col] = np.nan          # that leg goes one-sided
    b = Broker(s, CROSS)
    o = b.submit([(C(5040), -1), (C(5050), 1)], 10, "vertical")
    assert not o and "no tradeable quote" in o.reason
    assert b.held() == [] and b.cash == 0.0
    assert len(b.rejections) == 1


def test_a_missing_strike_is_named_not_silently_dropped():
    b = Broker(make_session(), CROSS)
    o = b.submit([(C(9999), 1)], 10)
    assert not o and "not in the chain" in o.reason


def test_close_all_refuses_as_a_whole_when_one_leg_has_gone_dark():
    s = make_session()
    b = Broker(s, CROSS)
    b.submit([(C(5040), -1), (C(5050), 1)], 10, "vertical")
    col = s.column(CALL, 5050)
    s.arrays[(CALL, "ask")][60, col] = np.nan
    o = b.close_all(60)
    assert not o
    assert len(b.held()) == 2      # still on: a rule that cannot exit has not


# --------------------------------------------------------------------- basis

def test_basis_averages_on_add_survives_a_partial_and_resets_on_flip():
    s = make_session(base=2.00, spread=0.0)
    b = Broker(s, CROSS)
    b.buy(C(5000), 1, 10)
    s.arrays[(CALL, "bid")][20] = 4.0
    s.arrays[(CALL, "ask")][20] = 4.0
    b.buy(C(5000), 1, 20)
    p = b.positions[C(5000)]
    assert p.qty == 2 and p.avg_price == pytest.approx(3.00)

    b.close(C(5000), 30, qty=1)
    p = b.positions[C(5000)]
    assert p.qty == 1 and p.avg_price == pytest.approx(3.00)   # unchanged

    s.arrays[(CALL, "bid")][40] = 5.0
    s.arrays[(CALL, "ask")][40] = 5.0
    b.sell(C(5000), 3, 40)                                     # flip to short 2
    p = b.positions[C(5000)]
    assert p.qty == -2 and p.avg_price == pytest.approx(5.00)
    assert p.opened_at == 40


# ---------------------------------------------------------------- settlement

def test_settlement_is_intrinsic_and_free():
    s = make_session(settle=5020.0)
    b = Broker(s, CROSS)
    b.buy(C(5000), 1, 10)
    before = b.commission
    b.settle()
    assert b.commission == before                       # expiry costs nothing
    assert b.settlement_cash == pytest.approx(20.0 * MULTIPLIER)


def test_settlement_pays_a_short_negatively():
    b = Broker(make_session(settle=5020.0), CROSS)
    b.sell(P(5050), 1, 10)
    b.settle()
    assert b.settlement_cash == pytest.approx(-30.0 * MULTIPLIER)


def test_out_of_the_money_expires_for_nothing():
    b = Broker(make_session(settle=4990.0), CROSS)
    b.buy(C(5050), 1, 10)
    b.settle()
    assert b.settlement_cash == 0.0


def test_no_orders_after_settlement():
    b = Broker(make_session(), CROSS)
    b.settle()
    with pytest.raises(RuntimeError):
        b.buy(C(5000), 1, 10)


# ---------------------------------------------------- structures, composed

def test_a_credit_vertical_pays_credit_minus_width_when_fully_breached():
    """Short 5000C / long 5010C. Composed from two calls to submit, and the
    engine is never told it is a vertical."""
    s = make_session(settle=5100.0, spread=0.0, base=2.00)
    b = Broker(s, CROSS)
    o = b.submit([(C(5000), -1), (C(5010), 1)], 10, "call spread")
    assert o.price == pytest.approx(0.0)      # flat fixture: no credit
    b.settle()
    # Fully through: short pays 100, long collects 90 -> -10 points.
    assert b.settlement_cash == pytest.approx(-10.0 * MULTIPLIER)
    assert b.pnl == pytest.approx(-10.0 * MULTIPLIER - o.commission)


def test_an_iron_condor_between_its_wings_keeps_the_whole_credit():
    s = make_session(settle=5000.0)
    b = Broker(s, CROSS)
    o = b.submit([(C(5050), -1), (C(5060), 1), (P(4950), -1), (P(4940), 1)],
                 10, "condor")
    assert o                                   # four legs, one atomic package
    credit = o.cash
    b.settle()
    assert b.settlement_cash == 0.0            # every leg expires worthless
    assert b.pnl == pytest.approx(credit)
    assert o.commission == pytest.approx(4 * leg_commission("SPXW", 1, 2.0))


def test_covering_a_long_into_a_spread_is_one_more_order():
    """C in the specification: sell the next OTM strike against a naked long."""
    s = make_session(settle=5100.0, spread=0.0, base=2.00)
    b = Broker(s, CROSS)
    b.buy(C(5000), 1, 10, "abot")
    b.sell(C(5010), 1, 60, "cover")
    b.settle()
    # Long 100 points, short -90: the spread is worth its full 10-point width.
    assert b.settlement_cash == pytest.approx(10.0 * MULTIPLIER)


def test_shorting_a_long_inside_it_caps_at_the_inner_strike():
    """S in the specification: sell a CLOSER-to-ATM strike against the long."""
    s = make_session(settle=5100.0, spread=0.0, base=2.00)
    b = Broker(s, CROSS)
    b.buy(C(5010), 1, 10, "abot")
    b.sell(C(5000), 1, 60, "short")
    b.settle()
    assert b.settlement_cash == pytest.approx(-10.0 * MULTIPLIER)


# ------------------------------------------------------------------ valuation

def test_mark_modes_bracket_each_other_and_blank_on_a_dark_leg():
    s = make_session(spread=0.20)
    b = Broker(s, CROSS)
    b.buy(C(5000), 1, 10)
    assert b.mark(20, "liquidate") < b.mark(20, "mid")
    col = s.column(CALL, 5000)
    s.arrays[(CALL, "bid")][30, col] = np.nan
    assert b.mark(30) is None and b.equity(30) is None


def test_equity_at_entry_is_minus_the_spread_and_the_fee():
    s = make_session(spread=0.20)
    b = Broker(s, CROSS)
    b.buy(C(5000), 1, 10)
    assert b.equity(10, "liquidate") == pytest.approx(
        -0.20 * MULTIPLIER - leg_commission("SPXW", 1, 2.10))


def test_pnl_equals_the_sum_of_the_ledger():
    s = make_session(settle=5020.0)
    b = Broker(s, CROSS)
    b.submit([(C(5050), -1), (C(5060), 1)], 10, "condor")
    b.buy(P(4950), 1, 30, "hedge")
    b.close(P(4950), 60)
    b.settle()
    assert b.pnl == pytest.approx(sum(r["cash"] for r in b.ledger()))


def test_price_of_prices_the_package_you_would_SEND():
    """`price_of` takes the legs to submit, not the legs held — and crossing
    means a submitted buy pays the ask while a submitted sell takes the bid."""
    s = make_session(spread=0.20, base=2.00)
    b = Broker(s, CROSS)
    open_legs = [(C(5000), 1)]
    assert b.price_of(open_legs, 10, "cross") == pytest.approx(-2.10)   # debit
    close_legs = [(C(5000), -1)]
    assert b.price_of(close_legs, 10, "cross") == pytest.approx(1.90)   # credit
    assert b.price_of(open_legs, 10, "mid") == pytest.approx(-2.00)


def test_price_of_matches_what_submitting_actually_costs():
    s = make_session(spread=0.20, base=2.00)
    b = Broker(s, CROSS)
    legs = [(C(5000), -1), (C(5010), 1)]
    quoted = b.price_of(legs, 10, "cross")
    o = b.submit(legs, 10)
    assert o.price == pytest.approx(quoted)


def test_the_walked_bracket_sits_between_the_two_bounds():
    """CROSS pays the full half-spread on EVERY leg, which for a multi-leg order
    is not a pessimistic assumption so much as a wrong model: a BAG has its own,
    tighter book. WALK is the placeholder centre pending calibration."""
    from lab.fills import WALK, BRACKET3
    assert MID.edge < WALK.edge < CROSS.edge
    assert BRACKET3 == (MID, WALK, CROSS)
    bid, ask = 2.95, 3.05
    for side in (BUY, SELL):
        m, w, c = (f.price(bid, ask, side) for f in BRACKET3)
        assert min(m, c) <= w <= max(m, c), (side, m, w, c)
