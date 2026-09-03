#!/usr/bin/env python3
"""Tests for W, L, COVER and SHORT.

This is the layer that decides every result in the programme: a stop that fires
a minute early, or a take-profit that books the price the market ran to instead
of the price the order was resting at, moves every number downstream while
every number still looks plausible. So the arithmetic is pinned exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from lab._synthetic import C, P, hold, make_session, set_path
from lab.broker import MULTIPLIER, Broker, leg_commission
from lab.exits import (CLOSE, COVER, HIT_GATE, HIT_L, HIT_W, NONE, SETTLED,
                       SHORT, ExitPolicy, Trade, close_value_series, manage,
                       settle)
from lab.fills import CROSS, MID
from lab.session import CALL, PUT


def _open(sess, contract, minute=10, qty=1, model=CROSS):
    b = Broker(sess, model)
    o = b.buy(contract, qty, minute, "t")
    t = Trade(tag="t", entry_minute=minute, legs=[(contract, qty)],
              entry_price=abs(o.price) / qty, credit=False)
    t.record(o)
    return b, t


# ------------------------------------------------------------------ valuing

def test_close_value_is_what_flattening_pays_in():
    s = make_session(spread=0.20, base=2.00)
    long_leg = close_value_series(s, [(C(5000), 1)], edge=1.0)
    assert long_leg[10] == pytest.approx(1.90)          # a long sells at the bid
    short_leg = close_value_series(s, [(C(5000), -1)], edge=1.0)
    assert short_leg[10] == pytest.approx(-2.10)        # a short buys at the ask
    assert close_value_series(s, [(C(5000), 1)], edge=0.0)[10] == pytest.approx(2.00)


def test_a_dark_leg_makes_the_package_priceless_not_half_priced():
    s = make_session(spread=0.20, base=2.00)
    set_path(s, CALL, 5000.0, [float("nan")], start=30)
    v = close_value_series(s, [(C(5000), 1), (C(5010), -1)], edge=1.0)
    assert np.isnan(v[30]) and np.isfinite(v[29])


# ------------------------------------------------------------- the W trigger

def test_w_fires_at_the_level_and_fills_AT_the_limit():
    """The order was resting at $4.00. When the bid gaps to $8.00 it filled at
    $4.00 somewhere inside that minute — booking $8.00 is a windfall a resting
    order never received."""
    s = make_session(spread=0.20, base=2.00, settle=5000.0)
    b, t = _open(s, C(5050))
    assert t.entry_price == pytest.approx(2.10)         # bought the ask
    hold(s, CALL, 5050.0, 8.00, start=40)               # bid becomes 7.90
    manage(b, t, ExitPolicy(w=2.0))
    assert t.exit_reason == HIT_W and t.exit_minute == 40
    assert t.exit_price == pytest.approx(4.20)          # 2.0 x the 2.10 entry
    assert t.legs == []


def test_w_does_not_fire_when_the_bid_never_reaches_it():
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 4.15, start=40)               # bid 4.05, level 4.20
    manage(b, t, ExitPolicy(w=2.0))
    assert t.exit_minute is None and t.exit_reason == SETTLED


def test_the_trigger_follows_the_book_the_fill_model_trades_on():
    """Under the crossing bracket a W needs a BID at its level; under the mid
    bracket the midpoint is what both the trigger and the fill use. They move
    together, which is the only self-consistent pairing."""
    s = make_session(spread=1.00, base=2.00)            # bid 1.5 / ask 2.5
    b, t = _open(s, C(5050), model=CROSS)               # entry 2.50, level 5.00
    hold(s, CALL, 5050.0, 5.30, start=40, spread=1.00)  # bid 4.80, mid 5.30
    manage(b, t, ExitPolicy(w=2.0))
    assert t.exit_minute is None                        # no bid at 5.00

    s2 = make_session(spread=1.00, base=2.00)
    b2, t2 = _open(s2, C(5050), model=MID)              # entry 2.00, level 4.00
    hold(s2, CALL, 5050.0, 5.30, start=40, spread=1.00)
    manage(b2, t2, ExitPolicy(w=2.0))
    assert t2.exit_minute == 40 and t2.exit_price == pytest.approx(4.00)


def test_a_mid_triggered_W_against_a_crossing_book_is_refused_not_filled():
    """The override exists so the smoothing question can be asked. When it
    names a price nobody is bidding, the order is refused and the trade rides
    on — which is what would have happened."""
    s = make_session(spread=1.00, base=2.00)
    b, t = _open(s, C(5050), model=CROSS)               # entry 2.50, level 5.00
    hold(s, CALL, 5050.0, 5.30, start=40, spread=1.00)  # bid 4.80, mid 5.30
    manage(b, t, ExitPolicy(w=2.0, trigger_edge=0.0))
    assert t.exit_minute is None and len(t.legs) == 1
    assert any("not reachable" in r.reason for r in b.rejections)


# ------------------------------------------------------------- the L trigger

def test_l_fires_at_the_level_and_fills_at_the_MARKET():
    """A stop is not a resting limit. It is a decision to be out, and it
    concedes the spread like any other crossing order."""
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050))                            # entry 2.10
    hold(s, CALL, 5050.0, 0.80, start=25)               # bid 0.70 <= 1.05
    manage(b, t, ExitPolicy(l=0.5))
    assert t.exit_reason == HIT_L and t.exit_minute == 25
    assert t.exit_price == pytest.approx(0.70)          # the bid, not the level


def test_the_adverse_trigger_wins_a_tie():
    """Both conditions true in the same minute. A minute has no internal order,
    and assuming the favourable one filled first would flatter every result.

    The tie is manufactured by putting both levels on the entry price
    (`w = l = 1.0`); on real data they can only coincide deliberately."""
    s = make_session(spread=2.00, base=2.00)            # bid 1.00 / ask 3.00
    b, t = _open(s, C(5050))                            # entry 3.00
    assert t.entry_price == pytest.approx(3.00)
    # Both levels sit on the entry price, and the very next minute's bid is
    # exactly that. Both conditions are true in minute 11; L must take it.
    hold(s, CALL, 5050.0, 4.00, start=11, spread=2.00)  # bid becomes exactly 3.00
    manage(b, t, ExitPolicy(w=1.0, l=1.0))
    assert t.exit_reason == HIT_L and t.exit_minute == 11


def test_l_action_none_leaves_the_trade_alone():
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 0.20, start=25)
    manage(b, t, ExitPolicy(l=0.5, l_action=NONE))
    assert t.exit_minute is None and t.legs


def test_zero_disables_a_trigger():
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 0.02, start=25)
    manage(b, t, ExitPolicy(w=0.0, l=0.0))
    assert t.exit_minute is None


def test_a_trigger_cannot_fire_on_the_entry_minute():
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050), minute=10)
    # The entry price is the ask, so the bid is already below l = 0.95 of it.
    manage(b, t, ExitPolicy(l=0.95))
    assert t.exit_minute != 10


# ------------------------------------------------------------ COVER and SHORT

def test_cover_sells_the_next_strike_OUT_of_the_money():
    s = make_session(spread=0.20, base=2.00, settle=5000.0)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 8.00, start=40)
    manage(b, t, ExitPolicy(w=2.0, w_action=COVER, cover_width=1))
    assert t.converted_to == 5055.0                     # calls go OTM upward
    assert t.exit_reason == "W:cover"
    assert sorted(q for _, q in t.legs) == [-1, 1]


def test_cover_on_a_put_goes_DOWNWARD():
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, P(4950))
    hold(s, PUT, 4950.0, 8.00, start=40)
    manage(b, t, ExitPolicy(w=2.0, w_action=COVER, cover_width=2))
    assert t.converted_to == 4940.0


def test_short_sells_a_strike_CLOSER_to_the_money():
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 0.50, start=25)
    manage(b, t, ExitPolicy(l=0.5, l_action=SHORT, short_width=1))
    assert t.converted_to == 5045.0
    assert t.exit_reason == "L:short"


@pytest.mark.parametrize("width", [1, 2, 3])
def test_spread_widths_run_to_three_strikes(width):
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 8.00, start=40)
    manage(b, t, ExitPolicy(w=2.0, w_action=COVER, cover_width=width))
    assert t.converted_to == 5050.0 + 5.0 * width


def test_a_converted_trade_is_held_to_expiry_and_triggers_no_more():
    """One trigger, and only one: every action either flattens the trade or
    turns it into a spread that rides to settlement."""
    s = make_session(spread=0.20, base=2.00, settle=5000.0)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 8.00, start=40)
    hold(s, CALL, 5050.0, 0.05, start=60)               # would have hit L later
    manage(b, t, ExitPolicy(w=2.0, w_action=COVER, l=0.5))
    assert t.exit_reason == "W:cover" and len(t.legs) == 2


def test_a_cover_off_the_end_of_the_chain_is_refused_not_invented():
    s = make_session(spread=0.20, base=2.00)
    top = float(s.strikes[CALL][-1])
    b, t = _open(s, C(top))
    hold(s, CALL, top, 8.00, start=40)
    manage(b, t, ExitPolicy(w=2.0, w_action=COVER))
    assert t.converted_to is None and len(t.legs) == 1


def test_only_a_naked_long_can_be_converted():
    s = make_session(spread=0.20, base=2.00)
    b = Broker(s, CROSS)
    o = b.submit([(C(5050), 1), (C(5060), -1)], 10, "spread")
    t = Trade(tag="t", entry_minute=10, legs=[(C(5050), 1), (C(5060), -1)],
              entry_price=abs(o.price), credit=False)
    t.record(o)
    hold(s, CALL, 5050.0, 20.00, start=40)
    manage(b, t, ExitPolicy(w=2.0, w_action=COVER))
    assert t.converted_to is None


# --------------------------------------------------------- credit structures

def test_a_credit_structure_with_a_w_or_l_is_refused_not_inverted():
    """W and L are long-only; a credit structure carries neither. A short spread's
    take-profit is a buy-back at a FRACTION of the credit — the opposite
    reading of the same number — so it gets its own policy, not this one."""
    s = make_session(spread=0.20, base=2.00)
    b = Broker(s, CROSS)
    o = b.submit([(C(5050), -1), (C(5060), 1)], 10, "credit spread")
    t = Trade(tag="cs", entry_minute=10, legs=[(C(5050), -1), (C(5060), 1)],
              entry_price=abs(o.price), credit=True)
    t.record(o)
    with pytest.raises(ValueError, match="long-only"):
        manage(b, t, ExitPolicy(w=0.5))
    with pytest.raises(ValueError, match="long-only"):
        manage(b, t, ExitPolicy(l=2.0))


def test_a_credit_structure_with_no_triggers_is_fine():
    """Credit structures hold to expiry, which must stay a legal thing to do."""
    s = make_session(spread=0.20, base=2.00, settle=5000.0)
    b = Broker(s, CROSS)
    o = b.submit([(C(5050), -1), (C(5060), 1)], 10, "credit spread")
    t = Trade(tag="cs", entry_minute=10, legs=[(C(5050), -1), (C(5060), 1)],
              entry_price=abs(o.price), credit=True)
    t.record(o)
    manage(b, t, ExitPolicy())
    settle(b, [t])
    assert t.pnl == pytest.approx(b.pnl)


# ------------------------------------------------------------ reconciliation

def test_a_trade_that_cannot_be_closed_rides_to_expiry():
    s = make_session(spread=0.20, base=2.00, settle=5100.0)
    b, t = _open(s, C(5050))
    hold(s, CALL, 5050.0, 8.00, start=40)
    set_path(s, CALL, 5050.0, [float("nan")] * (s.n_minutes - 40), start=40)
    manage(b, t, ExitPolicy(w=2.0))
    assert t.exit_minute is None and len(t.legs) == 1
    settle(b, [t])
    assert t.pnl == pytest.approx(b.pnl)


def test_settlement_is_attributed_per_trade_and_reconciles():
    s = make_session(spread=0.20, base=2.00, settle=5100.0)
    b = Broker(s, CROSS)
    trades = []
    for k in (5050.0, 5060.0):
        o = b.buy(C(k), 1, 10, "t")
        t = Trade(tag="t", entry_minute=10, legs=[(C(k), 1)],
                  entry_price=abs(o.price), credit=False)
        t.record(o)
        trades.append(t)
    settle(b, trades)
    assert sum(t.pnl for t in trades) == pytest.approx(b.pnl)


def test_a_bad_policy_is_refused():
    with pytest.raises(ValueError, match="w_action"):
        ExitPolicy(w_action="roll").validate()
    with pytest.raises(ValueError, match="l_action"):
        ExitPolicy(l_action="cover").validate()
    with pytest.raises(ValueError, match="widths"):
        ExitPolicy(cover_width=4).validate()
    with pytest.raises(ValueError, match="magnitudes"):
        ExitPolicy(w=-1).validate()


# ------------------------------------------------------- decaying W and L

def test_a_static_level_is_the_decaying_one_with_no_asymptote():
    """The constant case is a point in the search space, not a separate path."""
    pol = ExitPolicy(w=3.0)
    lv = pol.level_series(2.00, 10, 391, "w")
    assert np.allclose(lv, 6.00)


def test_the_w_level_decays_toward_its_asymptote_by_half_lives():
    """4.5x at entry, pulled toward 3x, halving the gap every 45 minutes."""
    pol = ExitPolicy(w=4.5, w_end=3.0, w_half_life=45.0)
    lv = pol.level_series(2.00, 10, 391, "w")
    assert lv[10] == pytest.approx(9.00)                    # 4.5 x 2.00
    assert lv[55] == pytest.approx((3.0 + 0.75) * 2.00)     # one half-life
    assert lv[100] == pytest.approx((3.0 + 0.375) * 2.00)   # two
    assert lv[-1] == pytest.approx(6.00, abs=0.02)          # ~the asymptote


def test_decay_is_measured_in_minutes_HELD_not_minutes_of_the_day():
    """Two trades an hour apart are the same trade at different times."""
    pol = ExitPolicy(w=4.5, w_end=3.0, w_half_life=45.0)
    early = pol.level_series(2.00, 10, 391, "w")
    late = pol.level_series(2.00, 70, 391, "w")
    assert early[10] == pytest.approx(late[70])
    assert early[55] == pytest.approx(late[115])
    assert late[10] == pytest.approx(late[70])   # flat before entry


def test_a_decaying_w_waits_where_a_static_one_at_its_asymptote_would_not():
    """Bid jumps to 7.00 at minute 40 and stays there. A static 3x (level 6.30)
    takes it immediately; a static 4.5x (9.45) never does; the decaying one
    holds out and then takes it once the level has fallen to the bid."""
    prices = [2.00] * 40 + [7.10] * 351          # bid 7.00, entry 2.10
    fired = {}
    for pol in (ExitPolicy(w=3.0), ExitPolicy(w=4.5),
                ExitPolicy(w=4.5, w_end=3.0, w_half_life=45)):
        s2 = make_session(spread=0.20, base=2.00)
        set_path(s2, CALL, 5050.0, prices, start=0)
        b, t = _open(s2, C(5050), minute=10)
        manage(b, t, pol)
        fired[pol.label()] = t.exit_minute
    assert fired["w3/l0"] == 40
    assert fired["w4.5/l0"] is None
    decayed = fired["w4.5>3@45/l0"]
    assert decayed is not None and decayed > 40


def test_a_decaying_w_books_the_level_it_had_decayed_to():
    """It was resting at that price when it filled, not at its starting ask."""
    s = make_session(spread=0.20, base=2.00)
    set_path(s, CALL, 5050.0, [2.00] * 40 + [7.10] * 351, start=0)
    b, t = _open(s, C(5050), minute=10)
    pol = ExitPolicy(w=4.5, w_end=3.0, w_half_life=45)
    manage(b, t, pol)
    lv = pol.level_series(t.entry_price, 10, s.n_minutes, "w")
    assert t.exit_price == pytest.approx(lv[t.exit_minute])
    assert 3.0 * t.entry_price < t.exit_price < 4.5 * t.entry_price


def test_a_decaying_w_does_fire_once_it_has_decayed_far_enough():
    s = make_session(spread=0.20, base=2.00)
    set_path(s, CALL, 5050.0, [2.00] * 200 + [6.60] * 191, start=0)
    b, t = _open(s, C(5050), minute=10)          # entry 2.10, asymptote 6.30
    manage(b, t, ExitPolicy(w=4.5, w_end=3.0, w_half_life=45))
    assert t.exit_reason == HIT_W and t.exit_minute == 200


def test_a_stop_can_tighten_with_time_held():
    """`l_end > l` pulls the stop up as expiry approaches."""
    pol = ExitPolicy(l=0.3, l_end=0.7, l_half_life=30.0)
    lv = pol.level_series(2.00, 0, 391, "l")
    assert lv[0] == pytest.approx(0.60)
    assert lv[30] == pytest.approx(0.7 * 2.0 - 0.2 * 2.0)   # half way
    assert lv[300] == pytest.approx(1.40, abs=0.01)


# -------------------------------------------------------- the indicator exit

def test_the_gate_exit_fires_on_the_first_flagged_minute_after_entry():
    s = make_session(spread=0.20, base=2.00, settle=5100.0)
    b, t = _open(s, C(5050), minute=10)
    gm = np.zeros(s.n_minutes, dtype=bool)
    gm[[5, 120, 200]] = True                      # 5 is before entry
    manage(b, t, ExitPolicy(exit_gate=_dummy_gate()), gate_minutes=gm)
    assert t.exit_reason == HIT_GATE and t.exit_minute == 120


def _dummy_gate():
    from lab.gates import Gate, GateSet
    return GateSet(gates=(Gate(column="zone", op="in", values=(0,)),))


def test_the_gate_exit_crosses_the_spread_like_a_stop():
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050), minute=10)
    hold(s, CALL, 5050.0, 5.00, start=100)
    gm = np.zeros(s.n_minutes, dtype=bool); gm[120] = True
    manage(b, t, ExitPolicy(exit_gate=_dummy_gate()), gate_minutes=gm)
    assert t.exit_price == pytest.approx(4.90)    # the bid, not a limit


# ------------------------------------------------------------- precedence

def test_the_adverse_exit_beats_the_gate_beats_the_w():
    """All three armed, all reachable in the same minute. Order: L, gate, W."""
    s = make_session(spread=2.00, base=2.00)      # bid 1.00 / ask 3.00
    b, t = _open(s, C(5050), minute=10)           # entry 3.00
    gm = np.zeros(s.n_minutes, dtype=bool); gm[11:] = True
    hold(s, CALL, 5050.0, 4.00, start=11, spread=2.00)   # bid exactly 3.00
    manage(b, t, ExitPolicy(w=1.0, l=1.0, exit_gate=_dummy_gate()),
           gate_minutes=gm)
    assert t.exit_reason == HIT_L


def test_the_gate_beats_the_w_at_the_same_minute():
    s = make_session(spread=2.00, base=2.00)
    b, t = _open(s, C(5050), minute=10)
    gm = np.zeros(s.n_minutes, dtype=bool); gm[11:] = True
    hold(s, CALL, 5050.0, 4.00, start=11, spread=2.00)
    manage(b, t, ExitPolicy(w=1.0, exit_gate=_dummy_gate()), gate_minutes=gm)
    assert t.exit_reason == HIT_GATE


def test_do_nothing_at_L_removes_only_the_L():
    """The gate still stands — `l_action='none'` disables the stop, not the
    whole policy."""
    s = make_session(spread=0.20, base=2.00)
    b, t = _open(s, C(5050), minute=10)
    hold(s, CALL, 5050.0, 0.20, start=25)         # would stop out
    gm = np.zeros(s.n_minutes, dtype=bool); gm[200] = True
    manage(b, t, ExitPolicy(l=0.5, l_action=NONE, exit_gate=_dummy_gate()),
           gate_minutes=gm)
    assert t.exit_reason == HIT_GATE and t.exit_minute == 200


def test_a_bad_schedule_is_refused():
    with pytest.raises(ValueError, match="half lives"):
        ExitPolicy(w=2.0, w_half_life=0).validate()
    with pytest.raises(ValueError, match="magnitude"):
        ExitPolicy(w=2.0, w_end=-1.0).validate()
