#!/usr/bin/env python3
"""
select.py — choosing strikes, shared by every method.

The methods differ in **when** they fire and **what shape** they open. They do
not differ in how a $2.00 call is found, or how a vertical is walked outward
until its credit stops qualifying — so that lives here, once, and is tested
once. A selection bug in five copies is five bugs.

Everything returns concrete legs or `None`. Nothing raises on an ordinary
"there is no such strike today": a method that cannot open is a no-trade, and a
no-trade is a normal outcome that has to be counted rather than crash a sweep.

PRICING A STRUCTURE FOR SELECTION — THE MIDPOINT, ALWAYS
--------------------------------------------------------
**A structure is qualified at the MIDPOINT of its legs. This is a standing
decision (METHODOLOGY §4.6) and not a parameter of a study.**

The reason is the instrument. SPXW strikes near the money trade tens of
thousands of contracts a day, a BAG has its own book far tighter than the sum
of its legs', and the live price walker fills combos and singles at the mid
and closes them at the mid, every session, observed rather than
modelled. Qualifying a spread at short-bid-minus-long-ask does not describe a
conservative version of that trade; it describes a different trade, at a price
no one was asked to pay. **Commission is the real drag on these structures**
— 5.2% of the credit on a 5-wide against 1.1% on a 25-wide — and it is charged
per leg and modelled exactly.

`crossable=True` survives on `net_credit` and on `credit_vertical` for one
purpose only: reproducing results recorded before this rule. It is not a
bracket end to be reported and it is not a stress case for SELECTION.

A second, independent reason for nearest-match searches specifically: the
crossable curve falls as a book widens, so searching it for a target credit
returns the strike with the worst book rather than the right value. See
`credit_vertical`, which measures it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .session import CALL, PUT, Contract, Session


@dataclass(frozen=True, slots=True)
class Leg:
    """One leg of a structure, with its side already decided."""

    right: str
    strike: float
    qty: int          # signed: + long, - short

    def contract(self, sess: Session) -> Contract:
        return sess.contract(self.right, self.strike)

    def _replace_qty(self, qty: int) -> "Leg":
        """Same leg at a different size, keeping the SIDE. A method scales a
        selected structure by its lot size without having to know which legs
        the selector made short."""
        sign = 1 if self.qty >= 0 else -1
        return Leg(self.right, self.strike, sign * abs(int(qty)))


def live_minute(sess: Session, minute: int, window: int = 5) -> Optional[int]:
    """The first minute at or after `minute` whose option book exists.

    Never assume 09:31 — one session in the sample has no book until 09:41."""
    return sess.first_live(minute, window=window)


def _price(sess: Session, right: str, strike: float, minute: int,
           side: int, crossable: bool) -> Optional[float]:
    """What one leg costs (side +1, buying) or pays (side -1, selling)."""
    q = sess.quote(sess.contract(right, strike), minute)
    bid, ask = q
    if bid is None or ask is None:
        return None
    if not crossable:
        return (bid + ask) / 2.0
    return ask if side > 0 else bid


def net_credit(sess: Session, legs: Sequence[Leg], minute: int,
               crossable: bool = True) -> Optional[float]:
    """Per-share credit of opening `legs`. Positive = money in."""
    total = 0.0
    for leg in legs:
        px = _price(sess, leg.right, leg.strike, minute,
                    1 if leg.qty > 0 else -1, crossable)
        if px is None:
            return None
        total += -leg.qty * px
    return total


# ------------------------------------------------------------- naked longs

def long_by_price(sess: Session, right: str, target: float, minute: int,
                  tol: float = 0.0) -> Optional[Tuple[Leg, float]]:
    """The listed strike trading nearest `target` dollars. `target <= 0` = ATM.

    Returns the leg and the miss in dollars, which is reported rather than
    enforced: whether a $0.50 bucket filled at $0.80 belongs in the study is a
    question for the analysis, not for the selection."""
    pick = sess.by_price(right, target, minute, tol=tol)
    if pick is None:
        return None
    return Leg(right, pick.strike, 1), pick.miss


def long_by_offset(sess: Session, right: str, points: float, minute: int
                   ) -> Optional[Leg]:
    """A strike `points` out of the money from spot. 0 = at the money."""
    k = sess.atm(right, minute)
    if k is None:
        return None
    k = k + points if right == CALL else k - points
    col = sess.column(right, k)
    if col is None:
        return None
    return Leg(right, float(k), 1)


# ------------------------------------------------------------- long pairs

def straddle(sess: Session, minute: int, offset: float = 0.0
             ) -> Optional[Tuple[Leg, Leg]]:
    """A long call and a long put, `offset` points out of the money each.

    `offset = 0` is a pure at-the-money straddle; larger values walk it out to
    a strangle. Returned as a PAIR and not a single structure because a method
    manages its two legs independently — they are two naked longs that happen
    to be opened together."""
    c = long_by_offset(sess, CALL, offset, minute)
    p = long_by_offset(sess, PUT, offset, minute)
    return (c, p) if c and p else None


# ------------------------------------------------------------ credit spreads

def credit_vertical(sess: Session, right: str, minute: int, width: float,
                    ratio: float, itm: bool = True, crossable: bool = False,
                    tol: float = 0.15, min_strike: Optional[float] = None,
                    max_strike: Optional[float] = None
                    ) -> Optional[Tuple[List[Leg], float]]:
    """Short vertical whose credit is closest to `ratio` x width.

    `itm=True` starts the short leg in the money, which is what makes a credit
    above half the width reachable at all: an out-of-the-money vertical cannot
    pay 0.6 of its width.

    Direction follows the right. A short PUT spread is bullish — it wants the
    reversion upward that a BL/UL open is betting on — and a short CALL spread
    is bearish. The long leg sits `width` points further out of the money, so
    the loss is capped at `width - credit`.

    **PRICED AT THE MIDPOINT** — see the module docstring for why that is a
    standing rule rather than a choice. `crossable=True` exists only to
    reproduce results recorded before rule 22.

    It matters twice over here, because this search is a NEAREST MATCH rather
    than a threshold. The crossable curve falls as a book widens, so a vertical
    worth $9.90 at the mid but quoted $5.00 wide on each leg offers a $3.00
    crossable credit and **wins a search for `ratio` 0.30**. Such a search does
    not find the strike with the right value; it finds the strike with the
    worst book. Measured over 1,894 sessions: crossable landed a median 11.8
    points in the money against 7.6 at the mid, with 22.1% of picks carrying a
    short leg wider than $2.00 against 6.3%, and some not two-sided at all.
    `iron_condor` walks a THRESHOLD instead, where a wide book disqualifies
    rather than selects, so it never had this failure mode.

    `min_strike` / `max_strike` bound the SHORT leg. They exist for a paired
    second entry: a session that already carries a short put vertical at `Kp`
    may only add a short call vertical at `Kc >= Kp`, because below that line
    both shorts finish in the money for any settlement between them and the
    combined liability has a floor of `Kp - Kc` **whatever price does**. `Kc ==
    Kp` is the short iron butterfly, which is allowed. The bound is applied to
    the CANDIDATE SET rather than checked afterwards, so the search returns the
    best legal spread instead of refusing a legal session.

    Returns the legs and the achieved credit ratio, or None if nothing in the
    chain lands within `tol` of the asked ratio."""
    spot = sess.spot(minute)
    strikes = sess.strikes[right]
    if not strikes.size or not np.isfinite(spot):
        return None
    # In the money means strike above spot for a put, below for a call.
    candidates = strikes[strikes >= spot] if right == PUT else strikes[strikes <= spot]
    if not itm:
        candidates = strikes[strikes < spot] if right == PUT else strikes[strikes > spot]
    if min_strike is not None:
        candidates = candidates[candidates >= min_strike]
    if max_strike is not None:
        candidates = candidates[candidates <= max_strike]
    best = None
    for short_k in candidates:
        long_k = short_k - width if right == PUT else short_k + width
        if sess.column(right, long_k) is None:
            continue
        legs = [Leg(right, float(short_k), -1), Leg(right, float(long_k), 1)]
        credit = net_credit(sess, legs, minute, crossable)
        if credit is None or credit <= 0:
            continue
        got = credit / width
        miss = abs(got - ratio)
        if best is None or miss < best[0]:
            best = (miss, legs, got)
    if best is None or best[0] > tol:
        return None
    return best[1], best[2]


def iron_condor(sess: Session, minute: int, width: float, min_ratio: float,
                crossable: bool = True, min_offset: float = 5.0
                ) -> Optional[Tuple[List[Leg], float]]:
    """The furthest-out short condor still paying `min_ratio` of one wing width.

    Both wings are the same width, so max loss is `width - credit` and a
    `min_ratio` of 0.5 is the "at least 1:1 risk/reward" condition.

    Walked OUTWARD and taking the LAST that qualifies, not the first: among
    condors that all meet the rule, the one with the most room is the one that
    best expresses "nothing ever happens". Taking the nearest would collect more
    credit for a structure far likelier to be breached."""
    spot = sess.spot(minute)
    if not np.isfinite(spot):
        return None
    calls = sess.strikes[CALL]
    puts = sess.strikes[PUT]
    ups = calls[calls >= spot + min_offset]
    downs = puts[puts <= spot - min_offset][::-1]
    best = None
    for i in range(min(ups.size, downs.size)):
        sc, sp = float(ups[i]), float(downs[i])
        lc, lp = sc + width, sp - width
        if sess.column(CALL, lc) is None or sess.column(PUT, lp) is None:
            continue
        legs = [Leg(CALL, sc, -1), Leg(CALL, lc, 1),
                Leg(PUT, sp, -1), Leg(PUT, lp, 1)]
        credit = net_credit(sess, legs, minute, crossable)
        if credit is None:
            continue
        got = credit / width
        if got >= min_ratio:
            best = (legs, got)      # keep walking; the last one wins
    return best


def tradeable(sess: Session, legs: Sequence[Leg], minute: int) -> bool:
    """Every leg has a real two-sided quote. Checked before a structure is
    offered, so a method's no-trade and the broker's refusal do not both have
    to be interpreted afterwards."""
    return all(sess.tradeable(l.contract(sess), minute) for l in legs)


def smallest_credit_at_least(sess: Session, right: str, minute: int,
                             width: float, need: float,
                             crossable: bool = False,
                             min_strike: Optional[float] = None,
                             max_strike: Optional[float] = None
                             ) -> Optional[Tuple[List[Leg], float]]:
    """The LEAST rich in-the-money vertical still paying `need` x width.

    For a paired second entry. The richest legal spread is the iron butterfly, which maximises the credit and puts both short strikes on one
    number — so at settlement one side is always in the money and the pair can
    never keep its whole credit. Taking the SMALLEST qualifying credit instead
    puts the short as far out as the target allows, opening the window between
    the two shorts where both expire worthless.

    A vertical is also refused if its credit reaches its own width: that is an
    arbitrage rather than a rich spread, and deep in-the-money 0DTE strikes far
    from spot quote it often enough to matter — 15.4% of sessions touch one.

    Returns the legs and the achieved credit ratio, or None."""
    spot = sess.spot(minute)
    strikes = sess.strikes[right]
    if not strikes.size or not np.isfinite(spot):
        return None
    candidates = strikes[strikes >= spot] if right == PUT else strikes[strikes <= spot]
    if min_strike is not None:
        candidates = candidates[candidates >= min_strike]
    if max_strike is not None:
        candidates = candidates[candidates <= max_strike]
    best = None
    for short_k in candidates:
        long_k = short_k - width if right == PUT else short_k + width
        if sess.column(right, long_k) is None:
            continue
        legs = [Leg(right, float(short_k), -1), Leg(right, float(long_k), 1)]
        credit = net_credit(sess, legs, minute, crossable)
        if credit is None or credit <= 0 or credit >= width:
            continue
        got = credit / width
        if got < need:
            continue
        if best is None or got < best[1]:
            best = (legs, got)
    return best
