#!/usr/bin/env python3
"""
exits.py — W, L, COVER and SHORT: the specification's exit conventions.

THE FOUR
--------
**W** — a close limit placed at entry, as a multiple of the entry premium. A
naked long bought for $2.00 with `w = 2` is a SELL LIMIT at $4.00.

**L** — a stop placed at entry, as a fraction of the entry premium. The same
long with `l = 0.5` stops out when the position is worth $1.00.

Both may **decay with time held** rather than sitting still. `w = 4.5`,
`w_end = 3.0`, `w_half_life = 45` starts the limit at 4.5x and pulls it toward
3x, halving the remaining distance every 45 minutes — an ask that begins
ambitious and becomes realistic as the option's remaining life shortens. A
static level is the same object with `w_end` unset, so the constant case is a
point in the search space rather than a separate code path.

**C (COVER)** — sell the next OTM strike against a naked long, and hold the
resulting long spread to expiry.

**S (SHORT)** — sell a closer-to-ATM strike against a naked long, and hold the
resulting short spread to expiry.

W and L are *levels*; what happens when one is hit is a separate choice:

    at W:  close | cover | short
    at L:  close | short | none

Both may be zero, because some methods never improve with a stop and some
(entered as short spreads) are always better held to expiry.

**One exit beyond the specification's four**, off by default: `exit_gate` closes on the
first bar open after entry at which an indicator condition passes (see
`gates.py`). Bar open and not any minute: the metrics are only valid there. It
crosses the spread when it fires, like a stop — it is a decision to be out, not
a resting order.

There is deliberately **no unconditional time exit**. Nothing is ever flattened
merely because a clock struck, and nothing needs to be: SPX is cash-settled, so
an untouched position expires for free. A time-of-day condition belongs in a
gate, alongside whatever else makes it a decision.

**Ordering when several fire in the same minute: L, then the gate, then W.**
The adverse one first, and the favourable one last, for the same reason ties go
to L: a minute has no internal order and resolving it in the trade's favour
would flatter every result.

TWO FILL MODELS, AND WHY THEY DIFFER
------------------------------------
**A W exit fills at its own limit price.** It was resting: the moment someone
bids $4.00 for the option, the order that has been sitting at $4.00 trades at
$4.00. If the minute grid first shows a bid of $8.00, the limit filled somewhere
inside that minute at $4.00 — booking $8.00 would be collecting a windfall the
resting order could not have received. This is the conservative choice and also
the realistic one.

**An L exit fills at the market.** A stop is not a resting limit; it is a
decision to be out, and it concedes the spread like any other crossing order.

That asymmetry is not a modelling convenience. It is the difference between the
two order types, and getting it backwards would flatter every stop.

W AND L ARE LONG-ONLY
---------------------
The specification removed the short side's W, and that was the only short method
that had one. Every structure carrying a W or an L is therefore entered for a
**debit**: the entry premium is what you paid, the position improves as it gets
richer, W is a multiple above 1 and L a fraction below it. The specification's ranges —
W in (1.5 … 5), L in (0.25 … 0.65) — are simply *the* ranges now.

A credit structure handed a W or an L is **refused**, not silently inverted. A
short spread's take-profit is a different object (buy it back for half the
credit, so `w < 1`), and the moment two opposite readings of one number coexist
in one field, a cell of a sweep means whichever the author last had in mind.

TRIGGERS READ THE 1-MINUTE OPTION DATA
--------------------------------------
Per the specification, and per the book that would actually trade: a long's exit is
tested against the **bid**, because a sell limit needs a buyer. The trigger book
is the same `edge` knob the fill model uses, so a study can ask what its stops
would have done on midpoints — but the default is the side that trades, and a
stop compared against a midpoint fires on a price nobody was offering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .broker import MULTIPLIER, Broker, Order
from .gates import GateSet
from .indicators import ChartSpec
from .session import CALL, Contract, Session

CLOSE, COVER, SHORT, NONE = "close", "cover", "short", "none"
W_ACTIONS = (CLOSE, COVER, SHORT)
L_ACTIONS = (CLOSE, SHORT, NONE)

#: Why a trade ended.
HIT_W, HIT_L, SETTLED, CONVERTED = "W", "L", "settle", "converted"
HIT_GATE = "gate"


@dataclass(frozen=True, slots=True)
class ExitPolicy:
    """One (W, L) cell of the baseline sweep, plus what to do at each."""

    w: float = 0.0
    """Multiple of the entry premium at which the close limit rests. 0 = none.
    With `w_end` set, this is the level AT ENTRY and it decays from here."""

    w_end: Optional[float] = None
    """Asymptote the W level decays toward. None = a static level."""

    w_half_life: float = 45.0
    """Minutes for the W level to cover half its remaining distance to `w_end`."""

    l: float = 0.0
    """Fraction of the entry premium at which the stop sits. 0 = none."""

    l_end: Optional[float] = None
    """Asymptote the L level decays toward. None = a static stop. A stop that
    RISES with time held (`l_end > l`) tightens as expiry approaches; one that
    falls gives a trade more room the longer it survives."""

    l_half_life: float = 45.0

    exit_gate: "GateSet" = field(default_factory=lambda: GateSet())
    """Indicator condition that closes the position on the first bar open after
    entry at which it passes. May read the SPX chart, the traded option's own
    chart, or both — see `gates.Gate.chart`."""

    option_chart: "ChartSpec" = field(default_factory=lambda: ChartSpec())
    """How to read the option's own chart, when the exit gate does. Session
    bounded, so 1m or 5m."""

    w_action: str = CLOSE
    l_action: str = CLOSE

    cover_width: int = 1
    """Strikes OTM to sell when covering. The specification allows up to three."""

    short_width: int = 1
    """Strikes toward the money to sell when shorting. Up to three."""

    credit: bool = False
    """Kept for the record only. **W and L are long-only** as of 22 Aug 2026:
    The one short method that allowed a W no longer does,
    so no credit structure carries either trigger. `manage` REFUSES a credit
    trade with a W or L rather than quietly inverting one, which turns the spec
    into an assertion instead of a comment."""

    trigger_edge: Optional[float] = None
    """Which book the triggers read. `None` — the default — means **the same
    book the fill model trades on**, which is the only self-consistent choice:
    a W is a resting limit and fills when the BID reaches it, so triggering it
    off a midpoint while filling at the bid produces a signal that then cannot
    be filled and is refused. Under the mid bracket both move together.

    An explicit value overrides it, which is how the smoothing question gets
    asked — "what if the stop measured midpoints while fills crossed" — with
    the understanding that a W so triggered may find no bid and be refused.
    That refusal is the honest answer, not a bug."""

    def validate(self) -> "ExitPolicy":
        if self.w_half_life <= 0 or self.l_half_life <= 0:
            raise ValueError("half lives are positive minutes")
        if self.w_end is not None and self.w_end < 0:
            raise ValueError("w_end is a magnitude")
        if self.l_end is not None and self.l_end < 0:
            raise ValueError("l_end is a magnitude")
        self.exit_gate.validate()
        if self.w_action not in W_ACTIONS:
            raise ValueError(f"w_action {self.w_action!r} not in {W_ACTIONS}")
        if self.l_action not in L_ACTIONS:
            raise ValueError(f"l_action {self.l_action!r} not in {L_ACTIONS}")
        if not 1 <= self.cover_width <= 3 or not 1 <= self.short_width <= 3:
            raise ValueError("spread widths run 1..3 strikes")
        if self.w < 0 or self.l < 0:
            raise ValueError("w and l are magnitudes; use 0 to disable")
        return self

    def levels(self, entry: float) -> Tuple[Optional[float], Optional[float]]:
        """(W, L) as absolute per-share prices AT ENTRY. Static view, for tests
        and for reporting the level a fired W was resting at."""
        e = abs(entry)
        return (self.w * e if self.w else None,
                self.l * e if self.l else None)

    def level_series(self, entry: float, entry_minute: int, n_minutes: int,
                     which: str) -> Optional[np.ndarray]:
        """The whole day's trigger level, per minute. None = disabled.

        Decay is exponential in MINUTES HELD, not minutes of the day: two
        trades opened an hour apart are the same trade at different times, and
        anchoring the schedule to the clock would give the later one a level it
        never asked for."""
        start = self.w if which == "w" else self.l
        if not start:
            return None
        end = self.w_end if which == "w" else self.l_end
        hl = self.w_half_life if which == "w" else self.l_half_life
        e = abs(entry)
        if end is None or end == start or hl <= 0:
            return np.full(n_minutes, start * e)
        held = np.arange(n_minutes, dtype=float) - entry_minute
        held[held < 0] = 0.0
        mult = end + (start - end) * np.exp(-np.log(2.0) * held / hl)
        return mult * e

    def label(self) -> str:
        def lvl(a, b, hl):
            if not a:
                return "0"
            return f"{a:g}" if b is None or b == a else f"{a:g}>{b:g}@{hl:g}"
        w = f"w{lvl(self.w, self.w_end, self.w_half_life)}"
        if self.w:
            w += {"close": "", "cover": "C", "short": "S"}[self.w_action]
        l = f"l{lvl(self.l, self.l_end, self.l_half_life)}"
        if self.l:
            l += {"close": "", "short": "S", "none": "N"}[self.l_action]
        return f"{w}/{l}" + ("/g" if self.exit_gate else "")


# ------------------------------------------------------------------- valuing

def close_value_series(sess: Session, legs: Sequence[Tuple[Contract, int]],
                       edge: float) -> np.ndarray:
    """Per-share proceeds of closing `legs`, for every minute of the session.

    Positive means money comes in. `legs` is what is HELD, so closing sells the
    longs and buys back the shorts; at `edge = 1` the longs go off at the bid
    and the shorts are bought at the ask. NaN wherever any leg lacks a
    two-sided quote — a package with a dark leg has no price, and a trigger
    tested against NaN is False in numpy, which is exactly the wanted answer.

    Vectorised over the whole day on purpose: a trade's exit is a
    first-crossing, so one array and one `argmax` replaces a per-minute loop
    over every open trade."""
    total = None
    for c, q in legs:
        col = sess.column(c.right, c.strike)
        if col is None:
            return np.full(sess.n_minutes, np.nan)
        bid = sess.arrays[(c.right, "bid")][:, col]
        ask = sess.arrays[(c.right, "ask")][:, col]
        mid = (bid + ask) / 2.0
        half = (ask - bid) / 2.0
        # Closing a long SELLS it (concede toward the bid); closing a short
        # BUYS it back (concede toward the ask).
        px = mid - edge * half if q > 0 else mid + edge * half
        leg = q * px
        total = leg if total is None else total + leg
    return total


def _first(mask: np.ndarray, after: int) -> Optional[int]:
    """Index of the first True strictly after `after`, or None."""
    if mask.size <= after + 1:
        return None
    tail = mask[after + 1:]
    if not tail.any():
        return None
    return int(np.argmax(tail)) + after + 1


# --------------------------------------------------------------------- trade

@dataclass(slots=True)
class Trade:
    """One signal, from entry to whatever ended it.

    Owns its own basis and its own cash. The broker owns the account; a trade
    owns the answer to "what did THIS idea make", which is the row the metrics
    layer groups by entry hour, by zone, by anything."""

    tag: str
    entry_minute: int
    legs: List[Tuple[Contract, int]]
    entry_price: float                  # per share, magnitude of the premium
    credit: bool
    cash: float = 0.0                   # signed dollars, money in positive
    fees: float = 0.0
    exit_minute: Optional[int] = None
    exit_reason: str = SETTLED
    exit_price: Optional[float] = None  # per share, proceeds of the close
    converted_to: Optional[float] = None   # strike sold in a COVER / SHORT
    target_miss: float = 0.0
    open_zone: int = -1
    bar_minute: int = -1
    """The bar that produced the signal — the join key to the indicator frame."""

    entry_legs: int = 1
    """How many legs the structure had at entry. `legs` is emptied on a close,
    so the shape has to be recorded rather than counted afterwards."""

    exit_refused: str = ""
    """Set when a trigger fired but the book could not fill the exit. The trade
    rides to expiry, and "held on purpose" stays distinguishable from "tried to
    get out and could not"."""

    def record(self, order: Order) -> None:
        self.cash += order.cash
        self.fees += order.commission

    @property
    def pnl(self) -> float:
        return self.cash


# ------------------------------------------------------------------ the walk

def manage(broker: Broker, trade: Trade, policy: ExitPolicy,
           gate_minutes: Optional[np.ndarray] = None) -> Trade:
    """Run one trade to its conclusion. Settlement is the caller's job.

    Only ONE trigger can ever fire, because every action either flattens the
    trade or converts it into a spread that is held to expiry. So this is four
    first-crossing computations over pre-built arrays, not a loop over minutes
    with a branch per open trade.

    `gate_minutes` is a per-minute boolean from `gates.minute_mask`, True only
    at bar opens where the exit condition passes."""
    policy = policy.validate()
    if (policy.w or policy.l) and (trade.credit or policy.credit):
        raise ValueError(
            "W and L are long-only; a credit structure carries neither. "
            "A credit structure's take-profit is a buy-back at a FRACTION of "
            "the credit, which is the opposite reading of the same number; "
            "give it its own policy rather than inverting this one.")

    sess = broker.session
    n = sess.n_minutes
    w_lv = policy.level_series(trade.entry_price, trade.entry_minute, n, "w")
    l_lv = policy.level_series(trade.entry_price, trade.entry_minute, n, "l")

    hit_g = None
    if policy.exit_gate and gate_minutes is not None:
        hit_g = _first(gate_minutes, trade.entry_minute)

    hit_w = hit_l = None
    if w_lv is not None or l_lv is not None:
        edge = (broker.fill_model.edge if policy.trigger_edge is None
                else policy.trigger_edge)
        v = close_value_series(sess, trade.legs, edge)
        if w_lv is not None:
            hit_w = _first(v >= w_lv, trade.entry_minute)
        if l_lv is not None:
            hit_l = _first(v <= l_lv, trade.entry_minute)

    # Adverse first, favourable last. A minute has no internal order, so any
    # tie resolved in the trade's favour would flatter every result.
    candidates = [(hit_l, HIT_L, policy.l_action, None),
                  (hit_g, HIT_GATE, CLOSE, None),
                  (hit_w, HIT_W, policy.w_action, w_lv)]
    live = [(m, r, a, lv) for m, r, a, lv in candidates if m is not None]
    if not live:
        return trade
    minute, reason, action, levels = min(live, key=lambda c: c[0])
    if action == NONE:
        # "Do nothing at L" removes only the L. Anything else still stands.
        rest = [c for c in live if c[1] != HIT_L]
        if not rest:
            return trade
        minute, reason, action, levels = min(rest, key=lambda c: c[0])
    _act(broker, trade, policy, minute, reason, action,
         limit=float(levels[minute]) if levels is not None else None)
    return trade


def _act(broker: Broker, trade: Trade, policy: ExitPolicy, minute: int,
         reason: str, action: str, limit: Optional[float]) -> None:
    if action == CLOSE:
        legs = [(c, -q) for c, q in trade.legs]
        # A W exit was resting and fills at its own price; an L exit crosses.
        o = broker.submit(legs, minute, tag=f"{trade.tag} {reason}", limit=limit)
        if not o:
            trade.exit_refused = reason   # cannot get out; it rides to expiry
            return
        trade.record(o)
        trade.legs = []
        trade.exit_minute, trade.exit_reason = minute, reason
        trade.exit_price = o.price
        return

    # COVER and SHORT both sell one option against a single naked long and
    # then hold the spread to expiry. Direction is the only difference, and
    # `Session.step` already owns it: +n is further out of the money on either
    # right, -n is toward it.
    if len(trade.legs) != 1 or trade.legs[0][1] <= 0:
        trade.exit_refused = f"{reason}:{action} needs a naked long"
        return
    held, qty = trade.legs[0]
    n = policy.cover_width if action == COVER else -policy.short_width
    k = broker.session.step(held.right, held.strike, n)
    if k is None:
        trade.exit_refused = f"{reason}:{action} off the chain"
        return
    against = broker.session.contract(held.right, k)
    o = broker.sell(against, qty, minute, tag=f"{trade.tag} {reason} {action}")
    if not o:
        trade.exit_refused = f"{reason}:{action}"
        return
    trade.record(o)
    trade.legs = [(held, qty), (against, -qty)]
    trade.exit_minute, trade.exit_reason = minute, f"{reason}:{action}"
    trade.converted_to = k


def settle(broker: Broker, trades: Sequence[Trade]) -> None:
    """Expire what every trade still holds, at the official close, for free.

    Attributed per trade rather than taken from the account, so the sum of the
    trades reconciles with the broker exactly — which is the invariant that
    catches an attribution mistake."""
    sess = broker.session
    for t in trades:
        for c, q in t.legs:
            t.cash += q * sess.intrinsic(c) * MULTIPLIER
    broker.settle()
