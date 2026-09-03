#!/usr/bin/env python3
"""
fills.py — what price you actually get, and the bracket you must report.

THE STANDING RULE THIS ENFORCES
-------------------------------
Striking at the midpoint versus crossing the spread was worth $23-45 per trade
in the previous programme — larger than any parameter effect it measured in
nine stages. So the fill assumption is not a sensitivity check bolted on at the
end; it is a first-class axis, and every result is reported as a bracket. A
strategy that survives only at the mid is an execution claim, not a strategy
claim, and `Bracket` exists so that sentence has a number attached.

THE MODEL
---------
One knob, `edge`, in [0, 1]:

    price = mid + side * edge * (ask - bid) / 2

    edge = 0.0   fill at the midpoint          — the optimistic bracket
    edge = 0.5   split the half-spread
    edge = 1.0   buy the ask, sell the bid     — the pessimistic bracket

`side` is +1 to buy and -1 to sell, so the same expression pays up on a buy and
concedes on a sell without a branch.

TICK CONCESSION
---------------
SPX options tick $0.05 below $3.00 and $0.10 at or above it, and an order that
is not on the grid is rejected outright (IBKR Error 110). Rounding therefore is
not cosmetic. It always CONCEDES — buys round up, sells round down — so the
snap can never manufacture a better price than the book offered. At edge = 1.0
the rounding is a no-op, because bid and ask are already on the grid.

WHAT IS REFUSED
---------------
A fill requires a two-sided quote. A contract with an ask and no bid can in
reality be lifted, but allowing that would make the mid and cross brackets
disagree about WHICH TRADES HAPPEN as well as at what price, and the bracket is
only interpretable when both legs of it see the same opportunity set. The flag
exists to answer the question later; it is off, and the refusal is recorded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

BUY, SELL = 1, -1

TICK_BREAK = 3.00
TICK_LOW, TICK_HIGH = 0.05, 0.10


def tick_size(price: float) -> float:
    """$0.05 under $3.00, $0.10 at or above. The break is on the PRICE being
    quoted, so a walk crossing $3.00 changes grid mid-walk — which is real."""
    return TICK_LOW if price < TICK_BREAK else TICK_HIGH


def round_conceding(price: float, side: int) -> float:
    """To the grid, always against yourself. Buys up, sells down."""
    t = tick_size(price)
    n = price / t
    # A price already on the grid must not move: 2.4000000000000004 / 0.05 is
    # 48.000000000000007, and a naive ceil would push it to 2.45.
    r = round(n)
    if abs(n - r) < 1e-9:
        return round(r * t, 4)
    from math import ceil, floor
    return round((ceil(n) if side == BUY else floor(n)) * t, 4)


@dataclass(frozen=True, slots=True)
class FillModel:
    """How a decision becomes a price."""

    edge: float = 1.0
    """0 = mid, 1 = full cross. The two ends are the reported bracket."""

    tick: bool = True
    allow_one_sided_buy: bool = False
    """Lift an offer with no bid behind it. Off: see the module docstring."""

    min_price: float = 0.05
    """Below one tick there is no order. A leg quoted under this cannot trade."""

    def price(self, bid: Optional[float], ask: Optional[float],
              side: int) -> Optional[float]:
        """The fill, or None if this quote cannot support one."""
        if ask is None or ask < self.min_price:
            return None
        if bid is None:
            if not (self.allow_one_sided_buy and side == BUY):
                return None
            px = ask
        else:
            mid = (bid + ask) / 2.0
            px = mid + side * self.edge * (ask - bid) / 2.0
            # A price we CONSTRUCT has to be placeable, so it is snapped to the
            # grid. A price we merely TAKE is already a price someone is
            # showing — snapping a 3.05 ask up to 3.10 would charge us for a
            # tick rule that governs orders, not quotes.
            if self.tick and abs(px - bid) > 1e-9 and abs(px - ask) > 1e-9:
                px = round_conceding(px, side)
                # Conceding to the grid must never concede past the book:
                # 2.95/3.05 at half-edge snaps to 3.10, which is a worse price
                # than simply crossing. Clamped, so `edge` stays monotone.
                px = min(px, ask) if side == BUY else max(px, bid)
        px = round(px, 4)
        return px if px >= self.min_price else None

    def label(self) -> str:
        return {0.0: "mid", 0.5: "half", 1.0: "cross"}.get(
            round(self.edge, 3), f"edge{self.edge:g}")


MID = FillModel(edge=0.0)
HALF = FillModel(edge=0.5)
CROSS = FillModel(edge=1.0)

WALK = FillModel(edge=0.25)
"""What a WALKED combo order actually costs, as opposed to the two bounds.

`CROSS` pays the full half-spread **on every leg**, which for a multi-leg order
is not a pessimistic assumption so much as a wrong one: a BAG has its own book,
and the combo market is far tighter than the sum of its legs' markets. The live
price walker posts at the combo mid and steps, and in live paper trading it has
never once ended up with a bid-plus-ask result.

So `MID` and `CROSS` bracket the truth but do not centre it, and `CROSS` is the
looser of the two bounds by a wide margin on four-leg structures. **0.25 is a
placeholder, not a measurement** -- it is deliberately nearer the pessimistic
side of what the walker reportedly achieves, and it must be replaced by a figure
derived from the logged live fills against the book at the time.
"""

BRACKET = (MID, CROSS)
BRACKET3 = (MID, WALK, CROSS)
"""Report all three. The two bounds keep a result honest; the middle one is what
a decision should be taken on, once it is calibrated."""
"""The two models every result is reported against. Never one alone."""
