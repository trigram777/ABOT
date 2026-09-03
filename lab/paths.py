#!/usr/bin/env python3
"""
paths.py — intra-minute price resolution and the tick grid, shared.

**A minute bar cannot hold a trail** (METHODOLOGY §4.1). Measured over 14.7M
live option minutes, a single minute's own traded range is 10-15% of a $1-4
option's mid, and a MAJORITY of individual minutes span the entire give-back of
a typical trailing stop all by themselves — rising above 95% in the most
volatile part of the session. A trail offset is therefore only meaningful
against a STATED resolution, and three are carried as a bracket exactly as
`fills.BRACKET3` carries the price.

This module exists as one shared implementation on purpose. Trailing logic was
duplicated across several ad-hoc scripts before it lived here, and the copies
drifted — one of them grew a floor bug the others did not have. A trail is
exactly the kind of object that must have one definition.

    BAR         minute mids only. The optimistic end, and what every trail
                result assumes until the resolution axis is made explicit.
    INTRA       the option's own traded high/low, DE-BOUNCED by the quoted
                spread. The realistic middle: a print at the bid and the next
                at the ask is not a move, and the raw range counts it as one.
    INTRA_RAW   the traded high/low as printed. The pessimistic end.
    NBBO        the LIVE SMOOTHED MID, rebuilt from the quote tape at the
                engine's own quote and decision cadences (METHODOLOGY §4.1).
                Not a bracket end and not an assumption -- a reconstruction of
                what the live decision loop actually sees. It is available only
                for sessions and legs that were pulled, so `price_paths`
                refuses it by name rather than degrading to a mid.

THE TICK GRID CUTS BOTH WAYS AND THE TWO SNAPS ARE NOT THE SAME FUNCTION.
A resting SELL concedes DOWNWARD; a buy-back stop is a BUY and concedes
UPWARD. Rounding a buy-back down would fill it a tick better than any exchange
would have — small, systematic, and in the direction that flatters every short
result.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .fills import TICK_BREAK, TICK_HIGH, TICK_LOW
from .session import Contract, Session

BAR, INTRA, INTRA_RAW = "bar", "intra", "intra_raw"
NBBO = "nbbo"
#: The minute-bar bracket. Deliberately stable: a study that sweeps
#: `RESOLUTIONS` should keep sweeping exactly the three it was written against.
RESOLUTIONS = (BAR, INTRA, INTRA_RAW)
#: Everything a `Spec` may legally name. `NBBO` needs data attached to the
#: session, so it is opt-in rather than swept by default.
ALL_RESOLUTIONS = RESOLUTIONS + (NBBO,)


def _tick(px: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(px) < TICK_BREAK, TICK_LOW, TICK_HIGH)


def snap_down(px: np.ndarray) -> np.ndarray:
    """A resting SELL level, on the tradeable grid, conceding.

    **A trail offset finer than one tick is not a trail, it is arithmetic.**
    SPX options tick $0.05 below $3.00 and $0.10 at or above, so on a $1.05
    option a 0.99 give asks for $1.0395 — a price no exchange accepts and no
    fill can happen at. Unsnapped, a sweep reported the tightest give on the
    grid as the optimum at every strike and every resolution, monotone with no
    turn, which is exactly what a level that cannot be traded at looks like
    from the inside.

    The same wall is visible from the other side: a 1% give-back on a $2 option
    is $0.02, INSIDE ONE TICK. The alternative is to stop the parameter grid by
    hand. Snapping makes the constraint part of the object instead, so the give
    may be swept anywhere and the surface simply goes flat where the ticks run
    out — a minimum realistic trail offset derived rather than imposed."""
    t = _tick(px)
    return np.floor(np.asarray(px) / t + 1e-9) * t


def snap_up(px: np.ndarray) -> np.ndarray:
    """A resting BUY level, on the tradeable grid, conceding.

    The mirror of `snap_down`, and it must exist separately: a short's
    ratcheting buy-back is a BUY, so the grid costs it money in the opposite
    direction. `min(1.5e, 2.25 * max(low, 0.30e))` on a $1.15 entry asks
    $1.7250; the tradeable level is **$1.75**, not $1.70."""
    t = _tick(px)
    return np.ceil(np.asarray(px) / t - 1e-9) * t


def price_paths(sess: Session, c: Contract, resolution: str,
                side: str = "mid"
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(high, low, ok)` per minute for one contract, under one resolution.

    `high` is the best price a rising trail could have ratcheted to inside that
    minute and the worst a BUY stop could be triggered against; `low` is the
    mirror. Under `BAR` both are the midpoint, so a minute is a point and no
    trail can fire inside the minute that set its own extreme.

    The traded range is used only where the bar is real — a finite, positive
    `t_low <= t_high` with volume behind it. A minute with no prints has no
    range, and giving it one from the quote would invent movement.

    `side` names THE SIDE OF THE BOOK THE ORDER TRANSACTS AGAINST — `"ask"`
    for a buy-back, `"bid"` for a sell, `"mid"` for the reference the engine
    currently uses. It is a property of the ORDER, not of the contract, which
    is why it is an argument here rather than a field on `Contract`. Only
    `NBBO` carries the two sides; the three minute-bar resolutions have one
    price per minute and ignore it, so existing callers are unaffected.

    Pricing each trail against the side its own order transacts on sounds
    obviously right and measures out worse than the mid on every axis
    (METHODOLOGY §4.6). The default is the reference, not the side."""
    if resolution not in ALL_RESOLUTIONS:
        raise ValueError(f"unknown resolution {resolution!r}")
    if resolution == NBBO:
        return _nbbo(sess, c, side)
    col = sess.column(c.right, c.strike)
    if col is None:
        n = sess.n_minutes
        nan = np.full(n, np.nan)
        return nan, nan.copy(), np.zeros(n, dtype=bool)
    bid = sess.arrays[(c.right, "bid")][:, col]
    ask = sess.arrays[(c.right, "ask")][:, col]
    mid = (bid + ask) / 2.0
    ok = np.isfinite(mid)
    if resolution == BAR:
        return mid, mid.copy(), ok

    hi = sess.arrays[(c.right, "t_high")][:, col]
    lo = sess.arrays[(c.right, "t_low")][:, col]
    vol = sess.arrays[(c.right, "volume")][:, col]
    real = (np.isfinite(hi) & np.isfinite(lo) & (lo > 0) & (hi >= lo)
            & np.isfinite(vol) & (vol > 0))
    if resolution == INTRA:
        # Remove the bid-ask bounce. A trade at the bid followed by one at the
        # ask is not a move, and the raw range books it as the full spread.
        # Collapse rather than cross when the range is narrower than the
        # spread: what is left is one price, not a negative one.
        half = np.where(np.isfinite(ask) & np.isfinite(bid), (ask - bid) / 2.0, 0.0)
        centre = (hi + lo) / 2.0
        hi = np.maximum(hi - half, centre)
        lo = np.minimum(lo + half, centre)

    high = np.where(real & ok, np.maximum(mid, hi), mid)
    low = np.where(real & ok, np.minimum(mid, lo), mid)
    return high, low, ok


def _nbbo(sess: Session, c: Contract, side: str = "mid"
          ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(high, low, ok)` from the reconstructed live smoothed-mid path.

    `Session.nbbo` maps `(right, strike)` to the per-minute extremes of the
    LIVE SMOOTHED MID over the engine's own decision instants (METHODOLOGY
    §4.1). A leg that was not pulled has no path, and giving it the minute mid
    would silently mix two resolutions inside one structure -- and the leg most
    likely to be missing is the quiet far one, which is precisely the leg a
    fallback would fake most convincingly. So a missing leg is `ok = False`
    everywhere, which the caller reads as "no quotes" and refuses to trade.
    """
    paths = getattr(sess, "nbbo", None)
    if paths is None:
        raise ValueError("resolution NBBO needs Session.nbbo; see "
                         "abot_nbbo_paths.py")
    n = sess.n_minutes
    got = paths.get((c.right, round(float(c.strike), 3)))
    if got is None:
        nan = np.full(n, np.nan)
        return nan, nan.copy(), np.zeros(n, dtype=bool)
    if isinstance(got, dict):
        # A leg pulled with both sides. Falling back to "mid" when a side is
        # absent is deliberate: it is the reference the engine actually uses,
        # so a partial artifact reproduces the deployed rule rather than half
        # of an experimental one.
        return got.get(side) or got["mid"]
    return got
