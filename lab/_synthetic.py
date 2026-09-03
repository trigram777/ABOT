#!/usr/bin/env python3
"""
_synthetic.py — hand-built sessions, for tests that need exact arithmetic.

Not a model of an option market and not trying to be. The point of a fixture
here is that the arithmetic of a fill, a basis, a trigger level and a
settlement can be written down by hand and checked; a Black-Scholes surface
would only make the expected numbers harder to state and no more correct.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional, Sequence

import numpy as np

from .session import CALL, PUT, Contract, Session

STRIKES = np.arange(4900.0, 5101.0, 5.0)
N_MIN = 391


def make_session(settle: float = 5000.0, spot: float = 5000.0,
                 spread: float = 0.20, base: Optional[float] = None) -> Session:
    """A flat chain: every strike two-sided, every minute identical.

    `base` pins every option at one price; the default gives intrinsic plus a
    flat $2 of time value, so verticals have a sane sign."""
    arrays = {}
    for right in (CALL, PUT):
        if base is None:
            intr = (np.maximum(0.0, spot - STRIKES) if right == CALL
                    else np.maximum(0.0, STRIKES - spot))
            row = intr + 2.0
        else:
            row = np.full(STRIKES.size, float(base))
        mid = np.tile(row, (N_MIN, 1))
        arrays[(right, "bid")] = mid - spread / 2
        arrays[(right, "ask")] = mid + spread / 2
        for f in ("last", "volume", "delta", "theta", "vega", "iv", "oi"):
            arrays[(right, f)] = np.zeros((N_MIN, STRIKES.size))
    minutes = [dt.datetime(2024, 1, 3, 9, 30) + dt.timedelta(minutes=i)
               for i in range(N_MIN)]
    ones = np.full(N_MIN, float(spot))
    return Session(date=dt.date(2024, 1, 3), symbol="SPXW", expiry="2024-01-03",
                   minutes=minutes, strikes={CALL: STRIKES, PUT: STRIKES},
                   arrays=arrays, spot_open=ones, spot_high=ones, spot_low=ones,
                   spot_close=ones, spot_reported=np.ones(N_MIN, bool),
                   book_reported=np.ones(N_MIN, bool), settle=settle, vix=15.0)


def set_path(sess: Session, right: str, strike: float, mids: Sequence[float],
             start: int = 0, spread: float = 0.20) -> None:
    """Give one contract a price path from `start` onward."""
    col = sess.column(right, strike)
    for i, m in enumerate(mids):
        t = start + i
        if t >= sess.n_minutes:
            break
        if m is None or not np.isfinite(m):
            sess.arrays[(right, "bid")][t, col] = np.nan
            sess.arrays[(right, "ask")][t, col] = np.nan
        else:
            sess.arrays[(right, "bid")][t, col] = m - spread / 2
            sess.arrays[(right, "ask")][t, col] = m + spread / 2


def hold(sess: Session, right: str, strike: float, mid: float,
         start: int, spread: float = 0.20) -> None:
    """Hold one contract at a price from `start` to the end of the day."""
    set_path(sess, right, strike, [mid] * (sess.n_minutes - start), start, spread)


def C(k: float) -> Contract:
    return Contract("SPXW", CALL, k)


def P(k: float) -> Contract:
    return Contract("SPXW", PUT, k)
