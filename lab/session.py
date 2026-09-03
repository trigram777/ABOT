#!/usr/bin/env python3
"""
session.py — one trading day, as arrays a strategy can index into.

WHAT THIS OWNS
--------------
A session is the option chain for one 0DTE expiry (dense `[minute, strike]`
float arrays, one per field per right), the SPX spot bars that the indicators
and entry triggers read, and the two scalars a day is scored against: the
official close it settles on and the opening VIX print.

It owns no strategy. There is no condor here, no wing, no credit rule — only
"what is the 3900 call bid at 10:17" and "which strike is closest to $2.00".

TIMING, AND WHY IT IS SAFE
--------------------------
The dataset's minute rows are the book at the START of the minute, not the end.
Measured on liquid near-the-money strikes across 2022/2024/2026 sessions:

    mean |quote_mid[m] - ohlc_open[m]|   = $0.07 / $0.52 / $0.07
    mean |quote_mid[m] - ohlc_close[m]|  = $0.31 / $1.31 / $0.41

The quote tracks its own bar's OPEN, and the previous bar's close, roughly ten
times better than it tracks its own close. So a decision taken at minute `m`
that reads `bid[m]`/`ask[m]` and fills against them is reading the book as it
stood when the decision was made. No shift is applied, and none is needed.

The same convention applies to the spot bars: `spot_open[m]` is the index at
`m:00`, which is what a rule firing at `m:00` could have seen. `spot_close[m]`
is the index at the END of that minute and is therefore LOOKAHEAD for any
decision made at `m` — it is carried because exits and excursion measurement
legitimately want it, and it is named so that using it is a choice.

QUOTE HYGIENE
-------------
Applied once, at load, exactly as `engine/chains.py` justified it: a zero or
absent bid is not a price, and NaN is worse than zero because `NaN > 0` is
False while `NaN <= 0` is also False — only the negated comparison catches it.
A crossed book (ask < bid) is bad data, not an arbitrage. All three collapse to
NaN, and every reader downstream treats NaN as "no price" and refuses rather
than inventing one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date as Date
from functools import lru_cache
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

#: Where the dataset lives. Defaults to the sample dataset bundled with this
#: repository -- synthetic, ~60 sessions, enough to run the engine end to end.
#: Point LAB_DATA_ROOT at a production store to run against real data; the
#: layout it expects is the five paths below.
ROOT = Path(os.environ.get("LAB_DATA_ROOT",
                           Path(__file__).resolve().parents[1] / "sample_data"))
SPXW_MINUTE = ROOT / "spxw_pass1" / "minute"
XSP_MINUTE = ROOT / "xsp_pass1" / "minute"
SPOT_1M = ROOT / "spx_ibkr_spot" / "spx_ibkr_1m.parquet"
SPOT_DERIVED = ROOT / "spx_ibkr_spot" / "derived"
MASTER_INDEX = ROOT / "dataset_catalog" / "master_session_index.parquet"

CALL, PUT = "C", "P"
RIGHTS = (CALL, PUT)
_WIRE = {"CALL": CALL, "PUT": PUT}

MINUTE_ROOT = {"SPXW": SPXW_MINUTE, "XSP": XSP_MINUTE}

#: dataset column -> array name. Everything a rule or an indicator might read.
FIELDS: Dict[str, str] = {
    "quote_bid": "bid",
    "quote_ask": "ask",
    "ohlc_open": "t_open",
    "ohlc_high": "t_high",
    "ohlc_low": "t_low",
    "ohlc_close": "last",
    "ohlc_volume": "volume",
    "greeks_delta": "delta",
    "greeks_theta": "theta",
    "greeks_vega": "vega",
    "iv_implied_vol": "iv",
    "oi_open_interest": "oi",
}

PRICED = ("bid", "ask", "last")


# ------------------------------------------------------------------ contracts

@dataclass(frozen=True, slots=True)
class Contract:
    """One option. The expiry is the session, so it is not carried here."""

    symbol: str
    right: str
    strike: float

    def __str__(self) -> str:
        return f"{self.symbol} {self.strike:g}{self.right}"


# -------------------------------------------------------------------- session

@dataclass(slots=True, eq=False)
class Session:
    """One 0DTE expiry: the chain, the spot path, and the settlement price.

    `eq=False` so a Session hashes by identity. It holds megabytes of arrays,
    field-wise equality would be meaningless and slow, and identity is exactly
    what a per-session cache key wants — `cached()` already guarantees one
    object per date per process."""

    date: Date
    symbol: str
    expiry: str
    minutes: List                       # tz-aware ET timestamps, 09:30..16:00
    strikes: Dict[str, np.ndarray]      # per right, ascending
    arrays: Dict[Tuple[str, str], np.ndarray]   # (right, field) -> [min, strike]
    spot_open: np.ndarray               # SPX at m:00 — lookahead-safe at m
    spot_high: np.ndarray
    spot_low: np.ndarray
    spot_close: np.ndarray              # SPX at m:59 — LOOKAHEAD at m
    spot_reported: np.ndarray           # did IBKR print a bar for that minute
    book_reported: np.ndarray           # did the option file report an underlying
    settle: float                       # official daily close — what 0DTE pays on
    vix: float                          # the OPENING print, lookahead-safe

    _index: Dict[str, Dict[float, int]] = None   # right -> strike -> column

    #: (right, strike) -> (high, low, ok) of the LIVE SMOOTHED MID, per minute
    #: (METHODOLOGY §4.1). Attached by the quote-tape reduction for the
    #: sessions that were pulled and `None` everywhere else; `paths.NBBO` is
    #: the only reader and it raises rather than degrade when it is absent.
    nbbo: Optional[Dict] = None

    # ------------------------------------------------------------- geometry
    def __post_init__(self) -> None:
        if self._index is None:
            self._index = {r: {round(float(s), 3): i
                               for i, s in enumerate(self.strikes[r])}
                           for r in self.strikes}

    @property
    def n_minutes(self) -> int:
        return len(self.minutes)

    def column(self, right: str, strike: float) -> Optional[int]:
        return self._index[right].get(round(float(strike), 3))

    def minute_of(self, hhmm: str) -> int:
        """Index of a wall-clock 'HH:MM' Eastern. Raises if the day lacks it."""
        h, m = (int(p) for p in hhmm.split(":"))
        for i, ts in enumerate(self.minutes):
            if (ts.hour, ts.minute) == (h, m):
                return i
        raise KeyError(f"{hhmm} is not a minute of {self.date}")

    def first_live(self, start: int = 0, window: int = 30) -> Optional[int]:
        """First minute at or after `start` whose option book actually exists.

        Never hardcode 09:31. It is right on 1,890 of the 1,894 eligible
        sessions and wrong on the rest — three open a minute earlier, and one
        does not report a book until **09:41**. A rule that assumed the
        constant would trade that day against the pre-rotation snapshot, which
        is the one thing `book_reported` exists to prevent."""
        for m in range(max(0, start), min(start + window, self.n_minutes)):
            if self.book_reported[m]:
                return m
        return None

    def entry_minute(self, hhmm: str, window: int = 30) -> Optional[int]:
        """The first live minute at or after a wall clock. The honest `minute_of`."""
        return self.first_live(self.minute_of(hhmm), window)

    def clock(self, minute: int) -> str:
        ts = self.minutes[minute]
        return f"{ts.hour:02d}:{ts.minute:02d}"

    # ---------------------------------------------------------------- prices
    def value(self, right: str, strike: float, field: str,
              minute: int) -> Optional[float]:
        col = self.column(right, strike)
        if col is None:
            return None
        v = self.arrays[(right, field)][minute, col]
        return None if not np.isfinite(v) else float(v)

    def quote(self, c: Contract, minute: int) -> Tuple[Optional[float], Optional[float]]:
        """(bid, ask) — either may be None, and often the pair is (None, None)."""
        col = self.column(c.right, c.strike)
        if col is None:
            return None, None
        b = self.arrays[(c.right, "bid")][minute, col]
        a = self.arrays[(c.right, "ask")][minute, col]
        return (None if not np.isfinite(b) else float(b),
                None if not np.isfinite(a) else float(a))

    def mid(self, c: Contract, minute: int) -> Optional[float]:
        b, a = self.quote(c, minute)
        return None if b is None or a is None else (b + a) / 2.0

    def row(self, right: str, field: str, minute: int) -> np.ndarray:
        """The whole strike axis of one field at one minute. NaN where absent."""
        return self.arrays[(right, field)][minute]

    def series(self, c: Contract, field: str) -> Optional[np.ndarray]:
        """One contract's whole day of one field — the 1m path a stop reads."""
        col = self.column(c.right, c.strike)
        return None if col is None else self.arrays[(c.right, field)][:, col]

    def tradeable(self, c: Contract, minute: int) -> bool:
        b, a = self.quote(c, minute)
        return b is not None and a is not None

    # ------------------------------------------------------- strike selection
    def spot(self, minute: int) -> float:
        return float(self.spot_open[minute])

    def atm(self, right: str, minute: int) -> Optional[float]:
        """Nearest listed strike to spot. The ATM floor every price sweep starts at."""
        s = self.spot(minute)
        ks = self.strikes[right]
        if not ks.size or not np.isfinite(s):
            return None
        return float(ks[int(np.argmin(np.abs(ks - s)))])

    def step(self, right: str, strike: float, n: int) -> Optional[float]:
        """`n` strikes further OUT of the money (negative = toward the money).

        Direction is per right — a call goes OTM upward, a put downward — so a
        COVER ("sell the next OTM strike against it") and a SHORT ("sell a
        closer-to-ATM strike") are `step(+n)` and `step(-n)` on either side,
        with no `if right == CALL` at the call site."""
        col = self.column(right, strike)
        if col is None:
            return None
        j = col + (n if right == CALL else -n)
        ks = self.strikes[right]
        return float(ks[j]) if 0 <= j < ks.size else None

    def by_price(self, right: str, target: float, minute: int,
                 price: str = "mid", otm_only: bool = True,
                 tol: float = 0.0) -> Optional["StrikePick"]:
        """The listed strike trading closest to `target` dollars.

        `target <= 0` means at the money. Otherwise every strike with a live
        two-sided quote is scored on |price - target| and the nearest wins;
        `tol`, if given, is the largest miss in dollars that still counts as
        the target having been met. The miss is always REPORTED rather than
        enforced, because whether a $0.50 bucket that could only be filled at
        $0.80 belongs in the study is a question for the analysis, not for the
        engine — and a rule that silently substituted would make the two
        indistinguishable afterwards."""
        ks = self.strikes[right]
        if not ks.size:
            return None
        s = self.spot(minute)
        bid = self.arrays[(right, "bid")][minute]
        ask = self.arrays[(right, "ask")][minute]
        live = np.isfinite(bid) & np.isfinite(ask)
        # `otm_only` shapes the price LADDER, but the at-the-money rung is the
        # ladder's floor and is defined by distance, not by side: restricting
        # it would return the first strike past spot rather than the nearest
        # one, and "ATM" would mean something different from `atm()`.
        if otm_only and target > 0 and np.isfinite(s):
            live &= (ks >= s) if right == CALL else (ks <= s)
        if not live.any():
            return None
        px = {"mid": (bid + ask) / 2.0, "bid": bid, "ask": ask}[price]
        if target <= 0:                       # at the money, by distance not price
            j = int(np.argmin(np.where(live, np.abs(ks - s), np.inf)))
        else:
            j = int(np.argmin(np.where(live, np.abs(px - target), np.inf)))
        miss = float(px[j] - target) if target > 0 else 0.0
        met = None if (target > 0 and tol <= 0) else (abs(miss) <= tol or target <= 0)
        return StrikePick(strike=float(ks[j]), price=float(px[j]),
                          target=float(target), miss=miss, target_met=met)

    def by_delta(self, right: str, target: float, minute: int) -> Optional[float]:
        """Strike whose |delta| is closest to `target`. Sign is ignored."""
        d = np.abs(self.arrays[(right, "delta")][minute])
        b = self.arrays[(right, "bid")][minute]
        a = self.arrays[(right, "ask")][minute]
        ok = np.isfinite(d) & np.isfinite(b) & np.isfinite(a)
        if not ok.any():
            return None
        j = int(np.argmin(np.where(ok, np.abs(d - abs(target)), np.inf)))
        return float(self.strikes[right][j])

    def contract(self, right: str, strike: float) -> Contract:
        return Contract(self.symbol, right, float(strike))

    # ------------------------------------------------------------- settlement
    def intrinsic(self, c: Contract, price: Optional[float] = None) -> float:
        """Per-share cash settlement value. Cash-settled index options pay this
        and nothing else, and pay it for free."""
        s = self.settle if price is None else price
        scale = 10.0 if c.symbol.upper() == "XSP" else 1.0
        k = c.strike * scale
        return max(0.0, s - k) / scale if c.right == CALL else max(0.0, k - s) / scale


@dataclass(frozen=True, slots=True)
class StrikePick:
    """What `by_price` found, including how far it missed."""

    strike: float
    price: float
    target: float
    miss: float          # price - target, dollars; positive = pricier than asked
    target_met: Optional[bool]
    """None when no tolerance was supplied — the miss is reported, not judged."""


# --------------------------------------------------------------------- loading

def _partition(symbol: str, day: Date) -> Path:
    root = MINUTE_ROOT[symbol.upper()]
    return root / f"year={day.year}" / f"{day.isoformat()}.parquet"


@lru_cache(maxsize=1)
def _spot_by_date() -> Dict[Date, Tuple[np.ndarray, ...]]:
    """The whole IBKR 1m spot file, split per session, loaded once per process.

    14 MB and 941k rows: about a second to read and partition, against ~2,400
    sessions that would otherwise each pay a predicate pushdown on the same
    file. A sweep worker pays it once."""
    df = (pl.read_parquet(SPOT_1M,
                          columns=["date", "timestamp", "open", "high", "low", "close"])
            .sort("timestamp"))
    out: Dict[Date, Tuple[np.ndarray, ...]] = {}
    for (d,), part in df.group_by(["date"], maintain_order=True):
        out[d] = (np.array([t for t in part["timestamp"].to_list()], dtype=object),
                  part["open"].to_numpy().astype(float),
                  part["high"].to_numpy().astype(float),
                  part["low"].to_numpy().astype(float),
                  part["close"].to_numpy().astype(float))
    return out


@lru_cache(maxsize=1)
def _master() -> pl.DataFrame:
    return pl.read_parquet(MASTER_INDEX)


@lru_cache(maxsize=1)
def _settle_by_date() -> Dict[Date, float]:
    m = _master()
    return dict(zip(m["date"].to_list(), m["official_close"].to_list()))


def calendar(symbol: str = "SPXW", eligible_only: bool = True,
             start: Optional[Date] = None, end: Optional[Date] = None) -> List[Date]:
    """The backtestable dates.

    `eligible_only` applies the master index's `intraday_research_eligible`,
    which is the filter the catalog asks path-dependent work to use: it drops
    sessions whose SPX minute path is incomplete, and an incomplete path is
    exactly what silently breaks a Bollinger band, a stop, or a trail."""
    m = _master()
    col = {"SPXW": "has_spxw", "XSP": "has_xsp"}[symbol.upper()]
    q = m.filter(pl.col(col))
    if eligible_only:
        q = q.filter(pl.col("intraday_research_eligible"))
    days = sorted(q["date"].to_list())
    if start:
        days = [d for d in days if d >= start]
    if end:
        days = [d for d in days if d <= end]
    return days


def load(day: Date, symbol: str = "SPXW") -> Session:
    """One session from Parquet. Raises if the partition is absent."""
    path = _partition(symbol, day)
    if not path.exists():
        raise FileNotFoundError(f"no {symbol} partition for {day}: {path}")

    cols = ["strike", "right", "timestamp", "expiration", "ctx_vix",
            "iv_underlying_price", "greeks_underlying_price", *FIELDS]
    df = pl.read_parquet(path, columns=cols)

    minutes = df["timestamp"].unique().sort().to_list()
    axis = {ts: i for i, ts in enumerate(minutes)}
    n = len(minutes)

    strikes: Dict[str, np.ndarray] = {}
    arrays: Dict[Tuple[str, str], np.ndarray] = {}
    for wire, right in _WIRE.items():
        side = df.filter(pl.col("right") == wire)
        ks = np.array(sorted(set(side["strike"].to_list())), dtype=float)
        col_of = {round(float(s), 3): i for i, s in enumerate(ks)}
        rows = np.fromiter((axis[t] for t in side["timestamp"].to_list()),
                           dtype=np.int32, count=side.height)
        cols_ = np.fromiter((col_of[round(float(s), 3)] for s in side["strike"].to_list()),
                            dtype=np.int32, count=side.height)
        for src, name in FIELDS.items():
            a = np.full((n, ks.size), np.nan)
            a[rows, cols_] = side[src].to_numpy().astype(float)
            arrays[(right, name)] = a
        strikes[right] = ks

        # Hygiene, once, here. `~(x > 0)` and not `x <= 0` — NaN fails both
        # comparisons and only the negated form catches it.
        bid, ask = arrays[(right, "bid")], arrays[(right, "ask")]
        bid[~(bid > 0)] = np.nan
        ask[~(ask > 0)] = np.nan
        crossed = ask < bid
        bid[crossed] = np.nan
        ask[crossed] = np.nan

    # Did the option file report an underlying at all? This is the liveness
    # signal: on essentially every session 09:30:00 reports none, because it is
    # the pre-rotation snapshot rather than a book. It is deliberately NOT the
    # spot series — IBKR does print a 09:30 bar, so the spot cannot tell you
    # the option book was not there yet.
    under = (df.group_by("timestamp")
               .agg(pl.col("iv_underlying_price").drop_nulls().first().alias("iv_u"),
                    pl.col("greeks_underlying_price").drop_nulls().first().alias("gk_u"))
               .sort("timestamp"))
    book = np.zeros(n, dtype=bool)
    idx = np.fromiter((axis[t] for t in under["timestamp"].to_list()),
                      dtype=np.int32, count=under.height)
    iv_u = under["iv_u"].fill_null(0.0).to_numpy().astype(float)
    gk_u = under["gk_u"].fill_null(0.0).to_numpy().astype(float)
    picked = np.where(np.isfinite(iv_u) & (iv_u > 0), iv_u, gk_u)
    book[idx] = np.isfinite(picked) & (picked > 0)

    o, h, l, c, reported = _spot_frame(day, minutes)

    return Session(
        date=day, symbol=symbol.upper(), expiry=df["expiration"][0],
        minutes=minutes, strikes=strikes, arrays=arrays,
        spot_open=o, spot_high=h, spot_low=l, spot_close=c,
        spot_reported=reported, book_reported=book,
        settle=float(_settle_by_date().get(day, float("nan"))),
        vix=_opening_vix(df),
    )


def _opening_vix(df: pl.DataFrame) -> float:
    if "ctx_vix" not in df.columns:
        return float("nan")
    v = df["ctx_vix"].drop_nulls()
    return float(v[0]) if len(v) else float("nan")


def _spot_frame(day: Date, minutes: Sequence) -> Tuple[np.ndarray, ...]:
    """IBKR spot bars projected onto the option minute axis.

    The option axis runs 09:30..16:00 (391 minutes); IBKR's RTH bars run
    09:30..15:59 (390). The 16:00 slot therefore has no bar of its own and
    carries 15:59's close forward — which is what the index was doing at
    16:00, not a guess. `spot_reported` marks it as carried."""
    n = len(minutes)
    o = np.full(n, np.nan); h = np.full(n, np.nan)
    l = np.full(n, np.nan); c = np.full(n, np.nan)
    reported = np.zeros(n, dtype=bool)

    bars = _spot_by_date().get(day)
    if bars is not None:
        ts, bo, bh, bl, bc = bars
        axis = {t: i for i, t in enumerate(minutes)}
        idx = np.fromiter((axis.get(t, -1) for t in ts), dtype=np.int64, count=ts.size)
        keep = idx >= 0
        o[idx[keep]] = bo[keep]; h[idx[keep]] = bh[keep]
        l[idx[keep]] = bl[keep]; c[idx[keep]] = bc[keep]
        reported[idx[keep]] = True

    for a in (o, h, l, c):
        _carry(a)
    # The open of a minute with no bar is the last close, not the last open.
    gap = ~reported
    o[gap] = c[gap]
    return o, h, l, c, reported


def _carry(a: np.ndarray) -> np.ndarray:
    """Last known value forward, then the first known one backward. In place."""
    ok = np.isfinite(a)
    if not ok.any():
        return a
    idx = np.where(ok, np.arange(a.size), 0)
    np.maximum.accumulate(idx, out=idx)
    a[:] = a[idx]
    a[:int(np.argmax(ok))] = a[int(np.argmax(ok))]
    return a


@lru_cache(maxsize=32)
def cached(day: Date, symbol: str = "SPXW") -> Session:
    """Process-local session cache. A sweep runs every parameter cell for one
    session before moving on, so the load is paid once per session, not once
    per cell."""
    return load(day, symbol)


# ------------------------------------------------------------------ spot bars

TIMEFRAMES = {1: SPOT_1M, 5: SPOT_DERIVED / "spx_5m.parquet",
              15: SPOT_DERIVED / "spx_15m.parquet",
              30: SPOT_DERIVED / "spx_30m.parquet",
              60: SPOT_DERIVED / "spx_60m.parquet"}


@lru_cache(maxsize=8)
def bars(minutes: int = 30) -> pl.DataFrame:
    """The whole continuous SPX bar series at one timeframe, 2017 -> now.

    Continuous and not per-session on purpose: a 20-period band at 60m needs
    three sessions of history and cannot exist inside one day. Sessions are a
    column, not a boundary."""
    path = TIMEFRAMES[minutes]
    cols = ["date", "timestamp", "open", "high", "low", "close"]
    df = pl.read_parquet(path).sort("timestamp")
    # `source_minutes` says how many 1m bars a bar was built from, and it is
    # not cosmetic: the 60m series opens each day with a 30-minute stub
    # (09:30-10:00) because RTH starts on the half hour. A rule that assumes
    # every 60m bar is an hour long is wrong on the first bar of every session.
    keep = cols + [c for c in ("source_minutes",) if c in df.columns]
    return df.select(keep)
