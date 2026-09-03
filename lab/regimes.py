#!/usr/bin/env python3
"""
regimes.py — which days were ordinary, and which were history.

WHY THIS EXISTS
---------------
A naked long held to expiry is paid by the tail, and the tail is a handful of
sessions. Measured on a naive hourly long over 1,894 sessions, **one single
session contributed nearly FOUR TIMES the entire total** — remove that one day
and a comfortably positive result becomes a large negative one. Naked puts have the same problem pointed the other way: they
will look magnificent because 2020 is in the sample.

So every result that involves a naked long held to expiry is reported **split by
regime**. Not to delete the extreme days — they are history and a method that
cannot survive one is not a method — but so that "this works" and "this caught
one tariff announcement" stop looking identical on a summary line.

TWO AXES, BECAUSE THEY CATCH DIFFERENT DAYS
-------------------------------------------
**`move_ratio`** — the realised move divided by the market's own priced move,
the opening ATM straddle. It says *the option market was wrong by this factor*.
It catches 2025-10-10 (a 191-point move against a $22.90 straddle: 8.4x) and
2018-10-10 (5.5x) — cheap options in a calm book that suddenly paid.

**`abs_return`** — the raw size of the day, **open to settlement**, not close
to close: a 0DTE position lives entirely inside the session, so an overnight
gap is not part of what it could have captured. It says *this was a historic move*
regardless of what it cost. It catches 2025-04-09 (+9.99%) and 2020-03-20
(−5.16%), where the straddle was already enormous so the mispricing was
moderate but the magnitude was not.

Measured over the sample, the two agree on only **6 of the 19** days each puts
in its own top 1%. Using either alone would miss most of the other's. A day is
EXTREME if it is extreme on **either**.

Thresholds are FIXED and interpretable, not sample percentiles: "the market's
priced move was wrong by 3.5x" and "a 3% day" mean the same thing next year,
whereas a p99 cut silently re-labels history every time the dataset grows.

EPISODES
--------
Extremes arrive in clusters — 2025-04-03 through 2025-04-09 is one event, not
four. `episode` groups EXTREME days separated by fewer than `episode_gap`
sessions, and `days_from_extreme` lets a report exclude the neighbourhood of an
event rather than only its worst day.

THE SAFETY RULE
---------------
Half of this table is computed from the realised day and is **reporting only**.
Feeding `move_ratio` to an entry rule would be perfect foresight of the exact
thing being predicted. The other half — the opening straddle, VIX, trailing
volatility — is known at the open and is safe.

`SAFE_FOR_RULES` and `REPORTING_ONLY` name which is which, and `rule_safe()`
returns a frame containing only the former, so the safe path is the easy one.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import polars as pl

from . import session as S

CACHE = Path(__file__).resolve().parent / "cache"
TABLE = CACHE / "regimes.parquet"

NORMAL, ELEVATED, EXTREME = "normal", "elevated", "extreme"
TIERS = (NORMAL, ELEVATED, EXTREME)

#: Known at the open. Safe to gate an entry on.
SAFE_FOR_RULES = ("date", "open_spot", "atm", "straddle", "priced_pct", "vix",
                  "prior_vol")

#: Computed from the realised day. Reporting only — never a rule input.
REPORTING_ONLY = ("settle", "realized", "abs_return", "move_ratio", "exc_ratio",
                  "excursion", "tier", "episode", "days_from_extreme")


@dataclass(frozen=True, slots=True)
class RegimeConfig:
    """Fixed thresholds. See the module docstring for why not percentiles."""

    extreme_ratio: float = 3.5      # ~p99 of move_ratio on this sample
    extreme_return: float = 0.030   # a 3% day; ~p99 of |return|
    elevated_ratio: float = 2.5     # ~p95
    elevated_return: float = 0.0175 # ~p95
    episode_gap: int = 5
    """Sessions. EXTREME days closer together than this are one event."""


# ------------------------------------------------------------------ building

def _one(day: dt.date) -> Optional[dict]:
    """One session's regime facts. Runs in a worker."""
    try:
        s = S.load(day)
    except Exception:
        return None
    m = s.first_live()
    if m is None:
        return None
    spot = s.spot(m)
    k = s.atm(S.CALL, m)
    if k is None or not np.isfinite(spot) or not np.isfinite(s.settle):
        return None
    call = s.mid(s.contract(S.CALL, k), m)
    put = s.mid(s.contract(S.PUT, k), m)
    if call is None or put is None:
        return None
    straddle = call + put
    path = s.spot_open[m:]
    up = float(np.nanmax(path) - spot)
    dn = float(spot - np.nanmin(path))
    return dict(date=day, open_spot=float(spot), settle=float(s.settle),
                atm=float(k), straddle=float(straddle), vix=float(s.vix),
                excursion=float(max(up, dn)))


def build(days: Optional[Sequence[dt.date]] = None, workers: int = 24) -> pl.DataFrame:
    """Compute the regime table from the chains. ~10 s for the whole calendar."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    days = list(days) if days is not None else S.calendar()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(_one, days, chunksize=16) if r]
    return _derive(pl.DataFrame(rows))


def _derive(df: pl.DataFrame, cfg: RegimeConfig = RegimeConfig()) -> pl.DataFrame:
    df = df.sort("date").with_columns(
        (pl.col("settle") - pl.col("open_spot")).abs().alias("realized"),
        ((pl.col("settle") - pl.col("open_spot")) / pl.col("open_spot"))
            .abs().alias("abs_return"),
        (pl.col("straddle") / pl.col("open_spot")).alias("priced_pct"),
    ).with_columns(
        (pl.col("realized") / pl.col("straddle")).alias("move_ratio"),
        (pl.col("excursion") / pl.col("straddle")).alias("exc_ratio"),
    ).join(_prior_vol(), on="date", how="left")
    extreme = ((pl.col("move_ratio") >= cfg.extreme_ratio)
               | (pl.col("abs_return") >= cfg.extreme_return))
    elevated = ((pl.col("move_ratio") >= cfg.elevated_ratio)
                | (pl.col("abs_return") >= cfg.elevated_return))
    df = df.with_columns(
        pl.when(extreme).then(pl.lit(EXTREME))
          .when(elevated).then(pl.lit(ELEVATED))
          .otherwise(pl.lit(NORMAL)).alias("tier"))
    return _episodes(df, cfg)


def _prior_vol() -> pl.DataFrame:
    """Trailing realised volatility from PRIOR sessions — the one dispersion
    measure here a rule may legitimately read.

    Computed over EVERY trading session from the master index, not over the
    option-session subset: before 2022 SPXW listed 0DTE only on Mon/Wed/Fri, so
    differencing the subset would treat a Friday-to-Monday move as one day and
    quietly inflate the early years' volatility."""
    m = (pl.read_parquet(S.MASTER_INDEX, columns=["date", "official_close"])
           .sort("date"))
    return m.with_columns(
        (pl.col("official_close").log().diff()
           .rolling_std(window_size=20, min_samples=20, ddof=0)
           .shift(1)).alias("prior_vol")).select("date", "prior_vol")


def _episodes(df: pl.DataFrame, cfg: RegimeConfig) -> pl.DataFrame:
    """Group EXTREME days into events, and measure distance to the nearest one."""
    is_ex = (df["tier"] == EXTREME).to_numpy()
    idx = np.arange(df.height)
    ex_idx = idx[is_ex]
    episode = np.full(df.height, -1, dtype=np.int32)
    if ex_idx.size:
        breaks = np.diff(ex_idx) >= cfg.episode_gap
        ep_id = np.concatenate([[0], np.cumsum(breaks)])
        episode[ex_idx] = ep_id
        # Distance in sessions to the nearest extreme, so a report can drop the
        # neighbourhood of an event and not only its single worst day.
        dist = np.abs(idx[:, None] - ex_idx[None, :]).min(axis=1)
    else:
        dist = np.full(df.height, 10_000)
    return df.with_columns(pl.Series("episode", episode),
                           pl.Series("days_from_extreme", dist.astype(np.int32)))


@lru_cache(maxsize=1)
def table() -> pl.DataFrame:
    """The cached regime table, built on first use."""
    if TABLE.exists():
        return pl.read_parquet(TABLE)
    CACHE.mkdir(parents=True, exist_ok=True)
    df = build()
    df.write_parquet(TABLE)
    return df


def refresh() -> pl.DataFrame:
    """Rebuild and overwrite the cache. Call after the dataset changes."""
    CACHE.mkdir(parents=True, exist_ok=True)
    df = build()
    df.write_parquet(TABLE)
    table.cache_clear()
    return df


# ------------------------------------------------------------------- using

def rule_safe(df: Optional[pl.DataFrame] = None) -> pl.DataFrame:
    """Only the columns a rule may read. The safe path, made the easy one."""
    d = table() if df is None else df
    return d.select([c for c in SAFE_FOR_RULES if c in d.columns])


def attach(trades: pl.DataFrame) -> pl.DataFrame:
    """Join the regime tier onto a per-trade frame."""
    keep = ["date", "tier", "episode", "days_from_extreme", "move_ratio",
            "abs_return", "straddle"]
    return trades.join(table().select(keep), on="date", how="left")


def split(trades: pl.DataFrame, exclude_near: int = 0) -> pl.DataFrame:
    """Score a per-trade frame per regime tier, plus an ex-extreme total.

    `exclude_near` also drops sessions within that many days of an extreme, so
    an event can be removed as an event rather than as its worst afternoon."""
    from .metrics import score
    df = attach(trades)
    out = []
    groups = [(t, df.filter(pl.col("tier") == t)) for t in TIERS]
    groups.append(("ex-extreme", df.filter(pl.col("tier") != EXTREME)))
    if exclude_near:
        groups.append((f"ex-episode({exclude_near}d)",
                       df.filter(pl.col("days_from_extreme") > exclude_near)))
    groups.append(("all", df))
    for name, g in groups:
        if not g.height:
            continue
        daily = (g.group_by("date").agg(pl.col("pnl").sum())
                  .sort("date")["pnl"].to_numpy())
        s = score(daily, g["pnl"].to_numpy(), float(g["fees"].sum()))
        out.append(dict(tier=name, sessions=int(g["date"].n_unique()),
                        trades=s.trades, total=s.total, per_trade=s.per_trade,
                        per_day=s.per_day, sharpe=s.sharpe,
                        max_dd=s.max_drawdown, win_trades=s.win_rate_trades,
                        ex_top1=s.total_ex_top1, days_to_half=s.days_to_half))
    return pl.DataFrame(out)
