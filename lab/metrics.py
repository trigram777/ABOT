#!/usr/bin/env python3
"""
metrics.py — scoring a cell, and being honest about what the score means.

Deliberately small. Sharpe and max drawdown on the DAILY P&L series, a t-stat
on the per-trade mean, and the counts needed to interpret them. The Deflated
Sharpe Ratio is not here yet because it needs something this module cannot
know — the total number of configurations tried across the whole programme —
and computing it from one study's trial count is the mistake it exists to
prevent.

Daily and not per-trade for the risk numbers: a method that opens thirteen
trades a day has thirteen correlated bets on one day's move, and a per-trade
Sharpe would count that as thirteen independent observations and inflate itself
by roughly the square root of that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import polars as pl

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class Score:
    trades: int
    days: int
    total: float
    per_trade: float
    per_day: float
    sharpe: float                 # annualised, on daily P&L
    max_drawdown: float           # dollars, on the cumulative daily curve
    t_stat: float                 # on the per-trade mean
    win_rate_trades: float
    win_rate_days: float
    fees: float

    # --- tail dependence. Reported ALWAYS, not on request.
    #
    # A Sharpe ratio and a t-stat both assume the mean is a property of the
    # sample. For a long-option method it is routinely a property of one day:
    # on a naive hourly long over 1,894 sessions, ONE session contributes
    # nearly four times the entire total — remove it and a comfortably positive
    # result is a large negative one. Deflated Sharpe does not catch this, because it
    # corrects for how many configurations were tried, not for an expectancy
    # that rests on one observation.
    total_ex_top1: float
    total_ex_top5: float
    top1_share: float
    """Best day's P&L as a share of the total. > 1 means the total is one day
    plus a pile of losses."""
    days_to_half: int
    """How many of the best days it takes to reach half of all GROSS GAINS
    (losing days excluded). A concentration measure: 1 is a coin flip wearing
    a backtest, and a few hundred is an edge that was actually earned."""

    def line(self, label: str = "") -> str:
        return (f"{label:<18} n{self.trades:>6}  total ${self.total:>11,.0f}  "
                f"per trade ${self.per_trade:>7.2f}  Sharpe {self.sharpe:>6.2f}  "
                f"maxDD ${self.max_drawdown:>10,.0f}  t {self.t_stat:>6.2f}  "
                f"ex-top1 ${self.total_ex_top1:>11,.0f}  "
                f"days/half {self.days_to_half:>4}")


def score(daily: np.ndarray, per_trade: np.ndarray, fees: float = 0.0) -> Score:
    """`daily` is one P&L per session; `per_trade` one per trade."""
    daily = np.asarray(daily, dtype=float)
    per_trade = np.asarray(per_trade, dtype=float)
    n_d, n_t = daily.size, per_trade.size
    sd = daily.std(ddof=1) if n_d > 1 else 0.0
    sharpe = (daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else 0.0
    curve = np.cumsum(daily)
    peak = np.maximum.accumulate(curve) if n_d else np.array([0.0])
    dd = float((curve - peak).min()) if n_d else 0.0
    sd_t = per_trade.std(ddof=1) if n_t > 1 else 0.0
    t = (per_trade.mean() / sd_t * np.sqrt(n_t)) if sd_t > 0 else 0.0
    order = np.sort(daily)[::-1] if n_d else np.array([0.0])
    total = float(daily.sum())
    gains = order[order > 0]
    half = np.searchsorted(np.cumsum(gains), 0.5 * gains.sum()) + 1 if gains.size else 0
    return Score(
        total_ex_top1=float(order[1:].sum()) if n_d > 1 else 0.0,
        total_ex_top5=float(order[5:].sum()) if n_d > 5 else 0.0,
        top1_share=float(order[0] / total) if total > 0 else float("nan"),
        days_to_half=int(half),
        trades=n_t, days=n_d, total=total,
        per_trade=float(per_trade.mean()) if n_t else 0.0,
        per_day=float(daily.mean()) if n_d else 0.0,
        sharpe=float(sharpe), max_drawdown=dd, t_stat=float(t),
        win_rate_trades=float((per_trade > 0).mean()) if n_t else 0.0,
        win_rate_days=float((daily > 0).mean()) if n_d else 0.0,
        fees=fees)


def score_frame(df: pl.DataFrame, by: Optional[list] = None) -> pl.DataFrame:
    """Score a per-trade frame, optionally grouped (e.g. by `entry_hour`).

    A session with no trade still counts as a day with zero P&L — dropping it
    would score the strategy only on the days it chose to act and quietly
    annualise a shorter, luckier sample."""
    keys = by or []
    out = []
    groups = ([(tuple(), df)] if not keys
              else [(k, g) for k, g in df.group_by(keys, maintain_order=True)])
    for key, g in groups:
        daily = (g.group_by("date").agg(pl.col("pnl").sum())
                  .sort("date")["pnl"].to_numpy())
        s = score(daily, g["pnl"].to_numpy(), float(g["fees"].sum()))
        row = {k: v for k, v in zip(keys, key if isinstance(key, tuple) else (key,))}
        row.update(vars(s) if hasattr(s, "__dict__") else
                   {f: getattr(s, f) for f in Score.__slots__})
        out.append(row)
    return pl.DataFrame(out)
