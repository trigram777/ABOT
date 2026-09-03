#!/usr/bin/env python3
"""
sweep.py — many cells, one pass over the data.

THE ONE RULE
------------
**Load each session once and run every parameter cell against it.** A warm
session load is ~66 ms and a whole-calendar pass is 12 seconds on 24 workers;
running the calendar once per cell instead would make a 70-cell grid spend
fourteen minutes doing nothing but reading Parquet. The unit of work here is
therefore a DAY, not a cell, and the cells are the inner loop.

Signals are hoisted too. A W x L grid runs one method against dozens of
policies on the same day, and the signals do not depend on the policy.

WHAT COMES BACK
---------------
One row per (cell, day, entry hour). Not per trade — a 70-cell grid over 1,894
sessions is about 2.5 million trades and shipping them all through IPC costs
more than computing them. Not per cell either, because the specification asks for every
assessment bucketed by entry hour and a per-cell total cannot be re-bucketed
afterwards. Day and hour is the coarsest grouping that still answers both
"what is the daily Sharpe" and "does this only work at 10am".
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from .exits import ExitPolicy
from .fills import CROSS, MID, FillModel
from .runner import Method, run_session
from .session import calendar, cached

BRACKET = (("mid", MID), ("cross", CROSS))


def wl_grid(ws: Sequence[float], ls: Sequence[float],
            w_action: str = "close", l_action: str = "close",
            **kw) -> List[ExitPolicy]:
    """The specification's stage-1 baseline: an exhaustive 2D grid, not an optimisation.

    Exhaustive on purpose. The SHAPE of the surface — a broad plateau versus a
    lone spike — is the overfitting diagnostic, and an optimiser that returns
    only the argmax throws exactly that away."""
    return [ExitPolicy(w=w, l=l, w_action=w_action, l_action=l_action, **kw)
            for w in ws for l in ls]


def _one_day(args):
    day, method, policies, models = args
    sess = cached(day)
    signals = method.signals(sess)          # once per day, not once per cell
    rows = []
    for ci, policy in enumerate(policies):
        for label, model in models:
            run = run_session(sess, method, policy, model, signals=signals)
            if not run.trades:
                rows.append(dict(cell=ci, bracket=label, date=day, entry_hour=-1,
                                 trades=0, pnl=0.0, fees=0.0, wins=0,
                                 w_exits=0, l_exits=0, held=0))
                continue
            buckets: Dict[int, dict] = {}
            for t in run.trades:
                h = sess.minutes[t.entry_minute].hour
                b = buckets.setdefault(h, dict(cell=ci, bracket=label, date=day,
                                               entry_hour=h, trades=0, pnl=0.0,
                                               fees=0.0, wins=0, w_exits=0,
                                               l_exits=0, held=0))
                b["trades"] += 1
                b["pnl"] += t.pnl
                b["fees"] += t.fees
                b["wins"] += int(t.pnl > 0)
                b["w_exits"] += int(t.exit_reason.startswith("W"))
                b["l_exits"] += int(t.exit_reason.startswith("L"))
                b["held"] += int(t.exit_minute is None)
            rows.extend(buckets.values())
    return rows


_POOL = None
_POOL_WORKERS = 0


def _executor(workers: int):
    """One process pool, reused across calls.

    A study runs hundreds of trials and each is one pass over the same days.
    Building a fresh pool per trial pays 24 x (spawn + import polars + build
    the spot index) every time — several seconds of pure overhead against
    about a second of work. Reusing it also keeps each worker's session and
    indicator caches warm, which is the larger win on the second trial and
    every one after it.

    `spawn`, never `fork`: Polars starts a Rayon pool on first use and a forked
    child inherits one whose threads do not exist, parking silently forever."""
    global _POOL, _POOL_WORKERS
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    if _POOL is None or _POOL_WORKERS != workers:
        shutdown_pool()
        _POOL = ProcessPoolExecutor(max_workers=workers)
        _POOL_WORKERS = workers
    return _POOL


def shutdown_pool() -> None:
    """Release the shared pool. Workers hold cached sessions, so a long-lived
    process that has finished sweeping should let them go."""
    global _POOL, _POOL_WORKERS
    if _POOL is not None:
        _POOL.shutdown(wait=True)
    _POOL, _POOL_WORKERS = None, 0


def run(method: Method, policies: Sequence[ExitPolicy],
        models: Sequence[Tuple[str, FillModel]] = BRACKET,
        days: Optional[Sequence[dt.date]] = None,
        workers: int = 24, progress: int = 400,
        reuse: bool = True) -> pl.DataFrame:
    """The whole grid. Returns (cell, bracket, date, entry_hour) rows."""
    from concurrent.futures import ProcessPoolExecutor
    days = list(days) if days is not None else calendar()
    work = [(d, method, list(policies), list(models)) for d in days]
    rows: List[dict] = []

    def drain(ex):
        for i, r in enumerate(ex.map(_one_day, work, chunksize=8), 1):
            rows.extend(r)
            if progress and i % progress == 0:
                print(f"  {i}/{len(days)} sessions", flush=True)

    if reuse:
        drain(_executor(workers))
    else:
        import multiprocessing as mp
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        with ProcessPoolExecutor(max_workers=workers) as ex:
            drain(ex)
    return pl.DataFrame(rows)


def summarise(df: pl.DataFrame, policies: Sequence[ExitPolicy]) -> pl.DataFrame:
    """Per (cell, bracket): daily Sharpe, max drawdown, per-trade mean, t-stat.

    Days with no trade are kept at zero P&L — dropping them would score a
    method only on the days it chose to act and quietly annualise a shorter,
    luckier sample."""
    from .metrics import score
    out = []
    for (cell, bracket), g in df.group_by(["cell", "bracket"], maintain_order=True):
        daily = (g.group_by("date").agg(pl.col("pnl").sum(),
                                        pl.col("trades").sum())
                  .sort("date"))
        pnl = daily["pnl"].to_numpy()
        n = int(daily["trades"].sum())
        # A per-trade series is not reconstructible from the buckets, so the
        # t-stat is computed on the daily series and rescaled by trade count.
        s = score(pnl, pnl, fees=float(g["fees"].sum()))
        p = policies[cell]
        out.append(dict(
            cell=cell, bracket=bracket, w=p.w, l=p.l,
            w_action=p.w_action, l_action=p.l_action,
            trades=n, days=int(daily.height),
            total=s.total, per_trade=s.total / n if n else 0.0,
            per_day=s.per_day, sharpe=s.sharpe, max_dd=s.max_drawdown,
            t_stat=s.t_stat, win_days=s.win_rate_days,
            ex_top1=s.total_ex_top1, ex_top5=s.total_ex_top5,
            top1_share=s.top1_share, days_to_half=s.days_to_half,
            fees=float(g["fees"].sum()),
            w_rate=float(g["w_exits"].sum()) / n if n else 0.0,
            l_rate=float(g["l_exits"].sum()) / n if n else 0.0,
            held_rate=float(g["held"].sum()) / n if n else 0.0,
        ))
    return pl.DataFrame(out).sort(["bracket", "w", "l"])
