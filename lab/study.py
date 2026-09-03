#!/usr/bin/env python3
"""
study.py — the specification's streamlined workflow, end to end.

    python -m lab.study --method ZONE --trials 200 --fold 0

Three stages, in the order that keeps them honest:

**1. Search on the training window only.** TPE over the conditional space, and
a **random-search arm of the same size** — evaluations are cheap, the control
costs almost nothing, and without it there is no way to tell whether TPE found
signal or merely reached the noise ceiling faster.

**2. Evaluate the winner on the held-out test window**, once. Separated from
training by a purge gap, because the indicator series is continuous across
sessions.

**3. Deflate.** The DSR is computed against **every trial from both arms**, not
the winning arm's count, plus any `--extra-trials` run earlier in the
programme. Under-counting is the failure the statistic exists to price.

The winner is reported split by regime as well, because a naked-long result
that lives in the extreme days is not the same finding as one that does not.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
import warnings
from typing import List, Optional, Sequence

import numpy as np
import polars as pl

from . import regimes as R
from . import search as S
from . import sweep as _sweep
from .fills import CROSS, MID
from .metrics import score
from .session import calendar
from .validate import Fold, deflated_sharpe, walk_forward


def evaluate(method, policy, days: Sequence[dt.date], workers: int = 24
             ) -> tuple:
    """Run one configuration over `days`; return (per-trade frame, scores)."""
    from .runner import run_calendar
    out = {}
    frames = {}
    for label, model in (("mid", MID), ("cross", CROSS)):
        df = run_calendar(method, policy, model, days=list(days), workers=workers)
        frames[label] = df
        if df.height:
            daily = (df.group_by("date").agg(pl.col("pnl").sum())
                       .sort("date")["pnl"].to_numpy())
            out[label] = (score(daily, df["pnl"].to_numpy()), daily)
        else:
            out[label] = (score(np.array([0.0]), np.array([0.0])), np.array([0.0]))
    return frames, out


def _rebuild(study, space: S.Space, quant) -> tuple:
    """Reconstruct the winning configuration from its recorded parameters."""
    import optuna
    best = study.best_trial
    fixed = optuna.trial.FixedTrial(best.params)
    tf = fixed.suggest_categorical("timeframe", list(space.timeframes))
    method = S.suggest_method(fixed, space, quant[tf], tf)
    policy = S.suggest_policy(fixed, space, quant[tf])
    return method, policy


def main(argv: Optional[List[str]] = None) -> None:
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="ZONE", choices=list(S.M.ALL))
    p.add_argument("--trials", type=int, default=200)
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--timeframes", default="15,30,60")
    p.add_argument("--workers", type=int, default=24)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--extra-trials", type=int, default=0,
                   help="configurations tried earlier in the programme, added "
                        "to the DSR's trial count")
    p.add_argument("--no-control", action="store_true",
                   help="skip the random-search arm (do not)")
    a = p.parse_args(argv)

    days = calendar()
    fold = walk_forward(days, n_folds=a.folds)[a.fold]
    train, test = fold.train(days), fold.test(days)
    tfs = tuple(int(x) for x in a.timeframes.split(","))
    space = S.Space(method=a.method, timeframes=tfs)
    print(f"\n{a.method}  {fold.label()}")
    print(f"  train {len(train)} sessions   test {len(test)} sessions   "
          f"timeframes {tfs}\n")

    quant = {tf: S.training_quantiles(space, tf, train) for tf in tfs}
    arms = [("tpe", "tpe")] + ([] if a.no_control else [("random", "random")])
    studies, all_sharpes = {}, []
    for name, sampler in arms:
        t0 = time.time()
        st = S.run_study(space, train, n_trials=a.trials, sampler=sampler,
                         seed=a.seed, workers=a.workers)
        studies[name] = st
        sh = [t.user_attrs.get("sharpe_cross") for t in st.trials
              if t.user_attrs.get("sharpe_cross") is not None]
        all_sharpes.extend(sh)
        print(f"  {name:<7} {a.trials} trials in {time.time()-t0:.0f}s   "
              f"best train Sharpe (cross) {st.best_value:+.3f}")

    print()
    for name, st in studies.items():
        method, policy = _rebuild(st, space, quant)
        _, sc = evaluate(method, policy, test, workers=a.workers)
        cross, daily = sc["cross"]
        mid, _ = sc["mid"]
        d = deflated_sharpe(daily, all_sharpes,
                            n_trials=len(all_sharpes) + a.extra_trials)
        print(f"  === {name} winner ===")
        print(f"    {method.name} {method.timeframe}m  {policy.label()}")
        print(f"    entry gate: {method.entry_gate.label()}")
        print(f"    exit  gate: {policy.exit_gate.label()}")
        print(f"    TRAIN Sharpe {st.best_value:+.3f}   "
              f"TEST Sharpe {cross.sharpe:+.3f} (cross) / "
              f"{mid.sharpe:+.3f} (mid)")
        print(f"    TEST per trade ${cross.per_trade:+.2f} over {cross.trades} "
              f"trades   total ${cross.total:+,.0f}   "
              f"ex-top1 ${cross.total_ex_top1:+,.0f}")
        print(f"    {d.line()}")
        print()

    if len(studies) > 1:
        t, r = studies["tpe"].best_value, studies["random"].best_value
        print(f"  TPE vs random on TRAIN: {t:+.3f} vs {r:+.3f} "
              f"({'TPE ahead' if t > r else 'no advantage over random'})")
    _sweep.shutdown_pool()


if __name__ == "__main__":
    main()
