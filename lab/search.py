#!/usr/bin/env python3
"""
search.py — the unified Bayesian sweep, over Optuna TPE.

WHY TPE AND NOT A GENETIC ALGORITHM
-----------------------------------
The space is **conditional**: an indicator's threshold exists only if that
indicator is switched on, and a COVER's width only matters if the W action is a
COVER. A GA mutates dead genes and has no principled way to express that; TPE's
define-by-run does it as ordinary Python control flow, which is what the
`suggest_*` functions below are.

The usual argument for Bayesian optimisation — expensive evaluations — does NOT
apply here. A trial is a few seconds. The binding constraint is overfitting, not
compute, and that is why `validate.py` exists and why three rules are wired in
rather than left to discipline:

**1. The objective is scored at the CROSS bracket.** Nothing whose edge depends
on midpoint fills can win a trial, because a variant that only survives at the
mid is an execution claim rather than a strategy claim. The mid figure is
recorded on every trial so the gap is always visible.

**2. Every trial's Sharpe is kept**, because the Deflated Sharpe Ratio needs the
spread across trials and the honest count of them. Optuna's storage is what
makes that count durable across sessions — under-counting trials is the exact
failure the DSR exists to price.

**3. A random-search arm is a first-class mode.** Evaluations are cheap, so the
control costs almost nothing, and without it there is no way to tell whether
TPE found signal or merely reached the noise ceiling faster.

THRESHOLDS ARE SEARCHED AS QUANTILES
------------------------------------
`s_bandwidth` lives near 0.01, `s_pctb` on [-1, 1], `prev_range` in index
points. Searching those on their raw scales would need a hand-written range per
column and would still be wrong in a different decade. Instead a gate's
threshold is suggested as a **quantile** and mapped through the column's own
empirical distribution — scale free, uniform across columns, and stable as the
index goes from 2,250 to 7,500.

**The quantiles are computed from the TRAINING window only.** Deriving them
from all history would leak the test period's distribution into the definition
of every threshold, which is a subtle enough leak to survive a code review and
large enough to matter.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import polars as pl

from . import indicators as I
from . import examples as M
from . import sweep as _sweep
from .exits import CLOSE, COVER, NONE, SHORT, ExitPolicy
from .fills import CROSS, MID
from .gates import OPTION, SPOT, Gate, GateSet
from .indicators import BandConfig, ChartSpec
from .metrics import score
from .validate import Fold, deflated_sharpe

#: Columns a gate may read, split by how a threshold is expressed on them.
NUMERIC = ("s_pctb", "f_pctb", "pctb_spread", "s_bandwidth", "f_bandwidth",
           "bandwidth_ratio", "gap_low", "gap_mid", "gap_high", "prev_range",
           "green_red_avg")
ORDERED = {"zone": 7, "prev_green": 3}
"""Ordered categoricals. A contiguous band is the natural condition on them —
zones run BL < UL < L < M < H < UH < BH — and it searches far better than a
free subset of seven booleans."""

SET_CAT = {"s_slope": 5, "f_slope": 5, "relation": 5, "slope_pair": 25}
CROSSING = ("cross_low", "cross_mid", "cross_high")


@dataclass(frozen=True, slots=True)
class Space:
    """What may vary. Everything absent from here is held fixed."""

    method: str = "ZONE"
    timeframes: Tuple[int, ...] = (15, 30, 60)

    # --- exits (long methods only)
    w: Tuple[float, ...] = (0.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5, 5.0)
    w_decay: bool = True
    """Allow the W to decay toward an asymptote as well as sit still."""

    l: Tuple[float, ...] = (0.0, 0.25, 0.33, 0.5, 0.65)
    w_actions: Tuple[str, ...] = (CLOSE, COVER, SHORT)
    l_actions: Tuple[str, ...] = (CLOSE, SHORT, NONE)
    widths: Tuple[int, ...] = (1, 2, 3)

    # --- gates
    gate_columns: Tuple[str, ...] = ()
    """Explicit column list. Leave empty to derive it from `metric_view`."""

    metric_view: str = "both"
    """Which view of the indicator set may be gated on.

    `zone` and `relation` are EXACT functions of continuous columns already in
    the set — verified reconstructing at 100% over 31,346 bars:

        zone     = f(s_pctb, f_pctb)     discretised at -1 / 0 / +1
        relation = f(sign(gap_low), sign(gap_high))

    so they add no information. What they add is **search efficiency**: one
    ordered-band parameter versus two continuous thresholds an optimiser has to
    land on exactly. The cost is that two views of one feature double the ways
    to express a rule, which is more room to fit noise.

    `both` keeps them (the default), `categorical` searches only the coarse
    trader vocabulary, `continuous` only the fine unbounded columns — which are
    strictly richer, since within zone BL the observed |s_pctb| runs to 7.11 and
    the zone cannot tell 7 half-widths from 1."""

    max_entry_gates: int = 2
    max_exit_gates: int = 1
    charts: Tuple[str, ...] = (SPOT,)
    """Add `OPTION` to let gates read the traded option's own chart."""

    # --- entry window (the specification's hour bucketing)
    windows: Tuple[Tuple[str, str], ...] = (("09:30", "15:00"),)

    # --- method-specific
    targets: Tuple[float, ...] = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0)
    offsets: Tuple[float, ...] = (0.0, 10.0, 20.0, 40.0, 60.0)
    ratios: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7)
    min_ratios: Tuple[float, ...] = (0.15, 0.25, 0.35, 0.5, 0.65)
    spread_widths: Tuple[float, ...] = (5.0, 10.0, 20.0)
    levels: Tuple[float, ...] = (0.2, 0.4, 0.6, 0.8)
    directions: Tuple[str, ...] = ("reversion", "momentum")

    # --- bands
    band_sources: Tuple[str, ...] = ("open",)
    """**Fixed, not searched** (23 Aug). See `indicators.BandConfig.source`: the
    band is built from the same quantity the rule is compared against. Left as a
    tuple rather than deleted so a deliberate re-test is one line."""

    band_mas: Tuple[str, ...] = ("sma",)
    """SMA is the standing choice, and the source and MA are NOT searched: the
    band is built from the same quantity the rule is compared against, so its
    metrics mean one thing rather than two."""

    def columns(self) -> Tuple[str, ...]:
        """The gateable columns this space actually searches."""
        if self.gate_columns:
            return self.gate_columns
        if self.metric_view == "categorical":
            return I.CATEGORICAL_VIEW
        if self.metric_view == "continuous":
            return I.CONTINUOUS_VIEW
        if self.metric_view != "both":
            raise ValueError(f"unknown metric_view {self.metric_view!r}")
        return I.CATEGORICAL_VIEW + I.CONTINUOUS_VIEW

    @property
    def credit_method(self) -> bool:
        """A credit structure carries no W or L: both are long-only, and a
        credit structure handed either raises."""
        return self.method in ("CREDIT",)


# ------------------------------------------------------------------ quantiles

class Quantiles:
    """Empirical quantiles per column, from the TRAINING window only."""

    __slots__ = ("_grid", "_by_col")

    def __init__(self, frame: pl.DataFrame, columns: Sequence[str],
                 grid: int = 101):
        self._grid = np.linspace(0.0, 1.0, grid)
        self._by_col: Dict[str, np.ndarray] = {}
        for c in columns:
            if c not in frame.columns:
                continue
            x = frame[c].to_numpy().astype(float)
            x = x[np.isfinite(x)]
            if x.size:
                self._by_col[c] = np.quantile(x, self._grid)

    def value(self, column: str, q: float) -> Optional[float]:
        arr = self._by_col.get(column)
        if arr is None:
            return None
        return float(np.interp(q, self._grid, arr))

    def has(self, column: str) -> bool:
        return column in self._by_col


def training_quantiles(space: Space, timeframe: int,
                       days: Sequence[dt.date]) -> Quantiles:
    """Quantiles of every gateable column over the training sessions."""
    frame = I.build(timeframe)
    lo, hi = min(days), max(days)
    frame = frame.filter((pl.col("date") >= lo) & (pl.col("date") <= hi))
    return Quantiles(frame, space.columns())


# ------------------------------------------------------------------ suggest

def _suggest_gate(trial, space: Space, quant: Quantiles, prefix: str
                  ) -> Optional[Gate]:
    col = trial.suggest_categorical(f"{prefix}_col", list(space.columns()))
    chart = (trial.suggest_categorical(f"{prefix}_chart", list(space.charts))
             if len(space.charts) > 1 else space.charts[0])
    if col in ORDERED:
        n = ORDERED[col]
        lo = trial.suggest_int(f"{prefix}_lo", 0, n - 1)
        hi = trial.suggest_int(f"{prefix}_hi", lo, n - 1)
        return Gate(column=col, op="in", values=tuple(range(lo, hi + 1)),
                    chart=chart)
    if col in SET_CAT:
        n = SET_CAT[col]
        mask = trial.suggest_int(f"{prefix}_mask", 1, 2 ** n - 2)
        vals = tuple(i for i in range(n) if mask >> i & 1)
        return Gate(column=col, op="in", values=vals, chart=chart)
    if col in CROSSING:
        v = trial.suggest_categorical(f"{prefix}_dir", [-1, 1])
        return Gate(column=col, op="in", values=(int(v),), chart=chart)
    if not quant.has(col):
        return None
    op = trial.suggest_categorical(f"{prefix}_op", ["ge", "le", "between"])
    if op == "between":
        q1 = trial.suggest_float(f"{prefix}_q1", 0.02, 0.98)
        q2 = trial.suggest_float(f"{prefix}_q2", 0.02, 0.98)
        lo_q, hi_q = min(q1, q2), max(q1, q2)
        return Gate(column=col, op="between", lo=quant.value(col, lo_q),
                    hi=quant.value(col, hi_q), chart=chart)
    q = trial.suggest_float(f"{prefix}_q", 0.02, 0.98)
    v = quant.value(col, q)
    return Gate(column=col, op=op, lo=v if op == "ge" else None,
                hi=v if op == "le" else None, chart=chart)


def _suggest_gateset(trial, space: Space, quant: Quantiles, prefix: str,
                     max_gates: int) -> GateSet:
    if max_gates <= 0:
        return GateSet()
    n = trial.suggest_int(f"{prefix}_n", 0, max_gates)
    gates = []
    for i in range(n):
        g = _suggest_gate(trial, space, quant, f"{prefix}{i}")
        if g is not None:
            gates.append(g)
    # `all` only: an `any` set cannot span both charts, and a mixed rule means
    # conjunction anyway.
    return GateSet(gates=tuple(gates), mode="all")


def suggest_policy(trial, space: Space, quant: Quantiles) -> ExitPolicy:
    """The exit half of a trial. Credit methods get an empty policy."""
    exit_gate = _suggest_gateset(trial, space, quant, "xg", space.max_exit_gates)
    if space.credit_method:
        return ExitPolicy(exit_gate=exit_gate,
                          option_chart=ChartSpec(timeframe=5))
    w = trial.suggest_categorical("w", list(space.w))
    l = trial.suggest_categorical("l", list(space.l))
    w_end = None
    half = 45.0
    if w and space.w_decay and trial.suggest_categorical("w_decays", [0, 1]):
        # The asymptote is a FRACTION of the start, so the two cannot cross and
        # a decaying W always relaxes rather than tightening.
        w_end = w * trial.suggest_float("w_end_frac", 0.4, 1.0)
        half = trial.suggest_float("w_half_life", 15.0, 180.0)
    return ExitPolicy(
        w=w, w_end=w_end, w_half_life=half, l=l,
        w_action=trial.suggest_categorical("w_action", list(space.w_actions))
        if w else CLOSE,
        l_action=trial.suggest_categorical("l_action", list(space.l_actions))
        if l else CLOSE,
        cover_width=trial.suggest_categorical("cover_w", list(space.widths)),
        short_width=trial.suggest_categorical("short_w", list(space.widths)),
        exit_gate=exit_gate, option_chart=ChartSpec(timeframe=5))


def suggest_method(trial, space: Space, quant: Quantiles, timeframe: int):
    """The entry half of a trial."""
    bands = BandConfig(
        fast=trial.suggest_int("fast", 5, 15),
        slow=trial.suggest_int("slow", 16, 40),
        k=trial.suggest_float("k", 1.0, 3.0),
        source=trial.suggest_categorical("band_source", list(space.band_sources)),
        ma=trial.suggest_categorical("band_ma", list(space.band_mas)),
        green_red_period=trial.suggest_int("gr_period", 3, 30))
    wi = trial.suggest_categorical("window", [f"{a}|{b}" for a, b in space.windows])
    first, last = wi.split("|")
    common = dict(timeframe=timeframe, bands=bands, first_entry=first,
                  last_entry=last,
                  entry_gate=_suggest_gateset(trial, space, quant, "eg",
                                              space.max_entry_gates),
                  option_chart=ChartSpec(timeframe=5))
    m = space.method
    if m == "ZONE":
        return M.ZoneEntry(target=trial.suggest_categorical("target", list(space.targets)),
                           **common)
    if m == "CREDIT":
        return M.CreditVertical(
            width=trial.suggest_categorical("width", list(space.spread_widths)),
            ratio=trial.suggest_categorical("ratio", list(space.ratios)),
            **common)
    raise ValueError(f"unknown method {m!r}")


# ----------------------------------------------------------------- objective

MIN_TRADES = 200
"""A configuration that trades a handful of times has a Sharpe made of noise.
Below this the trial is scored as a failure rather than allowed to win on four
lucky sessions."""


@dataclass
class Objective:
    """One trial: build a config, run it on the training days, score it.

    Scored at the **cross** bracket. The mid figure is computed and recorded but
    never optimised, so a variant that only works on midpoint fills cannot win
    — and the size of the gap is on every trial for inspection."""

    space: Space
    days: Sequence[dt.date]
    quantiles: Dict[int, Quantiles]
    workers: int = 24
    min_trades: int = MIN_TRADES

    def __call__(self, trial) -> float:
        tf = trial.suggest_categorical("timeframe", list(self.space.timeframes))
        quant = self.quantiles[tf]
        try:
            method = suggest_method(trial, self.space, quant, tf)
            policy = suggest_policy(trial, self.space, quant)
        except ValueError as exc:            # an impossible corner of the space
            trial.set_user_attr("invalid", str(exc))
            return -1e6
        df = _sweep.run(method, [policy], models=(("mid", MID), ("cross", CROSS)),
                        days=list(self.days), workers=self.workers, progress=0)
        out = {}
        for bracket in ("mid", "cross"):
            g = df.filter(pl.col("bracket") == bracket)
            daily = (g.group_by("date").agg(pl.col("pnl").sum())
                      .sort("date")["pnl"].to_numpy())
            n = int(g["trades"].sum())
            s = score(daily, daily)
            out[bracket] = (s, n)
        cross, n_trades = out["cross"]
        mid, _ = out["mid"]
        trial.set_user_attr("trades", n_trades)
        trial.set_user_attr("sharpe_mid", mid.sharpe)
        trial.set_user_attr("sharpe_cross", cross.sharpe)
        trial.set_user_attr("total_cross", cross.total)
        trial.set_user_attr("per_day_cross", cross.per_day)
        trial.set_user_attr("max_dd_cross", cross.max_drawdown)
        trial.set_user_attr("ex_top1_cross", cross.total_ex_top1)
        trial.set_user_attr("days_to_half", cross.days_to_half)
        trial.set_user_attr("bracket_gap", mid.sharpe - cross.sharpe)
        trial.set_user_attr("policy", policy.label())
        trial.set_user_attr("entry_gate", method.entry_gate.label())
        trial.set_user_attr("exit_gate", policy.exit_gate.label())
        if n_trades < self.min_trades:
            return -1e6
        return float(cross.sharpe)


# --------------------------------------------------------------------- study

def run_study(space: Space, days: Sequence[dt.date], n_trials: int = 200,
              sampler: str = "tpe", seed: int = 7, workers: int = 24,
              storage: Optional[str] = None, study_name: Optional[str] = None,
              show_progress: bool = False):
    """One study. `sampler='random'` is the control arm, not an afterthought."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if sampler == "tpe":
        # `multivariate` + `group` are what make TPE handle the CONDITIONAL
        # space properly: a threshold that only exists when its indicator is on
        # is modelled within its own group rather than marginalised over trials
        # where the parameter was absent.
        smp = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True)
    elif sampler == "random":
        smp = optuna.samplers.RandomSampler(seed=seed)
    else:
        raise ValueError(f"unknown sampler {sampler!r}")
    quant = {tf: training_quantiles(space, tf, days) for tf in space.timeframes}
    study = optuna.create_study(direction="maximize", sampler=smp,
                                storage=storage, study_name=study_name,
                                load_if_exists=storage is not None)
    study.optimize(Objective(space, days, quant, workers=workers),
                   n_trials=n_trials, show_progress_bar=show_progress)
    return study


def trial_frame(study) -> pl.DataFrame:
    rows = []
    for t in study.trials:
        if t.value is None:
            continue
        rows.append(dict(number=t.number, value=t.value, **t.user_attrs))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def deflate(study, returns: np.ndarray, extra_trials: int = 0):
    """DSR of a chosen configuration, priced against every trial that was run.

    `extra_trials` is where exploratory runs outside this study are added. The
    count is the input people fudge; Optuna's storage is what makes it honest."""
    sharpes = [t.user_attrs.get("sharpe_cross") for t in study.trials
               if t.user_attrs.get("sharpe_cross") is not None]
    return deflated_sharpe(returns, sharpes,
                           n_trials=len(sharpes) + extra_trials)
