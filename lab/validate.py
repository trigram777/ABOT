#!/usr/bin/env python3
"""
validate.py — walk-forward splits and the Deflated Sharpe Ratio.

WHY THIS EXISTS BEFORE THE OPTIMISER
------------------------------------
Evaluations here are cheap: a full-calendar cell is seconds. That is exactly
the regime in which an optimiser's sample-efficiency stops mattering and its
capacity to fit noise starts to. The binding constraint on this programme is
not compute, it is the number of configurations tried — so the machinery that
prices that has to exist before the machinery that spends it.

WALK-FORWARD, NOT ONE SPLIT
---------------------------
A single train/test cut answers one question once, on whichever regime happened
to land in the second half. `walk_forward` produces successive
(train, test) date ranges so a configuration has to survive being refitted
through 2018's volatility, 2020, 2022 and 2025 rather than through one of them.

**Splits are by DATE and are contiguous.** Randomly assigning sessions to folds
would put 2025-04-08 in train and 2025-04-09 in test, and a rule fitted on one
tariff headline would be validated on the next day of the same event.

**A purge gap sits between train and test.** Nothing in this engine holds a
position overnight, so leakage through open trades is impossible — but the
indicator series is continuous across sessions, and a 20-period 60m band on the
first test day is computed from bars inside the training window. The gap is
what makes the band on the first scored day independent of the fitted data.

THE DEFLATED SHARPE RATIO
-------------------------
`deflated_sharpe` asks: given that **N** configurations were tried, how
surprising is the best one's Sharpe? It corrects the threshold upward for the
number of trials and for the skew and kurtosis of the returns, both of which
matter enormously here — a 0DTE long's returns are violently right-skewed and
fat-tailed, and the standard Sharpe error formula assumes neither.

**N must be the honest total**, including every exploratory run, not the size
of the final study. That number is the one people fudge, which is why the
Optuna storage — a durable trial count — is the reason to use it.

**DSR does not rescue an expectancy that rests on one observation.** It prices
multiple testing, not a degenerate payoff distribution. It sits alongside the
tail-dependence figures in `metrics.Score`, never instead of them.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward step. Dates are inclusive."""

    index: int
    train_start: dt.date
    train_end: dt.date
    test_start: dt.date
    test_end: dt.date

    def train(self, days: Sequence[dt.date]) -> List[dt.date]:
        return [d for d in days if self.train_start <= d <= self.train_end]

    def test(self, days: Sequence[dt.date]) -> List[dt.date]:
        return [d for d in days if self.test_start <= d <= self.test_end]

    def label(self) -> str:
        return (f"fold{self.index}: train {self.train_start}..{self.train_end} "
                f"test {self.test_start}..{self.test_end}")


def walk_forward(days: Sequence[dt.date], n_folds: int = 5,
                 train_frac: float = 0.6, purge: int = 25,
                 anchored: bool = True) -> List[Fold]:
    """Successive (train, test) windows over a sorted list of session dates.

    `anchored` grows the training window from a fixed start (more data each
    fold, the usual choice when the process is assumed stable); unanchored
    slides a fixed-length window (fewer assumptions, less data).

    `purge` sessions are dropped between train and test. 25 is one trading
    month, comfortably longer than the longest band warmup in the programme
    (a 20-period 60m band spans about three sessions, a 20-period daily
    context more)."""
    days = sorted(days)
    n = len(days)
    if n_folds < 1 or not 0 < train_frac < 1:
        raise ValueError("n_folds >= 1 and 0 < train_frac < 1")
    first_train = int(n * train_frac)
    if first_train <= purge:
        raise ValueError("training window is shorter than the purge gap")
    remaining = n - first_train
    step = remaining // n_folds
    if step <= purge:
        raise ValueError(
            f"{n_folds} folds over {remaining} test sessions leaves {step} per "
            f"fold, which is not more than the {purge}-session purge gap")
    folds = []
    for i in range(n_folds):
        train_end_i = first_train + i * step - 1
        test_lo = train_end_i + 1 + purge
        test_hi = min(train_end_i + step, n - 1) if i < n_folds - 1 else n - 1
        if test_lo > test_hi:
            break
        train_lo = 0 if anchored else max(0, train_end_i - first_train + 1)
        folds.append(Fold(index=i,
                          train_start=days[train_lo], train_end=days[train_end_i],
                          test_start=days[test_lo], test_end=days[test_hi]))
    return folds


# ------------------------------------------------------------------- the DSR

def sharpe(returns: np.ndarray, periods: int = 252) -> float:
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0


def trial_variance(trial_sharpes: Sequence[float], robust: bool = True) -> float:
    """Spread of the trial Sharpes, for the selection-bias benchmark.

    **Robust by default, and it matters.** The classical estimator is the plain
    sample variance, which assumes the trials are draws from one well-behaved
    distribution of candidate strategies. A real search is not that: it produces
    a cluster of plausible configurations plus a handful of catastrophes. In the
    first study run here, trial Sharpes ran from about -0.5 down to -10, and the
    sample variance those implied put the selection benchmark at **6.44** — a
    bar no strategy of any kind would clear, which makes the DSR uninformative
    rather than strict.

    So the scale comes from the interquartile range (/1.349, the normal-consistent
    conversion), which describes the bulk of the candidates and ignores a few
    blow-ups. **The trial COUNT is untouched** — every configuration tried still
    counts toward N, because that is the multiple-testing term and dropping
    trials from it is precisely the dishonesty this statistic exists to catch.
    Only the *dispersion* is made robust."""
    t = np.asarray([x for x in trial_sharpes if np.isfinite(x)], float)
    if t.size < 2:
        return 0.0
    if not robust:
        return float(t.var(ddof=1))
    q1, q3 = np.percentile(t, [25, 75])
    sigma = (q3 - q1) / 1.349
    # A degenerate IQR (most trials identical) falls back rather than reporting
    # a zero benchmark, which would make every result look significant.
    return float(sigma ** 2) if sigma > 0 else float(t.var(ddof=1))


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """The Sharpe the BEST of `n_trials` independent random strategies would
    show, given the spread of Sharpes across those trials.

    Bailey & López de Prado's benchmark: with enough tries, a good-looking
    Sharpe is the expected outcome of trying, and this is how good-looking."""
    if n_trials < 2 or variance <= 0:
        return 0.0
    e = 0.5772156649015329                       # Euler-Mascheroni
    z1 = stats.norm.ppf(1.0 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(variance) * ((1 - e) * z1 + e * z2))


def probabilistic_sharpe(observed: float, benchmark: float, n: int,
                         skew: float, kurtosis: float) -> float:
    """P(true Sharpe > benchmark), correcting for skew and excess kurtosis.

    Both corrections matter here rather than being pedantry: a 0DTE naked
    long's returns are violently right-skewed and fat-tailed, and the textbook
    Sharpe standard error assumes neither. Positive skew makes a given Sharpe
    MORE credible, fat tails less."""
    if n < 3:
        return float("nan")
    denom = 1.0 - skew * observed + (kurtosis - 1.0) / 4.0 * observed ** 2
    if denom <= 0:
        return float("nan")
    z = (observed - benchmark) * np.sqrt(n - 1) / np.sqrt(denom)
    return float(stats.norm.cdf(z))


@dataclass(frozen=True, slots=True)
class Deflated:
    sharpe: float
    benchmark: float
    dsr: float
    n_trials: int
    n_obs: int
    skew: float
    kurtosis: float

    def verdict(self, alpha: float = 0.95) -> str:
        if not np.isfinite(self.dsr):
            return "undefined"
        return "survives" if self.dsr >= alpha else "not distinguishable from selection"

    def line(self) -> str:
        return (f"Sharpe {self.sharpe:.3f} vs benchmark {self.benchmark:.3f} "
                f"from {self.n_trials} trials -> DSR {self.dsr:.3f} "
                f"({self.verdict()})")


def deflated_sharpe(returns: np.ndarray, trial_sharpes: Sequence[float],
                    n_trials: Optional[int] = None, periods: int = 252,
                    robust: bool = True) -> Deflated:
    """The DSR of `returns`, given the spread of Sharpes across all trials.

    `n_trials` defaults to `len(trial_sharpes)` and should be **overridden with
    the honest total** whenever exploratory runs happened outside the study
    that produced `trial_sharpes`. Under-counting trials is the failure this
    statistic exists to prevent, and it is the easiest one to commit."""
    r = np.asarray(returns, float)
    r = r[np.isfinite(r)]
    n = int(n_trials if n_trials is not None else len(trial_sharpes))
    obs = sharpe(r, periods)
    var = trial_variance(trial_sharpes, robust=robust)
    bench = expected_max_sharpe(n, var)
    # The DSR compares NON-annualised Sharpe per observation, so both sides are
    # brought back to the same footing before the probability is taken.
    scale = np.sqrt(periods)
    dsr = probabilistic_sharpe(obs / scale, bench / scale, r.size,
                               float(stats.skew(r)) if r.size > 2 else 0.0,
                               float(stats.kurtosis(r, fisher=False))
                               if r.size > 3 else 3.0)
    return Deflated(sharpe=obs, benchmark=bench, dsr=dsr, n_trials=n,
                    n_obs=int(r.size),
                    skew=float(stats.skew(r)) if r.size > 2 else 0.0,
                    kurtosis=float(stats.kurtosis(r, fisher=False))
                    if r.size > 3 else 3.0)
