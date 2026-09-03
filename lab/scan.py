#!/usr/bin/env python3
"""
scan.py — which indicators have any conditional signal, before spending trials.

THIS COMES FIRST
----------------
The specification asks for an initial sweep that "attempts a new trade on every bar
[…] as well as **emergent statistical buckets that will suggest useful
indicators for entries**". That is a descriptive step, not an optimisation, and
it is the cheapest evidence in the programme: **one** ungated run, then the
per-trade P&L sliced by every indicator's own deciles.

WHY IT COMES BEFORE THE SEARCH
------------------------------
A trial costs ~16 seconds. Searching twenty indicator columns for a gate that
was never going to help is the expensive way to learn that. A scan of the same
twenty costs one run and consumes no trials, so it narrows the search space
without spending any of the multiple-testing budget the DSR has to price.

WHAT IT IS NOT
--------------
**It is not a significance test, and the t-statistics here are not evidence.**
Twenty columns times ten buckets is two hundred comparisons on one dataset, and
the best of them will look impressive by construction. The purpose is ORDERING —
which columns are worth putting in the search space — and a column that shows
nothing here almost certainly has nothing for a gate to find either.

Anything the scan suggests still has to win a trial, in a study whose trial
count includes it.

BUCKETING
---------
Numeric columns are cut into deciles of their own distribution, so every bucket
has the same number of trades and the comparison is about P&L rather than about
which bucket happened to be crowded. Categorical columns are cut by value.

**Monotonicity is reported.** A column whose P&L rises steadily across its
deciles is a far better gate candidate than one whose middle bucket happens to
be highest: the first is a relationship, the second is a coincidence with ten
chances to happen.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import polars as pl

from . import indicators as I
from . import regimes as R

#: Columns whose values are labels, not magnitudes. The third-order lags of a
#: categorical are still categorical — `prev_zone` is a zone, not a magnitude.
CATEGORICAL = set(I.CATEGORICAL_VIEW) | {f"prev_{c}" for c in I.CATEGORICAL_VIEW}


def attach_indicators(trades: pl.DataFrame, timeframe: int,
                      cfg: I.BandConfig = I.BandConfig(),
                      columns: Optional[Sequence[str]] = None,
                      third_order: bool = False) -> pl.DataFrame:
    """Join every indicator value at the signal's own bar onto each trade.

    Joined on (`date`, `bar_minute`) and not on `entry_minute`: the liveness
    gate can push a fill a minute or two later, and joining on the fill minute
    would silently drop every trade whose book opened late.

    `third_order` adds the specification's third order — the same metrics one bar back,
    as `prev_*`. It doubles the column count and therefore the number of
    comparisons, so it is off by default and reported as its own block."""
    cols = list(columns or (I.CATEGORICAL_VIEW + I.CONTINUOUS_VIEW))
    if third_order and columns is None:
        cols += [f"prev_{c}" for c in cols if c != "prev_range"]
    feats = (I.features(timeframe, cfg, third_order)
               .with_columns(((pl.col("timestamp").dt.hour().cast(pl.Int32) - 9) * 60
                              + pl.col("timestamp").dt.minute().cast(pl.Int32)
                              - 30).alias("bar_minute"))
               .select(["date", "bar_minute", *[c for c in cols
                                                if c in I.features(timeframe, cfg,
                                                                   third_order).columns]]))
    return trades.join(feats, on=["date", "bar_minute"], how="left")


def attach_asof(trades: pl.DataFrame, timeframe: int,
                cfg: I.BandConfig = I.BandConfig(),
                columns: Optional[Sequence[str]] = None,
                third_order: bool = False,
                suffix: str = "") -> pl.DataFrame:
    """Join the most recent CLOSED indicator bar at or before each entry.

    `attach_indicators` joins on an exact (`date`, `bar_minute`) match, which
    only works when the indicator timeframe IS the entry grid. **An indicator
    study needs them separated** (rule 6): entries stay on one clock while the
    metric set is read at 15m, 30m and 60m, and a 10:30 entry has no 60m bar of
    its own.

    Backward as-of on the timestamp, and lookahead-free for the same reason the
    exact join is: a metric row stamped T is computed from bars CLOSED before T
    and compared against bar T's OPEN, so it is known at T. Reading the 10:00
    row at 10:30 uses information that was 30 minutes old, which is the honest
    semantics of a slower chart, not a peek.

    `suffix` renames the joined columns, so several timeframes can sit on one
    frame at once."""
    cols = list(columns or (I.CATEGORICAL_VIEW + I.CONTINUOUS_VIEW))
    if third_order and columns is None:
        cols += [f"prev_{c}" for c in cols if c != "prev_range"]
    feats = I.features(timeframe, cfg, third_order)
    cols = [c for c in cols if c in feats.columns]
    feats = feats.select(["timestamp", *cols]).sort("timestamp")
    if suffix:
        feats = feats.rename({c: f"{c}{suffix}" for c in cols})
    # The feature frame's timestamps are zone-aware ET. Build the entry stamp in
    # the SAME dtype and zone: 09:30 ET plus `bar_minute`, localised rather than
    # converted, so the wall clock is preserved across a DST boundary.
    ts = feats.schema["timestamp"]
    stamped = trades.with_columns(
        (pl.col("date").cast(pl.Datetime(ts.time_unit))
         + pl.duration(minutes=pl.col("bar_minute").cast(pl.Int64) + 570))
        .dt.replace_time_zone(ts.time_zone).alias("_entry_ts")
    ).sort("_entry_ts")
    return (stamped.join_asof(feats, left_on="_entry_ts", right_on="timestamp",
                              strategy="backward")
                   .drop("_entry_ts"))


def triggers(trades: pl.DataFrame, columns: Optional[Sequence[str]] = None,
             buckets: int = 10, min_days: int = 80,
             suffix: str = "") -> pl.DataFrame:
    """Each bucket read as a ONE-A-DAY TRIGGER, scored per day.

    `scan()` averages every bar's trade, which is the right statistic for a
    method that trades all day and the wrong one for a method that gets **at
    most one** entry per session. Here each bucket becomes a rule --
    *"open on the FIRST bar of the day whose value falls in this bucket, and if
    no bar does, do not trade today"* -- and is scored on its per-DAY series.

    That is a conditional trigger with two properties the per-trade view cannot
    express: it can only be right once a day, and it may not fire at all. `days`
    is therefore a result, not a sample size to be maximised."""
    cols = [c for c in (columns or (I.CATEGORICAL_VIEW + I.CONTINUOUS_VIEW))
            if f"{c}{suffix}" in trades.columns]
    n_cal = trades["date"].n_unique()
    rows = []
    for c in cols:
        col = f"{c}{suffix}"
        x = trades[col].to_numpy().astype(float)
        b = _buckets(x, c, buckets)
        if b is None:
            continue
        t = trades.with_columns(pl.Series("_b", b))
        for lab in np.unique(b[np.isfinite(b)]):
            sel = (t.filter(pl.col("_b") == lab).sort("bar_minute")
                    .group_by("date", maintain_order=True).first())
            if sel.height < min_days:
                continue
            p = sel["pnl"].to_numpy()
            sd = p.std(ddof=1) if p.size > 1 else float("nan")
            rows.append(dict(
                column=c, bucket=float(lab), days=int(p.size),
                fire_rate=float(p.size / n_cal),
                per_day=float(p.mean()),
                per_cal_day=float(p.sum() / n_cal),
                total=float(p.sum()), win_rate=float((p > 0).mean()),
                sharpe=float(p.mean() / sd * np.sqrt(252.0)) if sd else float("nan"),
                t_stat=float(p.mean() / (sd / np.sqrt(p.size))) if sd else float("nan"),
                median_minute=float(np.median(sel["bar_minute"].to_numpy())),
                lo=float(np.nanmin(x[b == lab])), hi=float(np.nanmax(x[b == lab]))))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


@dataclass(frozen=True, slots=True)
class Cut:
    column: str
    bucket: str
    trades: int
    per_trade: float
    total: float
    win_rate: float


def _buckets(x: np.ndarray, column: str, n: int) -> Optional[np.ndarray]:
    """Bucket labels per row, or None if the column cannot be cut."""
    if column in CATEGORICAL:
        return x.astype(float)
    ok = np.isfinite(x)
    if ok.sum() < n * 10:
        return None
    edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, n + 1)))
    if edges.size < 3:
        return None
    out = np.full(x.size, np.nan)
    out[ok] = np.clip(np.searchsorted(edges, x[ok], side="right") - 1,
                      0, edges.size - 2)
    return out


def scan(trades: pl.DataFrame, columns: Optional[Sequence[str]] = None,
         buckets: int = 10, min_trades: int = 50,
         value: str = "pnl") -> pl.DataFrame:
    """Per-trade OUTCOME sliced by each indicator's own buckets.

    `value` names the outcome column and defaults to P&L. **It is not always
    P&L**: where a method's entry is trying to bring about a STATE rather than
    to profit on its own, the state is what the buckets have to be scored on.
    The case in point is a paired method whose FIRST spread exists to give the
    opposite spread a chance to be sold: its outcome column is a 0/1 flag for
    whether spot ever travelled outside the short strike, and `per_trade` reads
    as that rate. Scoring such an entry on its own P&L measures something the
    method is not trying to do (METHODOLOGY §3).

    Returns one row per (column, bucket) plus, per column, the spread between
    its best and worst bucket and how monotone the response is."""
    cols = [c for c in (columns or (I.CATEGORICAL_VIEW + I.CONTINUOUS_VIEW))
            if c in trades.columns]
    pnl = trades[value].to_numpy().astype(float)
    rows = []
    for c in cols:
        x = trades[c].to_numpy().astype(float)
        b = _buckets(x, c, buckets)
        if b is None:
            continue
        for lab in np.unique(b[np.isfinite(b)]):
            m = b == lab
            if m.sum() < min_trades:
                continue
            p = pnl[m]
            rows.append(dict(column=c, bucket=float(lab), trades=int(m.sum()),
                             per_trade=float(p.mean()), total=float(p.sum()),
                             win_rate=float((p > 0).mean()),
                             lo=float(np.nanmin(x[m])) if not np.isnan(x[m]).all() else np.nan,
                             hi=float(np.nanmax(x[m])) if not np.isnan(x[m]).all() else np.nan))
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def rank(cut: pl.DataFrame, min_buckets: int = 3) -> pl.DataFrame:
    """Order columns by how much their buckets separate, with monotonicity.

    `spread` is best minus worst bucket per trade. `monotone` is Spearman's rho
    between bucket order and per-trade P&L, in [-1, 1]: **a column whose P&L
    rises steadily across its deciles is a far better gate candidate than one
    whose middle bucket happens to be highest**, because the first is a
    relationship and the second is a coincidence with ten chances to happen.
    Categorical columns have no order, so their rho is reported as NaN.

    **`monotone_p` is reported because rho alone invites over-reading.** With
    ten buckets a pure-noise column has median |rho| **0.21**, a p90 of **0.52**
    and reaches **0.73** — measured over 60 synthetic draws of 2,000 trades with
    no relationship at all. A rho of 0.6 therefore means very little on its own,
    and this is not a hypothetical: the first real scan run here ranked `s_pctb` at
    rho = -0.60, which the p-value puts at **0.067** — not distinguishable from
    noise, and exactly the reading that would otherwise have gone into the
    search space as a finding.

    It is still not a significance test of the strategy: twenty columns times
    ten buckets is two hundred comparisons on one dataset. The p-value guards
    against reading a rank as a relationship, nothing more."""
    from scipy import stats
    out = []
    for (col,), g in cut.group_by(["column"], maintain_order=True):
        if g.height < min_buckets:
            continue
        g = g.sort("bucket")
        p = g["per_trade"].to_numpy()
        rho = pval = float("nan")
        if col not in CATEGORICAL and g.height > 2:
            res = stats.spearmanr(g["bucket"].to_numpy(), p)
            rho, pval = float(res.statistic), float(res.pvalue)
        best = g.filter(pl.col("per_trade") == p.max()).row(0, named=True)
        worst = g.filter(pl.col("per_trade") == p.min()).row(0, named=True)
        out.append(dict(column=col, buckets=g.height,
                        spread=float(p.max() - p.min()), monotone=rho,
                        monotone_p=pval,
                        best_bucket=best["bucket"], best_per_trade=best["per_trade"],
                        best_n=best["trades"], best_lo=best["lo"], best_hi=best["hi"],
                        worst_bucket=worst["bucket"],
                        worst_per_trade=worst["per_trade"],
                        overall=float((g["total"].sum() / g["trades"].sum()))))
    return (pl.DataFrame(out).sort("spread", descending=True)
            if out else pl.DataFrame())


def by_hour(trades: pl.DataFrame) -> pl.DataFrame:
    """P&L per entry hour — the specification's bucketing, for free from one run."""
    return (trades.group_by("entry_hour")
                  .agg(pl.len().alias("trades"),
                       pl.col("pnl").mean().round(2).alias("per_trade"),
                       pl.col("pnl").sum().round(0).alias("total"),
                       (pl.col("pnl") > 0).mean().round(3).alias("win_rate"))
                  .sort("entry_hour"))


def by_regime(trades: pl.DataFrame) -> pl.DataFrame:
    """The same slice by regime tier, so a conditional signal that lives only
    in the extreme days is visible as such before it is searched for."""
    return R.split(trades)
