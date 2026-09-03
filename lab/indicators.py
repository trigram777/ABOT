#!/usr/bin/env python3
"""
indicators.py — the dual Bollinger metric set, computed once for the whole decade.

WHAT IT IS
----------
Two Bollinger band families on one continuous SPX bar series: a FAST one (10
periods by default) and a SLOW one (20), each giving a low, mid and high line —
`FL/FM/FH` and `SL/SM/SH`. Everything in the specification's first, second and
third order metric lists is derived from those six lines and the bar's own open.

The output is one wide frame, one row per bar, for the whole 2017-2026 series.
It is computed ONCE per (timeframe, band config) and cached, so a sweep that
holds the band configuration fixed and varies the trading rule pays for the
indicators exactly once rather than once per session.

THE LOOKAHEAD RULE, AND WHAT IT MEANS HERE
------------------------------------------
The specification: *"Bollinger metrics must only use the opening Bollinger print per
bar, not where it drifted to by the bar's close."*

Implemented as: **every band at bar `t` is computed from bars `t-n .. t-1`** —
fully closed bars only — and the metrics then compare bar `t`'s OPEN against
those bands. A band therefore has exactly one value per bar and cannot drift
within it, which is the property the rule is asking for.

The alternative reading — that the band should include the current bar with its
open standing in for the running price, the way a live chart would paint it —
is also lookahead-safe, and differs only in whether the newest price
participates in its own band. It is not what is implemented. If you want it,
set `source="open"`: the band is then built from opens alone, so the two
readings differ by one bar of lag and nothing else.

Everything else obeys the same discipline. `prev_range`, `prev_green` and
`green_red_avg` read bar `t-1` and earlier. The bar's OPEN is the only value
from bar `t` that any metric touches, because it is the only one known when a
decision at `t` is taken.

CONTINUOUS ACROSS SESSIONS
--------------------------
A 20-period band at 60m needs three sessions of history and cannot exist inside
one day, so the series is unbroken and sessions are a column rather than a
boundary. Overnight gaps enter the standard deviation, which is real
information for a 0DTE opening trade. `session_bar` counts bars since the
session's open so a rule can still gate on "first bar of the day".

Note the 60m series opens each day with a 30-minute stub (09:30-10:00) before
the hourly grid resumes — an artefact of RTH starting on the half hour, carried
in `source_minutes` rather than smoothed away.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date as Date
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .session import bars

# --------------------------------------------------------------------- zones

#: Bar-open position relative to the six band lines. Deliberately DISCRETE and
#: not %b: the specification asks where the bar opened, in named regions, and a region
#: is what an indicator toggle can switch on.
BL, UL, L, M, H, UH, BH = range(7)
ZONE_NAMES = {BL: "BL", UL: "UL", L: "L", M: "M", H: "H", UH: "UH", BH: "BH"}

#: Band-family slope character.
FLAT, EXPANSION, CONTRACTION, TREND_UP, TREND_DOWN = range(5)
SLOPE_NAMES = {FLAT: "FLAT", EXPANSION: "EXPANSION", CONTRACTION: "CONTRACTION",
               TREND_UP: "TREND_UP", TREND_DOWN: "TREND_DOWN"}
"""CONTRACTION is an ADDITION to the specification's three (Flat, Expansion, Trend).
Diverging bands are named there and converging ones are not, which leaves a
squeeze — the most-watched Bollinger regime there is — with nowhere to go but
into FLAT, where it would be indistinguishable from a genuinely quiet band.
Flagged rather than folded in silently."""

#: How the fast family sits relative to the slow one.
OTHER, F_INSIDE_S, S_INSIDE_F, F_ABOVE_S, F_BELOW_S = range(5)
RELATION_NAMES = {OTHER: "OTHER", F_INSIDE_S: "F_INSIDE_S", S_INSIDE_F: "S_INSIDE_F",
                  F_ABOVE_S: "F_ABOVE_S", F_BELOW_S: "F_BELOW_S"}


@dataclass(frozen=True, slots=True)
class BandConfig:
    """Everything about how the bands are drawn. Hashable, so it keys the cache."""

    fast: int = 10
    slow: int = 20
    k: float = 2.0
    """Standard deviations to the outer lines."""

    source: str = "open"
    """**FIXED at `open` by decision, 23 Aug. Not a sweep axis.**

    The specification left it open and `search.Space` used to vary it over
    close/open/hlc3. It no longer does, for two reasons — neither of them the
    obvious one.

    **The obvious one is wrong and is withdrawn**: "a close-source band measures
    a series the decision rule never sees" is false. The band reads only CLOSED
    bars (`.shift(1)`), and those bars' closes are perfectly visible by the time
    the rule fires. There is no lookahead in a close-source band.

    **1. Faithful reconstruction.** The trader runs OPEN + SMA manually and every
    hypothesis in the specification was formed while looking at those charts. An
    open-source band is what the hypotheses were about; testing them on a
    close-source band would be testing a different hypothesis.

    **2. Units.** `%b` and `zone` place the CURRENT bar's open inside a
    distribution. If that distribution is built from closes, every inter-bar gap
    lands in the numerator and in nothing else, so the metric drifts with
    overnight and inter-bar gaps rather than with position in the band.

    Building the band from the same quantity the rule is compared against makes
    `%b`, `zone` and every gap mean one thing rather than two.

    `close` and `hlc3` remain expressible so that results computed before this
    decision stay reproducible — early work here used `close`. Nothing should
    search them.

    Note the band still excludes the current bar via `.shift(1)` — see
    `_family`. With an OPEN source that is a deliberate choice and not a
    leftover: including bar `t`'s own open would let the price being measured
    pull the band toward itself, and `s_pctb` would then be partly a statement
    about its own denominator."""

    ma: str = "sma"
    """`sma` or `ema`. **SMA is the standing choice**; EMA stays a sweep axis but
    only where the question is whether a faster
    response to a turn is worth the lag on every other bar. With `ema`, the
    dispersion is still the rolling standard deviation over the same window: an
    EMA of squared deviations would change two things at once and make the axis
    unreadable."""

    slope_lookback: int = 1
    """Bars back the slope is measured over, on the closed-bar band series."""

    flat_eps: float = 0.02
    """A band line moving less than this FRACTION OF ITS OWN BANDWIDTH over the
    lookback counts as flat. Scale-free on purpose: SPX ran 2,250 in 2017 and
    7,500 in 2026, and an absolute threshold in points would mean something
    different at each end of the sample."""

    green_red_period: int = 10
    """Bars in the green/red average. One method's entry signal; a sweep axis."""

    def validate(self) -> "BandConfig":
        if self.source not in ("close", "open", "hlc3"):
            raise ValueError(f"unknown source {self.source!r}")
        if self.ma not in ("sma", "ema"):
            raise ValueError(f"unknown ma {self.ma!r}")
        if self.fast >= self.slow:
            raise ValueError(f"fast ({self.fast}) must be shorter than slow ({self.slow})")
        return self


# ------------------------------------------------------------------ building

def _source(cfg: BandConfig) -> pl.Expr:
    if cfg.source == "hlc3":
        return (pl.col("high") + pl.col("low") + pl.col("close")) / 3.0
    return pl.col(cfg.source)


def _average(src: pl.Expr, n: int, kind: str) -> pl.Expr:
    if kind == "sma":
        return src.rolling_mean(window_size=n, min_samples=n)
    return src.ewm_mean(span=n, adjust=False, min_samples=n)


def _family(cfg: BandConfig, n: int, tag: str) -> List[pl.Expr]:
    """One band family's three lines, already shifted off the current bar.

    The `.shift(1)` is the whole lookahead rule in one call: the value carried
    on row `t` is the band as it stood when bar `t` opened, computed from bars
    that had already closed."""
    src = _source(cfg)
    mid = _average(src, n, cfg.ma)
    # Population sigma (ddof=0) — the Bollinger convention, and the one every
    # charting package draws.
    sd = src.rolling_std(window_size=n, min_samples=n, ddof=0)
    return [mid.shift(1).alias(f"{tag}M"),
            (mid - cfg.k * sd).shift(1).alias(f"{tag}L"),
            (mid + cfg.k * sd).shift(1).alias(f"{tag}H")]


def _slope_type(low: np.ndarray, high: np.ndarray, lb: int,
                eps: float) -> np.ndarray:
    """FLAT / EXPANSION / CONTRACTION / TREND_UP / TREND_DOWN per bar."""
    width = high - low
    dl = np.full(low.size, np.nan)
    dh = np.full(high.size, np.nan)
    dl[lb:] = low[lb:] - low[:-lb]
    dh[lb:] = high[lb:] - high[:-lb]
    with np.errstate(invalid="ignore", divide="ignore"):
        rl, rh = dl / width, dh / width
    out = np.full(low.size, -1, dtype=np.int8)
    ok = np.isfinite(rl) & np.isfinite(rh)
    flat_l, flat_h = np.abs(rl) < eps, np.abs(rh) < eps
    both_flat = flat_l & flat_h
    # A widening band whose low line is merely flat is still an expansion, so
    # direction is taken from whichever line is actually moving.
    diverge = (~both_flat) & (rh >= 0) & (rl <= 0)
    converge = (~both_flat) & (rh <= 0) & (rl >= 0)
    up = (~both_flat) & (rh > 0) & (rl > 0)
    down = (~both_flat) & (rh < 0) & (rl < 0)
    out[ok & both_flat] = FLAT
    out[ok & diverge] = EXPANSION
    out[ok & converge] = CONTRACTION
    out[ok & up] = TREND_UP
    out[ok & down] = TREND_DOWN
    return out


def _zone(price: np.ndarray, sl, sm, sh, fl, fm, fh) -> np.ndarray:
    """Which of the seven named regions the bar opened in.

    Counted rather than compared pairwise, so it stays well defined when the
    two families overlap oddly — a fast low line can sit above a slow mid in a
    sharp move, and a nest of `if` statements would fall through to nothing."""
    n_low = (price > sl).astype(np.int8) + (price > fl).astype(np.int8)
    n_mid = (price > sm).astype(np.int8) + (price > fm).astype(np.int8)
    n_high = (price > sh).astype(np.int8) + (price > fh).astype(np.int8)
    out = np.full(price.size, -1, dtype=np.int8)
    ok = np.isfinite(sl) & np.isfinite(fl) & np.isfinite(price)
    out[ok] = np.select(
        [n_low[ok] == 0, n_low[ok] == 1, n_mid[ok] == 0, n_mid[ok] == 1,
         n_high[ok] == 0, n_high[ok] == 1],
        [BL, UL, L, M, H, UH], default=BH)
    return out


def _relation(sl, sh, fl, fh) -> np.ndarray:
    out = np.full(sl.size, -1, dtype=np.int8)
    ok = np.isfinite(sl) & np.isfinite(fl)
    inside = (fl >= sl) & (fh <= sh)
    outside = (sl >= fl) & (sh <= fh)
    above = (fl > sl) & (fh > sh)
    below = (fl < sl) & (fh < sh)
    out[ok] = OTHER
    # Containment is checked first: a nested band is also trivially "above" on
    # one line, and calling it F_ABOVE_S would lose the fact that it is nested.
    out[ok & below] = F_BELOW_S
    out[ok & above] = F_ABOVE_S
    out[ok & outside] = S_INSIDE_F
    out[ok & inside] = F_INSIDE_S
    return out


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """+1 where `a` crossed above `b` this bar, -1 below, 0 otherwise."""
    d = np.sign(a - b)
    prev = np.full(d.size, np.nan)
    prev[1:] = d[:-1]
    out = np.zeros(d.size, dtype=np.int8)
    ok = np.isfinite(d) & np.isfinite(prev)
    out[ok & (prev <= 0) & (d > 0)] = 1
    out[ok & (prev >= 0) & (d < 0)] = -1
    return out


def build(timeframe: int, cfg: BandConfig = BandConfig()) -> pl.DataFrame:
    """The whole SPX feature matrix for one timeframe and band configuration."""
    return metrics(bars(timeframe), cfg)


def metrics(df: pl.DataFrame, cfg: BandConfig = BandConfig()) -> pl.DataFrame:
    """Every metric, over any OHLC bar frame.

    Shared by the SPX chart and the option's own chart on purpose: the specification
    asks for both, and for mixing them on one order — "price action on the SPX
    triggers an entry, but the exit is based on Bollingers from the option's
    price action". Two implementations of `zone` or `%b` would let the two
    vocabularies drift, and a gate named `s_pctb` would then mean different
    things depending on which chart it was pointed at."""
    cfg = cfg.validate()

    df = df.with_columns(
        *_family(cfg, cfg.slow, "S"),
        *_family(cfg, cfg.fast, "F"),
        # ---- first order, all from CLOSED bars
        (pl.col("close") - pl.col("open")).shift(1).alias("prev_range"),
        # The SIGNED body of the previous bar and its MAGNITUDE are different
        # questions and the sign is not always the interesting half. Scored
        # against a breach objective, `prev_range` traces a symmetric U —
        # both tails +10 to +15pp, the middle −12pp — which Spearman reads as
        # rho ≈ −0.15 and dismisses. `abs_prev_range` is the same information
        # ordered correctly: monotone across all ten deciles, −12pp to +14pp
        # Kept as its own column because a search space cannot apply
        # `abs()` to a threshold on its own.
        (pl.col("close") - pl.col("open")).abs().shift(1).alias("abs_prev_range"),
        ((pl.col("close") - pl.col("open")).sign()).shift(1).alias("prev_green"),
        ((pl.col("close") - pl.col("open")).sign()
            .rolling_mean(window_size=cfg.green_red_period,
                          min_samples=cfg.green_red_period)
            .shift(1)).alias("green_red_avg"),
        # int_range and not a cum_count over `timestamp`: an option's own chart
        # is indexed by minute and carries no timestamp column, and `metrics`
        # has to work identically on both frames or the two vocabularies drift.
        pl.int_range(pl.len()).over("date").alias("session_bar"),
    )

    a = {c: df[c].to_numpy().astype(float) for c in
         ("open", "SM", "SL", "SH", "FM", "FL", "FH")}
    op = a["open"]

    s_width, f_width = a["SH"] - a["SL"], a["FH"] - a["FL"]
    with np.errstate(invalid="ignore", divide="ignore"):
        s_bw, f_bw = s_width / a["SM"], f_width / a["FM"]
        # Centred %b: 0 at the mid, +/-1 at the bands. Identical to
        # 2*(%b - 0.5) for symmetric bands, and defined when width is zero.
        s_pb = (op - a["SM"]) / (s_width / 2.0)
        f_pb = (op - a["FM"]) / (f_width / 2.0)
        gap_l = (a["SL"] - a["FL"]) / a["SM"]
        gap_m = (a["SM"] - a["FM"]) / a["SM"]
        gap_h = (a["SH"] - a["FH"]) / a["SM"]
        pctb_spread = f_pb - s_pb

    lb, eps = cfg.slope_lookback, cfg.flat_eps
    out = df.with_columns(
        pl.Series("zone", _zone(op, a["SL"], a["SM"], a["SH"],
                                a["FL"], a["FM"], a["FH"])),
        pl.Series("s_slope", _slope_type(a["SL"], a["SH"], lb, eps)),
        pl.Series("f_slope", _slope_type(a["FL"], a["FH"], lb, eps)),
        pl.Series("relation", _relation(a["SL"], a["SH"], a["FL"], a["FH"])),
        pl.Series("s_bandwidth", s_bw),
        pl.Series("f_bandwidth", f_bw),
        pl.Series("bandwidth_ratio", np.divide(f_bw, s_bw,
                                               out=np.full_like(f_bw, np.nan),
                                               where=s_bw != 0)),
        pl.Series("s_pctb", s_pb),
        pl.Series("f_pctb", f_pb),
        pl.Series("pctb_spread", pctb_spread),
        pl.Series("gap_low", gap_l),
        pl.Series("gap_mid", gap_m),
        pl.Series("gap_high", gap_h),
        pl.Series("cross_low", _cross(a["SL"], a["FL"])),
        pl.Series("cross_mid", _cross(a["SM"], a["FM"])),
        pl.Series("cross_high", _cross(a["SH"], a["FH"])),
    )
    # The slope PAIR is a second-order metric in its own right: "S slope + F
    # slope" is one categorical, not two, because the interesting states are
    # combinations (slow trending while fast contracts, and so on).
    out = out.with_columns(
        (pl.col("s_slope").cast(pl.Int16) * 5
         + pl.col("f_slope").cast(pl.Int16)).alias("slope_pair"))
    return out


# ------------------------------------------------------ measured redundancy
#
# Two of these metrics are EXACT functions of others. Verified on 31,346 bars
# at 30m, reconstructing at 100.000000%:
#
#   zone     = f(s_pctb, f_pctb)          discretised at the cut points -1/0/+1
#   relation = f(sign(gap_low), sign(gap_high))     gap_mid is not involved
#
# `s_pctb > 1` IS `price > SH` by construction, so the six band comparisons the
# zone counts are already carried by the two centred %b values; and the gaps are
# signed, so their signs already say which family sits inside which.
#
# The reverse does NOT hold, and the difference matters: within zone BL the
# |s_pctb| actually observed runs from 1.00 to 7.11 (p50 1.39, p99 3.81). A bar
# seven half-widths below the mid and one barely below SL are the same zone, and
# that discarded magnitude is exactly the range these methods trade in.
#
# So the categoricals carry no INFORMATION the continuous columns lack. What
# they carry is SEARCH EFFICIENCY: `zone in {BL, UL}` is one ordered-band
# parameter, where the same condition through %b needs an optimiser to discover
# two continuous thresholds landing on exactly -1. They are a prior on where the
# interesting cuts are, and a good one, because those cut points are the ones
# the trade is actually described in.
#
# Both views are kept, and `search.Space.metric_view` makes choosing between
# them a controlled experiment rather than an accident.

#: Exact functional dependencies, for anything that needs to avoid double-counting.
DERIVED_FROM = {"zone": ("s_pctb", "f_pctb"),
                "relation": ("gap_low", "gap_high"),
                "abs_prev_range": ("prev_range",),
                "prev_green": ("prev_range",)}

#: The coarse, trader-vocabulary view.
CATEGORICAL_VIEW = ("zone", "relation", "s_slope", "f_slope", "slope_pair",
                    "cross_low", "cross_mid", "cross_high", "prev_green")

#: The fine, unbounded view. Strictly richer, and harder to search.
CONTINUOUS_VIEW = ("s_pctb", "f_pctb", "pctb_spread", "s_bandwidth",
                   "f_bandwidth", "bandwidth_ratio", "gap_low", "gap_mid",
                   "gap_high", "prev_range", "abs_prev_range",
                   "green_red_avg")


#: Every metric column `build` produces, in the specification's own ordering.
FIRST_ORDER = ["prev_range", "abs_prev_range", "prev_green", "zone",
               "s_slope", "f_slope",
               "s_bandwidth", "f_bandwidth", "s_pctb", "f_pctb"]
SECOND_ORDER = ["green_red_avg", "relation", "slope_pair", "cross_low",
                "cross_mid", "cross_high", "bandwidth_ratio", "gap_low",
                "gap_mid", "gap_high", "pctb_spread"]
METRICS = FIRST_ORDER + SECOND_ORDER

BANDS = ["SL", "SM", "SH", "FL", "FM", "FH"]


def with_third_order(df: pl.DataFrame, columns: Optional[List[str]] = None
                     ) -> pl.DataFrame:
    """The specification's third order: the same metrics one bar back.

    A plain lag of the whole metric set, and therefore lookahead-safe by
    construction. Lagged ACROSS the session boundary, like everything else
    here — the last bar of yesterday is a bar, and at 30m and 60m (the only
    timeframes the specification asks this of) refusing to look at it would blank the
    metric for the first bar of every day, which is the bar most rules trade."""
    cols = METRICS if columns is None else columns
    return df.with_columns([pl.col(c).shift(1).alias(f"prev_{c}") for c in cols])


@lru_cache(maxsize=16)
def features(timeframe: int, cfg: BandConfig = BandConfig(),
             third_order: bool = False) -> pl.DataFrame:
    """Cached feature matrix. Built once per (timeframe, config) per process."""
    df = build(timeframe, cfg)
    return with_third_order(df) if third_order else df


# ------------------------------------------------------------------- access

def for_session(day: Date, timeframe: int, cfg: BandConfig = BandConfig(),
                third_order: bool = False) -> pl.DataFrame:
    """One session's bars, with a `minute` column indexing the option session.

    `minute` is minutes since 09:30, which is exactly the row index into a
    `Session`'s arrays — so a rule that fires on a 15m bar knows without any
    further lookup which option-chain minute it is allowed to trade at."""
    df = features(timeframe, cfg, third_order).filter(pl.col("date") == day)
    # Cast BEFORE the arithmetic: `dt.hour()` is Int8, so `(hour - 9) * 60`
    # overflows for every bar from 12:00 on and the index silently goes
    # negative — 12:00 came out as minute -106.
    return df.with_columns(
        ((pl.col("timestamp").dt.hour().cast(pl.Int32) - 9) * 60
         + pl.col("timestamp").dt.minute().cast(pl.Int32) - 30).alias("minute"))


def describe(row: dict) -> str:
    """One bar's state, in the specification's vocabulary. For eyeballing, not sweeping."""
    return (f"{ZONE_NAMES.get(row['zone'], '?'):>2}  "
            f"S:{SLOPE_NAMES.get(row['s_slope'], '?'):<11} "
            f"F:{SLOPE_NAMES.get(row['f_slope'], '?'):<11} "
            f"{RELATION_NAMES.get(row['relation'], '?'):<10} "
            f"bw {row['s_bandwidth']:.4f}/{row['f_bandwidth']:.4f}  "
            f"%b {row['s_pctb']:+.2f}/{row['f_pctb']:+.2f}  "
            f"g/r {row['green_red_avg']:+.2f}")


# ------------------------------------------------------- the option's own chart

OPTION_SOURCES = ("mid", "trade")


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """How to read an option's own chart. Frozen, so it can key a cache."""

    timeframe: int = 5
    source: str = "mid"
    """`mid` — the quote midpoint, present every minute the book is two-sided.
    `trade` — the real print OHLC, higher fidelity and gappy at far strikes."""

    bands: BandConfig = field(default_factory=BandConfig)

    def validate(self) -> "ChartSpec":
        if self.source not in OPTION_SOURCES:
            raise ValueError(f"unknown option source {self.source!r}")
        if self.timeframe not in (1, 5, 15):
            raise ValueError(
                f"option charts are session-bounded, so {self.timeframe}m is "
                "not usable: a 0DTE contract exists for one day, and a "
                "20-period band at 30m or 60m would need more history than the "
                "contract has. 1m and 5m are the practical timeframes.")
        self.bands.validate()
        return self


def option_bars(sess, contract, spec: ChartSpec = ChartSpec()) -> pl.DataFrame:
    """One contract's own OHLC chart within one session.

    **Session-bounded, and it cannot be otherwise.** A 0DTE option exists for a
    single day, and yesterday's 4750C is a different contract with a different
    strike relative to spot — so unlike the SPX series there is no continuous
    history to warm a band from. A 20-period slow band is ready 20 bars into the
    session: 09:51 at 1m, 11:10 at 5m. That is late for an opening trade and
    fine for power hour, which is where these timeframes are wanted.

    Bars are aligned to the session open, so bar `k` covers minutes
    `[k*tf, (k+1)*tf)` and its opening minute is `k*tf` — the same convention
    the derived SPX bars use, so a spot gate and an option gate on the same
    timeframe refer to the same instants."""
    spec = spec.validate()
    n = sess.n_minutes
    col = sess.column(contract.right, contract.strike)
    if col is None:
        return pl.DataFrame(schema={"date": pl.Date, "minute": pl.Int32,
                                    "open": pl.Float64, "high": pl.Float64,
                                    "low": pl.Float64, "close": pl.Float64})
    if spec.source == "mid":
        bid = sess.arrays[(contract.right, "bid")][:, col]
        ask = sess.arrays[(contract.right, "ask")][:, col]
        mid = (bid + ask) / 2.0
        o = h = l = c = mid
    else:
        a = sess.arrays
        o = a[(contract.right, "t_open")][:, col]
        h = a[(contract.right, "t_high")][:, col]
        l = a[(contract.right, "t_low")][:, col]
        c = a[(contract.right, "last")][:, col]

    tf = spec.timeframe
    if tf == 1:
        minutes = np.arange(n, dtype=np.int32)
        bo, bh, bl, bc = o, h, l, c
    else:
        k = int(np.ceil(n / tf))
        pad = k * tf - n
        def block(x, how):
            y = np.concatenate([x, np.full(pad, np.nan)]).reshape(k, tf)
            if how == "first":
                return _first_finite(y)
            if how == "last":
                return _first_finite(y[:, ::-1])
            # An entirely unquoted bar is a real thing at a far strike, and
            # nanmax warns rather than returning NaN quietly. Masked instead.
            empty = ~np.isfinite(y).any(axis=1)
            filled = np.where(np.isfinite(y), y, -np.inf if how == "max" else np.inf)
            out = filled.max(axis=1) if how == "max" else filled.min(axis=1)
            out[empty] = np.nan
            return out
        bo, bh, bl, bc = (block(o, "first"), block(h, "max"),
                          block(l, "min"), block(c, "last"))
        minutes = (np.arange(k) * tf).astype(np.int32)

    return pl.DataFrame({
        "date": pl.Series([sess.date] * len(minutes), dtype=pl.Date),
        "minute": minutes, "open": bo, "high": bh, "low": bl, "close": bc,
    }).filter(pl.col("open").is_not_nan() | pl.col("close").is_not_nan())


def _first_finite(y: np.ndarray) -> np.ndarray:
    """First finite value of each row, NaN if the row is empty. A bar whose
    first minutes are unquoted opens at its first real price, not at NaN."""
    ok = np.isfinite(y)
    idx = np.argmax(ok, axis=1)
    out = y[np.arange(y.shape[0]), idx]
    out[~ok.any(axis=1)] = np.nan
    return out


@lru_cache(maxsize=512)
def option_features(sess, contract, spec: ChartSpec = ChartSpec()) -> pl.DataFrame:
    """The full metric set computed on one contract's own chart.

    Same columns and same meanings as the SPX frame — `metrics` is shared — so
    a gate reads identically whichever chart it is pointed at.

    Cached on (session identity, contract, spec): a sweep runs many exit
    policies against the same session, and several of them will hold the same
    contract, so the bands would otherwise be rebuilt once per cell per trade."""
    b = option_bars(sess, contract, spec)
    if not b.height:
        return b
    return metrics(b, spec.bands)
