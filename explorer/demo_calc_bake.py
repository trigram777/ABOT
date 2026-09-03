#!/usr/bin/env python3
"""
Generate a SYNTHETIC option-path panel for the exit-policy explorer.

WHY SYNTHETIC, AND WHY THAT COSTS NOTHING HERE
----------------------------------------------
The explorer's job is to resolve exit policies -- stops, trails, price targets,
clocks, chart conditions and spread conversions -- over the price path of a
short-dated option. That needs paths SHAPED like option paths: decaying toward
expiry, right-skewed, occasionally exploding, with an intra-minute low that is
consistent with the midpoint and with neighbouring strikes correctly ordered.

It does not need real market data. The production panel is built from a
commercial tick feed whose licence does not permit redistribution, so this file
simulates the panel instead.

The parameters below are chosen BY HAND for plausibility and are **not
calibrated to any vendor dataset**. Nothing here is a statement about how any
market has behaved, and no equity curve the explorer draws over this data is
evidence of anything. It is a demonstration of software.

WHAT IS FAITHFUL ANYWAY
-----------------------
Three properties are reproduced deliberately, because the explorer's controls
are meaningless without them:

  * **Expiry decay.** Options are priced at each minute with the remaining time
    shrinking to zero, so a position held to the close converges on intrinsic
    value and a trailing exit has something real to trail.
  * **The tick grid.** Every price is snapped to the exchange increment
    ($0.05 below $3.00, $0.10 at or above). A level finer than the grid is not
    a level, and the packer stores cents as int16 on the assumption that prices
    are already snapped.
  * **Closed-bar indicators.** Every band at bar `t` is computed from bars
    `t-n .. t-1` and compared against bar `t`'s OPEN. A band that reads its own
    bar is the most common lookahead bug in this kind of tool, and reproducing
    the discipline here keeps the demo honest about what it is showing.

OUTPUT SCHEMA -- the contract `demo_calc_pack.py` reads
-------------------------------------------------------
One row per (session, slot, selector, right):

    date slot right sel target strike entry miss offset spot settle live
    k_short   one strike TOWARD the money -- selling it makes a credit spread
    k_cover   one strike AWAY             -- selling it makes a debit spread
    mid       the option's midpoint path, HORIZON minutes from entry
    low       its intra-minute low  -- what a STOP fires on, not the midpoint
    short     k_short's midpoint path
    cover     k_cover's midpoint path

Plus a second frame of index-chart indicators, per session per timeframe, at
BAR BOUNDARIES ONLY -- the chart is the same for every contract and does not
change inside a bar, so this is 78 values an hour rather than 240.

    python demo_calc_bake.py [--sessions 600] [--seed 20260903]
"""
from __future__ import annotations

import argparse
import datetime as dt
from typing import Dict, List

import numpy as np
import polars as pl
from scipy.special import ndtr

# ----------------------------------------------------------------- constants

MINUTES = 391                    # 09:30 .. 16:00 inclusive
EXPIRY = MINUTES - 1             # the settlement minute
HORIZON = 61                     # minutes of path carried past each entry
SLOTS = (30, 90, 150, 210, 270, 330)          # 10:00 11:00 12:00 13:00 14:00 15:00
SELECTORS = [("delta", 0.10), ("delta", 0.25), ("delta", 0.40), ("price", 1.00)]
TFS = (5, 15, 30)
COLUMNS = ["s_pctb", "f_pctb", "pctb_spread", "s_bandwidth", "f_bandwidth",
           "bandwidth_ratio", "gap_low", "gap_mid", "gap_high", "prev_range",
           "green_red_avg", "zone", "slope_pair"]

STRIKE_STEP = 5.0                # the grid strikes are listed on
TICK_BREAK = 3.00                # below this the increment is 0.05, at or above 0.10
PRICE_CAP = 320.0                # the packer stores cents as int16; stay inside it
CALL, PUT = "C", "P"

FAST, SLOW, KDEV = 10, 20, 2.0   # band parameters, matching the production set
GREEN_RED = 10

QUOTE_SD = 0.0045                # quote-noise amplitude, as a fraction of price
QUOTE_AR = 0.90                  # its persistence; see `_path`
_AR_SCALE = (1.0 - QUOTE_AR ** 2) ** 0.5
QUIET = 0.45                     # share of minutes whose low IS the midpoint

# Options are PRICED above the vol the path actually realises. That ordering is
# the single most important economic property to get right here: implied sits
# above realised in index options, which is why a long decays and a short is
# paid. Get it backwards -- as the first draft of this file did, by adding jump
# variance the pricing vol never saw -- and every long in the panel prints a
# profit, which is both wrong and instantly obvious in the verifier.
IMPLIED_PREMIUM = 1.18
JUMP_P = 0.04                    # share of sessions carrying an intraday jump


# ------------------------------------------------------------ Black-Scholes

def _bs(spot, strike, t_years, iv, right):
    """Price and delta. `spot` and `t_years` broadcast; t=0 gives intrinsic."""
    spot = np.asarray(spot, dtype=float)
    t = np.maximum(np.asarray(t_years, dtype=float), 0.0)
    sig = np.maximum(iv, 1e-6) * np.sqrt(np.maximum(t, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + 0.5 * sig ** 2) / sig
        d2 = d1 - sig
    if right == CALL:
        px = spot * ndtr(d1) - strike * ndtr(d2)
        dl = ndtr(d1)
        intrinsic = np.maximum(spot - strike, 0.0)
    else:
        px = strike * ndtr(-d2) - spot * ndtr(-d1)
        dl = ndtr(d1) - 1.0
        intrinsic = np.maximum(strike - spot, 0.0)
    dead = t <= 0
    px = np.where(dead, intrinsic, px)
    dl = np.where(dead, np.sign(intrinsic) * (1.0 if right == CALL else -1.0), dl)
    return px, dl


def _iv(strike, spot, base):
    """A hand-set smile. Out-of-the-money strikes carry more implied vol, which
    is what makes a delta selector and a price selector disagree at all."""
    m = (strike - spot) / spot
    return base * (1.0 + 6.0 * m * m + 0.9 * np.maximum(-m, 0.0))


def _snap(px):
    """The exchange tick grid. Always snap: a level the grid cannot express is
    not a level, and the packer assumes prices are already on it."""
    px = np.clip(np.asarray(px, dtype=float), 0.0, PRICE_CAP)
    fine = np.round(px / 0.05) * 0.05
    coarse = np.round(px / 0.10) * 0.10
    return np.where(px < TICK_BREAK, fine, coarse)


def _tte(minutes):
    """Year fraction remaining, on a 252-session year."""
    return np.maximum(EXPIRY - np.asarray(minutes, dtype=float), 0.0) / (390.0 * 252.0)


# --------------------------------------------------------------- the session

def _session(rng: np.random.Generator, level: float):
    """One session's index path, and the vol its options are PRICED at.

    Vol is drawn per session from a fat-tailed distribution and a small
    fraction of sessions get an intraday jump, so the panel contains the quiet
    days, the trending days and the violent days the explorer's controls are
    there to be tested against.

    Returns the priced vol, not the realised one -- see IMPLIED_PREMIUM.
    """
    real_iv = float(np.exp(rng.normal(np.log(0.16), 0.42)))
    real_iv = min(real_iv, 1.30)
    per_min = real_iv / np.sqrt(390.0 * 252.0)

    steps = rng.standard_normal(EXPIRY) * per_min
    if rng.random() < JUMP_P:                     # an intraday jump
        at = rng.integers(20, EXPIRY - 20)
        steps[at] += rng.choice([-1.0, 1.0]) * rng.uniform(2.0, 6.0) * per_min * 8
    # A mild session-long lean. It is spread over EXPIRY steps, so the size
    # here is per-STEP and has to be divided down or the lean dominates the
    # diffusion -- at `per_min` it is a ~7% daily drift against a ~1% day.
    drift = rng.normal(0.0, 0.30) * per_min / np.sqrt(EXPIRY)
    path = level * np.exp(np.cumsum(steps + drift))
    return np.concatenate([[level], path]), real_iv * IMPLIED_PREMIUM


def _strikes_around(spot):
    lo = np.floor((spot * 0.90) / STRIKE_STEP) * STRIKE_STEP
    hi = np.ceil((spot * 1.10) / STRIKE_STEP) * STRIKE_STEP
    return np.arange(lo, hi + STRIKE_STEP, STRIKE_STEP)


def _pick(kind, target, right, spot, t, base_iv):
    """The selector, as a NEAREST rule in both forms -- it can overshoot the
    target in either direction, which is what the production selectors do."""
    grid = _strikes_around(spot)
    px, dl = _bs(spot, grid, t, _iv(grid, spot, base_iv), right)
    px = _snap(px)
    if kind == "delta":
        err = np.abs(np.abs(dl) - target)
        ok = px > 0
    else:
        err = np.abs(px - target)
        ok = px > 0
    if not ok.any():
        return None
    err = np.where(ok, err, np.inf)
    i = int(np.argmin(err))
    miss = (abs(float(dl[i])) - target) if kind == "delta" else (float(px[i]) - target)
    return float(grid[i]), miss


def _path(strike, right, spot_path, m, base_iv, rng):
    """One contract's midpoint path over the horizon, and its intra-minute low.

    The low is not the midpoint. A stop fires on what traded inside the minute,
    and a series that cannot go below its own minute-open cannot test a stop --
    which is the single largest artefact this kind of tool can carry.
    """
    end = min(m + HORIZON, MINUTES)
    idx = np.arange(m, end)
    spot = spot_path[idx]
    iv = _iv(strike, spot, base_iv)
    px, _ = _bs(spot, strike, _tte(idx), iv, right)

    # Quote noise is PERSISTENT, not per-minute independent, and the reason is
    # not cosmetic. A real book sits still for minutes at a time; i.i.d. jitter
    # flips the snapped price every minute, which is both unlike a book and
    # maximally incompressible -- it destroys the long constant runs the packer
    # depends on. Measured while building this: i.i.d. noise at the same
    # amplitude cost 2.2x the packed size.
    e = np.empty(px.size)
    e[0] = rng.normal(0.0, QUOTE_SD)
    for i in range(1, px.size):
        e[i] = QUOTE_AR * e[i - 1] + rng.normal(0.0, QUOTE_SD) * _AR_SCALE
    mid = _snap(np.maximum(px * (1.0 + e), 0.0))

    # The intra-minute low IS genuinely jumpy -- an extreme is noisy by nature --
    # but it is bounded by the midpoint it belongs to.
    dip = np.abs(rng.normal(0.0, 0.045, mid.shape))
    dip = np.where(rng.random(mid.shape) < QUIET, 0.0, dip)   # nothing traded through
    low = _snap(np.maximum(mid * (1.0 - np.clip(dip, 0.0, 0.30)), 0.0))
    low = np.minimum(low, mid)

    out_m = np.full(HORIZON, np.nan)
    out_l = np.full(HORIZON, np.nan)
    out_m[: idx.size] = mid
    out_l[: idx.size] = low
    return out_m, out_l


def _day(rng, day: dt.date, spot_path: np.ndarray, base_iv: float) -> List[dict]:
    settle = float(spot_path[EXPIRY])
    rows: List[dict] = []

    for m in SLOTS:
        spot = float(spot_path[m])
        t = float(_tte(m))
        for kind, target in SELECTORS:
            for right in (CALL, PUT):
                got = _pick(kind, target, right, spot, t, base_iv)
                if got is None:
                    continue
                strike, miss = got
                mid, low = _path(strike, right, spot_path, m, base_iv, rng)
                entry = float(mid[0])
                if not (entry > 0):
                    continue

                toward = -STRIKE_STEP if right == CALL else STRIKE_STEP
                k_short = strike + toward
                k_cover = strike - toward
                s_mid, _ = _path(k_short, right, spot_path, m, base_iv, rng)
                c_mid, _ = _path(k_cover, right, spot_path, m, base_iv, rng)

                rows.append(dict(
                    date=day, slot=m, right=right, sel=kind, target=target,
                    strike=float(strike), entry=entry, miss=float(miss),
                    offset=abs(float(strike) - spot), spot=spot, settle=settle,
                    live=int(min(HORIZON, MINUTES - m)),
                    k_short=float(k_short), k_cover=float(k_cover),
                    mid=mid.tolist(), low=low.tolist(),
                    short=s_mid.tolist(), cover=c_mid.tolist()))
    return rows


# ---------------------------------------------------------------- indicators

def _bars(spot_path: np.ndarray, tf: int):
    """Minute path -> OHLC bars of `tf` minutes, plus each bar's start minute."""
    edges = np.arange(0, EXPIRY + 1, tf)
    o, h, l, c, at = [], [], [], [], []
    for s in edges:
        e = min(s + tf, MINUTES)
        seg = spot_path[s:e]
        if seg.size == 0:
            continue
        o.append(seg[0]); h.append(seg.max()); l.append(seg.min()); c.append(seg[-1])
        at.append(int(s))
    return (np.array(o), np.array(h), np.array(l), np.array(c), np.array(at))


def _roll(a: np.ndarray, n: int, fn):
    """`fn` over the n CLOSED bars before each bar. Index t reads t-n .. t-1, so
    a band can never see the bar it is compared against."""
    out = np.full(a.size, np.nan)
    for i in range(n, a.size):
        out[i] = fn(a[i - n:i])
    return out


def _day_indicators(spot_path: np.ndarray, tf: int) -> Dict[str, np.ndarray]:
    o, h, l, c, at = _bars(spot_path, tf)
    band = {}
    for tag, n in (("s", SLOW), ("f", FAST)):
        ma = _roll(o, n, np.mean)
        sd = _roll(o, n, lambda x: float(np.std(x)))
        band[tag] = (ma, ma + KDEV * sd, ma - KDEV * sd)

    (sma, sup, slo), (fma, fup, flo) = band["s"], band["f"]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_pctb = (o - slo) / (sup - slo)
        f_pctb = (o - flo) / (fup - flo)
        s_bw = (sup - slo) / sma
        f_bw = (fup - flo) / fma
        ratio = f_bw / s_bw
        gap_low = (o - slo) / sma
        gap_mid = (o - sma) / sma
        gap_high = (sup - o) / sma

    prev_range = np.full(o.size, np.nan)
    prev_range[1:] = (h[:-1] - l[:-1]) / np.where(sma[1:] > 0, sma[1:], np.nan)

    sign = np.sign(c - o)
    green_red = _roll(sign, GREEN_RED, np.mean)

    zone = np.full(o.size, np.nan)
    ok = np.isfinite(s_pctb)
    zone[ok] = np.clip(np.floor(s_pctb[ok] * 4.0), -1.0, 4.0)

    sl_s = np.full(o.size, np.nan); sl_f = np.full(o.size, np.nan)
    sl_s[1:] = np.sign(np.diff(sma)); sl_f[1:] = np.sign(np.diff(fma))
    slope_pair = sl_s + sl_f

    return dict(s_pctb=s_pctb, f_pctb=f_pctb, pctb_spread=s_pctb - f_pctb,
                s_bandwidth=s_bw, f_bandwidth=f_bw, bandwidth_ratio=ratio,
                gap_low=gap_low, gap_mid=gap_mid, gap_high=gap_high,
                prev_range=prev_range, green_red_avg=green_red, zone=zone,
                slope_pair=slope_pair, bar=at.astype(float))


# ---------------------------------------------------------------------- main

def sessions(n: int, seed: int):
    """Synthetic weekday calendar and a slowly drifting index level."""
    rng = np.random.default_rng(seed)
    day = dt.date(2017, 1, 3)
    level = 2250.0
    out = []
    for _ in range(n):
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        out.append((day, level))
        level *= float(np.exp(rng.normal(0.00045, 0.010)))
        day += dt.timedelta(days=1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sessions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    from pathlib import Path
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    ind: List[dict] = []

    for i, (day, level) in enumerate(sessions(a.sessions, a.seed)):
        sub = np.random.default_rng(a.seed + i * 7919)
        path, base_iv = _session(sub, level)
        rows.extend(_day(sub, day, path, base_iv))
        for tf in TFS:
            d = _day_indicators(path, tf)
            bars = d.pop("bar")
            for j, b in enumerate(bars):
                if b < SLOTS[0] or b > 390:
                    continue
                r = {c: float(d[c][j]) for c in COLUMNS}
                r.update(date=day, bar=int(b), tf=tf)
                ind.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1} sessions  {len(rows):,} rows", flush=True)

    df = pl.DataFrame(rows)
    df.write_parquet(out / "demo_calc.parquet")
    pl.DataFrame(ind).write_parquet(out / "demo_calc_ind.parquet")
    print(f"wrote {df.height:,} entry rows over {a.sessions} synthetic sessions")
    print(f"      {len(ind):,} indicator bar-rows across {len(TFS)} timeframes")


if __name__ == "__main__":
    main()
