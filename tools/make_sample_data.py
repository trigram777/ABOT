#!/usr/bin/env python3
"""
Build the bundled SAMPLE DATASET -- synthetic, and deliberately so.

The production panel is built from a commercial tick feed whose licence does
not permit redistribution. Rather than ship a repository whose test suite
cannot run, this generates a dataset in exactly the layout `lab/session.py`
expects, from a Black-Scholes model over simulated index paths.

    python tools/make_sample_data.py [--sessions 60] [--seed 20260903]

WHAT IT IS NOT. It is not a model of any real market and nothing measured on
it is a statement about one. Its job is to be *shaped* correctly, so that the
engine's arithmetic -- fills, commissions, strike selection, path
reconstruction, band computation, settlement -- can be exercised end to end.

WHAT IT GETS RIGHT ON PURPOSE. Four properties, because tests depend on them:

  * **The book does not exist at 09:30.** On essentially every real session the
    first minute is a pre-rotation snapshot with no underlying reported, and
    `Session.book_reported` is how rules find the first tradeable minute. The
    generator reproduces that, including a handful of sessions where the book
    arrives several minutes late -- because a rule that hardcodes 09:31 is
    wrong on those, and there should be one here to catch it.
  * **Quotes are two-sided and uncrossed**, and go NaN where a strike is
    worthless -- `load()` scrubs non-positive and crossed quotes, and that path
    should be exercised rather than bypassed.
  * **The traded bar sits inside the quote.** `ohlc_low` is what a stop fires
    on, so it must be below the midpoint and consistent with the bar's own
    open and close, or the whole resolution axis is untestable.
  * **Prices are on the tick grid** ($0.05 below $3.00, $0.10 at or above).

LAYOUT PRODUCED

    sample_data/
      spxw_pass1/minute/year=YYYY/YYYY-MM-DD.parquet   the option chain
      spx_ibkr_spot/spx_ibkr_1m.parquet                the index, 1m
      spx_ibkr_spot/derived/spx_{5,15,30,60}m.parquet  resampled, continuous
      dataset_catalog/master_session_index.parquet     the calendar
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy.special import ndtr

ET = ZoneInfo("America/New_York")
MINUTES = 391                     # 09:30 .. 16:00 inclusive
EXPIRY = MINUTES - 1
RTH = 390                         # IBKR prints 09:30 .. 15:59
STRIKE_STEP = 5.0
MIN_HALF = 75.0                   # narrowest half-window, in points
MAX_HALF = 260.0                  # widest, so a violent session stays bounded
SIGMA_COVER = 3.6                 # half-windows of one session's own sigma
TICK_BREAK = 3.00
IMPLIED_PREMIUM = 1.18            # implied sits above realised; see explorer/
# The vol draw and the jump rate are set so the session mix lands where a real
# index sits -- roughly 85% quiet, ~12% elevated, under 3% extreme. That is a
# documented property of the instrument, not a tuned result: a fixture whose
# tails are fatter than the market's would make every regime split meaningless.
JUMP_P = 0.035
VOL_SIGMA = 0.38


# --------------------------------------------------------------- the model

def _bs(spot, strike, t, iv, call: bool):
    spot = np.asarray(spot, float)[..., None] if np.ndim(strike) else np.asarray(spot, float)
    t = np.maximum(np.asarray(t, float), 0.0)
    t = t[..., None] if np.ndim(strike) and np.ndim(t) else t
    sig = np.maximum(iv, 1e-6) * np.sqrt(np.maximum(t, 1e-12))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(spot / strike) + 0.5 * sig ** 2) / sig
        d2 = d1 - sig
    if call:
        px, dl = spot * ndtr(d1) - strike * ndtr(d2), ndtr(d1)
        intr = np.maximum(spot - strike, 0.0)
    else:
        px, dl = strike * ndtr(-d2) - spot * ndtr(-d1), ndtr(d1) - 1.0
        intr = np.maximum(strike - spot, 0.0)
    theta = -(spot * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) * np.maximum(iv, 1e-6)
              / (2 * np.sqrt(np.maximum(t, 1e-12)))) / 252.0
    vega = spot * np.exp(-0.5 * d1 ** 2) / np.sqrt(2 * np.pi) * np.sqrt(np.maximum(t, 1e-12)) / 100.0
    dead = np.broadcast_to(t <= 0, px.shape)
    px = np.where(dead, intr, px)
    dl = np.where(dead, np.where(intr > 0, 1.0 if call else -1.0, 0.0), dl)
    return px, dl, np.where(dead, 0.0, theta), np.where(dead, 0.0, vega)


def _iv(strikes, spot, base):
    m = (strikes - spot) / spot
    return base * (1.0 + 6.0 * m * m + 0.9 * np.maximum(-m, 0.0))


def _snap(px):
    px = np.clip(np.asarray(px, float), 0.0, None)
    return np.where(px < TICK_BREAK, np.round(px / 0.05) * 0.05, np.round(px / 0.10) * 0.10)


def _index_path(rng, level):
    real = min(float(np.exp(rng.normal(np.log(0.16), VOL_SIGMA))), 1.30)
    per = real / np.sqrt(390.0 * 252.0)
    steps = rng.standard_normal(EXPIRY) * per
    if rng.random() < JUMP_P:
        steps[rng.integers(20, EXPIRY - 20)] += rng.choice([-1., 1.]) * rng.uniform(2., 6.) * per * 8
    steps += rng.normal(0.0, 0.30) * per / np.sqrt(EXPIRY)
    path = level * np.exp(np.cumsum(steps))
    return np.concatenate([[level], path]), real * IMPLIED_PREMIUM


# ------------------------------------------------------------- one session

def _session_frames(rng, day: dt.date, level: float):
    path, iv_base = _index_path(rng, level)
    minutes = [dt.datetime.combine(day, dt.time(9, 30), ET) + dt.timedelta(minutes=int(i))
               for i in range(MINUTES)]

    # The book arrives after the open. Usually one minute; occasionally later,
    # because a rule that hardcodes 09:31 has to be catchable.
    first = 1 if rng.random() > 0.06 else int(rng.integers(2, 12))

    # The strike window follows the session's own volatility. A fixed window is
    # wrong in both directions: too narrow and a violent day settles off the end
    # of the grid, so a selector finds no strike where a real chain has plenty;
    # too wide and every quiet session carries hundreds of dead columns.
    daily = iv_base / IMPLIED_PREMIUM / np.sqrt(252.0)
    half = float(np.clip(SIGMA_COVER * daily * path[0], MIN_HALF, MAX_HALF))
    half = np.ceil(half / STRIKE_STEP) * STRIKE_STEP
    base = np.round(path[0] / STRIKE_STEP) * STRIKE_STEP
    strikes = np.arange(base - half, base + half + STRIKE_STEP, STRIKE_STEP)

    tte = np.maximum(EXPIRY - np.arange(MINUTES, dtype=float), 0.0) / (390.0 * 252.0)
    iv = _iv(strikes, path[:, None], iv_base)                       # [min, strike]

    # A quote wobble that is common to the whole chain at a given minute, NOT
    # independent per strike. Independent noise breaks the ordering of the
    # price surface across strikes, and a five-point vertical then pays more
    # than its own width -- an arbitrage the engine's selectors will happily
    # find and the tests will rightly reject.
    wob = np.zeros(MINUTES)
    for i in range(1, MINUTES):
        wob[i] = 0.90 * wob[i - 1] + rng.normal(0.0, 0.0045) * 0.436

    rows = []
    for right, call in (("CALL", True), ("PUT", False)):
        mid, delta, theta, vega = _bs(path, strikes, tte, iv, call)
        mid = np.maximum(mid, 0.0) * (1.0 + wob[:, None])

        spread = np.maximum(0.10, np.minimum(0.60, mid * 0.06))
        bid = _snap(np.maximum(mid - spread / 2, 0.0))
        ask = _snap(mid + spread / 2)
        bid[bid <= 0] = np.nan                                      # worthless: no bid
        ask[~np.isfinite(bid) & (ask < 0.05)] = np.nan

        # THE TRADED BAR SPANS [m, m+1), while the QUOTE row at m is the book
        # at the START of m. That is the convention the whole engine rests on:
        # a rule deciding at m:00 and filling on bid[m] is reading what it
        # could see, and `last[m-1]` is therefore what the quote at m tracks.
        # Build the bar from the interval, not from its opening instant.
        nxt = np.vstack([mid[1:], mid[-1:]])
        o, c = _snap(mid), _snap(nxt)
        span_hi, span_lo = np.maximum(o, c), np.minimum(o, c)
        pop = np.abs(rng.normal(0.0, 0.030, o.shape))
        dip = np.where(rng.random(o.shape) < 0.45, 0.0,
                       np.abs(rng.normal(0.0, 0.045, o.shape)))
        high = _snap(span_hi * (1.0 + np.clip(pop, 0, 0.25)))
        low = _snap(np.maximum(span_lo * (1.0 - np.clip(dip, 0, 0.30)), 0.0))

        # BEFORE THE BOOK OPENS THE CHAIN IS NOT EMPTY -- it carries the
        # previous session's closing snapshot, two-sided and entirely
        # plausible. What is missing is the UNDERLYING, which is why liveness
        # is read from `iv_underlying_price` and not from whether a quote
        # exists. A generator that blanks the quotes instead removes the exact
        # trap the gate is there for.
        dark = np.zeros(MINUTES, bool)
        dark[:first] = True
        for a in (bid, ask, o, high, low, c):
            a[dark, :] = a[first, :]

        n = MINUTES * strikes.size
        rows.append(pl.DataFrame({
            "strike": np.repeat(strikes[None, :], MINUTES, 0).ravel(),
            "right": np.full(n, right),
            # a Python list, not an object-dtype array: polars infers Datetime
            # from the former and an uncastable Object column from the latter.
            "timestamp": [t for t in minutes for _ in range(strikes.size)],
            "expiration": np.full(n, day.isoformat()),
            "ctx_vix": np.full(n, np.float32(iv_base * 100)),
            "iv_underlying_price": np.where(np.repeat(dark, strikes.size), np.nan,
                                            np.repeat(path, strikes.size)),
            "greeks_underlying_price": np.where(np.repeat(dark, strikes.size), np.nan,
                                                np.repeat(path, strikes.size)),
            "quote_bid": bid.ravel(), "quote_ask": ask.ravel(),
            "ohlc_open": o.ravel(), "ohlc_high": high.ravel(),
            "ohlc_low": low.ravel(), "ohlc_close": c.ravel(),
            "ohlc_volume": np.where(np.isfinite(o.ravel()),
                                    rng.integers(0, 900, n).astype(float), np.nan),
            "greeks_delta": delta.ravel(), "greeks_theta": theta.ravel(),
            "greeks_vega": vega.ravel(), "iv_implied_vol": iv.ravel(),
            "oi_open_interest": np.repeat(rng.integers(50, 9000, strikes.size)[None, :],
                                          MINUTES, 0).ravel().astype(float),
        }))

    chain = pl.concat(rows).with_columns(
        pl.col("timestamp").cast(pl.Datetime("us", "America/New_York")),
        *[pl.col(c).cast(pl.Float32) for c in
          ("strike", "ctx_vix", "iv_underlying_price", "greeks_underlying_price",
           "quote_bid", "quote_ask", "ohlc_open", "ohlc_high", "ohlc_low",
           "ohlc_close", "ohlc_volume", "greeks_delta", "greeks_theta",
           "greeks_vega", "iv_implied_vol", "oi_open_interest")])

    # the index, on IBKR's RTH axis: 09:30 .. 15:59, one minute short of the
    # option axis. `_spot_frame` carries 15:59's close into the 16:00 slot.
    step = path[:RTH]
    nxt = path[1:RTH + 1]
    wob = np.abs(rng.normal(0, 0.0004, RTH)) * step
    spot = pl.DataFrame({
        "date": [day] * RTH,
        "timestamp": minutes[:RTH],
        "open": step, "high": np.maximum(step, nxt) + wob,
        "low": np.minimum(step, nxt) - wob, "close": nxt,
    }).with_columns(pl.col("date").cast(pl.Date),
                    pl.col("timestamp").cast(pl.Datetime("us", "America/New_York")),
                    *[pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")])
    # The official settlement is a separate print, not the last 1m bar. Real
    # settlement is struck from opening prices the following morning and the
    # two are never quite equal; a sample where they ARE equal cannot exercise
    # the code that keeps them apart.
    settle = float(path[EXPIRY]) * (1.0 + rng.normal(0.0, 0.00012))
    return chain, spot, settle


# ---------------------------------------------------------------- resample

def _derive(spot: pl.DataFrame, tf: int) -> pl.DataFrame:
    """Continuous bars at `tf` minutes. `source_minutes` is carried because the
    first bar of a session is a stub -- RTH starts on the half hour, so a 60m
    series opens each day with 30 minutes, and a rule assuming otherwise is
    wrong on the first bar of every session."""
    return (spot.sort("timestamp")
            .group_by_dynamic("timestamp", every=f"{tf}m", group_by="date")
            .agg(pl.col("timestamp").first().alias("at"),
                 pl.col("open").first(), pl.col("high").max(),
                 pl.col("low").min(), pl.col("close").last(),
                 pl.len().alias("source_minutes"))
            # A bar is stamped with its FIRST DATAPOINT, not with the window it
            # fell in. The windows are epoch-aligned, so the opening stub lands
            # in the 09:00 window and would otherwise be stamped 09:00 -- half
            # an hour before the session it belongs to.
            .drop("timestamp").rename({"at": "timestamp"})
            .select("date", "timestamp", "open", "high", "low", "close", "source_minutes")
            .sort("timestamp"))


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", type=int, default=500,
                    help="eligible sessions. Walk-forward needs ~320; the "
                         "training-window tests compare calendar()[:400] "
                         "against calendar()[-400:], so they need >400.")
    ap.add_argument("--warmup", type=int, default=140,
                    help="leading sessions carrying INDEX bars only, so a "
                         "trailing-volatility and band windows are warm before the "
                         "eligible session and no test sits in the NaN prefix")
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    out = Path(a.out) if a.out else Path(__file__).resolve().parents[1] / "sample_data"
    (out / "dataset_catalog").mkdir(parents=True, exist_ok=True)
    (out / "spx_ibkr_spot" / "derived").mkdir(parents=True, exist_ok=True)

    day, level = dt.date(2023, 1, 3), 4750.0
    spots, index = [], []
    total = a.warmup + a.sessions
    for i in range(total):
        while day.weekday() >= 5:
            day += dt.timedelta(days=1)
        rng = np.random.default_rng(a.seed + i * 7919)
        warm = i < a.warmup
        chain, spot, settle = _session_frames(rng, day, level)

        if not warm:
            part = out / "spxw_pass1" / "minute" / f"year={day.year}"
            part.mkdir(parents=True, exist_ok=True)
            chain.write_parquet(part / f"{day.isoformat()}.parquet", compression="zstd")
        spots.append(spot)
        index.append(dict(date=day, has_spxw=not warm, has_xsp=False,
                          intraday_research_eligible=not warm,
                          official_close=settle))

        level = settle * float(np.exp(rng.normal(0.0002, 0.004)))
        day += dt.timedelta(days=1)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1} / {total} sessions", flush=True)

    allspot = pl.concat(spots).sort("timestamp")
    allspot.write_parquet(out / "spx_ibkr_spot" / "spx_ibkr_1m.parquet", compression="zstd")
    for tf in (5, 15, 30, 60):
        _derive(allspot, tf).write_parquet(
            out / "spx_ibkr_spot" / "derived" / f"spx_{tf}m.parquet", compression="zstd")

    (pl.DataFrame(index).with_columns(pl.col("date").cast(pl.Date))
        .write_parquet(out / "dataset_catalog" / "master_session_index.parquet"))

    # Derived caches are keyed to the dataset and nothing else, so a rebuilt
    # dataset must invalidate them or the next reader silently gets the old
    # one. `regimes.table()` is cached to disk on first use.
    cache = Path(__file__).resolve().parents[1] / "lab" / "cache"
    for stale in cache.glob("*.parquet"):
        stale.unlink()

    size = sum(f.stat().st_size for f in out.rglob("*.parquet"))
    print(f"wrote {a.sessions} eligible sessions (+{a.warmup} index-only warmup) "
          f"to {out}  ({size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
