#!/usr/bin/env python3
"""Tests for the dual Bollinger metric set.

The first test is the one that matters. Everything else here is arithmetic that
can be checked by hand; lookahead is the failure that would invalidate the whole
programme while every number still looked plausible.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from lab import indicators as I
from lab.indicators import (BH, BL, CONTRACTION, EXPANSION, FLAT, F_ABOVE_S,
                            F_BELOW_S, F_INSIDE_S, H, L, M, OTHER, S_INSIDE_F,
                            TREND_DOWN, TREND_UP, UH, UL, BandConfig)

TF = 30
DAY = dt.date(2024, 1, 3)


# ------------------------------------------------------------- THE lookahead

@pytest.mark.parametrize("tf", [5, 15, 30, 60])
def test_no_metric_changes_when_the_future_is_deleted(tf):
    """Truncate the series and every surviving row must be bit-identical.

    This is the property that makes the whole feature matrix usable: a metric
    on bar t must be computable from bars <= t, so removing bars after t cannot
    move it. A single unshifted rolling window fails this and nothing else in
    the pipeline would notice."""
    full = I.build(tf)
    cut = full.height // 2
    # Rebuild from a truncated bar series rather than slicing the output.
    bars = I.bars(tf).head(cut)
    monkey = _build_from(bars, tf)
    head = full.head(cut)
    for col in I.METRICS + I.BANDS:
        a = head[col].to_numpy().astype(float)
        b = monkey[col].to_numpy().astype(float)
        assert np.allclose(a, b, equal_nan=True), f"{col} moved when the future was cut"


@pytest.mark.parametrize("tf", [15, 60])
def test_no_metric_reads_its_own_bar_beyond_the_open(tf):
    """Perturb one bar's high/low/close, leaving its open alone.

    Nothing on THAT bar may move. A decision taken at `t:00` knows the open and
    nothing else about bar t, so any metric that shifted would be reading the
    rest of a bar that has not happened yet. This is the half of the lookahead
    property that truncation cannot see — deleting the future does not change
    bar t's own close, so a band that reads its own bar passes the truncation
    test and fails this one."""
    base = I.bars(tf)
    r = base.height // 2
    poked = base.with_columns(
        pl.when(pl.int_range(pl.len()) == r)
          .then(pl.col("close") * 1.02).otherwise(pl.col("close")).alias("close"),
        pl.when(pl.int_range(pl.len()) == r)
          .then(pl.col("high") * 1.02).otherwise(pl.col("high")).alias("high"),
        pl.when(pl.int_range(pl.len()) == r)
          .then(pl.col("low") * 0.98).otherwise(pl.col("low")).alias("low"))
    clean = _build_from(base, tf)
    dirty = _build_from(poked, tf)
    moved = []
    for col in I.METRICS + I.BANDS:
        a, b = (float(clean[col][r]), float(dirty[col][r]))
        if not (np.isnan(a) and np.isnan(b)) and a != b:
            moved.append(col)
    assert not moved, f"these read their own bar: {moved}"
    # And the perturbation must be detectable at all, or the test proves nothing.
    after = [c for c in I.METRICS + I.BANDS
             if float(clean[c][r + 1]) != float(dirty[c][r + 1])]
    assert after, "the poke changed nothing anywhere — the test is vacuous"


def _build_from(bar_frame: pl.DataFrame, tf: int) -> pl.DataFrame:
    """`build` against an injected bar series, so truncation is testable."""
    real = I.bars
    try:
        I.bars = lambda _tf: bar_frame            # noqa: ARG005
        return I.build(tf)
    finally:
        I.bars = real


def _probes(df, *at):
    """Sample positions that exist in THIS panel.

    The positions are arbitrary — the point is to check the band arithmetic at
    a few places, not at any particular bar. Hardcoding them assumes a panel
    length, so a smaller dataset turns a correctness test into an IndexError.
    Two surviving probes is the floor: one could be the warm-up prefix."""
    keep = [t for t in at if t < df.height]
    assert len(keep) >= 2, f"panel too short to probe: {df.height} bars"
    return keep

def test_bands_never_read_their_own_bar():
    """SM on row t is the average of the SOURCE over rows t-slow .. t-1."""
    cfg = BandConfig(fast=3, slow=5, source="close", ma="sma", k=2.0)
    df = I.build(TF, cfg)
    src = df["close"].to_numpy()
    for t in _probes(df, 100, 5000, 20000):
        assert df["SM"][t] == pytest.approx(src[t - 5:t].mean())
        assert df["FM"][t] == pytest.approx(src[t - 3:t].mean())
        sd = src[t - 5:t].std(ddof=0)
        assert df["SL"][t] == pytest.approx(src[t - 5:t].mean() - 2 * sd)
        assert df["SH"][t] == pytest.approx(src[t - 5:t].mean() + 2 * sd)


def test_prev_bar_metrics_are_the_previous_bar():
    df = I.build(TF)
    o, c = df["open"].to_numpy(), df["close"].to_numpy()
    for t in _probes(df, 50, 900, 4000):
        assert df["prev_range"][t] == pytest.approx(c[t - 1] - o[t - 1])
        assert df["prev_green"][t] == np.sign(c[t - 1] - o[t - 1])


def test_green_red_average_is_closed_bars_only():
    cfg = BandConfig(green_red_period=4)
    df = I.build(TF, cfg)
    sign = np.sign(df["close"].to_numpy() - df["open"].to_numpy())
    for t in _probes(df, 60, 4000, 20000):
        assert df["green_red_avg"][t] == pytest.approx(sign[t - 4:t].mean())


def test_third_order_is_a_plain_lag():
    df = I.features(TF, BandConfig(), third_order=True)
    for col in ("zone", "s_bandwidth", "green_red_avg"):
        a = df[col].to_numpy().astype(float)[:-1]
        b = df[f"prev_{col}"].to_numpy().astype(float)[1:]
        assert np.allclose(a, b, equal_nan=True)


# ------------------------------------------------------------------- zones

def _zone_of(price, sl, sm, sh, fl, fm, fh):
    return int(I._zone(*(np.array([x], float) for x in
                         (price, sl, sm, sh, fl, fm, fh)))[0])


def test_the_seven_zones_are_the_bibles_seven():
    # slow 90/100/110, fast 96/101/106 — the mid lines must differ, or the M
    # region has zero width and cannot be entered. That is correct behaviour,
    # not a bug: M is the gap BETWEEN the two mids.
    b = (90.0, 100.0, 110.0, 96.0, 101.0, 106.0)
    assert _zone_of(80.0, *b) == BL      # below both low bands
    assert _zone_of(93.0, *b) == UL      # between the two low bands
    assert _zone_of(98.0, *b) == L       # above both lows, below the mids
    assert _zone_of(100.5, *b) == M      # between the mid bands
    assert _zone_of(103.0, *b) == H      # above the mids, below both highs
    assert _zone_of(108.0, *b) == UH     # between the two high bands
    assert _zone_of(120.0, *b) == BH     # above both high bands


def test_the_m_region_is_empty_when_the_mid_lines_coincide():
    b = (90.0, 100.0, 110.0, 95.0, 100.0, 105.0)
    assert _zone_of(100.5, *b) == H      # no gap between the mids to sit in


def test_a_zone_is_assigned_even_when_the_families_overlap_oddly():
    """A fast low line above a slow mid is unusual but real in a sharp move.
    Counting keeps the mapping total; a pairwise if-chain would fall through."""
    z = _zone_of(103.0, 90.0, 95.0, 100.0, 101.0, 104.0, 107.0)
    assert z in (BL, UL, L, M, H, UH, BH)


def test_every_bar_of_the_real_series_gets_a_zone_after_warmup():
    df = I.build(TF)
    z = df["zone"].to_numpy()[25:]
    assert (z >= 0).all() and (z <= BH).all()


# ------------------------------------------------------------------ slopes

def _slope(low, high, eps=0.02):
    return int(I._slope_type(np.array(low, float), np.array(high, float), 1, eps)[-1])


def test_slope_types():
    assert _slope([90, 90], [110, 110]) == FLAT
    assert _slope([90, 88], [110, 112]) == EXPANSION
    assert _slope([90, 92], [110, 108]) == CONTRACTION
    assert _slope([90, 92], [110, 112]) == TREND_UP
    assert _slope([90, 88], [110, 108]) == TREND_DOWN


def test_flat_is_a_fraction_of_bandwidth_not_a_number_of_points():
    """The same relative move must classify identically at SPX 2,250 and 7,500."""
    small = _slope([2240, 2240.1], [2260, 2259.9])
    large = _slope([7480, 7480.33], [7520, 7519.67])
    assert small == large == FLAT


def test_a_widening_band_with_a_flat_low_line_is_still_expansion():
    assert _slope([90, 90], [110, 115]) == EXPANSION


# --------------------------------------------------------------- relations

def _rel(sl, sh, fl, fh):
    return int(I._relation(*(np.array([x], float) for x in (sl, sh, fl, fh)))[0])


def test_band_relations():
    assert _rel(90.0, 110.0, 95.0, 105.0) == F_INSIDE_S
    assert _rel(95.0, 105.0, 90.0, 110.0) == S_INSIDE_F
    assert _rel(90.0, 110.0, 95.0, 115.0) == F_ABOVE_S
    assert _rel(90.0, 110.0, 85.0, 105.0) == F_BELOW_S


def test_containment_beats_direction():
    """A nested fast band is also trivially above the slow low line; calling it
    F_ABOVE_S would lose the nesting, which is the more informative fact."""
    assert _rel(90.0, 110.0, 95.0, 110.0) == F_INSIDE_S


# --------------------------------------------------------------- crossings

def test_crossings_fire_once_on_the_bar_that_crosses():
    a = np.array([1.0, 1.0, 3.0, 3.0, 0.0], float)
    b = np.array([2.0, 2.0, 2.0, 2.0, 2.0], float)
    assert list(I._cross(a, b)) == [0, 0, 1, 0, -1]


# ------------------------------------------------------------ band measures

def test_centred_pctb_is_zero_at_the_mid_and_one_at_the_band():
    df = I.build(TF)
    op, sl, sm, sh = (df[c].to_numpy() for c in ("open", "SL", "SM", "SH"))
    pb = df["s_pctb"].to_numpy()
    ok = np.isfinite(pb)
    # The definition, checked against the columns rather than restated.
    assert np.allclose(pb[ok], ((op - sm) / ((sh - sl) / 2.0))[ok])
    # A bar opening on the mid scores 0; on the upper band, +1.
    assert np.allclose(((sm - sm) / ((sh - sl) / 2.0))[ok], 0.0)
    assert np.allclose(((sh - sm) / ((sh - sl) / 2.0))[ok], 1.0)
    # And it is genuinely bounded near +/-1 in practice, not unbounded noise.
    assert 0.75 < np.nanpercentile(np.abs(pb), 95) < 1.6


def test_bandwidth_is_scale_free_and_widens_with_the_timeframe():
    med = {tf: I.build(tf)["s_bandwidth"].median() for tf in (5, 15, 30, 60)}
    assert med[5] < med[15] < med[30] < med[60]
    assert 0.0 < med[5] < 0.1


# ------------------------------------------------------------- config axes

@pytest.mark.parametrize("source", ["open", "close", "hlc3"])
@pytest.mark.parametrize("ma", ["sma", "ema"])
def test_every_axis_combination_builds(source, ma):
    df = I.build(TF, BandConfig(source=source, ma=ma))
    assert df["SM"].null_count() < 30
    assert df["zone"].to_numpy()[30:].min() >= 0


def test_the_axes_actually_change_the_answer():
    a = I.build(TF, BandConfig(source="close", ma="sma"))["zone"].to_numpy()
    b = I.build(TF, BandConfig(source="close", ma="ema"))["zone"].to_numpy()
    c = I.build(TF, BandConfig(source="open", ma="sma"))["zone"].to_numpy()
    assert (a != b).mean() > 0.05 and (a != c).mean() > 0.01


def test_a_bad_config_is_refused_not_silently_coerced():
    with pytest.raises(ValueError, match="source"):
        I.build(TF, BandConfig(source="typical"))
    with pytest.raises(ValueError, match="fast"):
        I.build(TF, BandConfig(fast=20, slow=10))


# ------------------------------------------------------------ session access

def test_the_minute_index_maps_bars_onto_option_minutes():
    """The 12:00 bar must be minute 150, not -106: `dt.hour()` is Int8 and
    `(hour - 9) * 60` overflows from 12:00 on."""
    d = I.for_session(DAY, 30)
    assert d["minute"].to_list() == list(range(0, 391, 30))[:d.height]
    assert d["minute"].min() >= 0


def test_the_60m_series_opens_each_day_with_a_30_minute_stub():
    d = I.for_session(DAY, 60)
    assert d["source_minutes"].to_list() == [30] + [60] * (d.height - 1)
    assert d["minute"].to_list() == [0, 30, 90, 150, 210, 270, 330]


def test_bands_carry_across_the_overnight_gap():
    """The first bar of a session must have a band. At 60m it cannot have one
    any other way — 20 periods is three sessions of history."""
    d = I.for_session(DAY, 60)
    first = d.filter(pl.col("session_bar") == 0).to_dicts()[0]
    assert np.isfinite(first["SM"]) and np.isfinite(first["FM"])
    assert first["zone"] >= 0


def test_session_bar_counts_from_the_open():
    d = I.for_session(DAY, 15)
    assert d["session_bar"].to_list() == list(range(d.height))


# ------------------------------------------------------ measured redundancy

@pytest.mark.parametrize("tf", [5, 30, 60])
def test_zone_is_exactly_a_function_of_the_two_centred_pctb(tf):
    """`s_pctb > 1` IS `price > SH` by construction, so the six band
    comparisons a zone counts are already carried by the two %b values. The
    categorical adds search efficiency, not information."""
    df = I.build(tf)
    sp, fp = df["s_pctb"].to_numpy(), df["f_pctb"].to_numpy()
    zone = df["zone"].to_numpy()
    n_low = (sp > -1).astype(int) + (fp > -1).astype(int)
    n_mid = (sp > 0).astype(int) + (fp > 0).astype(int)
    n_high = (sp > 1).astype(int) + (fp > 1).astype(int)
    rebuilt = np.select(
        [n_low == 0, n_low == 1, n_mid == 0, n_mid == 1, n_high == 0, n_high == 1],
        [BL, UL, L, M, H, UH], default=BH)
    ok = np.isfinite(sp) & np.isfinite(fp) & (zone >= 0)
    assert ok.sum() > 1000
    assert (rebuilt[ok] == zone[ok]).all()


@pytest.mark.parametrize("tf", [5, 30, 60])
def test_relation_is_exactly_a_function_of_the_two_signed_gaps(tf):
    """And `gap_mid` is not involved at all."""
    df = I.build(tf)
    gl, gh = df["gap_low"].to_numpy(), df["gap_high"].to_numpy()
    rel = df["relation"].to_numpy()
    out = np.full(rel.size, I.OTHER)
    out[(gl > 0) & (gh > 0)] = F_BELOW_S
    out[(gl < 0) & (gh < 0)] = F_ABOVE_S
    out[(gl >= 0) & (gh <= 0)] = S_INSIDE_F
    out[(gl <= 0) & (gh >= 0)] = F_INSIDE_S
    ok = np.isfinite(gl) & np.isfinite(gh) & (rel >= 0)
    assert ok.sum() > 1000
    assert (out[ok] == rel[ok]).all()


def test_the_gaps_are_signed_so_they_carry_positionality():
    gl = I.build(30)["gap_low"].to_numpy()
    assert np.nanmin(gl) < 0 < np.nanmax(gl)


def test_the_continuous_view_is_strictly_richer_than_the_zone():
    """Within one zone the magnitude still varies enormously — a bar seven
    half-widths below the mid and one barely below SL are the same zone."""
    df = I.build(30)
    sp, zone = df["s_pctb"].to_numpy(), df["zone"].to_numpy()
    inside = np.abs(sp[np.isfinite(sp) & (zone == BL)])
    assert inside.max() > 3.0 and np.median(inside) < 2.0


def test_the_two_views_partition_the_documented_dependencies():
    """Every declared dependency names columns that exist in exactly one view,
    and the views stay disjoint. That is what lets `search.Space.metric_view`
    turn the redundancy into a controlled experiment instead of a hidden
    collinearity."""
    views = set(I.CATEGORICAL_VIEW) | set(I.CONTINUOUS_VIEW)
    for derived, sources in I.DERIVED_FROM.items():
        assert derived in views, derived
        assert all(s in views for s in sources), derived
    assert not set(I.CATEGORICAL_VIEW) & set(I.CONTINUOUS_VIEW)


def test_a_categorical_derived_column_coarsens_a_continuous_one():
    """The original and still the common case: `zone` and `relation` are the
    trader-vocabulary bucketing of metrics that are continuous underneath.
    `abs_prev_range` is deliberately NOT of this kind — it is a continuous
    transform of a continuous column — so the rule is asserted only where it
    is meant to hold."""
    for derived in ("zone", "relation", "prev_green"):
        assert derived in I.CATEGORICAL_VIEW
        assert all(s in I.CONTINUOUS_VIEW for s in I.DERIVED_FROM[derived])


def test_abs_prev_range_is_the_magnitude_of_the_signed_one():
    """The signed prior body and its magnitude answer different
    questions, and against a breach objective only the magnitude is ordered.
    Both are kept, and the dependency is declared so the redundancy is a
    controlled experiment rather than a hidden collinearity."""
    f = I.for_session(dt.date(2024, 1, 3), 60)
    a = f["prev_range"].to_numpy()
    b = f["abs_prev_range"].to_numpy()
    ok = np.isfinite(a) & np.isfinite(b)
    assert ok.sum() > 3
    assert np.allclose(np.abs(a[ok]), b[ok])
    assert I.DERIVED_FROM["abs_prev_range"] == ("prev_range",)
    assert "abs_prev_range" in I.CONTINUOUS_VIEW


def test_the_signed_prior_body_is_not_recoverable_from_its_magnitude():
    """The reverse dependency does not hold, which is why both are kept."""
    f = I.for_session(dt.date(2024, 1, 3), 60)
    a = f["prev_range"].to_numpy()
    ok = np.isfinite(a)
    assert (a[ok] < 0).any() and (a[ok] > 0).any()
