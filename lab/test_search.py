#!/usr/bin/env python3
"""Tests for the search space: what it can express and what it refuses."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

optuna = pytest.importorskip("optuna")

from lab import indicators as I
from lab import search as S
from lab.exits import CLOSE, ExitPolicy
from lab.gates import OPTION, SPOT
from lab.session import calendar

TF = 30


@pytest.fixture(scope="module")
def train():
    return calendar()[:400]


@pytest.fixture(scope="module")
def quant(train):
    return S.training_quantiles(S.Space(), TF, train)


def _fixed(params):
    return optuna.trial.FixedTrial(params)


# ------------------------------------------------------------- quantiles

def test_quantiles_are_monotone_and_bracket_the_column(quant):
    for col in ("s_pctb", "s_bandwidth"):
        vals = [quant.value(col, q) for q in (0.0, 0.25, 0.5, 0.75, 1.0)]
        assert vals == sorted(vals)


def test_quantiles_come_from_the_training_window_only(train):
    """Deriving thresholds from all history leaks the test period's
    distribution into the definition of every gate."""
    early = S.training_quantiles(S.Space(), TF, train)
    late = S.training_quantiles(S.Space(), TF, calendar()[-400:])
    assert early.value("s_bandwidth", 0.5) != late.value("s_bandwidth", 0.5)


def test_a_column_with_no_data_is_reported_missing(train):
    q = S.Quantiles(pl.DataFrame({"x": [1.0, 2.0]}), ["x", "absent"])
    assert q.has("x") and not q.has("absent")
    assert q.value("absent", 0.5) is None


# ----------------------------------------------------------- suggestion

def test_a_numeric_gate_maps_a_quantile_onto_the_columns_own_scale(quant):
    """`s_bandwidth` lives near 0.01 and `s_pctb` on [-1, 1]; searching raw
    scales would need a hand-written range per column."""
    g = S._suggest_gate(_fixed({"eg0_col": "s_bandwidth", "eg0_op": "ge",
                                "eg0_q": 0.5}), S.Space(), quant, "eg0")
    assert g.column == "s_bandwidth" and g.op == "ge"
    assert g.lo == pytest.approx(quant.value("s_bandwidth", 0.5))


def test_a_between_gate_orders_its_bounds(quant):
    g = S._suggest_gate(_fixed({"eg0_col": "s_pctb", "eg0_op": "between",
                                "eg0_q1": 0.8, "eg0_q2": 0.2}),
                        S.Space(), quant, "eg0")
    assert g.lo < g.hi


def test_an_ordered_categorical_becomes_a_contiguous_band(quant):
    """Zones run BL < UL < L < M < H < UH < BH, so a band searches far better
    than a free subset of seven booleans."""
    g = S._suggest_gate(_fixed({"eg0_col": "zone", "eg0_lo": 1, "eg0_hi": 3}),
                        S.Space(), quant, "eg0")
    assert g.op == "in" and g.values == (1, 2, 3)


def test_a_set_categorical_becomes_a_subset(quant):
    g = S._suggest_gate(_fixed({"eg0_col": "s_slope", "eg0_mask": 0b01010}),
                        S.Space(), quant, "eg0")
    assert set(g.values) == {1, 3}


def test_zero_gates_is_a_point_in_the_space(quant):
    gs = S._suggest_gateset(_fixed({"eg_n": 0}), S.Space(), quant, "eg", 2)
    assert not gs and gs.mask(I.build(TF)).all()


def test_gates_default_to_the_spot_chart_and_can_be_routed(quant):
    sp = S.Space(charts=(SPOT, OPTION))
    g = S._suggest_gate(_fixed({"eg0_col": "s_pctb", "eg0_chart": OPTION,
                                "eg0_op": "ge", "eg0_q": 0.5}), sp, quant, "eg0")
    assert g.chart == OPTION


# ------------------------------------------------------------- policies

def test_a_credit_method_gets_no_w_or_l(quant):
    """A credit structure is held to expiry; W and L are long-only."""
    for m in ("CREDIT",):
        pol = S.suggest_policy(_fixed({"xg_n": 0}), S.Space(method=m), quant)
        assert pol.w == 0.0 and pol.l == 0.0


def test_a_decaying_w_never_tightens(quant):
    """The asymptote is a FRACTION of the start, so the two cannot cross."""
    pol = S.suggest_policy(_fixed({
        "xg_n": 0, "w": 4.0, "l": 0.25, "w_decays": 1, "w_end_frac": 0.5,
        "w_half_life": 60.0, "w_action": CLOSE, "l_action": CLOSE,
        "cover_w": 1, "short_w": 1}), S.Space(), quant)
    assert pol.w_end is not None and pol.w_end < pol.w


def test_no_decay_leaves_a_static_level(quant):
    pol = S.suggest_policy(_fixed({
        "xg_n": 0, "w": 3.0, "l": 0.5, "w_decays": 0, "w_action": CLOSE,
        "l_action": CLOSE, "cover_w": 1, "short_w": 1}), S.Space(), quant)
    assert pol.w_end is None
    assert np.allclose(pol.level_series(2.0, 0, 391, "w"), 6.0)


# --------------------------------------------------------------- methods

@pytest.mark.parametrize("name", ["ZONE", "CREDIT"])
def test_every_method_can_be_suggested(quant, name):
    sp = S.Space(method=name)
    params = {"fast": 10, "slow": 20, "k": 2.0, "band_source": "open",
              "band_ma": "sma", "gr_period": 10, "window": "09:30|15:00",
              "eg_n": 0, "target": 2.0, "level": 0.6, "dir": "reversion",
              "width": 10.0, "ratio": 0.5, "min_ratio": 0.35, "offset": 0.0}
    m = S.suggest_method(_fixed(params), sp, quant, 30)
    assert m.name == name and m.timeframe == 30


def test_an_unknown_method_is_refused(quant):
    with pytest.raises(ValueError, match="unknown method"):
        S.suggest_method(_fixed({"fast": 10, "slow": 20, "k": 2.0,
                                 "band_source": "open", "band_ma": "sma",
                                 "gr_period": 10, "window": "09:30|15:00",
                                 "eg_n": 0}),
                         S.Space(method="NOPE"), quant, 30)


def test_neh_min_ratios_go_below_one_to_one():
    """The specification withdrew the 1:1 requirement on 23 Aug."""
    assert min(S.Space().min_ratios) < 0.5


# --------------------------------------------------------------- objective

def test_a_thin_configuration_cannot_win(train, quant):
    """A config that trades a handful of times has a Sharpe made of noise."""
    obj = S.Objective(S.Space(), train[:5], {30: quant}, workers=4,
                      min_trades=10_000_000)
    # min_trades, not an invalid parameter, is what must reject it.
    v = obj(_fixed({"timeframe": 30, "fast": 10, "slow": 20, "k": 2.0,
                    "band_source": "open", "band_ma": "sma", "gr_period": 10,
                    "window": "09:30|15:00", "eg_n": 0, "target": 2.0,
                    "xg_n": 0, "w": 3.0, "l": 0.5, "w_decays": 0,
                    "w_action": CLOSE, "l_action": CLOSE, "cover_w": 1,
                    "short_w": 1}))
    assert v == -1e6


def test_the_objective_scores_the_cross_bracket(train, quant):
    """A variant that only survives at the mid is an execution claim."""
    obj = S.Objective(S.Space(), train[:20], {30: quant}, workers=4,
                      min_trades=1)
    t = _fixed({"timeframe": 30, "fast": 10, "slow": 20, "k": 2.0,
                "band_source": "open", "band_ma": "sma", "gr_period": 10,
                "window": "09:30|15:00", "eg_n": 0, "target": 2.0, "xg_n": 0,
                "w": 3.0, "l": 0.5, "w_decays": 0, "w_action": CLOSE,
                "l_action": CLOSE, "cover_w": 1, "short_w": 1})
    v = obj(t)
    assert v == pytest.approx(t.user_attrs["sharpe_cross"])
    assert t.user_attrs["sharpe_mid"] >= t.user_attrs["sharpe_cross"]
    assert t.user_attrs["bracket_gap"] >= 0


def test_an_impossible_corner_of_the_space_is_recorded_not_crashed(train, quant):
    """A suggestion outside the declared choices fails the trial and says why,
    rather than taking down a study that is hours in."""
    obj = S.Objective(S.Space(), train[:5], {30: quant}, workers=4)
    t = _fixed({"timeframe": 30, "fast": 10, "slow": 20, "k": 2.0,
                "band_source": "open", "band_ma": "sma", "gr_period": 10,
                "window": "09:30|15:00", "eg_n": 0, "target": 2.0, "xg_n": 0,
                "w": 3.0, "l": 0.4, "w_decays": 0, "w_action": CLOSE,
                "l_action": CLOSE, "cover_w": 1, "short_w": 1})
    assert obj(t) == -1e6
    assert "not in" in t.user_attrs["invalid"]


def test_metric_view_selects_which_redundancy_is_searched():
    """`zone` and `relation` are exact functions of continuous columns, so the
    two views are a controlled experiment rather than an accident."""
    both = set(S.Space(metric_view="both").columns())
    cat = set(S.Space(metric_view="categorical").columns())
    cont = set(S.Space(metric_view="continuous").columns())
    assert cat | cont == both and not (cat & cont)
    assert "zone" in cat and "s_pctb" in cont
    assert "relation" in cat and "gap_low" in cont


def test_an_explicit_column_list_overrides_the_view():
    sp = S.Space(gate_columns=("s_pctb",), metric_view="categorical")
    assert sp.columns() == ("s_pctb",)


def test_an_unknown_view_is_refused():
    with pytest.raises(ValueError, match="metric_view"):
        S.Space(metric_view="hybrid").columns()


def test_band_source_is_fixed_at_open_and_not_searched():
    """Fixed by decision, 23 Aug. The band is built from the same quantity the
    rule is compared against, so `%b`, `zone` and the gaps mean one thing."""
    assert S.Space().band_sources == ("open",)
    assert S.Space().band_mas == ("sma",)
    assert I.BandConfig().source == "open"
    assert I.BandConfig().ma == "sma"


def test_close_and_hlc3_remain_expressible_for_reproducibility():
    """Early results here were computed on `close`. Refusing it outright would
    make them unreproducible."""
    for src in ("close", "open", "hlc3"):
        assert I.BandConfig(source=src).validate().source == src
    with pytest.raises(ValueError):
        I.BandConfig(source="typical").validate()


def test_ema_can_be_widened_back_in_for_one_study():
    """EMA stays available as a sweep axis without being a default."""
    sp = S.Space(band_mas=("sma", "ema"))
    assert set(sp.band_mas) == {"sma", "ema"}
