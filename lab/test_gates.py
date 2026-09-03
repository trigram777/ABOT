#!/usr/bin/env python3
"""Tests for indicator gating, including the boundary it must not cross."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
import pytest

from lab import gates as G
from lab import indicators as I
from lab import regimes as R
from lab.gates import Gate, GateSet

DAY = dt.date(2024, 1, 3)


@pytest.fixture(scope="module")
def feats():
    return I.for_session(DAY, 30)


def _f(**cols):
    return pl.DataFrame(cols)


# ------------------------------------------------------------------- ops

def test_numeric_ops():
    f = _f(x=[0.0, 1.0, 2.0, 3.0])
    assert list(Gate("x", "ge", lo=2.0).mask(f)) == [False, False, True, True]
    assert list(Gate("x", "le", hi=1.0).mask(f)) == [True, True, False, False]
    assert list(Gate("x", "between", lo=1.0, hi=2.0).mask(f)) == \
        [False, True, True, False]
    assert list(Gate("x", "outside", lo=1.0, hi=2.0).mask(f)) == \
        [True, False, False, True]


def test_membership_ops():
    f = _f(zone=[I.BL, I.M, I.BH])
    assert list(Gate("zone", "in", values=(I.BL, I.BH)).mask(f)) == \
        [True, False, True]
    assert list(Gate("zone", "not_in", values=(I.M,)).mask(f)) == \
        [True, False, True]


def test_a_missing_indicator_never_passes():
    """An absent metric is not a satisfied condition. NaN fails every
    comparison in numpy, and the negated form has to be handled explicitly or
    `outside` would quietly pass it."""
    f = _f(x=[float("nan"), 5.0])
    for g in (Gate("x", "ge", lo=1.0), Gate("x", "le", hi=99.0),
              Gate("x", "between", lo=0.0, hi=99.0),
              Gate("x", "outside", lo=0.0, hi=1.0)):
        assert g.mask(f)[0] == False, g.label()


def test_an_unknown_column_is_named_not_silently_false(feats):
    with pytest.raises(KeyError, match="s_pctbb"):
        Gate("s_pctbb", "ge", lo=0).mask(feats)


def test_a_malformed_gate_is_refused():
    with pytest.raises(ValueError, match="unknown op"):
        Gate("x", "approx", lo=1).validate()
    with pytest.raises(ValueError, match="needs values"):
        Gate("zone", "in").validate()
    with pytest.raises(ValueError, match="needs lo"):
        Gate("x", "ge").validate()
    with pytest.raises(ValueError, match="needs both"):
        Gate("x", "between", lo=1.0).validate()


# --------------------------------------------------------- the safety rule

def test_gating_on_a_reporting_only_column_is_refused():
    """`move_ratio` is the realised move over the priced move. Gating an entry
    on it is foresight of exactly the thing being predicted."""
    for col in R.REPORTING_ONLY:
        if col == "date":
            continue
        with pytest.raises(ValueError, match="reporting-only"):
            Gate(col, "ge", lo=1.0).validate()


def test_the_rule_safe_regime_columns_are_gateable():
    for col in ("vix", "prior_vol", "straddle"):
        Gate(col, "ge", lo=0.0).validate()


# ------------------------------------------------------------------- sets

def test_an_empty_gateset_passes_everything(feats):
    """Switching an indicator off is the same object with one fewer gate, so
    the ungated baseline is a point in the search space."""
    assert G.NONE.mask(feats).all()
    assert not G.NONE


def test_all_and_any(feats):
    lo = Gate("zone", "in", values=(I.BL, I.UL, I.L))
    tight = Gate("s_pctb", "le", hi=-0.8)
    both = G.all_of(lo, tight).mask(feats)
    either = G.any_of(lo, tight).mask(feats)
    assert both.sum() <= either.sum()
    assert (both & ~either).sum() == 0


def test_a_bad_mode_is_refused():
    with pytest.raises(ValueError, match="'all' or 'any'"):
        GateSet(gates=(Gate("x", "ge", lo=1),), mode="most").validate()


def test_a_gateset_is_hashable_so_it_can_key_a_sweep():
    """A gate holding a lambda would be unhashable and uncacheable, and could
    not be written into a results table."""
    a = G.all_of(G.zone_in(I.BL), G.pctb("s_pctb", hi=-0.5))
    b = G.all_of(G.zone_in(I.BL), G.pctb("s_pctb", hi=-0.5))
    assert hash(a) == hash(b) and a == b
    assert len({a, b}) == 1


# ------------------------------------------------------------ minute mask

def test_the_minute_mask_marks_only_bar_opens(feats):
    """The metrics are valid at a bar's open and nowhere else inside it."""
    gs = G.all_of(G.zone_in(I.BL, I.UL, I.L))
    m = G.minute_mask(feats, gs, 391)
    opens = set(feats["minute"].to_list())
    assert set(np.flatnonzero(m)).issubset(opens)
    assert m.sum() == int(gs.mask(feats).sum())


def test_an_empty_gateset_flags_no_minutes(feats):
    assert G.minute_mask(feats, G.NONE, 391).sum() == 0


def test_the_mask_lines_up_with_the_bars_it_came_from(feats):
    gs = G.all_of(Gate("zone", "in", values=(I.M,)))
    m = G.minute_mask(feats, gs, 391)
    for row, hit in zip(feats.iter_rows(named=True), gs.mask(feats)):
        assert bool(m[row["minute"]]) == bool(hit)
