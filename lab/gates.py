#!/usr/bin/env python3
"""
gates.py — indicator conditions, for entries and for exits.

A **gate** is one predicate over one indicator column, evaluated on bar opens.
Gates compose into a `GateSet` that is `all` or `any`, and the whole thing is a
frozen, hashable, flat-scalar object — which is the only reason an optimiser can
search it. A gate holding a lambda would be unhashable, uncacheable, and
impossible to write into a results table.

WHERE THEY APPLY
----------------
**Entry** — the signal is only emitted on bars where the gate passes. This is
the specification's "Indicator Toggles & Thresholds: Entry".

**Exit** — the position is closed on the first bar open after entry where the
gate passes. Note *bar open*: the metrics are only valid there, and the specification
puts entry and exit assessment on the 5/15/30/60m bars while reserving the
1-minute option data for limit and stop triggers. An indicator exit crosses the
spread, because like a stop it is a decision to be out rather than a resting
order.

WHAT MAY BE GATED ON
--------------------
Any column of the indicator frame, plus the rule-safe regime columns (opening
straddle, VIX, trailing volatility). Never the reporting-only regime columns —
`regimes.REPORTING_ONLY` is computed from the realised day, and gating on it is
foresight of exactly the thing being predicted. `Gate.validate` refuses them by
name rather than trusting the caller to remember.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np
import polars as pl

from . import regimes as _regimes

#: Numeric comparisons and set membership. Enough for every gate the specification
#: describes, and small enough that an optimiser can enumerate the choice.
OPS = ("ge", "le", "between", "outside", "in", "not_in")

#: Which price series a gate reads.
SPOT, OPTION = "spot", "option"
CHARTS = (SPOT, OPTION)


@dataclass(frozen=True, slots=True)
class Gate:
    """One condition on one indicator column."""

    column: str
    op: str = "ge"
    lo: Optional[float] = None
    """Threshold for `ge`, lower bound for `between` / `outside`.

    `None` and not NaN for "unset": `NaN != NaN`, so two identical gates built
    with NaN bounds compare unequal and hash to the same bucket — which would
    silently defeat every cache key and every dedup in a sweep."""

    hi: Optional[float] = None
    """Threshold for `le`, upper bound for `between` / `outside`."""

    values: Tuple[int, ...] = ()
    """Category codes for `in` / `not_in` — zones, slope types, relations."""

    chart: str = "spot"
    """Which chart the column is read from: the SPX series (`spot`) or the
    traded option's own price path (`option`).

    The specification asks for both and for **mixing them on one order** — "we use price
    action on the SPX to trigger an entry, but the exit is based on Bollingers
    from the option's price action". The metric names are identical because
    `indicators.metrics` computes both, so `s_pctb` means the same thing on
    either; this field is the only thing that says which."""

    def validate(self) -> "Gate":
        if self.chart not in CHARTS:
            raise ValueError(f"chart must be one of {CHARTS}, not {self.chart!r}")
        if self.op not in OPS:
            raise ValueError(f"unknown op {self.op!r}; expected one of {OPS}")
        if self.column in _regimes.REPORTING_ONLY:
            raise ValueError(
                f"{self.column!r} is computed from the realised day and is "
                "reporting-only; gating on it is foresight of the exact thing "
                "being predicted. See regimes.SAFE_FOR_RULES.")
        if self.op in ("in", "not_in") and not self.values:
            raise ValueError(f"{self.op} needs values")
        if self.op == "ge" and self.lo is None:
            raise ValueError("ge needs lo")
        if self.op == "le" and self.hi is None:
            raise ValueError("le needs hi")
        if self.op in ("between", "outside") and (
                self.lo is None or self.hi is None):
            raise ValueError(f"{self.op} needs both lo and hi")
        return self

    def mask(self, frame: pl.DataFrame) -> np.ndarray:
        """Per-row boolean. A null or NaN metric NEVER passes — an absent
        indicator is not a satisfied condition."""
        self.validate()
        if self.column not in frame.columns:
            raise KeyError(f"{self.column!r} is not in the frame; "
                           f"have {sorted(frame.columns)[:8]}...")
        col = frame[self.column].to_numpy()
        if self.op in ("in", "not_in"):
            hit = np.isin(col, np.asarray(self.values))
            return hit if self.op == "in" else ~hit
        x = col.astype(float)
        ok = np.isfinite(x)
        out = np.zeros(x.size, dtype=bool)
        if self.op == "ge":
            np.greater_equal(x, self.lo, out=out, where=ok)
        elif self.op == "le":
            np.less_equal(x, self.hi, out=out, where=ok)
        elif self.op == "between":
            out[ok] = (x[ok] >= self.lo) & (x[ok] <= self.hi)
        else:                                    # outside
            out[ok] = (x[ok] < self.lo) | (x[ok] > self.hi)
        return out

    def label(self) -> str:
        p = f"{self.column}" if self.chart == SPOT else f"opt:{self.column}"
        if self.op in ("in", "not_in"):
            return f"{p} {self.op} {list(self.values)}"
        if self.op == "ge":
            return f"{p}>={self.lo:g}"
        if self.op == "le":
            return f"{p}<={self.hi:g}"
        return f"{p} {self.op} [{self.lo:g},{self.hi:g}]"


@dataclass(frozen=True, slots=True)
class GateSet:
    """Zero or more gates, combined. Empty means "no condition" and passes all.

    Empty passing everything is what makes a toggle a toggle: switching an
    indicator off is the same object with one fewer gate, so the ungated
    baseline is a point in the search space rather than a separate code path."""

    gates: Tuple[Gate, ...] = ()
    mode: str = "all"

    def __bool__(self) -> bool:
        return bool(self.gates)

    def validate(self) -> "GateSet":
        if self.mode not in ("all", "any"):
            raise ValueError(f"mode must be 'all' or 'any', not {self.mode!r}")
        for g in self.gates:
            g.validate()
        return self

    def mask(self, frame: pl.DataFrame) -> np.ndarray:
        self.validate()
        if not self.gates:
            return np.ones(frame.height, dtype=bool)
        masks = [g.mask(frame) for g in self.gates]
        return (np.logical_and.reduce(masks) if self.mode == "all"
                else np.logical_or.reduce(masks))

    def label(self) -> str:
        if not self.gates:
            return "-"
        joiner = " & " if self.mode == "all" else " | "
        return joiner.join(g.label() for g in self.gates)

    def for_chart(self, chart: str) -> "GateSet":
        """The sub-set reading one chart. Empty if none do.

        A mixed set is split rather than evaluated together because the two
        frames have different row counts and different readiness: the SPX
        series is continuous across sessions and has a band from the first
        minute, while an option's chart starts when its contract does.

        Note this makes `mode="any"` across charts NOT expressible — an `any`
        set spanning both would have to be true on a chart the other half
        cannot see. `all` splits cleanly and is what a mixed rule means."""
        gates = tuple(g for g in self.gates if g.chart == chart)
        if gates and len(gates) != len(self.gates) and self.mode == "any":
            raise ValueError(
                "an 'any' GateSet cannot span both charts: the two frames have "
                "different rows, so the disjunction is not defined. Use 'all', "
                "or keep each chart in its own set.")
        return GateSet(gates=gates, mode=self.mode)


def minute_mask(frame: pl.DataFrame, gateset: GateSet, n_minutes: int
                ) -> np.ndarray:
    """Bar-level gate results, projected onto the session's minute axis.

    True only at a bar's OPENING minute. The metrics are valid there and
    nowhere else inside the bar, and the specification puts entry and exit assessment
    on the bars while reserving the 1-minute option data for limit and stop
    triggers — so an indicator exit can only fire when a bar opens."""
    out = np.zeros(n_minutes, dtype=bool)
    if not gateset:
        return out
    hit = gateset.mask(frame)
    minutes = frame["minute"].to_numpy()
    ok = (minutes >= 0) & (minutes < n_minutes) & hit
    out[minutes[ok]] = True
    return out


# ------------------------------------------------------- convenience builders

def zone_in(*zones: int, chart: str = SPOT) -> Gate:
    return Gate(column="zone", op="in", values=tuple(int(z) for z in zones),
                chart=chart)


def pctb(column: str = "s_pctb", lo: Optional[float] = None,
         hi: Optional[float] = None, chart: str = SPOT) -> Gate:
    """Centred %b band. `lo` only -> `ge`; `hi` only -> `le`; both -> `between`."""
    if lo is not None and hi is not None:
        return Gate(column=column, op="between", lo=lo, hi=hi, chart=chart)
    if lo is not None:
        return Gate(column=column, op="ge", lo=lo, chart=chart)
    if hi is None:
        raise ValueError("pctb needs at least one bound")
    return Gate(column=column, op="le", hi=hi, chart=chart)


def all_of(*gates: Gate) -> GateSet:
    return GateSet(gates=tuple(gates), mode="all")


def any_of(*gates: Gate) -> GateSet:
    return GateSet(gates=tuple(gates), mode="any")


NONE = GateSet()
