#!/usr/bin/env python3
"""
examples.py — two worked methods, so the engine has something to run.

**Neither of these is traded and neither is meant to be.** They exist to show
the shape a method takes here, and to give `runner`, `sweep`, `scan`, `search`
and their tests a subject. The research methods this engine was built for are
not part of this repository.

WHAT A METHOD IS. Everything below `_Base` is shared: the entry grid, the
window, the indicator frame, and the gate that decides which bars are offered.
A method adds one thing —

    signals(session) -> list[Signal]

— and a `Signal` is *minute plus legs*. The engine takes it from there:
`Broker` prices every leg at that minute, applies the tick concession in the
conceding direction, charges commission per leg on both ends, and settles what
is still open for free. A method never computes a price.

That division is the reason this package is small. Five very different
structures — a naked long, a debit vertical, a credit vertical, a condor, a
straddle — differ in *what they open and when*, and in nothing else.

THE TWO EXAMPLES ARE ONE OF EACH SIGN, deliberately, because the sign is what
changes the arithmetic everywhere downstream:

    ZoneEntry       a DEBIT structure. Pays to open, worth something or nothing
                    at expiry, diagnosed by deleting its best days.
    CreditVertical  a CREDIT structure. Paid to open, capped loss at the width,
                    diagnosed by deleting its WORST days.

`metrics.py` reports both tails for exactly this reason, and reporting only the
first on a short structure measures nothing.

A NOTE ON `ZoneEntry`, because it would be dishonest to present it as a finding:
its entry condition — buy the side the bar opened away from — was carried for a
long time in the programme this engine serves and then **dropped**, because a
random daily pick matched it once a day-permutation null was built. It is here
as a mechanism to demonstrate, not as an edge. See METHODOLOGY §4.3.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from . import indicators as I
from . import select
from .gates import SPOT, GateSet
from .indicators import BandConfig, ChartSpec
from .runner import Signal
from .select import Leg, live_minute
from .session import CALL, PUT, Session

#: "Opened beneath both middle bands" and "above both middle bands", as zones.
BELOW_MIDS = (I.BL, I.UL, I.L)
ABOVE_MIDS = (I.H, I.UH, I.BH)


@dataclass(frozen=True, slots=True)
class _Base:
    """What every method carries. Frozen and flat, so it can key a sweep."""

    timeframe: int = 30
    qty: int = 1
    first_entry: str = "09:30"
    last_entry: str = "15:00"
    """The entry window, which IS the entry-hour axis: restricting it produces
    rows that still carry `entry_hour`, so a bucketed run and a full run stay
    comparable. Report every bucket, never only the best one — seven buckets is
    a seven-way argmax."""

    bands: BandConfig = field(default_factory=BandConfig)
    entry_gate: GateSet = field(default_factory=GateSet)
    option_chart: ChartSpec = field(default_factory=ChartSpec)

    power_timeframe: Optional[int] = None
    """An optional SECOND entry grid, live from `power_from` to `last_entry`.

    A slow chart and a fast one say different things at different distances
    from a cash settlement: an hour before it, a 60-minute bar has no successor
    and a 1-minute bar has sixty. `None` is a single-grid method.

    It is an ENTRY grid only. `features()` still returns the main timeframe's
    frame, so an exit condition reads one chart for the whole session rather
    than changing vocabulary halfway through — the exit is a separate choice
    and should not silently inherit this one."""

    power_from: str = "15:00"
    """Where the main grid stops and the second grid starts. The boundary bar
    belongs to the SECOND grid, so no minute can be offered twice."""

    def features(self, sess: Session):
        """The MAIN timeframe's frame. See `power_timeframe`."""
        return I.for_session(sess.date, self.timeframe, self.bands)

    def _segments(self, sess: Session):
        """(timeframe, first minute, last minute) of each entry grid."""
        lo = sess.minute_of(self.first_entry)
        hi = sess.minute_of(self.last_entry)
        if self.power_timeframe is None:
            return ((self.timeframe, lo, hi),)
        cut = sess.minute_of(self.power_from)
        out = []
        if lo <= cut - 1:
            out.append((self.timeframe, lo, cut - 1))
        if cut <= hi:
            out.append((self.power_timeframe, cut, hi))
        return tuple(out)

    def _bars(self, sess: Session):
        """Bars inside the entry window that pass the SPOT half of the gate.

        Yields in minute order across every segment, so a caller cannot tell
        whether one grid or two produced the sequence."""
        seen = []
        for tf, lo, hi in self._segments(sess):
            feats = I.for_session(sess.date, tf, self.bands)
            allowed = self.entry_gate.for_chart(SPOT).mask(feats)
            for i, row in enumerate(feats.iter_rows(named=True)):
                m, z = row["minute"], row["zone"]
                if not allowed[i] or m is None or z is None or z < 0:
                    continue
                if m < lo or m > hi:
                    continue
                # Never assume the book exists at 09:31. It is true on almost
                # every session and wrong on a handful, and gating on the data
                # instead of on a constant costs nothing.
                live = live_minute(sess, m)
                if live is None:
                    continue
                seen.append((live, row))
        seen.sort(key=lambda p: (p[0], p[1]["minute"]))
        return seen


@dataclass(frozen=True, slots=True)
class ZoneEntry(_Base):
    """DEBIT example — buy the side the bar opened away from.

    A call when the bar opens beneath both middle bands, a put when above both,
    nothing in between. The strike is chosen by DOLLAR PRICE rather than by
    distance, so `target` means the same risk at index 2,250 and at 7,500 —
    which also means the strike's distance from spot varies with volatility,
    and any signal measured against it has to be controlled for that.

    `target = 0` selects the at-the-money strike.

    The entry condition does not survive a day-permutation null. It is here to
    be run, not believed."""

    target: float = 2.00

    short_offset: Optional[float] = None
    """Open a VERTICAL instead of a naked long once the chosen strike is at
    least this many points out of the money, capping the payoff at the width in
    exchange for a smaller debit.

    It is emitted in the SIGNAL rather than converted later, and that is not a
    stylistic choice: the engine then tick-rounds and charges both legs at
    entry. A conversion computed by hand after the fact quietly assumes the raw
    midpoint was transactable."""

    short_strikes: int = 2

    both_sides: bool = False
    """Ignore the zone condition and open BOTH a call and a put on every bar in
    the window. Useful for measuring an exit grid without confounding it with
    an entry rule."""

    name: str = "ZONE"

    def signals(self, sess: Session) -> List[Signal]:
        out = []
        for m, row in self._bars(sess):
            z = row["zone"]
            sides = ((CALL, PUT) if self.both_sides else
                     ((CALL,) if z in BELOW_MIDS else
                      (PUT,) if z in ABOVE_MIDS else ()))
            for right in sides:
                got = select.long_by_price(sess, right, self.target, m)
                if got is None:
                    continue
                leg, miss = got
                legs = (leg._replace_qty(self.qty),)
                tag = f"{self.name} {right}"
                if self.short_offset is not None:
                    spot = sess.spot(m)
                    off = ((leg.strike - spot) if right == CALL
                           else (spot - leg.strike))
                    if np.isfinite(off) and off >= self.short_offset:
                        inner = sess.step(right, leg.strike, -self.short_strikes)
                        if inner is not None:
                            legs = (leg._replace_qty(self.qty),
                                    Leg(right, float(inner), -self.qty))
                            tag = f"{self.name}V {right}"
                out.append(Signal(minute=m, legs=legs, tag=tag, zone=z,
                                  target_miss=miss, bar_minute=row["minute"]))
        return out


@dataclass(frozen=True, slots=True)
class CreditVertical(_Base):
    """CREDIT example — sell a defined-risk vertical of a fixed width.

    `ratio` is the credit demanded as a fraction of the width, so `0.30` on a
    10-point spread asks for $300 against $700 of risk. The selector walks out
    from the money and stops at the last strike that still pays it, which means
    the credit is very nearly PINNED by construction and what actually varies
    is distance from spot. That is worth knowing before sweeping `ratio` as
    though it were a payment threshold: a range chosen for the wrong quantity
    searches the wrong axis.

    Qualified and priced at the MIDPOINT. Crossing the spread on every leg
    independently is not a conservative version of this trade — a combination
    order has its own and much tighter book, and quoting it leg-by-leg silently
    selects different strikes."""

    width: float = 10.0
    ratio: float = 0.30
    side: str = "both"                 # "call", "put" or "both"
    name: str = "CREDIT"

    def signals(self, sess: Session) -> List[Signal]:
        rights = ((CALL, PUT) if self.side == "both"
                  else (CALL,) if self.side == "call" else (PUT,))
        out = []
        for m, row in self._bars(sess):
            for right in rights:
                got = select.credit_vertical(sess, right, m, self.width, self.ratio)
                if got is None:
                    continue
                legs = tuple(leg._replace_qty(self.qty * (1 if leg.qty > 0 else -1))
                             for leg in got[0])
                out.append(Signal(minute=m, legs=legs,
                                  tag=f"{self.name} {right}", zone=row["zone"],
                                  bar_minute=row["minute"]))
        return out


#: What `search.py` dispatches over.
ALL = {"ZONE": ZoneEntry, "CREDIT": CreditVertical}
