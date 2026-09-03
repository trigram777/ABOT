#!/usr/bin/env python3
"""
runner.py — a method plus an exit policy, run over sessions.

A **method** is anything that can answer "what would you open, and when". It is
a protocol, not a base class:

    class Method(Protocol):
        name: str
        def signals(self, sess: Session) -> list[Signal]: ...

`Signal` says which right, how the strike is chosen, and how big. Everything
after that — opening it, watching W and L, covering, shorting, settling — is
this module's job and is identical for every method. That is the point: five
very different methods differ in what they open and when, and in nothing else.

WHAT A RUN PRODUCES
-------------------
One row per TRADE, not per session. Entry hour is a column, because every
assessment wants bucketing by it and a per-session total cannot be re-bucketed
afterwards. So is the opening zone, so an indicator gate can be
fitted after the fact instead of re-running the sweep for each candidate.

THE RECONCILIATION
------------------
The trades' P&L must sum to the broker's, exactly. Trades own their own cash so
that the row-level numbers are attributable; the broker owns the account. If
those two ever disagree, one of them is wrong, and `SessionRun.reconciles`
is checked in the tests rather than assumed.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Protocol, Sequence, Tuple

import numpy as np
import polars as pl

from .broker import Broker
from .exits import (CONVERTED, ExitPolicy, SETTLED, Trade, manage, settle)
from . import indicators as _I
from .gates import OPTION, SPOT, minute_mask
from .fills import FillModel
from .select import Leg, tradeable
from .session import CALL, PUT, Session, calendar, cached


@dataclass(frozen=True, slots=True)
class Signal:
    """One structure a method wants to open, at one minute.

    The legs are CONCRETE — the method has already chosen the strikes, because
    choosing them is most of what distinguishes one method from another. A
    naked long is one leg, a credit vertical is two, a condor is four, and a
    method holding two independent longs emits two separate one-leg signals
    because they are managed independently rather than as a structure."""

    minute: int
    """Already resolved to a live minute by the method — see `select.live_minute`."""

    bar_minute: int = -1
    """The minute of the BAR that produced the signal, before the liveness gate
    moved the fill. The join key back to the indicator frame: `minute` can be a
    minute or two later and would miss the bar row entirely."""

    legs: Tuple[Leg, ...] = ()
    tag: str = ""
    zone: int = -1
    """The indicator state that produced it, carried through to the result row
    so gates can be fitted after the sweep rather than inside it."""

    credit: bool = False
    """Opened for a credit. Such a structure carries no W or L — both are
    long-only — and is held to expiry."""

    target_miss: float = 0.0
    entry_ratio: float = float("nan")
    """For a credit structure: the achieved credit as a fraction of one wing's
    width. What a credit structure is actually selected on."""

    chart_leg: int = 0
    """Which leg's own chart an option-chart gate reads. Unambiguous for a
    naked long, which is the only shape that currently uses one."""


class Method(Protocol):
    name: str

    def signals(self, sess: Session) -> List[Signal]:
        ...

    def revise(self, sess: Session, sig: Signal,
               open_trades: Sequence["Trade"]) -> Optional[Signal]:
        """Last look at a candidate, knowing what the session already holds.

        **Optional.** A method that does not define it is a pure function of
        the session, which is what a method is unless it needs to see its own
        open positions.

        `signals()` still runs once per day and stays cacheable across exit
        policies — it proposes. This decides, and it is the only place a method
        may depend on its own open positions. Return the signal to submit
        (possibly a DIFFERENT one, re-selected under a constraint the open
        positions imply), or `None` to decline.

        The motivating case is a paired two-sided method: at most one short
        call vertical and one short put vertical per session, the second may
        only open while the first is **currently** out of the money, and the
        two short strikes may never cross — below `Kc == Kp` both finish in the
        money for any settlement between them and the loss has a floor whatever
        price does. The first two are declines; the third is a RE-SELECTION,
        because refusing a session that had a legal spread available would
        understate the method rather than model it."""
        return sig

    def features(self, sess: Session):
        """The indicator frame this method reads, or None.

        The METHOD owns which timeframe and which band configuration; the
        POLICY owns which condition closes a position. An exit gate therefore
        needs both, and this is how the runner gets the half the policy does
        not carry."""
        return None


@dataclass(slots=True)
class SessionRun:
    date: dt.date
    trades: List[Trade]
    broker: Broker
    refused: int = 0

    @property
    def pnl(self) -> float:
        return self.broker.pnl

    @property
    def reconciles(self) -> bool:
        return abs(sum(t.pnl for t in self.trades) - self.broker.pnl) < 1e-6

    def rows(self) -> List[dict]:
        out = []
        for t in self.trades:
            out.append(dict(
                date=self.date,
                tag=t.tag,
                entry_minute=t.entry_minute,
                bar_minute=t.bar_minute,
                entry_clock=self.broker.session.clock(t.entry_minute),
                entry_hour=self.broker.session.minutes[t.entry_minute].hour,
                legs=t.entry_legs,
                credit=t.credit,
                entry_ratio=float("nan"),
                entry_price=t.entry_price,
                target_miss=t.target_miss,
                open_zone=t.open_zone,
                exit_minute=t.exit_minute,
                exit_reason=t.exit_reason,
                exit_refused=t.exit_refused,
                exit_price=t.exit_price,
                converted_to=t.converted_to,
                held_to_expiry=t.exit_minute is None,
                pnl=t.pnl,
                fees=t.fees,
            ))
        return out


def run_session(sess: Session, method: Method, policy: ExitPolicy,
                model: FillModel,
                signals: Optional[Sequence[Signal]] = None) -> SessionRun:
    """One method, one exit policy, one day.

    `signals` may be passed in already computed. A W x L grid runs the same
    method against dozens of policies on the same day, and the signals do not
    depend on the policy — recomputing them per cell would make the indicator
    lookup the dominant cost of the sweep."""
    broker = Broker(sess, model)
    trades: List[Trade] = []
    refused = 0

    # The SPX half of an exit gate is the same for every trade in the session,
    # so it is built once. The option half is per contract and is built when
    # the trade exists — that is the point of it.
    spot_exit = policy.exit_gate.for_chart(SPOT) if policy.exit_gate else None
    opt_exit = policy.exit_gate.for_chart(OPTION) if policy.exit_gate else None
    spot_exit_minutes = None
    if spot_exit:
        feats = method.features(sess) if hasattr(method, "features") else None
        if feats is None:
            raise ValueError(
                f"{getattr(method, 'name', method)} has an exit gate on the "
                "spot chart but exposes no features(); a gate needs the frame "
                "it reads")
        spot_exit_minutes = minute_mask(feats, spot_exit, sess.n_minutes)

    entry_opt = None
    eg = getattr(method, "entry_gate", None)
    if eg:
        entry_opt = eg.for_chart(OPTION) or None
    chart = getattr(method, "option_chart", None)

    revise = getattr(method, "revise", None)

    for sig in (method.signals(sess) if signals is None else signals):
        # A method that depends on its own open positions gets the last word,
        # and gets it BEFORE the liveness and gate checks — it may hand back a
        # different structure, whose legs are the ones that have to be
        # tradeable. `trades` is passed in submission order, which is the order
        # the session actually built them in.
        if revise is not None:
            sig = revise(sess, sig, trades)
            if sig is None:
                refused += 1
                continue
        m = sig.minute
        if not sig.legs or not tradeable(sess, sig.legs, m):
            refused += 1
            continue
        # An entry gate on the OPTION's chart can only be evaluated once the
        # strike is known, because until then there is no option to have a
        # chart. The method has chosen it; the runner reads it.
        if entry_opt is not None:
            leg = sig.legs[sig.chart_leg]
            mask = minute_mask(
                _I.option_features(sess, leg.contract(sess), chart or _I.ChartSpec()),
                entry_opt, sess.n_minutes)
            if not mask[m]:
                refused += 1
                continue
        legs = [(l.contract(sess), l.qty) for l in sig.legs]
        o = broker.submit(legs, m, tag=sig.tag or method.name)
        if not o:
            refused += 1
            continue
        lots = max(abs(l.qty) for l in sig.legs)
        t = Trade(tag=o.tag, entry_minute=m, legs=legs,
                  entry_price=abs(o.price) / lots, credit=sig.credit,
                  target_miss=sig.target_miss, open_zone=sig.zone,
                  entry_legs=len(sig.legs), bar_minute=sig.bar_minute)
        t.record(o)
        gm = spot_exit_minutes
        if opt_exit:
            om = minute_mask(
                _I.option_features(sess, sig.legs[sig.chart_leg].contract(sess),
                                   policy.option_chart),
                opt_exit, sess.n_minutes)
            gm = om if gm is None else (gm & om)
        manage(broker, t, policy, gm)
        trades.append(t)

    settle(broker, trades)
    return SessionRun(date=sess.date, trades=trades, broker=broker,
                      refused=refused)


# ------------------------------------------------------------------ parallel

def _one(args):
    day, method, policy, model = args
    sess = cached(day)
    run = run_session(sess, method, policy, model)
    return run.rows()


def run_calendar(method: Method, policy: ExitPolicy, model: FillModel,
                 days: Optional[Sequence[dt.date]] = None,
                 workers: int = 24) -> pl.DataFrame:
    """Every session, in parallel, as one trade-level frame.

    `spawn`, never `fork`: Polars has already started a Rayon pool by the time
    the calendar is read, and a forked child inherits a pool whose threads do
    not exist — N workers parked in `futex_do_wait` at load 0.1, with no error
    and no output. See lab/README.md."""
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    days = list(days) if days is not None else calendar()
    work = [(d, method, policy, model) for d in days]
    rows: List[dict] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_one, work, chunksize=16):
            rows.extend(r)
    return pl.DataFrame(rows) if rows else pl.DataFrame()
