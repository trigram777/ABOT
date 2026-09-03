#!/usr/bin/env python3
"""
broker.py — buy, sell, and settle. There is no structure in this file.

WHY THIS EXISTS IN THIS SHAPE
-----------------------------
The previous engine knew what an iron condor was: it had a `Condor`, a
`Strangle`, and six named `Piece`s, and every question had to be phrased in
those terms. That was right when there was one trade. It is wrong now, because
the study ahead sweeps naked longs, verticals up to three strikes wide, ITM
short spreads, butterflies, and long structures that get COVERED or SHORTED
into spreads halfway through the session — and enumerating those as types would
mean a new type for every idea.

So the vocabulary here is four verbs — buy, sell, close, settle — and one noun,
the position. A condor is four calls to `submit`. A COVER is one more. The
engine never learns what either is called.

THE THREE INVARIANTS
--------------------
1. **Cash is signed, money in positive.** A sale credits, a purchase debits, a
   fee always debits. Session P&L is then `sum(every cash flow) + settlement`,
   with no basis arithmetic anywhere in the total. Basis is tracked too, but
   only because W and L triggers are quoted as multiples of the entry price —
   never because the P&L needs it.

2. **An order is atomic.** A four-leg condor either goes on or it does not.
   Filling three legs of it and refusing the fourth would build a position no
   broker would have given you, and the naked leg it leaves behind would
   dominate every statistic it appeared in.

3. **Commission is charged PER LEG, and expiry is free.** Both are measured,
   not assumed — see COSTS.md. The per-leg minimum is why a structure's leg
   count is the most expensive decision in it, and the free expiry is why every
   early exit starts the day in deficit against doing nothing.

WHAT IT DELIBERATELY DOES NOT MODEL
-----------------------------------
Queue position, partial fills, and the walk of a limit order down a ladder. The
dataset is a minute grid; a fill model that pretended to know where in the
minute an order rested would be inventing precision the data cannot support.
The honest expression of that uncertainty is the mid/cross bracket, which is
reported on everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .fills import BUY, SELL, CROSS, FillModel
from .session import CALL, PUT, Contract, Session

# Charged per leg. See COSTS.md: the IBKR minimum applies per leg, not per
# order, so a four-leg structure at one lot pays it four times.
IBKR_PER_CONTRACT = 0.65
IBKR_MIN_PER_LEG = 1.00
THIRD_PARTY = {"XSP": 0.22, "SPXW_LOW": 0.54, "SPXW_HIGH": 0.63}
SPXW_PREMIUM_BREAK = 1.00

# THE XSP BULK SURCHARGE DOES NOT EXIST. It was refused rather than guessed
# until the trader filled multi-lot orders on both symbols and recorded the
# prints, which reconcile to the cent. XSP at 10
# contracts paid $8.70 = max(1.00, 0.65x10) + 10x0.22, the same formula as one
# contract. The threshold constant is kept as a record of what was tested.
XSP_MEASURED_TO = 10


def leg_commission(symbol: str, qty: int, premium: float) -> float:
    """One leg of one order, in dollars. Reproduces every observed IBKR print.

    **There is no multi-lot price break on either symbol.** Verified 25 Aug 2026
    against eight live prints spanning 1, 2, 10 and 12 contracts on SPXW and XSP:
    every one reconciles to the cent with this formula and per-contract cost is
    FLAT from two contracts up ($1.28 on SPXW above the premium break, $0.87 on
    XSP). The only quantity effect is the $1.00 per-leg minimum, and it only
    bites at one contract — which is why **one contract is the single worst size
    to trade**, at $1.63 a leg against $1.28.

    Quantity therefore amortises nothing beyond two, and a study that doubles
    size doubles commission exactly."""
    if qty <= 0:
        return 0.0
    sym = symbol.upper()
    if sym == "XSP":
        third = THIRD_PARTY["XSP"]
    else:
        third = THIRD_PARTY["SPXW_HIGH" if premium >= SPXW_PREMIUM_BREAK
                            else "SPXW_LOW"]
    return max(IBKR_MIN_PER_LEG, IBKR_PER_CONTRACT * qty) + qty * third


MULTIPLIER = 100.0


# --------------------------------------------------------------------- records

@dataclass(frozen=True, slots=True)
class Fill:
    """One leg of one order, done."""

    minute: int
    contract: Contract
    qty: int                 # signed: + bought, - sold
    price: float             # per share
    commission: float
    tag: str

    @property
    def cash(self) -> float:
        """Signed dollars, money in positive, commission already inside."""
        return -self.qty * self.price * MULTIPLIER - self.commission

    @property
    def notional(self) -> float:
        return abs(self.qty) * self.price * MULTIPLIER


@dataclass(frozen=True, slots=True)
class Rejection:
    minute: int
    reason: str
    legs: Tuple[Tuple[Contract, int], ...]
    tag: str


@dataclass(frozen=True, slots=True)
class Order:
    """The result of one `submit`. Either every leg filled or none did."""

    minute: int
    fills: Tuple[Fill, ...]
    rejection: Optional[Rejection]
    tag: str

    def __bool__(self) -> bool:
        return self.rejection is None

    @property
    def cash(self) -> float:
        return sum(f.cash for f in self.fills)

    @property
    def commission(self) -> float:
        return sum(f.commission for f in self.fills)

    @property
    def price(self) -> float:
        """Net per-share price of the package, signed the way a combo is quoted:
        POSITIVE means the package was sold for a credit, negative a debit."""
        return sum(-f.qty * f.price for f in self.fills)

    @property
    def reason(self) -> str:
        return self.rejection.reason if self.rejection else ""


@dataclass(slots=True)
class Position:
    """Net holding in one contract, with the basis a W/L trigger measures from."""

    contract: Contract
    qty: int = 0                  # signed
    avg_price: float = 0.0        # per share, of the CURRENT signed position
    opened_at: int = -1
    commission_paid: float = 0.0

    @property
    def is_long(self) -> bool:
        return self.qty > 0

    def basis(self) -> float:
        """Signed dollars paid to establish what is held. Negative = credit taken."""
        return -self.qty * self.avg_price * MULTIPLIER


# ---------------------------------------------------------------------- broker

class Broker:
    """One session's account. Positions, cash, ledger, settlement.

    Constructed per session per parameter cell; it holds no data of its own
    beyond a reference to the `Session`, so constructing one is free."""

    __slots__ = ("session", "fill_model", "positions", "orders", "rejections",
                 "cash", "_settled", "settlement_cash")

    def __init__(self, session: Session, fill_model: FillModel = CROSS):
        self.session = session
        self.fill_model = fill_model
        self.positions: Dict[Contract, Position] = {}
        self.orders: List[Order] = []
        self.rejections: List[Rejection] = []
        self.cash: float = 0.0
        self._settled = False
        self.settlement_cash: float = 0.0

    # ------------------------------------------------------------- submitting
    def submit(self, legs: Sequence[Tuple[Contract, int]], minute: int,
               tag: str = "", limit: Optional[float] = None) -> Order:
        """Trade a package atomically. `legs` are (contract, signed qty).

        Every leg is priced against the book at `minute` — which is the book at
        the START of that minute, so a rule deciding at `minute` is filling on
        what it could see. If any leg cannot be priced, nothing trades and the
        whole package is recorded as a rejection with the leg that killed it.

        `limit` is the net per-share price of an order that was ALREADY RESTING
        — same sign convention as `Order.price`, positive for a credit. It
        fills at exactly that price provided the book at `minute` could have
        supported it, and is refused otherwise. This is what a take-profit
        limit is: the moment the market reaches $4.00 the order sitting at
        $4.00 trades at $4.00, and booking the $8.00 the minute grid happens to
        show would collect a windfall the resting order never received."""
        if self._settled:
            raise RuntimeError("the session has settled; no further orders")
        legs = [(c, int(q)) for c, q in legs if q]
        if not legs:
            return self._reject(minute, "empty order", legs, tag)

        priced: List[Tuple[Contract, int, float]] = []
        for c, q in legs:
            bid, ask = self.session.quote(c, minute)
            if bid is None and ask is None and self.session.column(c.right, c.strike) is None:
                return self._reject(minute, f"{c} is not in the chain", legs, tag)
            px = self.fill_model.price(bid, ask, BUY if q > 0 else SELL)
            if px is None:
                return self._reject(minute, f"{c} has no tradeable quote "
                                            f"({bid}/{ask})", legs, tag)
            priced.append((c, q, px))

        if limit is not None:
            market = sum(-q * px for _, q, px in priced)
            if market < limit - 1e-9:
                return self._reject(minute, f"limit {limit:+.2f} not reachable; "
                                            f"book offers {market:+.2f}", legs, tag)
            # Spread the give-back across legs by size, so the net is exactly
            # the limit and each leg still carries a plausible per-leg price
            # (the commission band cares about it).
            total_qty = sum(abs(q) for _, q, _ in priced)
            shift = (limit - market) / total_qty
            adjusted = [(c, q, px - (1 if q > 0 else -1) * shift)
                        for c, q, px in priced]
            if any(px < self.fill_model.min_price for _, _, px in adjusted):
                return self._reject(minute, f"limit {limit:+.2f} prices a leg "
                                            "below one tick", legs, tag)
            priced = adjusted

        fills = []
        for c, q, px in priced:
            comm = leg_commission(c.symbol, abs(q), px)
            f = Fill(minute=minute, contract=c, qty=q, price=px,
                     commission=comm, tag=tag)
            self._apply(f)
            fills.append(f)

        order = Order(minute=minute, fills=tuple(fills), rejection=None, tag=tag)
        self.orders.append(order)
        return order

    def buy(self, contract: Contract, qty: int, minute: int, tag: str = "") -> Order:
        return self.submit([(contract, abs(qty))], minute, tag)

    def sell(self, contract: Contract, qty: int, minute: int, tag: str = "") -> Order:
        return self.submit([(contract, -abs(qty))], minute, tag)

    def close(self, contract: Contract, minute: int, qty: Optional[int] = None,
              tag: str = "close") -> Order:
        """Flatten a holding, or part of one. A no-op if nothing is held."""
        p = self.positions.get(contract)
        if p is None or not p.qty:
            return self._reject(minute, f"{contract} is not held", [], tag)
        n = p.qty if qty is None else int(np.sign(p.qty)) * min(abs(qty), abs(p.qty))
        return self.submit([(contract, -n)], minute, tag)

    def fill_at(self, contract: Contract, qty: int, minute: int, price: float,
                tag: str = "triggered") -> Order:
        """One leg at a STATED price, not at the book — a TRIGGERED order.

        **This is the one place a fill is not taken from the quote, and it
        exists because a trigger is not a limit.** `submit(limit=...)` models an
        order that was ALREADY RESTING at its price and the market came to it,
        so it refuses to fill when the book is worse than the limit. A trigger
        is the opposite: the market went PAST the level and the order fires
        into it — a buy-stop's market is above its level, a sell-trigger's is
        below. `limit` would refuse the second and silently accept the first,
        which is exactly the asymmetry that has to be avoided.

        What that chase costs cannot be read off a minute grid at all —
        filling at the level assumes it costs nothing, filling at the breaching
        quote assumes a full minute of latency, and the truth is between. So
        the caller owns the assumption and must report the bracket. Everything
        else is still enforced: the contract has to be in the chain and have a
        quote at `minute`, because nothing fills against a book that is not
        there; commission comes from the same schedule; and cash, basis and
        reconciliation are the broker's as usual."""
        if self._settled:
            raise RuntimeError("the session has settled; no further orders")
        if not qty:
            return self._reject(minute, "empty order", [], tag)
        if self.session.column(contract.right, contract.strike) is None:
            return self._reject(minute, f"{contract} is not in the chain", [], tag)
        bid, ask = self.session.quote(contract, minute)
        if bid is None and ask is None:
            return self._reject(minute, f"{contract} has no book at {minute}",
                                [(contract, qty)], tag)
        if not (price > 0):
            return self._reject(minute, f"triggered price {price} is not positive",
                                [(contract, qty)], tag)
        f = Fill(minute=minute, contract=contract, qty=int(qty),
                 price=float(price),
                 commission=leg_commission(contract.symbol, abs(int(qty)), price),
                 tag=tag)
        self._apply(f)
        order = Order(minute=minute, fills=(f,), rejection=None, tag=tag)
        self.orders.append(order)
        return order

    def stop_out(self, contract: Contract, minute: int, price: float,
                 tag: str = "stop") -> Order:
        """Flatten a holding at a STATED price. See `fill_at`."""
        p = self.positions.get(contract)
        if p is None or not p.qty:
            return self._reject(minute, f"{contract} is not held", [], tag)
        return self.fill_at(contract, -p.qty, minute, price, tag)


    def close_all(self, minute: int, tag: str = "close_all") -> Order:
        """Flatten everything in one package — one order, priced together, and
        refused as a whole if any leg has gone untradeable. That refusal is the
        point: a rule that cannot get out of one leg has not got out."""
        legs = [(p.contract, -p.qty) for p in self.positions.values() if p.qty]
        if not legs:
            return self._reject(minute, "flat", [], tag)
        return self.submit(legs, minute, tag)

    def _reject(self, minute: int, reason: str,
                legs: Iterable[Tuple[Contract, int]], tag: str) -> Order:
        r = Rejection(minute=minute, reason=reason, legs=tuple(legs), tag=tag)
        self.rejections.append(r)
        o = Order(minute=minute, fills=(), rejection=r, tag=tag)
        self.orders.append(o)
        return o

    def _apply(self, f: Fill) -> None:
        self.cash += f.cash
        p = self.positions.get(f.contract)
        if p is None:
            p = self.positions[f.contract] = Position(f.contract)
        p.commission_paid += f.commission
        old, new = p.qty, p.qty + f.qty
        if old == 0 or (old > 0) == (f.qty > 0):
            # Opening or adding: volume-weight the basis.
            p.avg_price = ((abs(old) * p.avg_price + abs(f.qty) * f.price)
                           / (abs(old) + abs(f.qty)))
            if old == 0:
                p.opened_at = f.minute
        elif abs(f.qty) >= abs(old):
            # Closed out, and possibly flipped. A flip starts a new basis.
            p.avg_price = f.price if new else 0.0
            p.opened_at = f.minute if new else -1
        # Reducing without closing leaves the basis alone, which is what a
        # partial close means: the remaining lot cost what it always cost.
        p.qty = new

    # ------------------------------------------------------------- valuation
    def held(self) -> List[Position]:
        return [p for p in self.positions.values() if p.qty]

    def contracts_open(self) -> int:
        return sum(abs(p.qty) for p in self.positions.values())

    def mark(self, minute: int, mode: str = "mid") -> Optional[float]:
        """Dollar value of what is held, in the same sign convention as cash:
        POSITIVE is money you would receive by flattening.

        `mid` values at the midpoint — the fair mark, for measurement.
        `liquidate` values at the side that would trade, longs at the bid and
        shorts at the ask — what flattening would actually pay, and therefore
        what a stop should be compared against. Commission is NOT included;
        `liquidation_cost` is the separate question.

        None if any held leg has no price: a partial total on a spread reads as
        a smaller loss than reality, which is the one lie a risk number must
        never tell."""
        total = 0.0
        for p in self.held():
            bid, ask = self.session.quote(p.contract, minute)
            if bid is None or ask is None:
                return None
            if mode == "mid":
                px = (bid + ask) / 2.0
            elif mode == "liquidate":
                px = bid if p.qty > 0 else ask
            else:
                raise ValueError(f"unknown mark mode {mode!r}")
            total += p.qty * px * MULTIPLIER
        return total

    def equity(self, minute: int, mode: str = "mid") -> Optional[float]:
        """Cash taken so far plus what is still on. The intraday P&L curve."""
        m = self.mark(minute, mode)
        return None if m is None else self.cash + m

    def price_of(self, legs: Sequence[Tuple[Contract, int]], minute: int,
                 mode: str = "mid") -> Optional[float]:
        """Per-share net price of SUBMITTING this package, without trading it.

        Same sign convention as `Order.price`: **positive means the package
        would be sold for a credit**, negative a debit. So the value of closing
        something you hold is `price_of([(c, -q) for c, q in held])` — the legs
        you would send, not the legs you are holding.

        `mode="cross"` prices each leg at the side that would actually trade
        for the SUBMITTED direction: a buy pays the ask, a sell receives the
        bid. (This had the two backwards, which made a closing package price
        out at the wrong end of every spread.)"""
        total = 0.0
        for c, q in legs:
            bid, ask = self.session.quote(c, minute)
            if bid is None or ask is None:
                return None
            if mode == "mid":
                px = (bid + ask) / 2.0
            elif mode == "cross":
                px = ask if q > 0 else bid       # buys pay up, sells concede
            else:
                raise ValueError(f"unknown price mode {mode!r}")
            total += -q * px
        return total

    # ------------------------------------------------------------- settlement
    def settle(self, price: Optional[float] = None) -> float:
        """Expire whatever is left, at the official close, for nothing.

        Cash-settled index options pay intrinsic and carry no commission — the
        OCC settlement rows in the broker statement have none. That asymmetry
        is the do-nothing baseline every exit rule has to beat."""
        if self._settled:
            return self.settlement_cash
        s = self.session.settle if price is None else price
        cash = 0.0
        for p in self.held():
            cash += p.qty * self.session.intrinsic(p.contract, s) * MULTIPLIER
        self.settlement_cash = cash
        self.cash += cash
        self._settled = True
        return cash

    @property
    def settled(self) -> bool:
        return self._settled

    # ------------------------------------------------------------- reporting
    @property
    def pnl(self) -> float:
        """Everything that happened, in dollars. Only meaningful after settle()
        or after the book is flat."""
        return self.cash

    @property
    def commission(self) -> float:
        return sum(o.commission for o in self.orders)

    @property
    def gross(self) -> float:
        """P&L before commission — the number that lies, kept so the size of
        the lie is reportable."""
        return self.cash + self.commission

    def ledger(self) -> List[dict]:
        rows = []
        for o in self.orders:
            if o.rejection is not None:
                rows.append(dict(minute=o.minute, clock=self.session.clock(o.minute),
                                 tag=o.tag, kind="reject", right="", strike=float("nan"),
                                 qty=0, price=float("nan"), commission=0.0, cash=0.0,
                                 reason=o.rejection.reason))
                continue
            for f in o.fills:
                rows.append(dict(minute=f.minute, clock=self.session.clock(f.minute),
                                 tag=f.tag, kind="buy" if f.qty > 0 else "sell",
                                 right=f.contract.right, strike=f.contract.strike,
                                 qty=f.qty, price=f.price, commission=f.commission,
                                 cash=f.cash, reason=""))
        if self._settled:
            rows.append(dict(minute=self.session.n_minutes - 1, clock="settle",
                             tag="settle", kind="settle", right="", strike=float("nan"),
                             qty=0, price=self.session.settle, commission=0.0,
                             cash=self.settlement_cash, reason=""))
        return rows
