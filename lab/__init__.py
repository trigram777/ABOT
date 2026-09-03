"""
lab — a general SPXW option execution engine for research.

The previous `engine/` package knew what an iron condor was. This one does not.
It buys and sells options at strikes, and every structure a study wants — a
naked long, a vertical, a condor, a butterfly, a strangle covered later into a
spread — is stitched together from those two verbs by the strategy that wants
it. That is the whole design change: structures are compositions, not types.

Layers, bottom up:

    session.py   one trading day as dense [minute, strike] arrays, plus the
                 SPX spot bars at 1/5/15/30/60m and the settlement close
    fills.py     what a price is: the mid/cross bracket, tick concession
    broker.py    positions, orders, commissions, settlement, ledger
"""

__all__ = ["session", "fills", "broker"]
