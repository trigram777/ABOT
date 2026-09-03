#!/usr/bin/env python3
"""
Pack the exit-policy explorer. Paths compress far better than they look.

Option prices are stored in CENTS as **raw int16, deliberately NOT delta
encoded.** Delta encoding looked like the obvious win and measurement said the
opposite: 4.96 MB raw against 6.08 MB delta-encoded. gzip already exploits the
long constant runs — a quiet book, and the padding past expiry — and differencing
destroys exactly that structure while forcing int32 to avoid overflow at the
NaN boundaries. It also removed a reconstruction bug that could not then exist.
The four series are 49 MB raw and ship as 5 MB.

Indicators are quantised to int8 over their own p0.5..p99.5, as in the sibling
bench — a threshold control has ~254 stops, finer than any threshold worth
setting — and they are stored **per day per timeframe at BAR boundaries only**,
because the index chart is the same for every contract and does not change inside
a bar. 78 values an hour, not 240.
"""
from __future__ import annotations

import base64, gzip, json
from pathlib import Path

import numpy as np
import polars as pl

from demo_calc_bake import COLUMNS, HORIZON, SELECTORS, SLOTS, TFS

OUT = Path("results")
NA = -32768


def main() -> None:
    df = pl.read_parquet(OUT / "demo_calc.parquet").sort(
        ["date", "slot", "sel", "target", "right"])
    days = df["date"].unique().sort().to_list()
    di = {d: i for i, d in enumerate(days)}
    si = {s: i for i, s in enumerate(SLOTS)}
    sel = [f"{k}:{v:g}" for k, v in SELECTORS]
    xi = {s: i for i, s in enumerate(sel)}
    n = df.height

    head = np.empty((n, 8), dtype=np.int16)
    head[:, 0] = [di[d] for d in df["date"].to_list()]
    head[:, 1] = [si[s] for s in df["slot"].to_list()]
    head[:, 2] = [0 if r == "C" else 1 for r in df["right"].to_list()]
    head[:, 3] = [xi[f"{a}:{b:g}"] for a, b in
                  zip(df["sel"].to_list(), df["target"].to_list())]
    cents = lambda c: np.nan_to_num(df[c].to_numpy() * 100, nan=NA).astype(np.int16)
    head[:, 4] = cents("entry")
    head[:, 5] = np.clip(df["offset"].to_numpy(), -3e4, 3e4).astype(np.int16)
    head[:, 6] = df["live"].to_numpy().astype(np.int16)
    head[:, 7] = np.clip(df["miss"].to_numpy() * 1000, -3e4, 3e4).astype(np.int16)

    # the strikes and the settle, needed to value a conversion at expiry
    aux = np.column_stack([
        df["strike"].to_numpy(), df["k_short"].to_numpy(),
        df["k_cover"].to_numpy(), df["settle"].to_numpy()]).astype(np.float32)

    series = {}
    for name in ("mid", "low", "short", "cover"):
        a = np.vstack(df[name].to_numpy())
        series[name] = np.where(np.isfinite(a), np.round(a * 100),
                                NA).astype(np.int16)

    ind = pl.read_parquet(OUT / "demo_calc_ind.parquet")
    bars = {tf: sorted(ind.filter(pl.col("tf") == tf)["bar"].unique().to_list())
            for tf in TFS}
    meta, blocks = [], []
    for tf in TFS:
        g = ind.filter(pl.col("tf") == tf)
        bi = {b: i for i, b in enumerate(bars[tf])}
        grid = np.full((len(days), len(bars[tf]), len(COLUMNS)), 255, np.uint8)
        rows = np.array([di.get(d, -1) for d in g["date"].to_list()])
        cols = np.array([bi[b] for b in g["bar"].to_list()])
        ok = rows >= 0
        for j, c in enumerate(COLUMNS):
            if c not in g.columns:
                meta.append(dict(col=c, tf=tf, lo=0.0, hi=1.0, ok=False)); continue
            x = g[c].to_numpy().astype(float)
            f = np.isfinite(x)
            if f.sum() < 100:
                meta.append(dict(col=c, tf=tf, lo=0.0, hi=1.0, ok=False)); continue
            lo, hi = np.quantile(x[f], [0.005, 0.995])
            if hi <= lo:
                hi = lo + 1
            q = np.clip(np.round((x - lo) / (hi - lo) * 254), 0, 254)
            grid[rows[ok], cols[ok], j] = np.where(f[ok], q[ok], 255).astype(np.uint8)
            meta.append(dict(col=c, tf=tf, lo=float(lo), hi=float(hi), ok=True))
        blocks.append(grid.tobytes())

    blob = b"".join(
        [head.astype("<i2").tobytes(), aux.astype("<f4").tobytes()]
        + [series[k].astype("<i2").tobytes() for k in ("mid", "low", "short", "cover")]
        + blocks)
    gz = gzip.compress(blob, 9)
    b64 = base64.b64encode(gz).decode()
    hdr = dict(n=n, days=[d.isoformat() for d in days], slots=list(SLOTS),
               sel=sel, horizon=HORIZON, columns=COLUMNS, tfs=list(TFS),
               bars={str(tf): bars[tf] for tf in TFS}, ind=meta,
               widths=dict(head=head.shape[1], aux=aux.shape[1]))
    (OUT / "demo_calc_header.json").write_text(json.dumps(hdr))
    (OUT / "demo_calc_blob.b64").write_text(b64)
    # the same bytes unencoded, for the split build. Named .bin and not .gz on
    # purpose -- see `decode()` in app.js.
    (OUT / "demo_calc_blob.bin").write_bytes(gz)
    print(f"rows {n:,}   raw {len(blob)/1e6:.1f} MB   gzip {len(gz)/1e6:.2f} MB   "
          f"base64 {len(b64)/1e6:.2f} MB   header {len(json.dumps(hdr))/1e3:.0f} kB")


if __name__ == "__main__":
    main()
