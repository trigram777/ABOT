#!/usr/bin/env python3
"""Assemble the explorer from its source parts and the packed dataset.

    python build.py            ->  results/explorer.html            (inline)
    python build.py --split    ->  results/explorer.html + .bin     (fetched)

The page is `src/shell.html` with four substitutions: the header JSON, the
blob (or a URL to it), and `app.js + ui.js` concatenated. Keeping the parts
separate is what makes them checkable -- `src/verify.mjs` runs the same engine
in node against the same bytes the browser gets.

INLINE is the safe default: one file, opens from file://, no server needed.
SPLIT drops the base64 tax (a flat 33%) and lets the blob cache separately from
the page, but `fetch` needs a real origin, so it must be served rather than
opened. Both builds share one `decode()`, so the format cannot drift between
them.
"""
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC, RES = ROOT / "src", ROOT / "results"
BLOB = "demo_calc_blob.bin"

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--split", action="store_true",
                help="fetch the blob instead of inlining it (needs a server)")
a = ap.parse_args()

page = ((SRC / "shell.html").read_text()
        .replace("__SCRIPT__", (SRC / "app.js").read_text() + (SRC / "ui.js").read_text())
        .replace("__HEADER__", (RES / "demo_calc_header.json").read_text())
        .replace("__BLOBURL__", BLOB if a.split else "")
        .replace("__BLOB__", "" if a.split else (RES / "demo_calc_blob.b64").read_text()))

out = RES / "explorer.html"
out.write_text(page)
total = out.stat().st_size
line = f"{out.name}  {total/1e6:.2f} MB"
if a.split:
    b = (RES / BLOB).stat().st_size
    total += b
    line += f"  +  {BLOB}  {b/1e6:.2f} MB   = {total/1e6:.2f} MB over the wire"
print(line)
