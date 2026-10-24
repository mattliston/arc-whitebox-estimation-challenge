#!/usr/bin/env python3
"""Build the trim card: graded 328326 base + t7 Mehler trim + output clamp.

No carrier, no cholesky, no new op types. Raw output changes by 5e-5 of the
error scale (measured trim ladder) plus a nonnegativity clamp that can only
reduce MSE. FLOPs drop 109.1e9: 2132.4e9 -> 2023.3e9, so per-MLP utilization
falls from 0.9697 to 0.9201 and adjusted improves about 5.1 percent.
"""
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent
    / "public_q24_native_transport_energy_final_workspace_v4"
    / "estimator.py"
)
BASE_SHA = "2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76"
OUT = HERE / "estimator.py"
Q24_TRIM_TERMS = 7


def main():
    source = BASE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA

    grid_match = re.search(r"_Q24_ORDER_GRID = \(\(.*?\)\)\n", source, re.S)
    rows = []
    for d in range(7):
        rows.append("(" + ", ".join(str(d + t) for t in range(Q24_TRIM_TERMS)) + ",)")
    new_grid = "_Q24_ORDER_GRID = (" + ",\n ".join(rows) + ")\n"
    patched = source.replace("_Q24_TERMS = 25\n", f"_Q24_TERMS = {Q24_TRIM_TERMS}\n")
    assert patched != source
    patched = patched.replace(grid_match.group(0), new_grid)
    slot = "        fnp.asarray(_INV_FACTORIAL, dtype=fnp.float64),\n"
    assert patched.count(slot) == 1
    patched = patched.replace(
        slot,
        "        fnp.asarray(_INV_FACTORIAL[:_Q24_TERMS], dtype=fnp.float64),\n",
    )
    out_anchor = "    output = fnp.asarray(host + correction, dtype=fnp.float32)\n"
    assert patched.count(out_anchor) == 1
    patched = patched.replace(
        out_anchor,
        "    output = fnp.asarray(\n"
        "        fnp.maximum(host + correction, 0.0), dtype=fnp.float32\n"
        "    )\n",
    )

    OUT.write_text(patched)
    receipt = {
        "schema": "trim-card-builder-receipt-v1",
        "base_sha256": BASE_SHA,
        "output_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "q24_terms": Q24_TRIM_TERMS,
        "patches": "t7 trim (3 anchored) + reported-output nonnegativity clamp",
        "cholesky_free": "cholesky" not in patched,
    }
    (HERE / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
