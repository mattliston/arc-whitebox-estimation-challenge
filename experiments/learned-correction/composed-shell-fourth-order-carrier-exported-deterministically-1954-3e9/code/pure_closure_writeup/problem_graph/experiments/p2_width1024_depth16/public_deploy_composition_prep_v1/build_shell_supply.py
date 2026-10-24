#!/usr/bin/env python3
"""Build estimator_shell_supply.py: supply card + lambda3 shell (NO carrier).

TAG=shell-supply-builder  The full-rank K22 supply supersedes the rank-4
carrier (supply+carrier would double-count the K22 residual), so this
composition is exactly the S1-S8 shell hunk set from build_composed.py on
top of public_supply_card_v1/estimator.py.  Same recovery: non-adapted hunk
payloads EXTRACTED from the audited estimator_x.py, pre-images asserted
byte-identical between the supply card and condrepair_v1 (the audited shell
base); the order-grid hunk t7-adapted (rows d=7,8,9 as 7-term rows, max
gather 15 <= 38).  The supply patches touched _reset_stars return, the
init/loop threading, and the assemble site - regions the shell never edits;
any anchor collision aborts with a report, never a forced apply.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_supply_card_v1" / "estimator.py"
X = P2 / "public_lambda3_shell_probe_v1" / "estimator_x.py"
V1 = P2 / "public_condrepair_deployment_card_v1" / "estimator.py"
OUT = HERE / "estimator_shell_supply.py"

BASE_SHA = "63e2b95bfc1b2c5b2b109f3cb824c3c3c79a1ff5e39043e8232ad4e5077a8ea0"
X_SHA = "f055bf1321500f74d8bdfd5e4bc209b5a44b5839d004ab8fa8c39bba4d0d24e5"
V1_SHA = "24ed2325e4db0c0287a222189a986c2d6462f80832c717827c27b783a6655ee2"

Q24_TRIM_TERMS = 7


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pin(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected, f"{path.name} drifted: {digest} != {expected}"


def region(source, start_marker, end_marker, include_end=True):
    count = source.count(start_marker)
    assert count == 1, f"start marker {start_marker[:40]!r} occurs {count}x"
    begin = source.index(start_marker)
    stop = source.index(end_marker, begin + len(start_marker))
    if include_end:
        stop += len(end_marker)
    return source[begin:stop]


def trim_grid(rows):
    out = []
    for d in range(rows):
        out.append(
            "(" + ", ".join(str(d + t) for t in range(Q24_TRIM_TERMS)) + ",)"
        )
    return "_Q24_ORDER_GRID = (" + ",\n ".join(out) + ")\n"


def main():
    pin(BASE, BASE_SHA)
    pin(X, X_SHA)
    pin(V1, V1_SHA)
    base = BASE.read_text()
    x = X.read_text()
    v1 = V1.read_text()

    patches = []
    collisions = []

    def preimage(tag, start, end, include_end=True):
        """old from the supply card; must equal the audited shell base's."""
        old = region(base, start, end, include_end)
        audited = region(v1, start, end, include_end)
        if old != audited:
            collisions.append(
                f"{tag}: supply-card pre-image differs from the audited "
                f"shell base (supply patches reached this region)"
            )
        return old

    channels_start, channels_end = "_DERIVATIVE_CHANNELS = (", "\n)\n"
    old = preimage("S1", channels_start, channels_end)
    new = region(x, channels_start, channels_end)
    assert "(0, 7)," in new and "(9, 0)," in new
    patches.append(("S1-derivative-channels", old, new))

    grids_start, grids_end = "_LEFT_DERIVATIVE_GRID = (", "\n_Q24_ORDER_GRID"
    old = preimage("S2", grids_start, grids_end, include_end=False)
    new = region(x, grids_start, grids_end, include_end=False)
    assert new.count("9, 8, 7, 6, 5, 4, 3, 2, 1, 0),)") == 1
    patches.append(("S2-derivative-grids", old, new))

    old = region(base, "_Q24_ORDER_GRID = (", "))\n")
    if old != trim_grid(7):
        collisions.append("S3: supply-card order grid is not the t7 spelling")
    new = trim_grid(10)
    assert new.rstrip().endswith("(9, 10, 11, 12, 13, 14, 15,))")
    patches.append(("S3-order-grid-ADAPTED", old, new))

    old = "    weights_workspace = fnp.empty((17, width, width), dtype=fnp.float64)\n"
    new = "    weights_workspace = fnp.empty((35, width, width), dtype=fnp.float64)\n"
    assert old in v1 and new in x
    patches.append(("S4-weights-workspace", old, new))

    sig_start = "def _inverse_sigma_context(sigma):"
    sig_end = "    return inverse_sigma, inv1, inv2, inv3, inv4, inv6\n"
    old = preimage("S5", sig_start, sig_end)
    new = region(x, sig_start, sig_end)
    assert "inv9 = inv6 * inv3" in new and new.count("inv7, inv8, inv9") == 1
    patches.append(("S5-inverse-sigma", old, new))

    wt_start = "    fnp.add(weights[12:15], 0.0, out=weights[12:15])\n"
    wt_end = "    return weights\n"
    old = preimage("S6", wt_start, wt_end)
    new = region(x, wt_start, wt_end)
    assert "# Lambda-cubed shell." in new and "weights[34], third" in new
    patches.append(("S6-lambda3-weight-blocks", old, new))

    old = (
        "    for start in range(0, 17, 3):\n"
        "        requests = _DERIVATIVE_CHANNELS[start:start + 3]\n"
    )
    new = (
        "    for start in range(0, 35, 3):\n"
        "        requests = _DERIVATIVE_CHANNELS[start:start + 3]\n"
    )
    assert old in v1 and new in x
    patches.append(("S7-g-four-all-loop", old, new))
    old = (
        "    for start in range(0, 17, 3):\n"
        "        stop = min(start + 3, 17)\n"
    )
    new = (
        "    for start in range(0, 35, 3):\n"
        "        stop = min(start + 3, 35)\n"
    )
    assert old in v1 and new in x
    patches.append(("S8-four-pair-loop", old, new))

    doc_line = (
        '"""FB-EC4-Q24 with a target-free native raw20/sigma instrumentation'
        " seam.\n"
    )
    patches.append((
        "D1-docstring-provenance",
        doc_line,
        doc_line
        + "\nComposed candidate: supply card (full-rank K22 residual state"
        " supply) +\nlambda-cubed shell (order grid adapted to the t7 trim);"
        " no rank-4 carrier -\nthe supply supersedes it.\n",
    ))

    if collisions:
        report = {
            "schema": "shell-supply-builder-COLLISION-report-v1",
            "status": "ABORTED_NO_FORCE",
            "collisions": collisions,
        }
        (HERE / "SHELL_SUPPLY_BUILD_RECEIPT.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n"
        )
        print(json.dumps(report, indent=1))
        raise SystemExit(2)

    composed = base
    for tag, old, new in patches:
        count = composed.count(old)
        assert count == 1, f"anchor {tag} matched {count} times, need 1"
        composed = composed.replace(old, new)

    compile(composed, str(OUT), "exec")
    assert "_Q24_TERMS = 7\n" in composed
    assert composed.count("range(0, 35, 3)") == 2
    assert composed.count("range(0, 17, 3)") == 0
    grid = region(composed, "_Q24_ORDER_GRID = (", "))\n")
    grid_rows = eval(grid.split("=", 1)[1])
    assert len(grid_rows) == 10 and all(
        len(row) == Q24_TRIM_TERMS for row in grid_rows
    )
    assert max(max(row) for row in grid_rows) == 15
    shell_block = region(x, "    # Lambda-cubed shell.", "    return weights\n")
    assert shell_block in composed

    tmp = OUT.with_name(f".{OUT.name}.tmp")
    tmp.write_text(composed)
    tmp.replace(OUT)

    receipt = {
        "schema": "shell-supply-builder-receipt-v1",
        "status": "BUILT",
        "inputs": {
            "base_supply_card": {"path": str(BASE), "sha256": BASE_SHA},
            "lambda3_shell_source": {"path": str(X), "sha256": X_SHA},
            "lambda3_shell_base": {"path": str(V1), "sha256": V1_SHA},
        },
        "patches": [tag for tag, _o, _n in patches],
        "no_carrier_note": "rank-4 K22 carrier intentionally absent: the "
        "full-rank supply supersedes it (double-count hazard)",
        "collisions": [],
        "output_sha256": sha256_text(composed),
        "output_bytes": len(composed.encode()),
    }
    (HERE / "SHELL_SUPPLY_BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: receipt[k] for k in ("status", "output_sha256", "patches")}, indent=1))


if __name__ == "__main__":
    main()
