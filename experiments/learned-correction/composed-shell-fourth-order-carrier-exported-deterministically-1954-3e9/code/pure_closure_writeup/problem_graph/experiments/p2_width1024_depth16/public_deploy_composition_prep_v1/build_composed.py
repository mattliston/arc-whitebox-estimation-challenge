#!/usr/bin/env python3
"""Build estimator_composed.py: no-eigh card + lambda3 shell + rank-4 K22 carrier.

TAG=deploy-composition-builder  Deterministic three-way composition for the
deployment candidate host:

  base   = public_noeigh_card_v1/estimator.py            (sha 373ad4d5...)
  shell  = the lambda-cubed Edgeworth shell, recovered as the audited textual
           diff between public_lambda3_shell_probe_v1/instrumented_estimator_x.py
           and its frozen base; per the X audit (public_lambda3_native_pilot_v1/
           AUDIT_RECEIPT.json, "composition/estimator_x") those lambda3 hunks
           are byte-identical to the diff estimator_x.py minus
           public_condrepair_deployment_card_v1/estimator.py, which is the
           condrepair-free spelling used here.  Every non-adapted hunk payload
           is EXTRACTED from estimator_x.py at build time (never retyped) and
           its pre-image is asserted byte-identical between the no-eigh card
           and condrepair_v1, proving the audited hunks transfer unchanged.
  adapt  = ONE hunk cannot transfer verbatim: the no-eigh card carries the t7
           trim (_Q24_TERMS = 7), so _Q24_ORDER_GRID rows are 7-term rows
           (d..d+6) in the trim builder's format, not the 25-term rows the
           shell was written against.  The shell's three added rows (d=7,8,9)
           are re-emitted in the trim format: 7 terms each, topping out at
           gather index 9+6=15, inside the q<=38 coefficient table.  See
           ADAPTATIONS below; everything else transfers byte-identically.
  k22    = the K22-residual second-star carrier: REPLACEMENTS and APPENDED are
           imported VERBATIM from public_k22_carrier_ab_probe_v1/
           build_estimator_k22r.py (sha-pinned), never copied by hand.

Every replacement anchor must occur exactly once in the working source; any
drift aborts the build.
"""
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_noeigh_card_v1" / "estimator.py"
X = P2 / "public_lambda3_shell_probe_v1" / "estimator_x.py"
V1 = P2 / "public_condrepair_deployment_card_v1" / "estimator.py"
K22_BUILDER = P2 / "public_k22_carrier_ab_probe_v1" / "build_estimator_k22r.py"
OUT = HERE / "estimator_composed.py"

BASE_SHA = "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
X_SHA = "f055bf1321500f74d8bdfd5e4bc209b5a44b5839d004ab8fa8c39bba4d0d24e5"
V1_SHA = "24ed2325e4db0c0287a222189a986c2d6462f80832c717827c27b783a6655ee2"
K22_BUILDER_SHA = "c6c9868c4b8784c87aea39f605adce0f330fade65e966236527495062e3326a8"

Q24_TRIM_TERMS = 7

ADAPTATIONS = [
    "A1 order-grid row length: the shell probe extended _Q24_ORDER_GRID with "
    "rows d=7,8,9 of 25 terms (d..d+24, max gather 9+24=33).  The no-eigh "
    "card's t7 trim makes every row 7 terms (_Q24_TERMS=7, the einsum t-axis "
    "and _INV_FACTORIAL[:_Q24_TERMS] slice), so the added rows are re-emitted "
    "as (7..13), (8..14), (9..15): 7 terms each, max gather 15 <= 38, inside "
    "the untouched q<=38 coefficient table.",
    "A2 order-grid formatting: rows are emitted in the trim builder's exact "
    "format ('(d, d+1, ..., d+6,)' joined with ',\\n ') so rows d=0..6 are "
    "byte-identical to the trim card's existing grid; the probe's reflow of "
    "rows d=0..6 to 25-term single lines is NOT ported (those rows would "
    "contradict _Q24_TERMS=7).",
    "A3 condrepair hunk not ported: the instrumented-x diff also carries the "
    "conditional-repair Cholesky block; the no-eigh lineage deliberately "
    "removed that whole repair family (trim receipt cholesky_free=true, "
    "no-eigh receipt removes the unconditional eigh block), so the shell is "
    "applied in its condrepair-free spelling (estimator_x hunks), which the "
    "X audit certifies byte-identical to the instrumented file's lambda3 "
    "hunks.",
]


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def pin(path, expected):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == expected, f"{path.name} drifted: {digest} != {expected}"
    return digest


def region(source, start_marker, end_marker, include_end=True):
    """The unique start_marker through the next end_marker (byte slice)."""
    count = source.count(start_marker)
    assert count == 1, f"start marker {start_marker[:40]!r} occurs {count}x"
    begin = source.index(start_marker)
    stop = source.index(end_marker, begin + len(start_marker))
    if include_end:
        stop += len(end_marker)
    return source[begin:stop]


def trim_grid(rows):
    """_Q24_ORDER_GRID text in build_estimator_trim.py's exact format."""
    out = []
    for d in range(rows):
        out.append(
            "(" + ", ".join(str(d + t) for t in range(Q24_TRIM_TERMS)) + ",)"
        )
    return "_Q24_ORDER_GRID = (" + ",\n ".join(out) + ")\n"


def apply(source, old, new, tag):
    count = source.count(old)
    assert count == 1, f"anchor {tag} matched {count} times, need 1"
    return source.replace(old, new)


def main():
    pin(BASE, BASE_SHA)
    pin(X, X_SHA)
    pin(V1, V1_SHA)
    pin(K22_BUILDER, K22_BUILDER_SHA)
    base = BASE.read_text()
    x = X.read_text()
    v1 = V1.read_text()

    spec = importlib.util.spec_from_file_location("k22_builder", K22_BUILDER)
    k22 = importlib.util.module_from_spec(spec)
    sys.modules["k22_builder"] = k22
    spec.loader.exec_module(k22)
    assert len(k22.REPLACEMENTS) == 7 and k22.APPENDED.startswith("\n\n# ---")

    patches = []

    # --- lambda3 shell, hunks extracted from the audited estimator_x -------
    # S1: _DERIVATIVE_CHANNELS gains the a+b=7 and a+b=9 rows.
    channels_start, channels_end = "_DERIVATIVE_CHANNELS = (", "\n)\n"
    old = region(base, channels_start, channels_end)
    new = region(x, channels_start, channels_end)
    assert old == region(v1, channels_start, channels_end), (
        "S1 pre-image drifted from the audited shell base"
    )
    assert "(0, 7)," in new and "(9, 0)," in new
    patches.append(("S1-derivative-channels", old, new))

    # S2: left/right derivative gather rows extend from 17 to 35 entries.
    grids_start, grids_end = "_LEFT_DERIVATIVE_GRID = (", "\n_Q24_ORDER_GRID"
    old = region(base, grids_start, grids_end, include_end=False)
    new = region(x, grids_start, grids_end, include_end=False)
    assert old == region(v1, grids_start, grids_end, include_end=False), (
        "S2 pre-image drifted from the audited shell base"
    )
    assert new.count("9, 8, 7, 6, 5, 4, 3, 2, 1, 0),)") == 1
    patches.append(("S2-derivative-grids", old, new))

    # S3 (ADAPTED, see A1/A2): _Q24_ORDER_GRID rows d=7,8,9 in trim format.
    old = region(base, "_Q24_ORDER_GRID = (", "))\n")
    assert old == trim_grid(7), "no-eigh grid is not the expected t7 spelling"
    x_grid = region(x, "_Q24_ORDER_GRID = (", "))\n")
    assert x_grid.count("(") == 11 and x_grid.rstrip().endswith(
        "30, 31, 32, 33))"
    ), "estimator_x grid is not the audited 10-row d<=9 q<=33 spelling"
    new = trim_grid(10)
    assert new.rstrip().endswith("(9, 10, 11, 12, 13, 14, 15,))")
    patches.append(("S3-order-grid-ADAPTED", old, new))

    # S4: the fused weights workspace grows from 17 to 35 planes.
    old = "    weights_workspace = fnp.empty((17, width, width), dtype=fnp.float64)\n"
    new = "    weights_workspace = fnp.empty((35, width, width), dtype=fnp.float64)\n"
    assert old in v1 and new in x
    patches.append(("S4-weights-workspace", old, new))

    # S5: _inverse_sigma_context gains inv7/inv8/inv9 rows (whole function).
    sig_start = "def _inverse_sigma_context(sigma):"
    sig_end = "    return inverse_sigma, inv1, inv2, inv3, inv4, inv6\n"
    old = region(base, sig_start, sig_end)
    new = region(x, sig_start, sig_end)
    assert old == region(v1, sig_start, sig_end), (
        "S5 pre-image drifted from the audited shell base"
    )
    assert "inv9 = inv6 * inv3" in new and new.count("inv7, inv8, inv9") == 1
    patches.append(("S5-inverse-sigma", old, new))

    # S6: the lambda-cubed weight blocks (planes 17..34) enter
    # _edgeworth_weights between the L3^2 finalization and the return.
    wt_start = "    fnp.add(weights[12:15], 0.0, out=weights[12:15])\n"
    wt_end = "    return weights\n"
    old = region(base, wt_start, wt_end)
    new = region(x, wt_start, wt_end)
    assert old == region(v1, wt_start, wt_end), (
        "S6 pre-image drifted from the audited shell base"
    )
    assert old == wt_start + wt_end
    assert "# Lambda-cubed shell." in new and "weights[34], third" in new
    patches.append(("S6-lambda3-weight-blocks", old, new))

    # S7/S8: both consumption loops cover the 35 channels.
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

    # --- K22 carrier: the probe builder's anchors, verbatim ----------------
    for index, (old, new) in enumerate(k22.REPLACEMENTS, start=1):
        patches.append((f"K{index}-k22-carrier", old, new))

    # --- provenance note in the module docstring ---------------------------
    doc_line = (
        '"""FB-EC4-Q24 with a target-free native raw20/sigma instrumentation'
        " seam.\n"
    )
    patches.append((
        "D1-docstring-provenance",
        doc_line,
        doc_line
        + "\nComposed deployment candidate: no-eigh card + lambda-cubed shell"
        " (order\ngrid adapted to the t7 trim) + rank-4 K22-residual"
        " second-star carrier.\n",
    ))

    composed = base
    for tag, old, new in patches:
        composed = apply(composed, old, new, tag)
    composed += k22.APPENDED

    # --- static closure checks --------------------------------------------
    compile(composed, str(OUT), "exec")
    assert "_Q24_TERMS = 7\n" in composed
    assert composed.count("range(0, 35, 3)") == 2
    assert composed.count("range(0, 17, 3)") == 0
    assert composed.count("K22R_RANK = 4") == 1
    assert composed.count("k22_residual_factors = _reset_stars") == 2
    grid = region(composed, "_Q24_ORDER_GRID = (", "))\n")
    grid_rows = eval(grid.split("=", 1)[1])
    assert len(grid_rows) == 10 and all(
        len(row) == Q24_TRIM_TERMS for row in grid_rows
    )
    assert max(max(row) for row in grid_rows) == 15
    left = eval(region(composed, "_LEFT_DERIVATIVE_GRID = (", ")\n").split("=", 1)[1])
    assert len(left[0]) == 35 and max(left[0]) == 9 < len(grid_rows)
    shell_block = region(x, "    # Lambda-cubed shell.", "    return weights\n")
    assert shell_block in composed, "shell weight block not byte-identical"

    tmp = OUT.with_name(f".{OUT.name}.tmp")
    tmp.write_text(composed)
    tmp.replace(OUT)

    receipt = {
        "schema": "deploy-composition-builder-receipt-v1",
        "inputs": {
            "base_noeigh_card": {"path": str(BASE), "sha256": BASE_SHA},
            "lambda3_shell_source": {"path": str(X), "sha256": X_SHA},
            "lambda3_shell_base": {"path": str(V1), "sha256": V1_SHA},
            "k22_builder": {"path": str(K22_BUILDER), "sha256": K22_BUILDER_SHA},
        },
        "shell_recovery_note": (
            "lambda3 hunks extracted from estimator_x.py, whose diff against "
            "condrepair_v1 the X audit certifies byte-identical to the "
            "instrumented_estimator_x.py lambda3 hunks (AUDIT_RECEIPT.json "
            "composition/estimator_x); pre-images asserted unchanged between "
            "condrepair_v1 and the no-eigh card for every non-adapted hunk"
        ),
        "patches": [tag for tag, _old, _new in patches],
        "adaptations": ADAPTATIONS,
        "k22_appended_bytes": len(k22.APPENDED),
        "output_sha256": sha256_text(composed),
        "output_bytes": len(composed.encode()),
        "transport_payload_state": (
            "carries the no-eigh card's frozen _TRANSPORT_PAYLOAD unchanged; "
            "STALE for this host by construction - retrain and re-splice per "
            "EXPORT_PLAYBOOK.md before any deployment"
        ),
    }
    receipt_path = HERE / "BUILD_RECEIPT.json"
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if (
            previous.get("output_sha256") == receipt["output_sha256"]
            and "smoke" in previous
        ):
            receipt["smoke"] = previous["smoke"]
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({k: receipt[k] for k in (
        "output_sha256", "output_bytes", "patches")}, indent=1))


if __name__ == "__main__":
    main()
