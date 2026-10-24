#!/usr/bin/env python3
"""Build the combined deployment candidate: banked + t7 Mehler trim + K22 carrier.

Composition (familiar op types only; NO cholesky anywhere):
  base   = the graded 328326 card (workspace_v4, sha 2fd2d40d...)
  trim   = Mehler bivariate series t=25 -> 7 terms (measured: saves 109.1e9,
           output effect 5.1e-5 of the error scale on the probe row)
  carrier= the pilot-validated rank-4 K22-residual second-star carrier
           (corrected k31 3.0 coefficients) with the two dense d-correction
           GEMMs REMOVED (k31/k22 channels keep the rank-4 approximation's
           small spurious diagonal content; the k4 channel keeps its cheap
           vector correction) - saves 60.2e9, bringing the carrier to ~152e9.
Budget: 2132.4 - 109.1 + ~152 = ~2175e9 <= 2^41 = 2199.0e9.
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
K22_BUILDER = HERE.parent / "public_k22_carrier_ab_probe_v1" / "build_estimator_k22r.py"
Q24_TRIM_TERMS = 7

APPENDED = '''

# --- K22-residual second-star carrier (deployment spelling) ---------------
# The reset fits one scalar pairing_alpha to the k22 residual and discards
# the rest; the transport-fidelity probe measured that discarded channel
# transported at exactly zero, and the sealed 200-row pilot measured the
# rank-4 carrier at -35.6% raw final-layer MSE (200/200 rows). This
# deployment spelling drops the two dense diagonal-correction GEMMs: the
# rank-4 factors' diagonal is a small artifact of the truncation and its
# k31/k22 spur is accepted as part of the approximation.

K22R_RANK = 4
_K22R_WHITEN_FLOOR = 1.0e-12


def _factor_k22_residual(
    symmetric_k22, pairing_alpha, pairing_plane, diagonal_context
):
    if K22R_RANK <= 0:
        return None
    residual32 = fnp.asarray(
        _off_diagonal(
            symmetric_k22 - pairing_alpha * pairing_plane, diagonal_context
        ),
        dtype=fnp.float32,
    )
    start = residual32[:, :K22R_RANK]
    y = residual32 @ start
    y = residual32 @ y
    gram = fnp.asarray(fnp.transpose(y) @ y, dtype=fnp.float64)
    gram_vals, gram_vecs = fnp.linalg.eigh(gram)
    top = fnp.maximum(fnp.max(gram_vals), 1.0e-30)
    floored = fnp.maximum(gram_vals, _K22R_WHITEN_FLOOR * top)
    whiten = gram_vecs @ (fnp.transpose(gram_vecs) / fnp.sqrt(floored)[:, None])
    y_orth = y @ fnp.asarray(whiten, dtype=fnp.float32)
    small = fnp.transpose(y_orth) @ (residual32 @ y_orth)
    small = fnp.asarray(
        0.5 * (small + fnp.transpose(small)), dtype=fnp.float64
    )
    lam64, vec64 = fnp.linalg.eigh(small)
    v_basis = y_orth @ fnp.asarray(vec64, dtype=fnp.float32)
    return v_basis, fnp.asarray(lam64, dtype=fnp.float32)


def _push_k22_residual(factors, weight_context, diagonal_context):
    weight32, _weight32_t, weight2, weight2_t, weight3, weight3_t = (
        weight_context
    )
    v_basis, lam = factors
    u_basis = weight2_t @ v_basis
    v_scaled = v_basis * lam[None, :]
    u_scaled = u_basis * lam[None, :]
    diag_q = fnp.sum(v_scaled * v_basis, axis=1)
    k4_channel = fnp.asarray(
        3.0 * fnp.sum(u_scaled * u_basis, axis=1)
        - 3.0 * (fnp.transpose(weight2 * weight2) @ diag_q),
        dtype=fnp.float64,
    )
    w2tq = u_scaled @ fnp.transpose(v_basis)
    s_one = (w2tq * fnp.transpose(weight32)) @ weight32
    # Entry coefficient 3 (the ordered sum covers both leg placements);
    # the rank-truncation diagonal's k31/k22 spur is accepted, so no dense
    # correction GEMMs here.
    k31_channel = fnp.asarray(3.0 * s_one, dtype=fnp.float64)
    pairing_form = None
    for rank_index in range(K22R_RANK):
        g_rank = fnp.transpose(weight32) @ (
            v_basis[:, rank_index][:, None] * weight32
        )
        term = lam[rank_index] * (g_rank * g_rank)
        pairing_form = term if pairing_form is None else pairing_form + term
    k22_channel = fnp.asarray(
        u_scaled @ fnp.transpose(u_basis) + 2.0 * pairing_form,
        dtype=fnp.float64,
    )
    return (
        k4_channel,
        _off_diagonal(k31_channel, diagonal_context),
        _off_diagonal(k22_channel, diagonal_context),
    )


def _push_k22_residual_diagonal(factors, weight_context):
    _w32, _wt, weight2, weight2_t, _w3, _w3t = weight_context
    v_basis, lam = factors
    u_basis = weight2_t @ v_basis
    v_scaled = v_basis * lam[None, :]
    u_scaled = u_basis * lam[None, :]
    diag_q = fnp.sum(v_scaled * v_basis, axis=1)
    return fnp.asarray(
        3.0 * fnp.sum(u_scaled * u_basis, axis=1)
        - 3.0 * (fnp.transpose(weight2 * weight2) @ diag_q),
        dtype=fnp.float64,
    )
'''


def load_k22_wiring_replacements():
    """Reuse the audited K22 builder's R1-R7 wiring anchors verbatim."""
    import ast

    source = K22_BUILDER.read_text()
    match = re.search(r"REPLACEMENTS = (\[.*?\n\])\n", source, re.S)
    return ast.literal_eval(match.group(1))


def main():
    source = BASE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA

    # Mehler t-trim (3 anchored patches, measured spelling).
    grid_match = re.search(r"_Q24_ORDER_GRID = \(\(.*?\)\)\n", source, re.S)
    rows = []
    for d in range(7):
        rows.append("(" + ", ".join(str(d + t) for t in range(Q24_TRIM_TERMS)) + ",)")
    new_grid = "_Q24_ORDER_GRID = (" + ",\n ".join(rows) + ")\n"
    patched = source.replace("_Q24_TERMS = 25\n", f"_Q24_TERMS = {Q24_TRIM_TERMS}\n")
    patched = patched.replace(grid_match.group(0), new_grid)
    slot = "        fnp.asarray(_INV_FACTORIAL, dtype=fnp.float64),\n"
    assert patched.count(slot) == 1
    patched = patched.replace(
        slot,
        "        fnp.asarray(_INV_FACTORIAL[:_Q24_TERMS], dtype=fnp.float64),\n",
    )

    # Nonnegativity projection at the REPORTED output (the deployed path is
    # Estimator.predict -> _predict_corrected, whose transport correction is
    # added after the raw host): post-ReLU layer means are >= 0 by
    # construction, so clamping the reported means at zero is exact
    # knowledge and can only reduce MSE (the carrier's accepted d-spur
    # nudges a few hundred near-zero unit means to about -1e-5).
    out_anchor = "    output = fnp.asarray(host + correction, dtype=fnp.float32)\n"
    assert patched.count(out_anchor) == 1
    patched = patched.replace(
        out_anchor,
        "    output = fnp.asarray(\n"
        "        fnp.maximum(host + correction, 0.0), dtype=fnp.float32\n"
        "    )\n",
    )

    # K22 carrier wiring (the audited R1-R7), then the deployment appended
    # block (replaces the probe builder's appended block content).
    applied = 0
    for index, (old, new) in enumerate(load_k22_wiring_replacements(), start=1):
        count = patched.count(old)
        assert count == 1, f"wiring anchor R{index} matched {count} times"
        patched = patched.replace(old, new)
        applied += 1
    patched += APPENDED

    OUT.write_text(patched)
    receipt = {
        "schema": "combined-card-builder-receipt-v1",
        "base_sha256": BASE_SHA,
        "output_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "q24_terms": Q24_TRIM_TERMS,
        "carrier": "rank 4, corrected k31 coefficient 3.0, d-correction GEMMs removed (k4 vector correction kept)",
        "wiring_anchors_applied": applied,
        "cholesky_free": "cholesky" not in patched,
    }
    (HERE / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
