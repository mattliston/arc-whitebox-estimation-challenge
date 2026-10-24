#!/usr/bin/env python3
"""Build estimator_k22r.py: the base host + the K22-residual second-star carrier.

TAG=k22r-builder  Every replacement anchor must occur exactly once in the
base source; any drift aborts the build. Rank K22R_RANK=0 must reproduce the
base bit-for-bit (the factors stay None and every added branch is skipped);
the harness checks that before the A/B.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = (
    HERE.parent
    / "public_condrepair_deployment_card_v3"
    / "estimator.py"
)
BASE_SHA = "e4d250563279a8c99dceb72bd71428855bd798d651730fca9c57d6577aa6f62d"
OUT = HERE / "estimator_k22r.py"

APPENDED = '''

# --- K22-residual second-star carrier (P2R0F3A probe) ---------------------
# The reset currently fits one scalar pairing_alpha to the k22 residual and
# discards the rest; the fidelity probe measured that discarded channel
# transported at exactly zero. This carrier factors the discarded residual
# to rank K22R_RANK at each reset and pushes it one layer with the exact
# iijj-family transport (value convention, diagonal-spur corrected).

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
    # The ordered sum S1 already contains both leg placements of each
    # unordered pair, so the entry coefficient is 3, not 6 (audit-corrected).
    k31_channel = fnp.asarray(
        3.0 * s_one - 3.0 * (weight3_t @ (diag_q[:, None] * weight32)),
        dtype=fnp.float64,
    )
    pairing_form = None
    for rank_index in range(K22R_RANK):
        g_rank = fnp.transpose(weight32) @ (
            v_basis[:, rank_index][:, None] * weight32
        )
        term = lam[rank_index] * (g_rank * g_rank)
        pairing_form = term if pairing_form is None else pairing_form + term
    k22_channel = fnp.asarray(
        u_scaled @ fnp.transpose(u_basis)
        + 2.0 * pairing_form
        - 3.0 * (weight2_t @ (diag_q[:, None] * weight2)),
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

REPLACEMENTS = [
    # R1: _reset_stars emits the factored residual.
    (
        "    return cubic, quartic, pairing_alpha\n",
        "    k22_residual_factors = _factor_k22_residual(\n"
        "        symmetric_k22, pairing_alpha, pairing_plane, diagonal_context\n"
        "    )\n"
        "    return cubic, quartic, pairing_alpha, k22_residual_factors\n",
    ),
    # R2: _initialize_h1 internal unpack.
    (
        "    cubic, quartic, pairing_alpha = _reset_stars(\n"
        "        k21,\n",
        "    cubic, quartic, pairing_alpha, k22_residual_factors = _reset_stars(\n"
        "        k21,\n",
    ),
    # R3: _initialize_h1 return tuple gains the factors (W3 spelling keeps
    # the trailing native-context element).
    (
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        (coefficients, gaussian_raw, sigma, alpha, pdf),\n"
        "    )\n",
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        k22_residual_factors,\n"
        "        (coefficients, gaussian_raw, sigma, alpha, pdf),\n"
        "    )\n",
    ),
    # R4: the caller unpacks the extra element.
    (
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        initial_native_context,\n"
        "    ) = _initialize_h1(\n",
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        k22_residual_factors,\n"
        "        initial_native_context,\n"
        "    ) = _initialize_h1(\n",
    ),
    # R5: layer-loop reset unpack.
    (
        "        cubic, quartic, pairing_alpha = _reset_stars(\n"
        "            k21_target,\n",
        "        cubic, quartic, pairing_alpha, k22_residual_factors = _reset_stars(\n"
        "            k21_target,\n",
    ),
    # R6: add the pushed residual planes into the pre-activation provider.
    (
        "            cp4_pre_provider_planes,\n"
        "        )\n"
        "\n"
        "        (\n"
        "            coefficients, gaussian_raw, sigma, alpha, pdf,\n",
        "            cp4_pre_provider_planes,\n"
        "        )\n"
        "        if k22_residual_factors is not None:\n"
        "            k22r_k4, k22r_k31, k22r_k22 = _push_k22_residual(\n"
        "                k22_residual_factors, weight_context, diagonal_context\n"
        "            )\n"
        "            pre_k4 = pre_k4 + k22r_k4\n"
        "            pre_k31 = pre_k31 + k22r_k31\n"
        "            pre_k22 = pre_k22 + k22r_k22\n"
        "\n"
        "        (\n"
        "            coefficients, gaussian_raw, sigma, alpha, pdf,\n",
    ),
    # R7: final layer consumes the k4 diagonal channel.
    (
        "                + _cp4_read_diagonal(cp4_lambda, pushed_u)\n"
        "            )\n",
        "                + _cp4_read_diagonal(cp4_lambda, pushed_u)\n"
        "            )\n"
        "            if k22_residual_factors is not None:\n"
        "                final_k4 = final_k4 + _push_k22_residual_diagonal(\n"
        "                    k22_residual_factors, weight_context\n"
        "                )\n",
    ),
]


def main():
    source = BASE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA, (
        "base estimator drifted; re-pin BASE_SHA deliberately"
    )
    patched = source
    for index, (old, new) in enumerate(REPLACEMENTS, start=1):
        count = patched.count(old)
        assert count == 1, f"anchor R{index} matched {count} times, need 1"
        patched = patched.replace(old, new)
    patched += APPENDED
    OUT.write_text(patched)
    receipt = {
        "schema": "k22r-builder-receipt-v1",
        "base_path": str(BASE),
        "base_sha256": BASE_SHA,
        "output_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "replacements": len(REPLACEMENTS),
        "rank": 4,
    }
    (HERE / "BUILD_RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
