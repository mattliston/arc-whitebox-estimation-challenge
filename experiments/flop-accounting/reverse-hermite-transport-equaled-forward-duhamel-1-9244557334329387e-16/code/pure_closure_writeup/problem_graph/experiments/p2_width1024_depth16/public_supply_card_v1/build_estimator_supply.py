#!/usr/bin/env python3
"""Build the supply card: no-eigh base + full-rank K22-residual state supply.

The reset already computes the dense residual (symmetric_k22 minus
alpha*pairing) and discards it; this card carries it one layer and adds its
three exact supply channels to the assembled quartic planes:
  k4  += 3*diag(W2^T K W2)
  k31 += 3*((W2^T K) o W^T) @ W     (off-diagonal)
  k22 += W2^T K W2                  (off-diagonal)
about 11.5e9 FLOPs per layer. Measured on the monkeypatch probe: kappa4
state decay eliminated (oracle ratios flat ~0.94) and host raw -61.0%.
"""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "public_noeigh_card_v1" / "estimator.py"
BASE_SHA = "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
OUT = HERE / "estimator.py"

APPENDED = '''

# --- full-rank K22-residual state supply (P2R0F3A6B) ----------------------

def _push_k22_residual_supply(residual, weight_context, diagonal_context):
    weight32, _wt, weight2, weight2_t, _w3, _w3t = weight_context
    residual32 = fnp.asarray(residual, dtype=fnp.float32)
    kw2 = residual32 @ weight2
    supply_k4 = 3.0 * fnp.sum(weight2 * kw2, axis=0)
    w2tk = weight2_t @ residual32
    supply_k31 = 3.0 * ((w2tk * fnp.transpose(weight32)) @ weight32)
    supply_k22 = weight2_t @ kw2
    return (
        fnp.asarray(supply_k4, dtype=fnp.float64),
        _off_diagonal(fnp.asarray(supply_k31, dtype=fnp.float64), diagonal_context),
        _off_diagonal(fnp.asarray(supply_k22, dtype=fnp.float64), diagonal_context),
    )


def _push_k22_residual_supply_diagonal(residual, weight_context):
    _w32, _wt, weight2, _w2t, _w3, _w3t = weight_context
    residual32 = fnp.asarray(residual, dtype=fnp.float32)
    return fnp.asarray(
        3.0 * fnp.sum(weight2 * (residual32 @ weight2), axis=0),
        dtype=fnp.float64,
    )
'''

REPLACEMENTS = [
    # S1: the reset emits the dense residual.
    (
        "    return cubic, quartic, pairing_alpha\n",
        "    k22_supply_residual = _off_diagonal(\n"
        "        symmetric_k22 - pairing_alpha * pairing_plane, diagonal_context\n"
        "    )\n"
        "    return cubic, quartic, pairing_alpha, k22_supply_residual\n",
    ),
    # S2: init internal unpack.
    (
        "    cubic, quartic, pairing_alpha = _reset_stars(\n"
        "        k21,\n",
        "    cubic, quartic, pairing_alpha, k22_supply_residual = _reset_stars(\n"
        "        k21,\n",
    ),
    # S3: init return gains the residual.
    (
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        (coefficients, gaussian_raw, sigma, alpha, pdf),\n"
        "    )\n",
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        k22_supply_residual,\n"
        "        (coefficients, gaussian_raw, sigma, alpha, pdf),\n"
        "    )\n",
    ),
    # S4: init caller unpack.
    (
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        initial_native_context,\n"
        "    ) = _initialize_h1(\n",
        "        quartic,\n"
        "        pairing_alpha,\n"
        "        k22_supply_residual,\n"
        "        initial_native_context,\n"
        "    ) = _initialize_h1(\n",
    ),
    # S5: layer-loop reset unpack.
    (
        "        cubic, quartic, pairing_alpha = _reset_stars(\n"
        "            k21_target,\n",
        "        cubic, quartic, pairing_alpha, k22_supply_residual = _reset_stars(\n"
        "            k21_target,\n",
    ),
    # S6: the assembled planes gain the supply.
    (
        "            cp4_pre_provider_planes,\n"
        "        )\n"
        "\n"
        "        (\n"
        "            coefficients, gaussian_raw, sigma, alpha, pdf,\n",
        "            cp4_pre_provider_planes,\n"
        "        )\n"
        "        supply_k4, supply_k31, supply_k22 = _push_k22_residual_supply(\n"
        "            k22_supply_residual, weight_context, diagonal_context\n"
        "        )\n"
        "        pre_k4 = pre_k4 + supply_k4\n"
        "        pre_k31 = pre_k31 + supply_k31\n"
        "        pre_k22 = pre_k22 + supply_k22\n"
        "\n"
        "        (\n"
        "            coefficients, gaussian_raw, sigma, alpha, pdf,\n",
    ),
    # S7: final layer diagonal supply.
    (
        "                + _cp4_read_diagonal(cp4_lambda, pushed_u)\n"
        "            )\n",
        "                + _cp4_read_diagonal(cp4_lambda, pushed_u)\n"
        "                + _push_k22_residual_supply_diagonal(\n"
        "                    k22_supply_residual, weight_context\n"
        "                )\n"
        "            )\n",
    ),
]


def main():
    source = BASE.read_text()
    assert hashlib.sha256(source.encode()).hexdigest() == BASE_SHA
    patched = source
    for index, (old, new) in enumerate(REPLACEMENTS, start=1):
        count = patched.count(old)
        assert count == 1, f"anchor S{index} matched {count} times"
        patched = patched.replace(old, new)
    patched += APPENDED
    OUT.write_text(patched)
    receipt = {
        "schema": "supply-card-builder-receipt-v1",
        "base": "public_noeigh_card_v1/estimator.py (graded 328439)",
        "base_sha256": BASE_SHA,
        "output_sha256": hashlib.sha256(patched.encode()).hexdigest(),
        "anchors": len(REPLACEMENTS),
        "cholesky_free": "cholesky" not in patched,
    }
    (HERE / "BUILD_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
