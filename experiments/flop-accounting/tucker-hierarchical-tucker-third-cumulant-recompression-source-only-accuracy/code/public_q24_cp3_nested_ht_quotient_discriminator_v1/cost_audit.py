#!/usr/bin/env python3
"""Exact dense-FlopScope ledger for the quotient nested-HT fallback."""

from __future__ import annotations

import json


N = 1024
R40_NON_CP3_BASE = 425_860_164_095
ACTUAL_BOUNDARY = 573_816_383_391
ROUNDED_BOUNDARY = 575_754_209_978


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def row(r: int, k: int) -> dict[str, int]:
    n = N
    basis = 15 * (
        mm(n, n, r)
        + mm(r, n, r)
        + 18 * r**3
        + mm(n, r, r)
        + n * r
        + 4 * r
    )
    weight_push = mm(n, n, r)
    basis_projection = n * r + mm(r, n, r)
    core_transport = 2 * mm(r, r, r * k) + mm(k, k, r * r)
    raw_birth = (
        n * n
        + mm(r, n, n)
        + n * r * r
        + n * k
        + mm(r * r, n, k)
    )
    core_add = r * r * k
    quotient_subtraction = mm(n, n, k) + mm(r * r, n, k) + n + r * r * k
    full_read = (
        n * r * r
        + mm(n, r * r, k)
        + mm(n, k, n)
        + n * r * k
        + mm(n, r * k, r)
        + mm(n, r, n)
        + 2 * n * n
    )
    final_read = n * r * r + mm(n, r * r, k) + (2 * n * k - n)
    raw = (
        basis
        + 15 * weight_push
        + 14 * (basis_projection + core_transport + core_add)
        + 15 * raw_birth
        + 29 * full_read
        + final_read
    )
    quotient = raw + 15 * quotient_subtraction
    total = R40_NON_CP3_BASE + quotient
    sym_dof = r * (r + 1) * (r + 2) // 6 - (
        (r - k) * (r - k + 1) * (r - k + 2) // 6
    )
    return {
        "r": r,
        "k": k,
        "basis_construction": basis,
        "gate_aware_basis_extra": 15 * n * r,
        "weight_push_each": weight_push,
        "basis_projection_each": basis_projection,
        "core_transport_each": core_transport,
        "raw_birth_each": raw_birth,
        "quotient_subtraction_each": quotient_subtraction,
        "full_read_each": full_read,
        "final_read": final_read,
        "raw_core_cp3": raw,
        "quotient_cp3": quotient,
        "combined_r40": total,
        "actual_margin": ACTUAL_BOUNDARY - total,
        "rounded_margin": ROUNDED_BOUNDARY - total,
        "symmetric_span_dof": sym_dof,
        "core_float32_bytes": 4 * r * r * k,
        "row_pair_float32_bytes": 4 * n * r * r,
        "row_mixed_float32_bytes": 4 * n * r * k,
    }


def main() -> None:
    result = {
        "schema": "q24-cp3-quotient-nested-ht-cost-v1",
        "mm": "m*n*(2*k-1)",
        "r40_non_cp3_base": R40_NON_CP3_BASE,
        "actual_boundary": ACTUAL_BOUNDARY,
        "rounded_boundary": ROUNDED_BOUNDARY,
        "rows": [
            row(96, 64),
            row(112, 48),
            row(128, 32),
            row(136, 24),
            row(144, 24),
            row(160, 16),
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
