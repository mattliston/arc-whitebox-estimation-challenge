"""Literal unit-reference nonnegative-law moment diagnostics for FB-EC4-32.

This pure-stdlib gate neither imports nor executes the estimator.  It freezes
the diagnostic formulas that a later literal closure replay must report.
Any nonfinite value or negative Hankel margin is a blocker; raw third/fourth
moments are never repaired by independent scalar clipping.
"""

from __future__ import annotations

import json
import math


LITERAL_REFERENCE_LAWS = (
    ("relu_like_unit_law", ((0.0, 0.46), (0.31, 0.19), (0.92, 0.23), (1.77, 0.12))),
    ("sparse_unit_law", ((0.0, 0.71), (0.04, 0.08), (0.63, 0.13), (2.80, 0.08))),
    ("broad_unit_law", ((0.0, 0.37), (0.22, 0.17), (0.88, 0.29), (1.54, 0.11), (3.10, 0.06))),
    ("tail_unit_law", ((0.0, 0.58), (0.015, 0.07), (0.41, 0.18), (1.33, 0.12), (4.75, 0.05))),
)


def raw_moments(support: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    probability = sum(weight for _value, weight in support)
    assert abs(probability - 1.0) <= 1.0e-15
    return tuple(
        sum(weight * value ** order for value, weight in support)
        for order in range(1, 5)
    )


def determinant3(matrix: tuple[tuple[float, ...], ...]) -> float:
    return (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )


def diagnose(name: str, support: tuple[tuple[float, float], ...]) -> dict[str, float | str]:
    r1, r2, r3, r4 = raw_moments(support)
    variance = r2 - r1 * r1
    assert r1 > 0.0 and variance > 0.0
    shifted_determinant = r1 * r3 - r2 * r2
    full_determinant = determinant3(
        ((1.0, r1, r2), (r1, r2, r3), (r2, r3, r4))
    )
    shifted_margin = r3 - r2 * r2 / r1
    full_margin = r4 - r2 * r2 - (r3 - r1 * r2) ** 2 / variance
    k3 = r3 - 3.0 * r1 * r2 + 2.0 * r1 ** 3
    k4 = r4 - 4.0 * r1 * r3 - 3.0 * r2 * r2 + 12.0 * r1 * r1 * r2 - 6.0 * r1 ** 4
    standardized_k3 = k3 / variance ** 1.5
    standardized_k4 = k4 / (variance * variance)
    values = (
        r1, r2, r3, r4, variance, shifted_determinant, full_determinant,
        shifted_margin, full_margin, standardized_k3, standardized_k4,
    )
    assert all(math.isfinite(value) for value in values)
    tolerance = 2.0e-14 * max(1.0, r4)
    assert shifted_determinant >= -tolerance
    assert full_determinant >= -tolerance
    assert shifted_margin >= -tolerance
    assert full_margin >= -tolerance
    return {
        "name": name,
        "raw1": r1,
        "raw2": r2,
        "raw3": r3,
        "raw4": r4,
        "variance": variance,
        "shifted_hankel_determinant": shifted_determinant,
        "full_hankel_determinant": full_determinant,
        "shifted_hankel_margin": shifted_margin,
        "full_hankel_margin": full_margin,
        "standardized_k3": standardized_k3,
        "standardized_k4": standardized_k4,
    }


def verify() -> dict[str, object]:
    rows = [diagnose(name, support) for name, support in LITERAL_REFERENCE_LAWS]
    summary = {
        "schema": "fb-ec4-32-moment-feasibility-formula-unit-v1",
        "status": "PASS_DIAGNOSTIC_FORMULA_ONLY__ACTUAL_CLOSURE_REPLAY_REQUIRED",
        "rows": rows,
        "summary": {
            "min_shifted_hankel_determinant": min(row["shifted_hankel_determinant"] for row in rows),
            "min_full_hankel_determinant": min(row["full_hankel_determinant"] for row in rows),
            "min_shifted_hankel_margin": min(row["shifted_hankel_margin"] for row in rows),
            "min_full_hankel_margin": min(row["full_hankel_margin"] for row in rows),
            "min_standardized_k3": min(row["standardized_k3"] for row in rows),
            "max_standardized_k3": max(row["standardized_k3"] for row in rows),
            "min_standardized_k4": min(row["standardized_k4"] for row in rows),
            "max_standardized_k4": max(row["standardized_k4"] for row in rows),
        },
        "qualification_rule": (
            "Any nonfinite or negative shifted/full Hankel margin on the later "
            "literal official replay is HOLD; do not scalar-clip raw3/raw4."
        ),
    }
    return summary


def main() -> None:
    print(json.dumps(verify(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
