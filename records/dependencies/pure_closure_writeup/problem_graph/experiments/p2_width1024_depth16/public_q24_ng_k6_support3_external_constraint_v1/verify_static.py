#!/usr/bin/env python3
"""Target-free exact checks for the non-Gaussian support-3+ K6 constraint."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
RECEIPT = HERE / "STATIC_CONSTRAINT_RECEIPT.json"

N = 1024
TRANSITIONS = 15
CAP = 2**41
D4_MAX = 1_356_595_288_584
EXPECTED = {
    "symmetric_order6_entries": 1_624_866_254_968_320,
    "support_at_most_two_entries": 2_619_904,
    "support_at_least_three_entries": 1_624_866_252_348_416,
    "dense_fp64_bytes": 12_998_930_018_787_328,
    "dense_ordinary_ops_per_transition": 3_327_726_084_809_554_944,
    "dense_ordinary_ops_15_transitions": 49_915_891_272_143_324_160,
    "d4_headroom": 842_427_966_968,
    "cp_official_action_flops_per_rank": 377_487_360,
    "cp_max_rank_before_any_source": 2231,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def quotient_formula(z: tuple[int, ...]) -> int:
    p = {a: sum(value**a for value in z) for a in range(1, 7)}
    return (
        p[1] ** 6
        - 6 * p[1] * p[5]
        - 15 * p[2] * p[4]
        - 10 * p[3] ** 2
        + 30 * p[6]
    )


def quotient_bruteforce(z: tuple[int, ...]) -> int:
    total = 0
    for indices in itertools.product(range(len(z)), repeat=6):
        if len(set(indices)) >= 3:
            product = 1
            for index in indices:
                product *= z[index]
            total += product
    return total


def parity_law(theta: int, signs: tuple[int, ...]) -> Fraction:
    parity = math.prod(signs)
    return Fraction(1 + theta * parity, 64)


def nonidentifiability_checks() -> dict[str, bool]:
    states = list(itertools.product((-1, 1), repeat=6))
    # Every proper-coordinate monomial agrees, so in particular every <=5
    # marginal agrees.  Test all 64 proper subsets exactly.
    proper_equal = True
    for mask in range(1 << 6):
        if mask == (1 << 6) - 1:
            continue
        moments = []
        for theta in (-1, 1):
            value = Fraction(0)
            for signs in states:
                monomial = math.prod(
                    signs[i] for i in range(6) if mask & (1 << i)
                )
                value += parity_law(theta, signs) * monomial
            moments.append(value)
        proper_equal &= moments[0] == moments[1]

    all_distinct = {}
    relu = {}
    weights = (1, 1, 1, -1, -1, -1)
    for theta in (-1, 1):
        sixth = Fraction(0)
        mean = Fraction(0)
        for signs in states:
            probability = parity_law(theta, signs)
            sixth += probability * math.prod(signs)
            affine = sum(w * (1 + x) for w, x in zip(weights, signs))
            mean += probability * max(affine, 0)
        all_distinct[theta] = sixth
        relu[theta] = mean
    return {
        "all_proper_coordinate_moments_equal": proper_equal,
        "all_distinct_k6_is_theta": all_distinct == {-1: -1, 1: 1},
        "relu_theta_plus_is_three_quarters": relu[1] == Fraction(3, 4),
        "relu_theta_minus_is_nine_eighths": relu[-1] == Fraction(9, 8),
    }


def main() -> None:
    total = math.comb(N + 5, 6)
    low = N + 5 * math.comb(N, 2)
    high = total - low
    dense_one = N * (2 * high - 1)
    dense_all = TRANSITIONS * dense_one
    headroom = CAP - D4_MAX
    cp_per_rank = 24 * TRANSITIONS * N * N
    derived = {
        "symmetric_order6_entries": total,
        "support_at_most_two_entries": low,
        "support_at_least_three_entries": high,
        "dense_fp64_bytes": 8 * high,
        "dense_ordinary_ops_per_transition": dense_one,
        "dense_ordinary_ops_15_transitions": dense_all,
        "d4_headroom": headroom,
        "cp_official_action_flops_per_rank": cp_per_rank,
        "cp_max_rank_before_any_source": headroom // cp_per_rank,
    }
    assert derived == EXPECTED
    assert 2231 * cp_per_rank == 842_174_300_160
    assert 2232 * cp_per_rank == 842_551_787_520
    assert 2231 * cp_per_rank <= headroom < 2232 * cp_per_rank

    quotient_vectors = (
        (1, 2, -1),
        (2, -3, 1, 4),
        (0, 1, -2, 3, -4),
    )
    quotient_exact = all(
        quotient_formula(z) == quotient_bruteforce(z) for z in quotient_vectors
    )
    identity = nonidentifiability_checks()
    assert quotient_exact and all(identity.values())

    receipt = json.loads(RECEIPT.read_text())
    assert receipt["status"] == "PASS_STATIC_EXTERNAL_CONSTRAINT_ONLY"
    assert receipt["arithmetic"] == derived
    assert receipt["checks"]["exact_support3_quotient_parity"] is True
    assert receipt["checks"]["exact_lower_state_nonidentifiability"] is True
    for relative, expected_sha in receipt["evidence_sha256"].items():
        assert sha256(ROOT / relative) == expected_sha
    assert sha256(HERE / "DERIVATION.md") == receipt["bindings"]["derivation_sha256"]
    assert sha256(Path(__file__)) == receipt["bindings"]["verifier_sha256"]
    assert receipt["firewall"] == {
        "estimator_called": False,
        "graph_mutated": False,
        "rank_or_row_selected": False,
        "release_or_run_created": False,
        "submission_made": False,
        "target_score_weights_or_names_read": False,
    }
    print("PASS_STATIC_EXTERNAL_CONSTRAINT_ONLY")


if __name__ == "__main__":
    main()
