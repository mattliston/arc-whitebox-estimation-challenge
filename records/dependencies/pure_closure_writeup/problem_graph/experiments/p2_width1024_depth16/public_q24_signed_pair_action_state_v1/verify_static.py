#!/usr/bin/env python3
"""Static verifier for the signed pair-action recurrent-state freeze.

This script performs integer/rational algebra and reads only the sibling
preregistration plus frozen static authority.  It imports no estimator and
opens no public row, target, score, name, board, package, or network path.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PREREG = HERE / "PREREGISTRATION.json"
PACKED = HERE.parent / "public_q24_d4_lifetime_packed_reinvestment_v1" / "PACKED_D4_STATIC_RECEIPT.json"

N = 1024
PAIR_LAYERS = 13
STATE_LAYERS = 14
HEADS = 16
SECTORS = 8
PROBES = 2
J_FEATURES = 36
STATE = 16
HIDDEN = 64
BUDGET = 2_199_023_255_552
PACKED_HOT_MAX = 1_356_595_288_584
ADDED_CAP = 1_520_000_000
PACKED_PEAK = 5_411_373_056
ALLOCATION_UPPER = 62_627_840


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matvec(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def row_summary(row):
    return {
        "sum": sum(row),
        "sum_squares": sum(x * x for x in row),
        "sum_absolute": sum(abs(x) for x in row),
        "max_absolute": max(abs(x) for x in row),
        "positive_count": sum(x > 0 for x in row),
        "negative_count": sum(x < 0 for x in row),
    }


def relu_expectation(matching):
    # Xi=s_i*eps_i*A_edge and affine coefficients cancel s_i to effective
    # integer weights [1,2,3,4].  There are 2^2 amplitude and 2^4 sign states.
    total = Fraction(0)
    count = 0
    for amplitudes in itertools.product((1, 2), repeat=2):
        coordinate_amplitude = [None] * 4
        for amplitude, edge in zip(amplitudes, matching):
            for coordinate in edge:
                coordinate_amplitude[coordinate] = amplitude
        for signs in itertools.product((-1, 1), repeat=4):
            value = sum(
                Fraction(weight * amplitude * sign)
                for weight, amplitude, sign in zip(
                    (1, 2, 3, 4), coordinate_amplitude, signs
                )
            )
            total += max(value, 0)
            count += 1
    return total / count


def verify_collision():
    probe = [1, -1, 1, -1]
    matrix_a = [
        [0, -1, -1, 1],
        [-1, 0, 1, -1],
        [-1, 1, 0, -1],
        [1, -1, -1, 0],
    ]
    matrix_b = [
        [0, -1, 1, -1],
        [-1, 0, -1, 1],
        [1, -1, 0, -1],
        [-1, 1, -1, 0],
    ]
    wanted = {
        "sum": -1,
        "sum_squares": 3,
        "sum_absolute": 3,
        "max_absolute": 1,
        "positive_count": 1,
        "negative_count": 2,
    }
    assert all(row_summary(row) == wanted for row in matrix_a + matrix_b)
    assert matvec(matrix_a, probe) == [-1, 1, -1, 1]
    assert matvec(matrix_b, probe) == [3, -3, 3, -3]

    # Check equivariance exhaustively for all 4! permutations.
    for permutation in itertools.permutations(range(4)):
        permuted_probe = [probe[permutation[i]] for i in range(4)]
        permuted_matrix = [
            [matrix_a[permutation[i]][permutation[j]] for j in range(4)]
            for i in range(4)
        ]
        expected = [matvec(matrix_a, probe)[permutation[i]] for i in range(4)]
        assert matvec(permuted_matrix, permuted_probe) == expected

    return {
        "a_action": matvec(matrix_a, probe),
        "b_action": matvec(matrix_b, probe),
        "permutations_checked": 24,
    }


def verify_physical_witness():
    # E[A^2]=5/2, E[A^4]=17/2.
    variance = Fraction(5, 2)
    raw_fourth = Fraction(17, 2)
    marginal_k4 = raw_fourth - 3 * variance * variance
    shared_k22 = raw_fourth - variance * variance
    standardized_k4 = marginal_k4 / (variance * variance)
    standardized_k22 = shared_k22 / (variance * variance)
    assert marginal_k4 == Fraction(-41, 4)
    assert standardized_k4 == Fraction(-41, 25)
    assert shared_k22 == Fraction(9, 4)
    assert standardized_k22 == Fraction(9, 25)

    matching_zero = ((0, 1), (2, 3))
    matching_one = ((0, 2), (1, 3))
    mean_zero = relu_expectation(matching_zero)
    mean_one = relu_expectation(matching_one)
    assert mean_zero == Fraction(109, 32)
    assert mean_one == Fraction(7, 2)
    assert mean_one - mean_zero == Fraction(3, 32)

    # A rational scale-contrast probe.  sigma is proportional to s, so the
    # common sqrt(5/2) cancels from sigma/mean(sigma).
    scales = [Fraction(1), Fraction(2), Fraction(4), Fraction(8)]
    mean_scale = sum(scales) / 4
    contrasts = [scale / mean_scale - 1 for scale in scales]
    sat = lambda x: x / (1 + abs(x))
    probe = [sat(value) for value in contrasts]

    def k22_action(matching):
        partner = {}
        for i, j in matching:
            partner[i] = j
            partner[j] = i
        return [standardized_k22 * probe[partner[i]] for i in range(4)]

    action_zero = k22_action(matching_zero)
    action_one = k22_action(matching_one)
    assert action_zero != action_one
    return {
        "standardized_k4": str(standardized_k4),
        "matched_standardized_k22": str(standardized_k22),
        "relu_mean_zero": str(mean_zero),
        "relu_mean_one": str(mean_one),
        "separation": str(mean_one - mean_zero),
        "k22_action_zero": [str(value) for value in action_zero],
        "k22_action_one": [str(value) for value in action_one],
    }


def verify_resources(prereg):
    pair_action_per_layer = (
        34 * N * N
        + SECTORS * PROBES * N * (2 * N - 1)
        + 134 * N
    )
    pair_action = PAIR_LAYERS * pair_action_per_layer
    update_per_layer = (
        68 * N
        + N * HIDDEN * (2 * 68 - 1)
        + 16 * HIDDEN * N
        + N * STATE * (2 * HIDDEN - 1)
        + 16 * STATE * N
        + N * (2 * STATE - 1)
        + 20 * N
    )
    updates = STATE_LAYERS * update_per_layer
    transport_per_layer = N * STATE * (2 * N - 1) + 30 * N * STATE + 10 * N
    transports = PAIR_LAYERS * transport_per_layer
    j_feature_copy = STATE_LAYERS * J_FEATURES * N
    derived = pair_action + updates + transports + j_feature_copy
    assert derived == 1_517_332_480
    assert prereg["resource_ledger"]["derived_added_flops"] == derived
    assert prereg["resource_ledger"]["added_flop_cap"] == ADDED_CAP
    assert ADDED_CAP - derived == 2_667_520

    combined = PACKED_HOT_MAX + ADDED_CAP
    assert combined == 1_358_115_288_584
    assert BUDGET - combined == 840_907_966_968
    assert prereg["resource_ledger"]["combined_maximum_flops"] == combined
    assert prereg["resource_ledger"]["combined_flop_headroom"] == BUDGET - combined
    assert PACKED_PEAK + ALLOCATION_UPPER == 5_474_000_896
    assert 2**33 - (PACKED_PEAK + ALLOCATION_UPPER) == 3_115_933_696
    return {
        "pair_action_flops": pair_action,
        "state_update_flops": updates,
        "state_transport_flops": transports,
        "j_feature_copy_flops": j_feature_copy,
        "derived_added_flops": derived,
        "sealed_added_cap": ADDED_CAP,
        "combined_maximum_flops": combined,
        "combined_flop_headroom": BUDGET - combined,
        "allocation_upper_bytes": ALLOCATION_UPPER,
        "combined_projected_peak_bytes": PACKED_PEAK + ALLOCATION_UPPER,
    }


def main():
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    packed = json.loads(PACKED.read_text(encoding="utf-8"))
    assert prereg["status"] == "FROZEN_STATIC_ONLY_BEFORE_SOURCE_OR_TARGET_RUN"
    assert prereg["decision"] == "GO_STATIC_BUILD"
    assert prereg["scope"] == {
        "estimator_imports": 0,
        "estimator_calls": 0,
        "public_rows_opened": 0,
        "targets_opened": 0,
        "observer_selection_opened": 0,
        "names_opened": 0,
        "board_opened": False,
        "network": False,
        "package": False,
        "submission": False,
    }
    assert packed["production_exact_static_proof"]["maximum_call_flops"] == PACKED_HOT_MAX
    assert packed["projections_explicitly_not_proven"]["packed_process_peak_bytes"] == PACKED_PEAK
    assert prereg["authority"]["packed_d4_receipt_sha256"] == sha256(PACKED)
    assert prereg["state_law"]["learned_parameter_count"] == 19_200
    collision = verify_collision()
    physical = verify_physical_witness()
    resources = verify_resources(prereg)
    print(json.dumps({
        "status": "PASS_STATIC_ONLY",
        "decision": "GO_STATIC_BUILD_SOURCE_GATES_ONLY",
        "preregistration_sha256": sha256(PREREG),
        "packed_d4_receipt_sha256": sha256(PACKED),
        "collision": collision,
        "physical_witness": physical,
        "resources": resources,
        "scope": prereg["scope"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
