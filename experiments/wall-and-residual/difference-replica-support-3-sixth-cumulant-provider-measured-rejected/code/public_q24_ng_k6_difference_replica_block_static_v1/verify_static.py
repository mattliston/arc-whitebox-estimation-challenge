#!/usr/bin/env python3
"""Exact static checks for the difference-replica support-3+ K6 design."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path


ROOT = Path(__file__).resolve().parent
N = 1024
LAYERS = 15
R6, R42, R222 = 512, 192, 128
SECTORS = (
    (4, 1, 1),
    (3, 2, 1),
    (2, 2, 2),
    (3, 1, 1, 1),
    (2, 2, 1, 1),
    (2, 1, 1, 1, 1),
    (1, 1, 1, 1, 1, 1),
)


def set_partitions(items):
    items = tuple(items)
    if not items:
        yield ()
        return
    first = items[0]
    for rest in set_partitions(items[1:]):
        yield ((first,),) + rest
        for index in range(len(rest)):
            block = tuple(sorted((first,) + rest[index]))
            yield tuple(sorted(rest[:index] + (block,) + rest[index + 1 :]))


PARTITIONS = {
    size: tuple(sorted(set(set_partitions(range(size)))))
    for size in range(7)
}


def product(values):
    result = Fraction(1)
    for value in values:
        result *= value
    return result


def dot(left, right):
    return sum((a * b for a, b in zip(left, right)), Fraction(0))


def alpha_grid(lam):
    return itertools.product(*(range(value + 1) for value in lam))


def power_sum(factors, w, alpha):
    if not any(alpha):
        return Fraction(len(w))
    return sum(
        product((factors[b][i] * w[i]) ** alpha[b] for b in range(len(alpha)))
        for i in range(len(w))
    )


def support3_block_formula(factors, lam, w):
    p = {alpha: power_sum(factors, w, alpha) for alpha in alpha_grid(lam)}
    full = product(dot(factor, w) ** multiplicity for factor, multiplicity in zip(factors, lam))
    p_lam = p[tuple(lam)]
    pair_twice = Fraction(0)
    for alpha in alpha_grid(lam):
        complement = tuple(lam[b] - alpha[b] for b in range(len(lam)))
        coefficient = math.prod(math.comb(lam[b], alpha[b]) for b in range(len(lam)))
        pair_twice += coefficient * (p[alpha] * p[complement] - p_lam)
    return full - pair_twice / 2 + (len(w) - 2) * p_lam


def support3_block_direct(factors, lam, w):
    slots = []
    for factor, multiplicity in zip(factors, lam):
        slots.extend([factor] * multiplicity)
    answer = Fraction(0)
    for coordinates in itertools.product(range(len(w)), repeat=6):
        if len(set(coordinates)) < 3:
            continue
        answer += product(
            slots[slot][coordinate] * w[coordinate]
            for slot, coordinate in enumerate(coordinates)
        )
    return answer


def sector_block_direct(factors, lam, w, sector):
    slots = []
    for factor, multiplicity in zip(factors, lam):
        slots.extend([factor] * multiplicity)
    answer = Fraction(0)
    for coordinates in itertools.product(range(len(w)), repeat=6):
        counts = tuple(sorted(Counter(coordinates).values(), reverse=True))
        if counts != sector:
            continue
        answer += product(
            slots[slot][coordinate] * w[coordinate]
            for slot, coordinate in enumerate(coordinates)
        )
    return answer


def sector_table(lam, sector):
    labels = []
    for factor_index, multiplicity in enumerate(lam):
        labels.extend([factor_index] * multiplicity)
    terms = defaultdict(int)
    for outer in PARTITIONS[6]:
        if tuple(sorted(map(len, outer), reverse=True)) != sector:
            continue
        exponent_vectors = []
        for block in outer:
            exponent = [0] * len(lam)
            for slot in block:
                exponent[labels[slot]] += 1
            exponent_vectors.append(tuple(exponent))
        for inner in PARTITIONS[len(outer)]:
            coefficient = 1
            factors = []
            for merged_blocks in inner:
                coefficient *= (-1) ** (len(merged_blocks) - 1) * math.factorial(
                    len(merged_blocks) - 1
                )
                factors.append(
                    tuple(
                        sum(exponent_vectors[index][axis] for index in merged_blocks)
                        for axis in range(len(lam))
                    )
                )
            terms[tuple(sorted(factors))] += coefficient
    return {key: value for key, value in terms.items() if value}


def sector_block_formula(factors, lam, w, sector):
    p = {alpha: power_sum(factors, w, alpha) for alpha in alpha_grid(lam)}
    answer = Fraction(0)
    for monomial, coefficient in sector_table(lam, sector).items():
        answer += coefficient * product(p[alpha] for alpha in monomial)
    return answer


def raw_moments(law, w):
    return [
        sum(probability * dot(point, w) ** degree for point, probability in law)
        for degree in range(7)
    ]


def cumulant6_from_moments(m):
    return (
        m[6]
        - 6 * m[5] * m[1]
        - 15 * m[4] * m[2]
        - 10 * m[3] * m[3]
        + 30 * m[4] * m[1] * m[1]
        + 120 * m[3] * m[2] * m[1]
        - 120 * m[3] * m[1] ** 3
        + 30 * m[2] ** 3
        - 270 * m[2] * m[2] * m[1] * m[1]
        + 360 * m[2] * m[1] ** 4
        - 120 * m[1] ** 6
    )


def difference_moments(m):
    result = []
    for degree in range(7):
        result.append(
            sum(
                Fraction(math.comb(degree, left))
                * (-1) ** (degree - left)
                * m[left]
                * m[degree - left]
                for left in range(degree + 1)
            )
        )
    return result


def difference_replica_cumulant6(law, w):
    d = difference_moments(raw_moments(law, w))
    assert d[1] == d[3] == d[5] == 0
    return (d[6] - 15 * d[4] * d[2] + 30 * d[2] ** 3) / 2


def cumulant6(law, w):
    return cumulant6_from_moments(raw_moments(law, w))


def support3_cumulant_action(law, w, provider=False):
    fn = difference_replica_cumulant6 if provider else cumulant6
    width = len(w)
    full = fn(law, w)
    singles = sum(
        fn(law, tuple(w[i] if i == j else Fraction(0) for i in range(width)))
        for j in range(width)
    )
    pairs = sum(
        fn(
            law,
            tuple(
                w[index] if index == left or index == right else Fraction(0)
                for index in range(width)
            ),
        )
        for left in range(width)
        for right in range(left + 1, width)
    )
    return full - pairs + (width - 2) * singles


def parity_law(theta):
    law = []
    for signs in itertools.product((-1, 1), repeat=6):
        parity = math.prod(signs)
        probability = Fraction(1 + theta * parity, 64)
        if probability:
            law.append((tuple(Fraction(1 + value) for value in signs), probability))
    return law


def push_relu(law, matrix):
    result = []
    for point, probability in law:
        output = tuple(
            max(
                Fraction(0),
                sum(point[row] * matrix[row][column] for row in range(len(point))),
            )
            for column in range(len(matrix[0]))
        )
        result.append((output, probability))
    return result


def marginal(law, coordinates):
    result = defaultdict(Fraction)
    for point, probability in law:
        result[tuple(point[index] for index in coordinates)] += probability
    return dict(result)


def verify_algebra():
    factors_by_type = {
        (6,): ((Fraction(1), Fraction(2), Fraction(-1), Fraction(3)),),
        (4, 2): (
            (Fraction(1), Fraction(-2), Fraction(3), Fraction(1)),
            (Fraction(2), Fraction(1), Fraction(-1), Fraction(4)),
        ),
        (2, 2, 2): (
            (Fraction(1), Fraction(2), Fraction(1), Fraction(3)),
            (Fraction(3), Fraction(-1), Fraction(2), Fraction(1)),
            (Fraction(2), Fraction(1), Fraction(4), Fraction(-2)),
        ),
    }
    w = (Fraction(2), Fraction(-1), Fraction(3), Fraction(1))
    term_counts = {}
    for lam, factors in factors_by_type.items():
        assert support3_block_formula(factors, lam, w) == support3_block_direct(
            factors, lam, w
        )
        sector_sum = Fraction(0)
        counts = []
        for sector in SECTORS:
            direct = sector_block_direct(factors, lam, w, sector)
            formula = sector_block_formula(factors, lam, w, sector)
            assert formula == direct
            sector_sum += formula
            counts.append(len(sector_table(lam, sector)))
        assert sector_sum == support3_block_formula(factors, lam, w)
        term_counts["".join(map(str, lam))] = counts
    assert term_counts == {
        "6": [4, 5, 3, 7, 8, 10, 11],
        "42": [9, 13, 6, 19, 22, 28, 29],
        "222": [16, 29, 12, 42, 52, 65, 66],
    }
    return term_counts


def verify_identifiability_and_resets():
    minus = parity_law(-1)
    plus = parity_law(1)
    for size in range(6):
        for coordinates in itertools.combinations(range(6), size):
            assert marginal(minus, coordinates) == marginal(plus, coordinates)
    witness = tuple(map(Fraction, (1, 1, 1, -1, -1, -1)))
    relu_mean_minus = sum(
        probability * max(Fraction(0), dot(point, witness))
        for point, probability in minus
    )
    relu_mean_plus = sum(
        probability * max(Fraction(0), dot(point, witness))
        for point, probability in plus
    )
    assert relu_mean_plus == Fraction(3, 4)
    assert relu_mean_minus == Fraction(9, 8)

    action_w = tuple(map(Fraction, (1, -2, 1, 1, -1, 2)))
    matrices = (
        (
            (1, 0, -1, 1, 0, 1),
            (0, 1, 1, -1, 1, 0),
            (1, -1, 0, 0, 1, 1),
            (-1, 1, 0, 1, 1, 0),
            (0, 1, -1, 1, 0, -1),
            (1, 0, 1, 0, -1, 1),
        ),
        (
            (1, 1, 0, -1, 0, 1),
            (0, 1, -1, 1, 1, 0),
            (1, 0, 1, 0, -1, -1),
            (0, -1, 1, 1, 0, 1),
            (1, 1, 0, 0, 1, -1),
            (-1, 0, 1, 1, 1, 0),
        ),
    )
    reset_actions = {}
    for theta, law in ((-1, minus), (1, plus)):
        values = []
        for matrix in matrices:
            law = push_relu(law, matrix)
            direct = support3_cumulant_action(law, action_w, provider=False)
            replica = support3_cumulant_action(law, action_w, provider=True)
            assert replica == direct
            assert direct != 0
            values.append(str(direct))
        reset_actions[str(theta)] = values
    assert reset_actions == {
        "-1": ["13361768745/8192", "323817022545/131072"],
        "1": ["-7839816255/8192", "-14395982445/8192"],
    }

    # A fixed law with all seven exact nonzero sector actions.  If S is a
    # Rademacher scalar and X=S*a, cum6(X)=16*a^6.
    a = tuple(map(Fraction, (1, 2, 3, 4, 5, 6)))
    ones = (Fraction(1),) * 6
    rademacher_sector_actions = {}
    for sector in SECTORS:
        value = 16 * sector_block_direct((a,), (6,), ones, sector)
        assert value > 0
        rademacher_sector_actions["".join(map(str, sector))] = str(value)
    return {
        "relu_means": {"plus": str(relu_mean_plus), "minus": str(relu_mean_minus)},
        "two_reset_support3_actions": reset_actions,
        "rademacher_sector_actions": rademacher_sector_actions,
    }


def verify_cost():
    differences = R6 + R42 + R222
    paths = 2 * differences
    map_units = 6 * R6 + 14 * R42 + 26 * R222
    count42 = Counter(
        left + right
        for left in range(5)
        for right in range(3)
        if left + right
    )
    count222 = Counter(
        first + second + third
        for first in range(3)
        for second in range(3)
        for third in range(3)
        if first + second + third
    )
    rows = {
        degree: R6 + count42[degree] * R42 + count222[degree] * R222
        for degree in range(1, 7)
    }
    assert rows == {1: 1280, 2: 1856, 3: 1984, 4: 1856, 5: 1280, 6: 832}
    assert sum(rows.values()) == map_units == 9088

    costs = {
        "iid_normal_generation": 16 * paths * N,
        "path_float32_matmuls": LAYERS * paths * N * (2 * N - 1),
        "path_relu": LAYERS * paths * N,
        "difference_subtractions": LAYERS * differences * N,
        "weight64_cast_and_powers": LAYERS * 12 * N * N,
        "difference64_casts": LAYERS * 2 * differences * N,
        "factor_power_and_mixed_products": LAYERS
        * N
        * (10 * R6 + 24 * R42 + 62 * R222),
        "six_degree_batched_float64_matmuls": LAYERS
        * 2
        * N
        * (2 * N - 1)
        * map_units,
        "total_q3plus_polynomial_assembly": LAYERS
        * N
        * (70 * R6 + 136 * R42 + 234 * R222),
        "type_reductions_and_scales": LAYERS * N * 2 * (differences + 2),
        "d4_association_upper": LAYERS * (12 * N * N + 20 * N),
    }
    assert sum(costs.values()) == 625_985_232_896
    padding = 2_786_811_904
    incremental_cap = sum(costs.values()) + padding
    assert incremental_cap == 628_772_044_800
    d4_max = 1_356_595_288_584
    budget = 2**41
    assert d4_max + incremental_cap == 1_985_367_333_384
    assert budget - d4_max - incremental_cap == 213_655_922_168

    memory_components = {
        "path_pingpong_fp32": 2 * paths * N * 4,
        "difference_fp32": differences * N * 4,
        "weight_powers_fp64": 6 * N * N * 8,
        "factor_input_fp64": max(rows.values()) * N * 8,
        "power_sum_outputs_fp64": map_units * N * 8,
        "all_factor_powers_fp64": (6 * R6 + 6 * R42 + 6 * R222) * N * 8,
        "sector_tuple_scratch_fp64": R6 * N * 8,
        "split_sector_outputs_fp64": 2 * 7 * N * 8,
        "index_coefficient_reserve": 16 * 1024 * 1024,
    }
    assert sum(memory_components.values()) == 220_053_504
    arena_cap = 256 * 1024 * 1024
    packed_d4_peak = 5_411_373_056
    assert packed_d4_peak + arena_cap == 5_679_808_512
    assert 8 * 1024**3 - packed_d4_peak - arena_cap == 2_910_126_080
    return {
        "packed_rows_by_degree": {str(key): value for key, value in rows.items()},
        "map_row_units": map_units,
        "costs": costs,
        "enumerated_incremental_core_flops": sum(costs.values()),
        "fixed_padding_flops": padding,
        "incremental_admission_cap_flops": incremental_cap,
        "combined_max_flops": d4_max + incremental_cap,
        "margin_to_2pow41": budget - d4_max - incremental_cap,
        "memory_components": memory_components,
        "explicit_incremental_simultaneous_bytes": sum(memory_components.values()),
        "incremental_arena_cap_bytes": arena_cap,
        "combined_projected_peak_bytes": packed_d4_peak + arena_cap,
    }


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    term_counts = verify_algebra()
    oracle = verify_identifiability_and_resets()
    resource = verify_cost()
    preregistration = json.loads((ROOT / "PREREGISTRATION.json").read_text())
    assert preregistration["access"] == {
        "rows_opened": 0,
        "weights_opened": 0,
        "targets_opened": 0,
        "scores_computed": 0,
        "estimators_or_fixtures_called": 0,
        "source_gates_run": 0,
        "graph_edits": 0,
        "releases": 0,
        "submissions": 0,
    }
    receipt = {
        "schema": "q24-ng-k6-difference-replica-block-static-verification-v1",
        "status": "PASS_STATIC_DESIGN_ONLY__SOURCE_GATE_NOT_RUN",
        "checks": {
            "difference_replica_unbiasedness_exact": True,
            "support3_block_formula_exact": True,
            "all_seven_sector_partition_tables_exact": True,
            "p_plus_minus_lower_marginal_nonidentifiability_exact": True,
            "p_plus_minus_relu_witness_exact": True,
            "two_rational_relu_resets_exact": True,
            "fixed_rademacher_all_sector_oracle_nonzero": True,
            "six_gemm_row_pack_exact": True,
            "conservative_integer_cost_and_memory_ledgers_exact": True,
            "row_target_weight_estimator_or_gate_access": False,
        },
        "sector_term_counts": term_counts,
        "finite_law_oracle": oracle,
        "resource": resource,
        "bindings": {
            "DESIGN.md": sha256(ROOT / "DESIGN.md"),
            "PREREGISTRATION.json": sha256(ROOT / "PREREGISTRATION.json"),
            "verify_static.py": sha256(ROOT / "verify_static.py"),
        },
        "next_action": "Independent static review; only then authorize the fixed target-blind 315-cell split action-energy gate.",
    }
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
