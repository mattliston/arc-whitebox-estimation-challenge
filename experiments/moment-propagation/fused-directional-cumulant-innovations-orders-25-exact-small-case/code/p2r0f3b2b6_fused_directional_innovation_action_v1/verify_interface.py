#!/usr/bin/env python3
"""Exact target-free verifier for the fused directional innovation interface.

The verifier uses Fraction arithmetic at n=5.  It proves that a connected
cumulant of a scalar projection contracts the complete dense symmetric
cumulant tensor, including every collision class through order five.  It also
checks the incumbent diag/plane projection formulas, the mixed-shell delta
identities, and the frozen production arithmetic ledger.
"""

from collections import Counter
from fractions import Fraction as F
from itertools import product
import json
from math import comb
from pathlib import Path


def partitions(items):
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for part in partitions(rest):
        for slot in range(len(part)):
            yield part[:slot] + [[first] + part[slot]] + part[slot + 1 :]
        yield [[first]] + part


def mean(values):
    return sum(values, F(0)) / len(values)


def joint_cumulant(variables):
    """Ordinary set-partition/Mobius cumulant of equal-length variables."""
    total = F(0)
    for part in partitions(list(range(len(variables)))):
        coefficient = F((-1) ** (len(part) - 1))
        factorial = 1
        for value in range(2, len(part)):
            factorial *= value
        coefficient *= factorial
        term = coefficient
        for block in part:
            term *= mean(
                [
                    product_values([variables[index][sample] for index in block])
                    for sample in range(len(variables[0]))
                ]
            )
        total += term
    return total


def product_values(values):
    out = F(1)
    for value in values:
        out *= value
    return out


def collision_label(index_tuple):
    counts = sorted(Counter(index_tuple).values(), reverse=True)
    return "[" + "".join(str(value) for value in counts) + "]"


def full_tensor(columns, order):
    n = len(columns)
    return {
        index_tuple: joint_cumulant([columns[index] for index in index_tuple])
        for index_tuple in product(range(n), repeat=order)
    }


def dense_contraction(tensor, direction):
    return sum(
        value * product_values([direction[index] for index in index_tuple])
        for index_tuple, value in tensor.items()
    )


def collision_contractions(tensor, direction):
    out = {}
    for index_tuple, value in tensor.items():
        label = collision_label(index_tuple)
        out[label] = out.get(label, F(0)) + value * product_values(
            [direction[index] for index in index_tuple]
        )
    return out


def explicit_incumbent_projection(tensor, direction, order):
    n = len(direction)
    if order == 3:
        return sum(tensor[(i, i, i)] * direction[i] ** 3 for i in range(n)) + 3 * sum(
            tensor[(i, i, j)] * direction[i] ** 2 * direction[j]
            for i in range(n)
            for j in range(n)
            if i != j
        )
    if order == 4:
        return (
            sum(tensor[(i, i, i, i)] * direction[i] ** 4 for i in range(n))
            + 4
            * sum(
                tensor[(i, i, i, j)] * direction[i] ** 3 * direction[j]
                for i in range(n)
                for j in range(n)
                if i != j
            )
            + 3
            * sum(
                tensor[(i, i, j, j)] * direction[i] ** 2 * direction[j] ** 2
                for i in range(n)
                for j in range(n)
                if i != j
            )
        )
    if order == 5:
        return (
            sum(tensor[(i, i, i, i, i)] * direction[i] ** 5 for i in range(n))
            + 5
            * sum(
                tensor[(i, i, i, i, j)] * direction[i] ** 4 * direction[j]
                for i in range(n)
                for j in range(n)
                if i != j
            )
            + 10
            * sum(
                tensor[(i, i, i, j, j)] * direction[i] ** 3 * direction[j] ** 2
                for i in range(n)
                for j in range(n)
                if i != j
            )
        )
    raise ValueError(order)


def expected_sector_counts(n, order):
    if order == 2:
        return {"[2]": n, "[11]": comb(n, 2)}
    if order == 3:
        return {"[3]": n, "[21]": n * (n - 1), "[111]": comb(n, 3)}
    if order == 4:
        return {
            "[4]": n,
            "[31]": n * (n - 1),
            "[22]": comb(n, 2),
            "[211]": n * comb(n - 1, 2),
            "[1111]": comb(n, 4),
        }
    if order == 5:
        return {
            "[5]": n,
            "[41]": n * (n - 1),
            "[32]": n * (n - 1),
            "[311]": n * comb(n - 1, 2),
            "[221]": comb(n, 2) * (n - 2),
            "[2111]": n * comb(n - 1, 3),
            "[11111]": comb(n, 5),
        }
    raise ValueError(order)


def production_cost(n, samples, rays, layers):
    projection = layers * samples * rays * (2 * n - 1)
    # Per ray/layer: N for the mean reduction; 9N for centering, powers
    # two through five, and four accumulations; four normalizations and six
    # connected-conversion operations.
    statistics = layers * rays * (10 * samples + 10)
    # Complete incumbent reference projection: powers; mean; dense covariance
    # quadratic form; three diagonals; five directed planes.
    incumbent_reference = layers * rays * (17 * n * n - 2 * n + 5)
    return {
        "sample_projection_flops": projection,
        "sample_statistics_flops": statistics,
        "sample_query_total_flops": projection + statistics,
        "incumbent_reference_projection_flops": incumbent_reference,
        "query_plus_reference_flops": projection + statistics + incumbent_reference,
    }


def main():
    n = 5
    sample_count = 9
    rows = [
        [
            F(
                (
                    (t + 1) * (i + 2) * (t + i + 3)
                    + (t + 2) ** (1 + (i % 3))
                    + (i + 1) ** 2
                )
                % 23
                - 11
            )
            for i in range(n)
        ]
        for t in range(sample_count)
    ]
    columns = [[row[i] for row in rows] for i in range(n)]
    direction = [F(2), F(-3), F(5), F(7), F(-11)]
    projected = [sum(direction[i] * row[i] for i in range(n)) for row in rows]

    parity = {}
    sector_nonzero = {}
    incumbent_parity = {}
    for order in range(2, 6):
        tensor = full_tensor(columns, order)
        dense = dense_contraction(tensor, direction)
        scalar = joint_cumulant([projected] * order)
        if dense != scalar:
            raise AssertionError(f"order {order} directional parity failed")
        sectors = collision_contractions(tensor, direction)
        expected = expected_sector_counts(n, order)
        if set(sectors) != set(expected):
            raise AssertionError((order, set(sectors), set(expected)))
        if sum(sectors.values(), F(0)) != dense:
            raise AssertionError(f"order {order} collision sum failed")
        if any(value == 0 for value in sectors.values()):
            raise AssertionError(f"order {order} has a zero collision witness")
        parity[str(order)] = True
        sector_nonzero[str(order)] = {key: True for key in sorted(sectors)}
        if order >= 3:
            supported = {
                3: {"[3]", "[21]"},
                4: {"[4]", "[31]", "[22]"},
                5: {"[5]", "[41]", "[32]"},
            }[order]
            collision_supported = sum(
                value for key, value in sectors.items() if key in supported
            )
            explicit = explicit_incumbent_projection(tensor, direction, order)
            if collision_supported != explicit:
                raise AssertionError(f"order {order} incumbent formula failed")
            incumbent_parity[str(order)] = True

    k3_sample = joint_cumulant([projected] * 3)
    k4_sample = joint_cumulant([projected] * 4)
    tensor3 = full_tensor(columns, 3)
    tensor4 = full_tensor(columns, 4)
    k3_reference = explicit_incumbent_projection(tensor3, direction, 3)
    k4_reference = explicit_incumbent_projection(tensor4, direction, 4)
    d3 = k3_sample - k3_reference
    d4 = k4_sample - k4_reference
    mixed_shell_parity = {
        "k3_squared": k3_sample**2 - k3_reference**2
        == d3 * (k3_sample + k3_reference),
        "k3_k4": k3_sample * k4_sample - k3_reference * k4_reference
        == d3 * k4_reference + k3_reference * d4 + d3 * d4,
        "k3_cubed": k3_sample**3 - k3_reference**3
        == d3
        * (k3_sample**2 + k3_sample * k3_reference + k3_reference**2),
    }
    if not all(mixed_shell_parity.values()):
        raise AssertionError("mixed shell delta identity failed")

    width = 1024
    full_counts = {
        str(order): expected_sector_counts(width, order) for order in range(2, 6)
    }
    for order, counts in full_counts.items():
        if sum(counts.values()) != comb(width + int(order) - 1, int(order)):
            raise AssertionError(f"width-1024 count failure at order {order}")

    result = {
        "schema": "p2-fused-directional-innovation-action-interface-v1",
        "status": "PASS_EXACT_INTERFACE_VERIFICATION",
        "access": {
            "targets_opened": 0,
            "observer_selection_opened": 0,
            "estimators_run": 0,
            "fixture_calls": 0,
            "network_actions": 0,
        },
        "exact_n5": {
            "arithmetic": "fractions.Fraction",
            "samples": sample_count,
            "directional_dense_parity": parity,
            "all_collision_sectors_nonzero": sector_nonzero,
            "incumbent_diag_plane_projection_parity": incumbent_parity,
            "mixed_shell_delta_parity": mixed_shell_parity,
        },
        "width1024_collision_coordinate_counts": full_counts,
        "rank13_scalar_coordinates": {
            "per_layer_mean_cov_k3_k4_k5": 65,
            "h2_through_h7_six_layers": 390,
            "h2_through_h9_eight_layers": 520,
            "composite_shells": "derived_without_new_sample_coordinates",
        },
        "cost_N65536_rank13": {
            "h2_through_h7": production_cost(1024, 65536, 13, 6),
            "h2_through_h9": production_cost(1024, 65536, 13, 8),
        },
        "cost_N32768_rank13": {
            "h2_through_h7": production_cost(1024, 32768, 13, 6),
            "h2_through_h9": production_cost(1024, 32768, 13, 8),
        },
    }
    stored_path = Path(__file__).with_name("VERIFICATION.json")
    if stored_path.is_file() and json.loads(stored_path.read_text()) != result:
        raise AssertionError("stored VERIFICATION.json does not match recomputation")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
