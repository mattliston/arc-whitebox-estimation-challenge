#!/usr/bin/env python3
"""Reproduce the IG-LC3 primitive, connected, incidence, and cost counts.

This is a finite combinatorial verifier only. It opens no datasets, targets,
estimators, fits, scores, submissions, or graph mutation surfaces.
"""

from __future__ import annotations

import json
from itertools import combinations, combinations_with_replacement
from math import comb


def weak_compositions(total: int, width: int, prefix=()):
    if width == 1:
        yield prefix + (total,)
        return
    for value in range(total + 1):
        yield from weak_compositions(total - value, width - 1,
                                     prefix + (value,))


def support(exponent):
    return frozenset(i for i, value in enumerate(exponent) if value)


def connected(exponents, width: int) -> bool:
    parent = list(range(width))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    touched = set()
    for exponent in exponents:
        vertices = list(support(exponent))
        touched.update(vertices)
        if vertices:
            root = vertices[0]
            for vertex in vertices[1:]:
                left, right = find(root), find(vertex)
                if left != right:
                    parent[right] = left
    return (len(touched) == width and
            len({find(i) for i in range(width)}) == 1)


def incidence(primitives):
    return tuple(map(sum, zip(*(primitive[1] for primitive in primitives))))


def enumerate_width(width: int):
    rho = [
        ("R", tuple(1 if i in pair else 0 for i in range(width)))
        for pair in combinations(range(width), 2)
    ]
    k3 = [("K3", exponent) for exponent in weak_compositions(3, width)]
    k4 = [("K4", exponent) for exponent in weak_compositions(4, width)]
    k5 = [("K5", exponent) for exponent in weak_compositions(5, width)]
    grade1 = rho + k3
    grade2 = k4
    grade3 = k5

    multisets = {
        1: [(primitive,) for primitive in grade1],
        2: ([(primitive,) for primitive in grade2] +
            list(combinations_with_replacement(grade1, 2))),
        3: ([(primitive,) for primitive in grade3] +
            [(left, right) for left in grade1 for right in grade2] +
            list(combinations_with_replacement(grade1, 3))),
    }

    result = {
        "primitive_counts": [len(grade1), len(grade2), len(grade3)],
        "raw_counts": [],
        "connected_counts": [],
        "incidence_counts": [],
        "connected_signatures": {},
    }
    for grade in (1, 2, 3):
        connected_rows = [
            row for row in multisets[grade]
            if connected([primitive[1] for primitive in row], width)
        ]
        signatures = {}
        for row in connected_rows:
            signature = "*".join(sorted(primitive[0] for primitive in row))
            signatures[signature] = signatures.get(signature, 0) + 1
        result["raw_counts"].append(len(multisets[grade]))
        result["connected_counts"].append(len(connected_rows))
        result["incidence_counts"].append(
            len({incidence(row) for row in connected_rows})
        )
        result["connected_signatures"][str(grade)] = signatures
    return result


EXPECTED = {
    1: {
        "primitive_counts": [1, 1, 1],
        "raw_counts": [1, 2, 3],
        "connected_counts": [1, 2, 3],
        "incidence_counts": [1, 2, 3],
    },
    2: {
        "primitive_counts": [5, 5, 6],
        "raw_counts": [5, 20, 66],
        "connected_counts": [3, 15, 56],
        "incidence_counts": [3, 12, 30],
    },
    3: {
        "primitive_counts": [13, 15, 21],
        "raw_counts": [13, 106, 671],
        "connected_counts": [1, 43, 418],
        "incidence_counts": [1, 19, 80],
    },
    4: {
        "primitive_counts": [26, 35, 56],
        "raw_counts": [26, 386, 4242],
        "connected_counts": [0, 43, 1474],
        "incidence_counts": [0, 15, 125],
    },
    5: {
        "primitive_counts": [45, 70, 126],
        "raw_counts": [45, 1105, 19491],
        "connected_counts": [0, 15, 2546],
        "incidence_counts": [0, 5, 126],
    },
}


def verify_formula_counts(width: int, result):
    a = comb(width, 2) + comb(width + 2, 3)
    b = comb(width + 3, 4)
    c = comb(width + 4, 5)
    assert result["primitive_counts"] == [a, b, c]
    assert result["raw_counts"] == [
        a,
        b + comb(a + 1, 2),
        c + a * b + comb(a + 2, 3),
    ]


def verify_cost_witnesses(results):
    s5 = results[5]
    signatures = s5["connected_signatures"]
    assert signatures["2"] == {"K3*K3": 15}
    assert signatures["3"] == {
        "K5": 1,
        "K4*R": 20,
        "K3*K4": 160,
        "K3*R*R": 150,
        "K3*K3*R": 870,
        "K3*K3*K3": 1345,
    }

    k3_pair_cache = comb(35 + 1, 2)
    rho_pair_cache = comb(10 + 1, 2)
    optimistic_multiplies = (
        k3_pair_cache + rho_pair_cache + 1345 + 870 + 150 + 160 + 20
    )
    optimistic_bin_adds = 15 + 2546
    response_operations = (5 + 1) * (5 + 126)  # 5 mults + 1 add/bin
    normalization_upper = 35 + 70 + 126
    optimistic_subtotal = (
        optimistic_multiplies + optimistic_bin_adds +
        response_operations + normalization_upper
    )
    assert k3_pair_cache == 630
    assert rho_pair_cache == 55
    assert optimistic_multiplies == 3230
    assert optimistic_bin_adds == 2561
    assert response_operations == 786
    assert normalization_upper == 231
    assert optimistic_subtotal == 6808

    fixed_entry_flops = 3 * (2 * 16**2) + 2 * 16
    direct_reads = 231 * fixed_entry_flops
    prefix_counts = [6, 21, 55, 120, 231]
    trie_flops = (sum(prefix_counts[1:4]) * (2 * 16**2) +
                  prefix_counts[4] * (2 * 16))
    varying = [comb(7, 3) - comb(6, 3),
               comb(8, 4) - comb(7, 4),
               comb(9, 5) - comb(8, 5)]
    fiber_flops = sum(varying) * (2 * 16**2)
    assert fixed_entry_flops == 1568
    assert direct_reads == 362208
    assert trie_flops == 107744
    assert varying == [15, 35, 70]
    assert fiber_flops == 61440

    margin = 68634022825
    queries = 39360000
    extra_per_query = margin // queries
    assert extra_per_query == 1743
    assert 8192 + extra_per_query == 9935

    return {
        "optimistic_response_subtotal_excluding_carrier_and_log_kernel":
            optimistic_subtotal,
        "fixed_tt_entry_flops": fixed_entry_flops,
        "direct_231_tt_reads": direct_reads,
        "symmetry_prefix_trie_flops": trie_flops,
        "optimistic_varying_fiber_flops": fiber_flops,
        "printed_margin_extra_flops_per_query": extra_per_query,
    }


def main():
    results = {}
    for width in range(1, 6):
        result = enumerate_width(width)
        verify_formula_counts(width, result)
        for key, expected in EXPECTED[width].items():
            assert result[key] == expected, (width, key, result[key], expected)
        results[width] = result

    costs = verify_cost_witnesses(results)
    output = {
        "schema": "ig-lc3-connected-hypergraph-enumeration-v1",
        "status": "PASS_EXACT_COUNTS_AND_GENERIC_TT_COST_WITNESSES",
        "widths": results,
        "cost_witnesses": costs,
        "scope_note": (
            "Connected counts are exact generic primitive-monomial support "
            "counts. The 6808 response subtotal is an optimistic estimate, "
            "not a compiled formal-log/Mobius circuit and not an integrated "
            "carrier cost."
        ),
        "access": {
            "targets": 0,
            "fits": 0,
            "scores": 0,
            "submissions": 0,
            "runs": 0,
            "graph_edits": 0,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
