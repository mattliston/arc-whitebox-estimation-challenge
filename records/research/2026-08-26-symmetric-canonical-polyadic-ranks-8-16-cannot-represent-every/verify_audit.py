#!/usr/bin/env python3
"""Deterministic, target-free verifier for the symmetric-CP prior audit."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]

SOURCE_HASHES = {
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "fullbudget_k21_k4_closure_v1/DESIGN.md":
        "5c3ae5b91a32f4216b8739e2d2013eb2cadbae64a0430c0aa7e71b3f15e2046f",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "FULL_BUDGET_CP4_CAP2048_RESIDUAL_CARRIER_AUDIT.md":
        "9f044c2665c2143413e5b7da398487c38545994b20ff177f3494c004b0811588",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "cp3_full_birth_cap_ladder/SYNTHETIC_RESOURCE_LADDER.json":
        "9fa0588a008f1a9934dc440cff7afb3e023bf6ff28c1fb72c84bae9e1f1126e6",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "public_oracle_kappa_probe_v1/ORACLE_KAPPA_VARIANTS.json":
        "48cab175c5f06ac3024c070e7cf116455666f86da9c18deafbdb28b9fe7b4d73",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "p2r0f3b2b_partition_complete_tt5_design_v1/TERMINAL_DESIGN_AUDIT.md":
        "ddfdfc64dec12efb5dbc7b96384e6087bc0eaad19a28a44e9d2eccd2839c16e6",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "p2r0f3b2b2_ig_lc3_reset_derivation_v1/DERIVATION_RECEIPT.json":
        "41a9045df1105671fd45645142db2721cec13fba3e3e35c941c5c7bdabd50703",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/"
    "p2r0f3b2b2_independent_gaussian_linked_cluster_grade3_v1/CONSTRAINT_RECEIPT.md":
        "d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba",
    "pure_closure_writeup/problem_graph/experiments/x3_all_distinct_cp_width_scaling/README.md":
        "643ecaa2e37118f5cb82914112b94a038ce63e49ca8bfbb6d45803367c29dfc1",
    "pure_closure_writeup/problem_graph/experiments/x3_all_distinct_cp_residual_capacity/README.md":
        "489fac12e2097889469937d95bda3256f3b76b628f3e3dc84aaa5685d62e719c",
    "pure_closure_writeup/problem_graph/experiments/x3_all_distinct_cp_source_constructor/README.md":
        "6eaa2f1deb5aedc1f300ae481d993a2cd6edb668841fb6b789cce56ce699d03f",
    "pure_closure_writeup/problem_graph/experiments/x3_rank12_cp_capacity/README.md":
        "483d0b54df0ffa73d75a04107dedac49c31cd71908efca39904b4ccf2f2854c4",
    "pure_closure_writeup/problem_graph/experiments/x1_residual_cp_dimension/README.md":
        "cb6e5ac94fc8593433fa46ee4f0c68217836adb850733c2339bd28f364c904dc",
    "pure_closure_writeup/problem_graph/experiments/x1_joint_rank8_positive_capacity/r2/README.md":
        "d98f2e4aeadd8676f7b618c8dfaa63a715cb4b620c24f0a8eae7de87e09d1ad4",
    "pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/"
    "fused_two_channel_h_action_cp_rank16/RESULTS.json":
        "22d11a68c143f84f53b52fb2efff899f03e4613f8ed0f4c5db561f5b395ab44d",
    "pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/"
    "fused_two_channel_h_action_cp_rank16/METER.json":
        "32eef43223f4fdc11786369699de2e9fee924e4387feb11fc6759852f9a1ff98",
    "pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/"
    "diagonal_cp4_rolling_action/DERIVATION.md":
        "cc09105940a42a7f1ee917da6045beb108f9bdfae7c2865ad6457332f27573dc",
    "pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/"
    "rank1_pairing_cp4_escape/RESULTS.json":
        "17e92d18132a376a21c562a258d49e897e4f90292805acdb47e1708cb023a332",
    "pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/"
    "coherent_rank24_carrier_cp4_k31_provider/FINDING.md":
        "7dd37b112f7655dba6df4e1fd801da02506acfa60facc8dacc582139b44df148",
    "pure_closure_writeup/problem_graph/experiments/x3_r0c1_kappa3_ceiling/"
    "sparse_polynomial_semiring_tt4_rank24/RESULTS.json":
        "e85cd3d4200c9ee6e79c64bfd02b8241c49edaedffd57d84352fa21c7bbf00bd",
}

EXPECTED_COSTS = {
    8: {
        "factor_push_once": 16_769_024,
        "independent_factor_pushes_15": 754_606_080,
        "shared_factor_pushes_15": 251_535_360,
        "visible_plane_reads_14": 1_101_004_800,
        "independent_carrier_total": 1_855_610_880,
        "shared_carrier_total": 1_352_540_160,
        "independent_storage_bytes": 98_400,
        "shared_storage_bytes": 32_864,
        "independent_query_dof_per_reset": 24_576,
        "independent_query_cap_work_14": 2_818_572_288,
        "shared_query_dof_per_reset": 8_208,
        "shared_query_cap_work_14": 941_359_104,
        "independent_carrier_plus_query_proxy": 4_674_183_168,
        "shared_carrier_plus_query_proxy": 2_293_899_264,
    },
    16: {
        "factor_push_once": 33_538_048,
        "independent_factor_pushes_15": 1_509_212_160,
        "shared_factor_pushes_15": 503_070_720,
        "visible_plane_reads_14": 2_275_409_920,
        "independent_carrier_total": 3_784_622_080,
        "shared_carrier_total": 2_778_480_640,
        "independent_storage_bytes": 196_800,
        "shared_storage_bytes": 65_728,
        "independent_query_dof_per_reset": 49_152,
        "independent_query_cap_work_14": 5_637_144_576,
        "shared_query_dof_per_reset": 16_416,
        "shared_query_cap_work_14": 1_882_718_208,
        "independent_carrier_plus_query_proxy": 9_421_766_656,
        "shared_carrier_plus_query_proxy": 4_661_198_848,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dense_rank_exact(matrix: list[list[int]]) -> int:
    work = [[Fraction(value) for value in row] for row in matrix]
    n_rows = len(work)
    n_cols = len(work[0]) if work else 0
    pivot_row = 0
    for column in range(n_cols):
        pivot = next(
            (row for row in range(pivot_row, n_rows) if work[row][column]),
            None,
        )
        if pivot is None:
            continue
        work[pivot_row], work[pivot] = work[pivot], work[pivot_row]
        pivot_value = work[pivot_row][column]
        work[pivot_row] = [value / pivot_value for value in work[pivot_row]]
        for row in range(n_rows):
            if row == pivot_row or not work[row][column]:
                continue
            scale = work[row][column]
            work[row] = [
                value - scale * pivot_value
                for value, pivot_value in zip(work[row], work[pivot_row])
            ]
        pivot_row += 1
        if pivot_row == n_rows:
            break
    return pivot_row


def diagonal_mode1_matrix(n: int, order: int) -> list[list[int]]:
    columns = n ** (order - 1)
    matrix = [[0] * columns for _ in range(n)]
    repeated_index_stride = sum(n**power for power in range(order - 1))
    for index in range(n):
        matrix[index][index * repeated_index_stride] = index + 1
    return matrix


def rank_checks() -> dict[str, object]:
    tiny_ranks = {
        str(order): dense_rank_exact(diagonal_mode1_matrix(5, order))
        for order in (3, 4, 5)
    }
    if tiny_ranks != {"3": 5, "4": 5, "5": 5}:
        raise AssertionError(f"unexpected exact tiny ranks: {tiny_ranks}")

    n = 1024
    sparse_pivots = {
        str(order): len({
            index * sum(n**power for power in range(order - 1))
            for index in range(n)
        })
        for order in (3, 4, 5)
    }
    if sparse_pivots != {"3": n, "4": n, "5": n}:
        raise AssertionError(f"diagonal pivot collision: {sparse_pivots}")
    if not all(rank < n for rank in (8, 16)):
        raise AssertionError("CP8/16 should be strictly below diagonal unfolding rank")
    return {
        "tiny_dense_exact_mode1_ranks_n5": tiny_ranks,
        "symbolic_diagonal_mode1_ranks_n1024": sparse_pivots,
        "cp_rank_candidates": [8, 16],
        "universal_exact_whole_tensor_cp8_cp16_possible": False,
    }


def compute_costs(rank: int) -> dict[str, int]:
    n = 1024
    affine_pushes = 15
    resets = 14
    entry_cap = 8192
    factor_push = n * rank * (2 * n - 1)
    independent_pushes = 3 * affine_pushes * factor_push
    shared_pushes = affine_pushes * factor_push
    plane_reads = resets * 5 * n * n * (2 * rank - 1)
    independent_carrier = independent_pushes + plane_reads
    shared_carrier = shared_pushes + plane_reads
    independent_query_dof = 3 * n * rank
    shared_query_dof = rank * (n + 2)
    independent_query_work = resets * independent_query_dof * entry_cap
    shared_query_work = resets * shared_query_dof * entry_cap
    return {
        "factor_push_once": factor_push,
        "independent_factor_pushes_15": independent_pushes,
        "shared_factor_pushes_15": shared_pushes,
        "visible_plane_reads_14": plane_reads,
        "independent_carrier_total": independent_carrier,
        "shared_carrier_total": shared_carrier,
        "independent_storage_bytes": 4 * 3 * rank * (n + 1),
        "shared_storage_bytes": 4 * rank * (n + 3),
        "independent_query_dof_per_reset": independent_query_dof,
        "independent_query_cap_work_14": independent_query_work,
        "shared_query_dof_per_reset": shared_query_dof,
        "shared_query_cap_work_14": shared_query_work,
        "independent_carrier_plus_query_proxy": independent_carrier
        + independent_query_work,
        "shared_carrier_plus_query_proxy": shared_carrier + shared_query_work,
    }


def main() -> int:
    failures: list[str] = []
    observed_hashes: dict[str, str] = {}
    for relative, expected in SOURCE_HASHES.items():
        path = REPO_ROOT / relative
        if not path.is_file():
            failures.append(f"missing source: {relative}")
            continue
        observed = sha256(path)
        observed_hashes[relative] = observed
        if observed != expected:
            failures.append(
                f"source hash mismatch: {relative}: {observed} != {expected}"
            )

    try:
        ranks = rank_checks()
    except AssertionError as error:
        failures.append(str(error))
        ranks = {}

    costs = {str(rank): compute_costs(rank) for rank in (8, 16)}
    for rank in (8, 16):
        if costs[str(rank)] != EXPECTED_COSTS[rank]:
            failures.append(f"cost mismatch at rank {rank}")

    report = {
        "schema": "p2r0f3b2b3-symmetric-cp-residual-prior-audit-verification-v1",
        "status": "PASS" if not failures else "FAIL",
        "target_or_network_access": False,
        "source_hash_count": len(SOURCE_HASHES),
        "source_hashes_match": len(observed_hashes) == len(SOURCE_HASHES)
        and not any("source" in failure for failure in failures),
        "rank_checks": ranks,
        "cost_checks": costs,
        "failures": failures,
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
