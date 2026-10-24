#!/usr/bin/env python3
"""Deterministic tiny/static verification for the unreleased K42+K51 leaf.

This verifier uses synthetic arrays and one-dimensional quadrature only.  It
does not import or execute the frozen estimator, open a row, or read targets.
"""

from __future__ import annotations

import ast
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py"
BASE_RESULT = (
    HERE.parent
    / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
    / "D4_CANDIDATE_RESULT.json"
)
PRICING = (
    HERE.parent
    / "p2r0c1d29_k42_k51_pair_inheritance_static_pricing_v1"
    / "K42_K51_STATIC_DESIGN_PRICING_RECEIPT.json"
)
SOURCE = HERE / "k42_k51_instrumented_source.py"
DERIVATION = HERE / "DERIVATION.md"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUNNER = HERE / "source_action_runner_scaffold.py"
RELEASE = HERE / "K42_K51_SOURCE_ACTION_RELEASE.json"
INDEPENDENT_HOLD = HERE / "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json"
ACCOUNTING_ERRATUM = HERE / "K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json"
METER_AMENDMENT = HERE / "K42_K51_RUNTIME_METER_GATE_AMENDMENT_V1.json"
INTEGRATED_ACCOUNTING = HERE / "K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json"
BASE_SHA = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
BASE_RESULT_SHA = "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1"
PRICING_SHA = "bb8a55e3224da56139dc79fa2b65acc4cf8320f0991c7d77da5ec0892ad3b6cc"
INDEPENDENT_HOLD_SHA = "1d5786631f2923c606d4be53f428f9e6fe88f9418dbf15ecc6abb111a592f7ff"
ACCOUNTING_ERRATUM_SHA = "63fbf1da297b292b6340c688da3e20b16a850c296e165e05f111b0847534b8c3"
METER_AMENDMENT_SHA = "6ae06a43e0093849437a3eae784a19604e1bd2b451cf13cd9b484586e2f80de2"

EXPECTED_OPERATION_CALL_DELTAS = {
    "empty": 16, "add": 165, "concatenate": 30, "copy": 15,
    "getitem": 43, "multiply": 1020, "reshape": 15, "stack": 45,
    "subtract": 210, "einsum": 29, "diagonal": 15,
    "fill_diagonal": 30, "transpose": 15, "copyto": 30,
    "matmul": 15, "sum": 15,
}

LEFT = (0, 0, 1, 2, 3, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6)
RIGHT = (0, 3, 2, 1, 0, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_error(left: np.ndarray | float, right: np.ndarray | float) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = np.maximum(1.0, np.maximum(np.abs(a), np.abs(b)))
    return float(np.max(np.abs(a - b) / scale))


def hermite_e(order: int, value: Any) -> Any:
    if order == 0:
        return np.ones_like(value, dtype=np.float64)
    previous, current = np.ones_like(value, dtype=np.float64), value
    for index in range(1, order):
        previous, current = current, value * current - index * previous
    return current


def normal_pdf(value: Any) -> Any:
    return np.exp(-0.5 * np.asarray(value) ** 2) / math.sqrt(2.0 * math.pi)


def normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def gauss_legendre_integral(function: Any, left: float, right: float) -> float:
    nodes, weights = np.polynomial.legendre.leggauss(256)
    midpoint = 0.5 * (left + right)
    radius = 0.5 * (right - left)
    values = function(midpoint + radius * nodes)
    return float(radius * np.sum(weights * values))


def coefficient_and_centering_proof() -> dict[str, float]:
    pre_mean = 0.37
    sigma = 1.31
    corrected_mean = 0.23
    alpha = pre_mean / sigma
    threshold = -alpha
    raw = [normal_cdf(alpha)]
    raw.extend(
        gauss_legendre_integral(
            lambda z, p=power: (pre_mean + sigma * z) ** p * normal_pdf(z),
            threshold,
            12.0,
        )
        for power in range(1, 6)
    )

    coefficient = np.zeros((6, 13), dtype=np.float64)
    coefficient[0, 0] = 1.0
    for power in range(1, 6):
        for order in range(13):
            if order <= power:
                coefficient[power, order] = (
                    math.factorial(power)
                    / math.factorial(power - order)
                    * sigma**order
                    * raw[power - order]
                )
            else:
                coefficient[power, order] = (
                    math.factorial(power)
                    * sigma**power
                    * normal_pdf(alpha)
                    * hermite_e(order - power - 1, -alpha)
                )

    direct_coefficient = np.zeros((5, 13), dtype=np.float64)
    for power in range(1, 6):
        for order in range(13):
            direct_coefficient[power - 1, order] = gauss_legendre_integral(
                lambda z, p=power, q=order: (
                    (pre_mean + sigma * z) ** p
                    * hermite_e(q, z)
                    * normal_pdf(z)
                ),
                threshold,
                12.0,
            )

    centered = {}
    direct_centered = {}
    for power in (1, 2, 4, 5):
        centered[power] = sum(
            math.comb(power, lower)
            * (-corrected_mean) ** (power - lower)
            * coefficient[lower]
            for lower in range(power + 1)
        )
        direct_centered[power] = np.asarray(
            [
                gauss_legendre_integral(
                    lambda z, p=power, q=order: (
                        (-corrected_mean) ** p
                        * hermite_e(q, z)
                        * normal_pdf(z)
                    ),
                    -12.0,
                    threshold,
                )
                + gauss_legendre_integral(
                    lambda z, p=power, q=order: (
                        (pre_mean + sigma * z - corrected_mean) ** p
                        * hermite_e(q, z)
                        * normal_pdf(z)
                    ),
                    threshold,
                    12.0,
                )
                for order in range(13)
            ],
            dtype=np.float64,
        )

    raw5_recurrence = pre_mean * raw[4] + 4.0 * sigma * sigma * raw[3]
    c4_tail = 4.0 * sigma * coefficient[3, 3:12]
    c5_tail = 5.0 * sigma * coefficient[4, 4:12]
    return {
        "coefficient_quadrature_max_relative_error": relative_error(
            coefficient[1:], direct_coefficient
        ),
        "centered_quadrature_max_relative_error": max(
            relative_error(centered[p], direct_centered[p]) for p in centered
        ),
        "raw5_recurrence_relative_error": relative_error(raw5_recurrence, raw[5]),
        "c4_tail_recurrence_relative_error": relative_error(
            c4_tail, coefficient[4, 4:13]
        ),
        "c5_tail_recurrence_relative_error": relative_error(
            c5_tail, coefficient[5, 5:13]
        ),
    }


def set_partitions(items: tuple[int, ...]):
    if not items:
        yield []
        return
    first = items[0]
    for rest in set_partitions(items[1:]):
        yield [[first], *[block[:] for block in rest]]
        for index in range(len(rest)):
            copied = [block[:] for block in rest]
            copied[index] = [first, *copied[index]]
            yield copied


def partition_signature(partition: list[list[int]], labels: tuple[str, ...]):
    blocks = []
    for block in partition:
        count_i = sum(labels[index] == "i" for index in block)
        count_j = len(block) - count_i
        if count_i + count_j == 1:
            return None
        blocks.append((count_i, count_j))
    return tuple(sorted(blocks))


def partition_proof() -> dict[str, object]:
    expected42 = Counter(
        {
            ((4, 2),): 1,
            ((0, 2), (4, 0)): 1,
            ((1, 1), (3, 1)): 8,
            ((1, 2), (3, 0)): 4,
            ((2, 0), (2, 2)): 6,
            ((2, 1), (2, 1)): 6,
            ((0, 2), (2, 0), (2, 0)): 3,
            ((1, 1), (1, 1), (2, 0)): 12,
        }
    )
    expected51 = Counter(
        {
            ((5, 1),): 1,
            ((1, 1), (4, 0)): 5,
            ((2, 0), (3, 1)): 10,
            ((2, 1), (3, 0)): 10,
            ((1, 1), (2, 0), (2, 0)): 15,
        }
    )
    counted = {}
    total_partitions = None
    for name, labels, expected in (
        ("k42", ("i",) * 4 + ("j",) * 2, expected42),
        ("k51", ("i",) * 5 + ("j",), expected51),
    ):
        signatures = Counter()
        all_partitions = list(set_partitions(tuple(range(6))))
        total_partitions = len(all_partitions)
        for partition in all_partitions:
            signature = partition_signature(partition, labels)
            if signature is not None:
                signatures[signature] += 1
        if signatures != expected:
            raise AssertionError(f"{name} set-partition signature mismatch")
        counted[name] = int(sum(signatures.values()))

    rng = np.random.default_rng(420051)
    k = {
        key: float(value)
        for key, value in zip(
            ((0, 2), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2),
             (3, 0), (3, 1), (4, 0), (4, 2), (5, 1)),
            rng.normal(size=11),
        )
    }
    c42 = (
        k[(4, 2)]
        + k[(0, 2)] * k[(4, 0)]
        + 8.0 * k[(1, 1)] * k[(3, 1)]
        + 4.0 * k[(1, 2)] * k[(3, 0)]
        + 6.0 * k[(2, 0)] * k[(2, 2)]
        + 6.0 * k[(2, 1)] ** 2
        + 3.0 * k[(0, 2)] * k[(2, 0)] ** 2
        + 12.0 * k[(1, 1)] ** 2 * k[(2, 0)]
    )
    recovered42 = (
        c42
        - k[(0, 2)] * k[(4, 0)]
        - 3.0 * k[(2, 0)] ** 2 * k[(0, 2)]
        - 8.0 * k[(1, 1)] * k[(3, 1)]
        - 12.0 * k[(2, 0)] * k[(1, 1)] ** 2
        - 4.0 * k[(1, 2)] * k[(3, 0)]
        - 6.0 * k[(2, 0)] * k[(2, 2)]
        - 6.0 * k[(2, 1)] ** 2
    )
    c51 = (
        k[(5, 1)]
        + 5.0 * k[(1, 1)] * k[(4, 0)]
        + 10.0 * k[(2, 0)] * k[(3, 1)]
        + 10.0 * k[(2, 1)] * k[(3, 0)]
        + 15.0 * k[(1, 1)] * k[(2, 0)] ** 2
    )
    recovered51 = (
        c51
        - 5.0 * k[(1, 1)] * k[(4, 0)]
        - 15.0 * k[(1, 1)] * k[(2, 0)] ** 2
        - 10.0 * k[(2, 0)] * k[(3, 1)]
        - 10.0 * k[(2, 1)] * k[(3, 0)]
    )
    return {
        "all_set_partitions": int(total_partitions),
        "k42_singleton_free_partitions": counted["k42"],
        "k51_singleton_free_partitions": counted["k51"],
        "k42_formula_relative_error": relative_error(recovered42, k[(4, 2)]),
        "k51_formula_relative_error": relative_error(recovered51, k[(5, 1)]),
    }


def source_contraction_proof() -> dict[str, float]:
    rng = np.random.default_rng(4251)
    n = 4
    centered = rng.normal(size=(2, 2, 13, n))
    sigma = np.exp(rng.normal(scale=0.3, size=n))
    covariance = rng.normal(scale=0.12, size=(n, n))
    covariance = 0.5 * (covariance + covariance.T)
    np.fill_diagonal(covariance, sigma * sigma)
    correlation = covariance / np.outer(sigma, sigma)
    correlation = np.clip(correlation, -0.75, 0.75)
    np.fill_diagonal(correlation, 1.0)
    rho_powers = np.stack([correlation**t for t in range(7)], axis=0)
    inverse_factorial = np.asarray([1.0 / math.factorial(t) for t in range(7)])
    order_grid = np.add.outer(np.arange(7), np.arange(7))
    inverse = np.stack([sigma ** (-a) for a in range(7)], axis=0)
    grid = centered[:, :, order_grid] * inverse[None, None, :, None, :]
    left = grid[:, 0][:, LEFT]
    right = grid[:, 1][:, RIGHT]
    weights = rng.normal(scale=0.03, size=(17, n, n))
    kernels = np.einsum(
        "sdti,sdtj,tij,t->sdij",
        left,
        right,
        rho_powers,
        inverse_factorial,
        optimize=True,
    )
    batched = np.einsum("dij,sdij->sij", weights, kernels, optimize=True)
    scalar = np.zeros_like(batched)
    for sector in range(2):
        for i in range(n):
            for j in range(n):
                for derivative, (a, b) in enumerate(zip(LEFT, RIGHT)):
                    for t in range(7):
                        scalar[sector, i, j] += (
                            weights[derivative, i, j]
                            * correlation[i, j] ** t
                            / math.factorial(t)
                            * centered[sector, 0, t + a, i]
                            * sigma[i] ** (-a)
                            * centered[sector, 1, t + b, j]
                            * sigma[j] ** (-b)
                        )
    initial = np.einsum(
        "sti,stj,tij,t->sij",
        grid[:, 0, 0],
        grid[:, 1, 0],
        rho_powers,
        inverse_factorial,
        optimize=True,
    )
    initial_scalar = np.zeros_like(initial)
    for sector in range(2):
        for i in range(n):
            for j in range(n):
                initial_scalar[sector, i, j] = sum(
                    correlation[i, j] ** t
                    / math.factorial(t)
                    * centered[sector, 0, t, i]
                    * centered[sector, 1, t, j]
                    for t in range(7)
                )
    return {
        "later_batched_vs_scalar_relative_error": relative_error(batched, scalar),
        "initial_batched_vs_scalar_relative_error": relative_error(
            initial, initial_scalar
        ),
    }


def connected_gaussian_null_proof() -> dict[str, float]:
    rng = np.random.default_rng(20341)
    n = 5
    variance = np.exp(rng.normal(size=n))
    covariance = rng.normal(scale=0.1, size=(n, n))
    covariance = 0.5 * (covariance + covariance.T)
    np.fill_diagonal(covariance, variance)
    vi = variance[:, None]
    vj = variance[None, :]
    c42 = 3.0 * vi * vi * vj + 12.0 * vi * covariance * covariance
    c51 = 15.0 * covariance * vi * vi
    k42 = c42 - 3.0 * vi * vi * vj - 12.0 * vi * covariance * covariance
    k51 = c51 - 15.0 * covariance * vi * vi
    return {
        "k42_correlated_gaussian_null_max_abs": float(np.max(np.abs(k42))),
        "k51_correlated_gaussian_null_max_abs": float(np.max(np.abs(k51))),
    }


def action_and_orientation_proof() -> dict[str, float]:
    rng = np.random.default_rng(154206)
    n = 4
    outputs = 3
    k42 = rng.normal(size=(n, n))
    k51 = rng.normal(size=(n, n))
    np.fill_diagonal(k42, 0.0)
    np.fill_diagonal(k51, 0.0)
    weight = rng.normal(size=(n, outputs))
    w2 = weight * weight
    w3 = w2 * weight
    w4 = w2 * w2
    w5 = w3 * w2
    action42 = 15.0 * np.sum(w4 * (k42 @ w2), axis=0)
    action51 = 6.0 * np.sum(w5 * (k51 @ weight), axis=0)

    brute42 = np.zeros(outputs, dtype=np.float64)
    brute51 = np.zeros(outputs, dtype=np.float64)
    for output in range(outputs):
        for indices in itertools.product(range(n), repeat=6):
            counts = Counter(indices)
            if len(counts) != 2:
                continue
            by_count = {count: index for index, count in counts.items()}
            product = math.prod(weight[index, output] for index in indices)
            if set(by_count) == {2, 4}:
                brute42[output] += product * k42[by_count[4], by_count[2]]
            if set(by_count) == {1, 5}:
                brute51[output] += product * k51[by_count[5], by_count[1]]

    k24 = k42.T
    k15 = k51.T
    transposed42 = 15.0 * np.sum(w2 * (k24 @ w4), axis=0)
    transposed51 = 6.0 * np.sum(weight * (k15 @ w5), axis=0)

    tau = np.exp(rng.normal(scale=0.2, size=n))
    next_sigma = np.exp(rng.normal(scale=0.2, size=outputs))
    standardized42 = k42 / (tau[:, None] ** 4 * tau[None, :] ** 2)
    standardized51 = k51 / (tau[:, None] ** 5 * tau[None, :])
    b4 = w4 * tau[:, None] ** 4 / next_sigma[None, :] ** 4
    b2 = w2 * tau[:, None] ** 2 / next_sigma[None, :] ** 2
    b5 = w5 * tau[:, None] ** 5 / next_sigma[None, :] ** 5
    b1 = weight * tau[:, None] / next_sigma[None, :]
    standardized_action42 = 15.0 * np.sum(b4 * (standardized42 @ b2), axis=0)
    standardized_action51 = 6.0 * np.sum(b5 * (standardized51 @ b1), axis=0)
    return {
        "factor15_brute_relative_error": relative_error(action42, brute42),
        "factor6_brute_relative_error": relative_error(action51, brute51),
        "k24_transpose_action_relative_error": relative_error(action42, transposed42),
        "k15_transpose_action_relative_error": relative_error(action51, transposed51),
        "k42_raw_standardized_relative_error": relative_error(
            action42 / next_sigma**6, standardized_action42
        ),
        "k51_raw_standardized_relative_error": relative_error(
            action51 / next_sigma**6, standardized_action51
        ),
        "k42_diagonal_max_abs": float(np.max(np.abs(np.diag(k42)))),
        "k51_diagonal_max_abs": float(np.max(np.abs(np.diag(k51)))),
    }


def endpoint_proof() -> dict[str, float]:
    rng = np.random.default_rng(720)
    raw42 = rng.normal(size=17)
    raw51 = rng.normal(size=17)
    raw_total = raw42 + raw51
    sigma = np.exp(rng.normal(scale=0.4, size=17))
    c6 = rng.normal(size=(4, 17))
    direct = (raw_total[None, :] / 720.0) * (c6 / sigma[None, :] ** 6)
    standardized = (raw_total / sigma**6)[None, :] * c6 / 720.0
    return {
        "single_raw_endpoint_scale_relative_error": relative_error(
            direct, standardized
        ),
        "shared_vector_sum_relative_error": relative_error(
            raw_total, np.sum(np.stack((raw42, raw51)), axis=0)
        ),
    }


def integrated_accounting_receipt_proof() -> dict[str, object]:
    """Semantically join the independent receipt to the final V3 inputs.

    The verifier source intentionally contains no receipt-file hash.  The
    receipt binds this verifier's source hash; the static result later records
    the receipt hash; and the freeze pins both.  That ordering is acyclic.
    """

    receipt = json.loads(INTEGRATED_ACCOUNTING.read_text())
    expected_top_level = {
        "schema",
        "status",
        "created_utc",
        "review",
        "binding_graph",
        "bindings",
        "official_environment",
        "integrated_source_and_reduction_derivation",
        "complete_incremental_identity",
        "pre_outcome_hard_cap_proof",
        "exact_official_operation_call_delta",
        "inventory_interpretation",
        "resource_and_authority_boundary",
        "terminal_decision",
    }
    if set(receipt) != expected_top_level:
        raise AssertionError("integrated receipt top-level schema drift")
    if receipt["schema"] != "q24-k42-k51-independent-integrated-accounting-receipt-v1":
        raise AssertionError("integrated receipt schema mismatch")
    if receipt["status"] != (
        "PASS_STATIC_INTEGRATED_ASSOCIATION_AND_CAP_PROOF__"
        "NO_ROW_OR_ESTIMATOR_AUTHORITY"
    ):
        raise AssertionError("integrated receipt status mismatch")

    review = receipt["review"]
    expected_review = {
        "reviewer_identity": "/root/k4251_integrated_accounting",
        "scope": (
            "independent literal official-FlopScope integrated association, "
            "operation-call delta, and pre-row cap proof"
        ),
        "source_execution_performed": False,
        "row_opened": False,
        "estimator_called": False,
    }
    if review != expected_review:
        raise AssertionError("integrated receipt review boundary mismatch")

    binding_graph = receipt["binding_graph"]
    expected_binding_graph = {
        "order": [
            "final derivation/preregistration/source/runner/verifier",
            "integrated accounting receipt",
            "STATIC_VERIFICATION_V3.json records receipt SHA-256",
            "SOURCE_FREEZE_V3.json pins both files",
        ],
        "receipt_self_hash_bound": False,
        "verifier_source_embeds_receipt_sha256": False,
        "verifier_semantically_recomputes_receipt_bindings": True,
        "static_verification_records_receipt_sha256": True,
        "freeze_pins_verifier_receipt_and_static_result": True,
    }
    if binding_graph != expected_binding_graph:
        raise AssertionError("integrated receipt binding graph mismatch")

    expected_bindings = {
        "d4_source_sha256": sha256(BASE),
        "d4_result_sha256": sha256(BASE_RESULT),
        "d29_pricing_sha256": sha256(PRICING),
        "independent_hold_sha256": sha256(INDEPENDENT_HOLD),
        "accounting_erratum_sha256": sha256(ACCOUNTING_ERRATUM),
        "meter_gate_amendment_sha256": sha256(METER_AMENDMENT),
        "derivation_sha256": sha256(DERIVATION),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "instrumented_source_sha256": sha256(SOURCE),
        "exec_cold_sink_and_worker_runner_sha256": sha256(RUNNER),
        "tiny_static_verifier_sha256": sha256(Path(__file__).resolve()),
    }
    if receipt["bindings"] != expected_bindings:
        raise AssertionError("integrated receipt final-file binding mismatch")

    expected_environment = {
        "interpreter": (
            "pure_closure_writeup/problem_graph/experiments/"
            "p2_width1024_depth16/data_gate/.venv/bin/python"
        ),
        "flopscope_version": "0.12.0",
        "whestbench_version": "0.16.0",
        "numpy_version": "2.4.6",
        "width": 1024,
        "binary64_provenance_pinned": True,
        "advanced_float64_gather_units_per_output": 8,
        "mixed_float64_float32_matmul_result": "float64",
    }
    if receipt["official_environment"] != expected_environment:
        raise AssertionError("integrated receipt official environment mismatch")

    n = 1024
    initial = 80 * n * n + 28 * n
    later_fused = 1360 * n * n + 476 * n
    later_reduction = 132 * n * n
    later_combined = 1492 * n * n + 476 * n
    all_later = 14 * later_combined
    integrated_source = initial + all_later
    expected_derivation = {
        "initial_formula": "80*n^2 + 28*n",
        "initial_width1024_flops": initial,
        "later_fused_kernel_formula_each": "1360*n^2 + 476*n",
        "later_fused_kernel_width1024_flops_each": later_fused,
        "later_reduction_formula_each": "132*n^2",
        "later_reduction_width1024_flops_each": later_reduction,
        "later_combined_formula_each": "1492*n^2 + 476*n",
        "later_combined_width1024_flops_each": later_combined,
        "later_source_count": 14,
        "all_later_width1024_flops": all_later,
        "integrated_source_and_reduction_width1024_flops": integrated_source,
        "standalone_dense_later_formula_each": "1506*n^2",
        "official_selection_law": (
            "min(dense 1506*n^2, explicit fused 1492*n^2+476*n)"
        ),
        "explicit_path_first_no_more_expensive_at_integer_width": 34,
    }
    if receipt["integrated_source_and_reduction_derivation"] != expected_derivation:
        raise AssertionError("integrated source/reduction derivation mismatch")

    expected_identity = {
        "shared_coefficient_and_context_flops": 101869568,
        "integrated_source_and_reduction_flops": 21993394176,
        "connected_and_diagonal_flops": 692367360,
        "actions_and_shared_sum_flops": 129037793280,
        "production_endpoint_flops": 0,
        "future_separately_licensed_shared_endpoint_formula": (
            "15*3*n + 14*(n + 2*n^2)"
        ),
        "future_separately_licensed_shared_endpoint_flops": 29420544,
        "future_shared_endpoint_excluded_from_current_incremental_identity": True,
        "predicted_integrated_incremental_flops": 151825424384,
        "identity": (
            "101869568 + 21993394176 + 692367360 + 129037793280 "
            "= 151825424384"
        ),
        "standalone_dense_diagnostic_flops": 152024121344,
        "standalone_dense_minus_integrated_prediction_flops": 198696960,
        "standalone_dense_is_an_upper_bound_under_pinned_f64_provenance": True,
    }
    if receipt["complete_incremental_identity"] != expected_identity:
        raise AssertionError("integrated receipt complete identity mismatch")

    expected_cap = {
        "hard_incremental_flop_cap": 152200000000,
        "cap_margin_over_integrated_prediction": 374575616,
        "cap_margin_over_standalone_dense_upper": 175878656,
        "required_runtime_law": (
            "A1 and A2 bills equal; B-A1 and B-A2 equal, nonnegative, "
            "and each <=152200000000"
        ),
        "exact_runtime_equality_to_prediction_required": False,
        "exact_runtime_equality_to_dense_diagnostic_required": False,
    }
    if receipt["pre_outcome_hard_cap_proof"] != expected_cap:
        raise AssertionError("integrated receipt cap proof mismatch")
    if receipt["exact_official_operation_call_delta"] != EXPECTED_OPERATION_CALL_DELTAS:
        raise AssertionError("integrated receipt operation-call delta mismatch")

    expected_inventory = {
        "source_einsum_calls": 29,
        "batched_matmul_calls": 15,
        "logical_dense_actions": 30,
        "syntactic_fill_diagonal_sites": 2,
        "runtime_fill_diagonal_calls": 30,
        "sink_work_is_outside_official_operation_delta": True,
        "source_and_runner_are_exec_cold_and_hash_pinned": True,
    }
    if receipt["inventory_interpretation"] != expected_inventory:
        raise AssertionError("integrated receipt inventory interpretation mismatch")

    expected_boundary = {
        "remote_scientific_scratch_ceiling_bytes": 309092352,
        "largest_new_remote_array_bytes": 285212672,
        "legacy_D4_plus_remote_projection_excluding_client_diagnostics_bytes": 7043162112,
        "client_diagnostic_working_ceiling_bytes": 25165824,
        "conservative_process_projection_including_client_diagnostics_bytes": 7068327936,
        "conservative_projection_margin_below_8gib_bytes": 1521606656,
        "process_rss_hard_gate_bytes": 8589934592,
        "official_residual_hard_gate_seconds_strictly_below": 0.4,
        "release_created": False,
        "row_opened": False,
        "weights_opened": False,
        "target_or_score_read": False,
        "estimator_imported": False,
        "estimator_called": False,
        "numerical_result_claimed": False,
        "independent_transaction_and_science_review_required_before_release": True,
    }
    if receipt["resource_and_authority_boundary"] != expected_boundary:
        raise AssertionError("integrated receipt resource/authority boundary mismatch")
    if receipt["terminal_decision"] != (
        "PASS_INTEGRATED_STATIC_ACCOUNTING_ONLY__BIND_V3_REVIEW_PACKAGE_AND_"
        "HOLD_RELEASE_ROWS_ESTIMATOR_TARGETS_AUTONOMY_SUBMISSION"
    ):
        raise AssertionError("integrated receipt terminal decision mismatch")

    return {
        "semantic_join_passed": True,
        "receipt_sha256": sha256(INTEGRATED_ACCOUNTING),
        "bound_file_hashes": expected_bindings,
        "predicted_integrated_incremental_flops": 151825424384,
        "hard_incremental_flop_cap": 152200000000,
        "cap_margin_over_integrated_prediction": 374575616,
        "standalone_dense_minus_integrated_prediction_flops": 198696960,
        "official_operation_call_delta": EXPECTED_OPERATION_CALL_DELTAS,
        "syntactic_fill_diagonal_sites": 2,
        "runtime_fill_diagonal_calls": 30,
        "remote_scientific_scratch_ceiling_bytes": 309092352,
        "conservative_process_projection_including_client_diagnostics_bytes": 7068327936,
        "acyclic_binding_graph": True,
    }


def source_ast_and_ledger_proof() -> dict[str, object]:
    source_text = SOURCE.read_text()
    tree = ast.parse(source_text)
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    einsum_syntax_calls = sum(
        isinstance(call.func, ast.Attribute) and call.func.attr == "einsum"
        for call in calls
    )
    matmul_syntax_calls = sum(
        isinstance(call.func, ast.Attribute) and call.func.attr == "matmul"
        for call in calls
    )
    assignments = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "OPERATION_LEDGER"
                for target in node.targets)
    ]
    if len(assignments) != 1:
        raise AssertionError("missing unique OPERATION_LEDGER")
    ledger = ast.literal_eval(assignments[0].value)
    components = (
        ledger["shared_coefficient_and_context_flops"]
        + ledger["source_kernel_and_reduction_flops"]
        + ledger["connected_and_diagonal_flops"]
        + ledger["actions_and_shared_sum_flops"]
    )
    pricing = json.loads(PRICING.read_text())
    erratum = json.loads(ACCOUNTING_ERRATUM.read_text())
    preregistration = json.loads(PREREGISTRATION.read_text())
    corrected = erratum["corrected_width1024_coordinate_only_ledger"]
    n = 1024
    fitted_components = {
        "shared_coefficient_and_context_flops": 15 * 3078 * n + 14 * 3808 * n,
        "source_kernel_and_reduction_flops": 80 * n * n + 28 * n + 14 * 1506 * n * n,
        "connected_and_diagonal_flops": 15 * (44 * n * n + 20 * n),
        "actions_and_shared_sum_flops": 15 * (
            4 * n * n * (2 * n - 1) + 16 * n * n + 2 * n
        ),
    }
    integrated_source_reduction = 80 * n * n + 28 * n + 14 * (
        1492 * n * n + 476 * n
    )
    integrated_components = {
        **fitted_components,
        "source_kernel_and_reduction_flops": integrated_source_reduction,
    }
    integrated_increment = int(sum(integrated_components.values()))
    base_totals = pricing["d4_host_fit"]["base_flops"]
    totals = [int(value + components) for value in base_totals]
    expected_totals = [1471895831927, 1508619409928, 1469973080576]
    integrated_totals = [int(value + integrated_increment) for value in base_totals]
    expected_integrated_totals = [1471697134967, 1508420712968, 1469774383616]
    memory = erratum["literal_call_and_memory_geometry"]

    forbidden = {
        "univariate_hook": "base._univariate_edgeworth =" in source_text,
        "bivariate_06_mutation": "weights[10]" in source_text,
        "bivariate_60_mutation": "weights[16]" in source_text,
        "endpoint_divisor": "/ 720" in source_text or "/720" in source_text,
        "rank_projection": "svd(" in source_text.lower(),
        "readonly_diagonal_copy": "fnp.copyto(fnp.diagonal" in source_text,
    }
    if any(forbidden.values()):
        raise AssertionError(f"forbidden source construct present: {forbidden}")
    required_tokens = (
        '"sdti,sdtj,tij,t->sdij"',
        '"dij,sdij->sij"',
        "fnp.matmul(planes, right_powers, out=pushed)",
        "raw_total = raw_by_sector[0] + raw_by_sector[1]",
        "base.Estimator().predict(mlp, base._PHASE2_FLOP_BUDGET)",
        "base._predict = predict_with_tape_capture",
        "base.transport_correction = transport_with_capture",
        "fnp.fill_diagonal(k42, 0.0)",
        "fnp.fill_diagonal(k51, 0.0)",
        "centered[:, base._Q24_ORDER_GRID, :]",
        "sector_left[:, LEFT_DERIVATIVES, :, :]",
        "sector_right[:, RIGHT_DERIVATIVES, :, :]",
    )
    missing = [token for token in required_tokens if token not in source_text]
    if missing:
        raise AssertionError(f"missing literal source bindings: {missing}")
    fill_diagonal_sites = source_text.count("fnp.fill_diagonal(")
    runner_text = RUNNER.read_text()
    runner_tree = ast.parse(runner_text)
    runner_functions = {
        node.name: ast.get_source_segment(runner_text, node) or ""
        for node in runner_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    forbidden_runtime_artifacts = [RELEASE]
    for rank in (201, 643):
        forbidden_runtime_artifacts.extend((
            HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_BUNDLE",
            HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_FAILURE.json",
            HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_RUN.lock",
        ))
        forbidden_runtime_artifacts.extend(
            HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_{phase}_WORKER_CONSUMED.json"
            for phase in ("A1", "B", "A2")
        )
    if any(path.exists() for path in forbidden_runtime_artifacts):
        raise AssertionError("release/claim/result must remain absent during static verification")
    prereg_source_hash = preregistration["authority"]["instrumented_source_sha256"]
    if prereg_source_hash != sha256(SOURCE):
        raise AssertionError("preregistration does not pin the literal source")
    required_runner_tokens = (
        "REQUIRED_FROZEN_FILES",
        "_verify_release_freeze",
        "exclusive_json(parent_lock_path, lock_payload)",
        'with path.open("x", encoding="utf-8")',
        "_validate_rank201_prerequisite",
        "_validated_embedded_rank201_release_payload",
        "require_distinct_phase_capabilities",
        "validate_worker_ticket(ticket_path, phase, rank)",
        'load_sealed_row(rank, authority["release"])',
        "SEALED_ROW_ZERO_ACCESS_SENTINELS",
        "validate_sealed_row_zero_access_sentinels(row)",
        "sealed_row_firewall",
        "mutant_sealed_row_zero_access_sentinel_rejected_before_weights",
        "subprocess.Popen(",
        "str(OFFICIAL_INTERPRETER)",
        'for phase in ("A1", "B", "A2")',
        'run_subprocess_phase(\n                "A1"',
        'run_subprocess_phase(\n                "B"',
        'run_subprocess_phase(\n                "A2"',
        "PREDICTED_INTEGRATED_INCREMENTAL_FLOPS = 151_825_424_384",
        "INCREMENTAL_FLOP_CAP = 152_200_000_000",
        "delta_a1 == delta_a2",
        "operation_delta_a1 == EXPECTED_OPERATION_CALL_DELTAS",
        "operation_delta_a2 == EXPECTED_OPERATION_CALL_DELTAS",
        "checked_local_array",
        "dtype changed before copy",
        "runtime_science_proof",
        "os.replace(staged_bundle, final_bundle)",
        "persist_failure_if_lock_owned",
        "lock_acquired_by_this_process",
        "transaction_primitive_selftest",
    )
    missing_runner = [token for token in required_runner_tokens if token not in runner_text]
    if missing_runner:
        raise AssertionError(f"runner ownership/hash gates missing: {missing_runner}")
    execute_text = runner_functions["execute"]
    worker_text = runner_functions["worker_phase"]
    row_loader_text = runner_functions["load_sealed_row"]
    if execute_text.index("exclusive_json(parent_lock_path, lock_payload)") >= execute_text.index('run_subprocess_phase(\n                "A1"'):
        raise AssertionError("runner claim must precede the first worker launch")
    if worker_text.index("validate_worker_ticket(ticket_path, phase, rank)") >= worker_text.index('load_sealed_row(rank, authority["release"])'):
        raise AssertionError("worker capability consumption must precede row open")
    if row_loader_text.index("validate_sealed_row_zero_access_sentinels(row)") >= row_loader_text.index("np.load(weights_path"):
        raise AssertionError("sealed-row zero sentinels must be validated before weights load")
    if "Callable" in runner_text or "spawn_fresh_base" in runner_text or "load_bound_mlp" in runner_text:
        raise AssertionError("runner exposes a forbidden injected callback authority")
    required_frozen_names = {
        "DERIVATION.md",
        "PREREGISTRATION.json",
        "k42_k51_instrumented_source.py",
        "source_action_runner_scaffold.py",
        "verify_static.py",
        "STATIC_VERIFICATION_V3.json",
        "K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json",
        "K42_K51_RUNTIME_METER_GATE_AMENDMENT_V1.json",
        "K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json",
        "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json",
    }
    if not all(f'"{name}"' in runner_text for name in required_frozen_names):
        raise AssertionError("runner does not enumerate the complete frozen file set")
    return {
        "source_sha256": sha256(SOURCE),
        "derivation_sha256": sha256(DERIVATION),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "runner_scaffold_sha256": sha256(RUNNER),
        "independent_hold_sha256": sha256(INDEPENDENT_HOLD),
        "accounting_erratum_sha256": sha256(ACCOUNTING_ERRATUM),
        "meter_gate_amendment_sha256": sha256(METER_AMENDMENT),
        "integrated_accounting_receipt_sha256": sha256(INTEGRATED_ACCOUNTING),
        "runtime_artifacts_absent": not any(path.exists() for path in forbidden_runtime_artifacts),
        "syntactic_einsum_sites": int(einsum_syntax_calls),
        "syntactic_matmul_sites": int(matmul_syntax_calls),
        "syntactic_fill_diagonal_sites": int(fill_diagonal_sites),
        "logical_source_einsum_calls": int(ledger["total_source_einsum_calls"]),
        "logical_batched_matmul_calls": int(ledger["batched_dense_matmul_calls"]),
        "logical_dense_matmuls": int(ledger["logical_dense_matmuls"]),
        "runtime_fill_diagonal_calls": int(ledger["runtime_fill_diagonal_calls"]),
        "official_operation_call_deltas": ledger["official_operation_call_deltas"],
        "binary64_operation_rate": int(ledger["binary64_operation_rate"]),
        "advanced_float64_gather_units_per_output": int(
            ledger["advanced_float64_gather_units_per_output"]
        ),
        "fitted_component_flops": fitted_components,
        "integrated_component_flops": integrated_components,
        "ledger_component_sum_flops": int(components),
        "integrated_incremental_flops_prediction": integrated_increment,
        "standalone_dense_minus_integrated_prediction": int(components - integrated_increment),
        "hard_incremental_flop_cap": 152200000000,
        "cap_margin_over_integrated_prediction": int(152200000000 - integrated_increment),
        "erratum_coordinate_only_flops": int(corrected["coordinate_only_incremental_flops"]),
        "instrumented_total_flops": totals,
        "expected_instrumented_total_flops": expected_totals,
        "integrated_instrumented_total_flops_prediction": integrated_totals,
        "expected_integrated_instrumented_total_flops": expected_integrated_totals,
        "minimum_flop_margin": int(2**41 - max(totals)),
        "incremental_memory_ceiling_bytes": int(memory["remote_incremental_scratch_ceiling_bytes"]),
        "legacy_projected_peak_rss_excluding_client_diagnostics_bytes": int(memory["projected_peak_rss_bytes"]),
        "client_diagnostic_working_ceiling_bytes": 25165824,
        "conservative_projected_peak_rss_bytes": int(memory["projected_peak_rss_bytes"]) + 25165824,
        "largest_new_array_bytes": int(memory["largest_new_array_bytes"]),
        "source_kernel_bank_dtype_shape": memory["source_kernel_bank"],
        "persistent_plane_dtype_shape": memory["persistent_oriented_planes"],
        "runner_complete_hash_and_transaction_tokens": True,
        "production_endpoint_present": False,
        "future_shared_endpoint_formula": ledger["future_separately_licensed_shared_endpoint_formula"],
        "future_shared_endpoint_flops": int(ledger["future_separately_licensed_shared_endpoint_flops"]),
        "future_shared_endpoint_excluded_from_current_bill": bool(ledger["future_shared_endpoint_excluded_from_current_bill"]),
        "forbidden_constructs": forbidden,
    }


def main() -> None:
    if sha256(BASE) != BASE_SHA:
        raise AssertionError("frozen D4 hash mismatch")
    if sha256(BASE_RESULT) != BASE_RESULT_SHA:
        raise AssertionError("frozen D4 result hash mismatch")
    if sha256(PRICING) != PRICING_SHA:
        raise AssertionError("frozen D29 hash mismatch")
    if sha256(INDEPENDENT_HOLD) != INDEPENDENT_HOLD_SHA:
        raise AssertionError("independent HOLD receipt hash mismatch")
    if sha256(ACCOUNTING_ERRATUM) != ACCOUNTING_ERRATUM_SHA:
        raise AssertionError("corrected accounting erratum hash mismatch")
    if sha256(METER_AMENDMENT) != METER_AMENDMENT_SHA:
        raise AssertionError("pre-outcome meter-gate amendment hash mismatch")
    if not INTEGRATED_ACCOUNTING.is_file():
        raise AssertionError("integrated accounting receipt is absent")

    accounting_receipt = integrated_accounting_receipt_proof()
    coefficient = coefficient_and_centering_proof()
    partition = partition_proof()
    contraction = source_contraction_proof()
    gaussian = connected_gaussian_null_proof()
    action = action_and_orientation_proof()
    endpoint = endpoint_proof()
    static = source_ast_and_ledger_proof()
    runner_spec = importlib.util.spec_from_file_location("k42_k51_v3_runner_static", RUNNER)
    if runner_spec is None or runner_spec.loader is None:
        raise AssertionError("cannot load V3 runner for tiny transaction selftest")
    runner_module = importlib.util.module_from_spec(runner_spec)
    runner_spec.loader.exec_module(runner_module)
    transaction_selftest = runner_module.transaction_primitive_selftest()

    thresholds = {
        "coefficient_quadrature": coefficient["coefficient_quadrature_max_relative_error"] <= 5e-11,
        "centered_quadrature": coefficient["centered_quadrature_max_relative_error"] <= 5e-11,
        "raw5_recurrence": coefficient["raw5_recurrence_relative_error"] <= 5e-13,
        "coefficient_tail_recurrence": max(
            coefficient["c4_tail_recurrence_relative_error"],
            coefficient["c5_tail_recurrence_relative_error"],
        ) <= 5e-13,
        "partition_count": partition["all_set_partitions"] == 203,
        "partition_formulas": max(
            partition["k42_formula_relative_error"],
            partition["k51_formula_relative_error"],
        ) <= 5e-15,
        "source_contractions": max(contraction.values()) <= 5e-13,
        "gaussian_nulls": max(gaussian.values()) <= 5e-13,
        "action_factors": max(
            action["factor15_brute_relative_error"],
            action["factor6_brute_relative_error"],
        ) <= 5e-13,
        "orientation": max(
            action["k24_transpose_action_relative_error"],
            action["k15_transpose_action_relative_error"],
        ) <= 5e-13,
        "raw_standardized_parity": max(
            action["k42_raw_standardized_relative_error"],
            action["k51_raw_standardized_relative_error"],
        ) <= 5e-13,
        "diagonal_zero": max(
            action["k42_diagonal_max_abs"], action["k51_diagonal_max_abs"]
        ) == 0.0,
        "single_shared_endpoint": max(endpoint.values()) <= 5e-15,
        "literal_ledger": (
            static["ledger_component_sum_flops"] == 152024121344
            and static["erratum_coordinate_only_flops"] == 152024121344
            and static["fitted_component_flops"] == {
                "shared_coefficient_and_context_flops": 101869568,
                "source_kernel_and_reduction_flops": 22192091136,
                "connected_and_diagonal_flops": 692367360,
                "actions_and_shared_sum_flops": 129037793280,
            }
            and static["instrumented_total_flops"]
            == static["expected_instrumented_total_flops"]
            and static["minimum_flop_margin"] == 690403845624
            and static["binary64_operation_rate"] == 2
            and static["advanced_float64_gather_units_per_output"] == 8
        ),
        "integrated_accounting_and_hard_cap": (
            accounting_receipt["semantic_join_passed"]
            and static["integrated_incremental_flops_prediction"] == 151825424384
            and static["standalone_dense_minus_integrated_prediction"] == 198696960
            and static["hard_incremental_flop_cap"] == 152200000000
            and static["cap_margin_over_integrated_prediction"] == 374575616
            and static["integrated_component_flops"]["source_kernel_and_reduction_flops"] == 21993394176
            and static["integrated_instrumented_total_flops_prediction"]
            == static["expected_integrated_instrumented_total_flops"]
        ),
        "call_inventory": (
            static["syntactic_einsum_sites"] == 3
            and static["syntactic_matmul_sites"] == 1
            and static["syntactic_fill_diagonal_sites"] == 2
            and static["logical_source_einsum_calls"] == 29
            and static["logical_batched_matmul_calls"] == 15
            and static["logical_dense_matmuls"] == 30
            and static["runtime_fill_diagonal_calls"] == 30
            and static["official_operation_call_deltas"] == EXPECTED_OPERATION_CALL_DELTAS
        ),
        "memory_ceiling": (
            static["incremental_memory_ceiling_bytes"] == 309092352
            and static["legacy_projected_peak_rss_excluding_client_diagnostics_bytes"] == 7043162112
            and static["client_diagnostic_working_ceiling_bytes"] == 25165824
            and static["conservative_projected_peak_rss_bytes"] == 7068327936
            and static["largest_new_array_bytes"] == 285212672
            and static["conservative_projected_peak_rss_bytes"] < 2**33
            and static["largest_new_array_bytes"] < 2**32
        ),
        "transaction_mock_replay_failure_orphan_controls": all(transaction_selftest.values()),
        "no_production_endpoint": (
            not static["production_endpoint_present"]
            and static["future_shared_endpoint_formula"] == "15*3*n + 14*(n + 2*n^2)"
            and static["future_shared_endpoint_flops"] == 29420544
            and static["future_shared_endpoint_excluded_from_current_bill"]
        ),
        "no_forbidden_constructs": not any(static["forbidden_constructs"].values()),
        "legal_diagonal_spelling": static["syntactic_fill_diagonal_sites"] == 2,
        "runner_complete_hash_transaction_ownership": (
            static["runtime_artifacts_absent"]
            and static["runner_complete_hash_and_transaction_tokens"]
        ),
    }
    if not all(thresholds.values()):
        failed = [name for name, passed in thresholds.items() if not passed]
        raise AssertionError(f"static verification failed: {failed}")

    result = {
        "schema": "q24-k42-k51-pair-source-action-static-verification-v3",
        "status": "PASS_STATIC_REPAIR_V3_ONLY__NO_RELEASE_NO_ROW_OR_ESTIMATOR_CALL",
        "authorities": {
            "d4_source_sha256": sha256(BASE),
            "d4_result_sha256": sha256(BASE_RESULT),
            "d29_pricing_sha256": sha256(PRICING),
            "independent_hold_sha256": sha256(INDEPENDENT_HOLD),
            "accounting_erratum_sha256": sha256(ACCOUNTING_ERRATUM),
            "meter_gate_amendment_sha256": sha256(METER_AMENDMENT),
            "integrated_accounting_receipt_sha256": sha256(INTEGRATED_ACCOUNTING),
            "instrumented_source_sha256": sha256(SOURCE),
            "derivation_sha256": sha256(DERIVATION),
            "preregistration_sha256": sha256(PREREGISTRATION),
            "runner_scaffold_sha256": sha256(RUNNER),
        },
        "coefficient_and_centering": coefficient,
        "partition_enumeration": partition,
        "batched_source_contraction": contraction,
        "correlated_gaussian_null": gaussian,
        "actions_orientation_and_scale": action,
        "shared_endpoint_identity": endpoint,
        "integrated_accounting_receipt_proof": accounting_receipt,
        "static_source_and_ledger": static,
        "transaction_primitive_selftest": transaction_selftest,
        "gates": thresholds,
        "firewall": {
            "estimator_imported": False,
            "estimator_run": False,
            "row_opened": False,
            "weights_opened": False,
            "names_opened": False,
            "targets_opened": False,
            "scores_opened": False,
            "training_steps": 0,
            "release_created": False,
            "submission_made": False,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
