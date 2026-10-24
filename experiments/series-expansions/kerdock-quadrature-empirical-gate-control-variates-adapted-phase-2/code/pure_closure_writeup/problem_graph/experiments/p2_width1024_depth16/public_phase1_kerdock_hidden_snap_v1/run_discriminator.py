#!/Users/matt/anaconda3/bin/python3
"""Phase-1-faithful two-block MUB state/local-innovation discriminator.

This is an observer-fit scientific runner, not a submitted estimator.  It
generates two independently oriented blocks of eight exact real MUB bases in
R^1024, includes every antipode, applies the Phase-1 exact h1-mean and h2-
preactivation-variance snaps, and persists post-ReLU means/gates through h16.

The target-free tapes can be replayed against any compatible analytic host.
When ``--open-target`` is passed, rank-level diagnostics compare both direct
state observations and non-double-counted local innovations to the opened
observer_fit truth.  The frozen scalar LocalInnovationTransport checkpoint is
included only as a capacity discriminator; it is not the deployed four-state
J corrector.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
FEATURE_ROOT = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
SCALAR_ROOT = P2 / "public_exposure_trained_residual_state_v1"
NATIVE_J_ROOT = P2 / "public_q24_native_transport_energy_final_v1"
NATIVE_J_MODEL = NATIVE_J_ROOT / "FINAL_MODEL.pt"
NATIVE_J_SOURCE = P2 / "public_full688_transport_energy_coherence_cv_v1/run_experiment.py"
NATIVE_ROW_ROOT = NATIVE_J_ROOT / "runs/native_rows_remaining_v1"
RUN_ROOT = HERE / "runs"

WIDTH = 1024
DEPTH = 16
FIELD_SIZE = 32
FIELD_MASK = FIELD_SIZE - 1
PRIMITIVE_POLYNOMIAL = 0b100101  # t^5 + t^2 + 1
BASES_PER_BLOCK = 8
PATHS_PER_BASE = 2 * WIDTH
PATHS_PER_BLOCK = BASES_PER_BLOCK * PATHS_PER_BASE
BLOCKS = 2
TOTAL_BASES = BLOCKS * BASES_PER_BLOCK
TOTAL_PATHS = BLOCKS * PATHS_PER_BLOCK
FIRST_CORRECTED_LAYER = 2
SEED = 0x325170_2026

# Independently audited six-level Winograd ledger.  Mean/gate readouts are
# reported separately; the scientific runner itself uses optimized torch GEMM.
WINOGRAD_FULL_DEPTH_CORE = 486_330_178_560
MEAN_READOUT_PER_LAYER = 67_108_864
GATE_READOUT_PER_LAYER = 67_108_864


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def gf_multiply(left: int, right: int) -> int:
    """Multiply two polynomial-basis elements of GF(32)."""
    result = 0
    a = int(left) & FIELD_MASK
    b = int(right) & FIELD_MASK
    for _ in range(5):
        if b & 1:
            result ^= a
        b >>= 1
        a <<= 1
        if a & FIELD_SIZE:
            a ^= PRIMITIVE_POLYNOMIAL
    return result & FIELD_MASK


def gf_trace(value: int) -> int:
    """Absolute trace GF(32)->GF(2)."""
    term = int(value) & FIELD_MASK
    result = term
    for _ in range(1, 5):
        term = gf_multiply(term, term)
        result ^= term
    if result not in (0, 1):
        raise AssertionError(f"invalid GF(32) trace {result}")
    return result


def parity_table() -> np.ndarray:
    return np.asarray([index.bit_count() & 1 for index in range(WIDTH)], dtype=np.int8)


def walsh_matrix() -> np.ndarray:
    indices = np.arange(WIDTH, dtype=np.uint16)
    parity = parity_table()
    return np.asarray(1 - 2 * parity[np.bitwise_and(indices[:, None], indices[None, :])], dtype=np.float32)


def chirps() -> np.ndarray:
    """Return c_a(x,y)=(-1)^Tr(a*x*y), a=0,...,15."""
    result = np.empty((2 * BASES_PER_BLOCK, WIDTH), dtype=np.int8)
    for a in range(result.shape[0]):
        for coordinate in range(WIDTH):
            x = coordinate & FIELD_MASK
            y = coordinate >> 5
            phase = gf_trace(gf_multiply(a, gf_multiply(x, y)))
            result[a, coordinate] = 1 if phase == 0 else -1
    return result


def fwht_in_place(values: np.ndarray) -> None:
    width = int(values.shape[0])
    span = 1
    while span < width:
        view = values.reshape(-1, 2 * span)
        left = view[:, :span].copy()
        right = view[:, span : 2 * span].copy()
        view[:, :span] = left + right
        view[:, span : 2 * span] = left - right
        span *= 2


def verify_mub_chirps(
    signs: np.ndarray, *, include_cross_block_pairs: bool = False
) -> dict[str, object]:
    """Exact integer proof that both eight-basis blocks are real MUBs."""
    if signs.shape != (16, WIDTH) or signs.dtype != np.int8:
        raise AssertionError("chirp schema changed")
    pair_count = 0
    spectra: set[int] = set()
    ranges = (
        [(0, TOTAL_BASES)]
        if include_cross_block_pairs
        else [(0, BASES_PER_BLOCK), (BASES_PER_BLOCK, TOTAL_BASES)]
    )
    for start, stop in ranges:
        for first in range(start, stop):
            for second in range(first + 1, stop):
                product = np.asarray(signs[first] * signs[second], dtype=np.int32)
                fwht_in_place(product)
                unique = np.unique(np.abs(product))
                if unique.tolist() != [32]:
                    raise AssertionError(
                        f"bases {first},{second} are not mutually unbiased: {unique}"
                    )
                spectra.update(int(value) for value in unique)
                pair_count += 1
    return {
        "construction": "H_1024 diag(c_a)/32",
        "coordinates": "(x,y) in GF(32)^2",
        "chirp": "c_a(x,y)=(-1)^Tr(a*x*y)",
        "primitive_polynomial_binary": bin(PRIMITIVE_POLYNOMIAL),
        "block_a_field_elements": list(range(0, 8)),
        "block_b_field_elements": list(range(8, 16)),
        "pairs_checked_within_blocks": pair_count,
        "unique_abs_unnormalized_cross_spectrum": sorted(spectra),
        "normalized_cross_gram_abs": 1.0 / math.sqrt(WIDTH),
        "status": "pass",
    }


def expected_chi(dimension: int) -> float:
    return math.sqrt(2.0) * math.exp(
        math.lgamma((dimension + 1.0) / 2.0) - math.lgamma(dimension / 2.0)
    )


def orientation(rank: int, block: int) -> tuple[np.ndarray, np.ndarray, int]:
    seed = int(SEED ^ (rank * 0x9E3779B1) ^ (block * 0x85EBCA77))
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(WIDTH).astype(np.int32)
    coordinate_sign = rng.choice(np.asarray([-1, 1], dtype=np.int8), size=WIDTH)
    return permutation, coordinate_sign, seed


def build_points(
    rank: int, signs: np.ndarray, *, common_orientation: bool = False
) -> tuple[np.ndarray, dict[str, object]]:
    """Build two N=16384 independently oriented antipodal MUB blocks."""
    hadamard = walsh_matrix()
    radius = expected_chi(WIDTH)
    scale = np.float32(radius / math.sqrt(WIDTH))
    blocks: list[np.ndarray] = []
    orientation_receipts: list[dict[str, object]] = []
    for block in range(BLOCKS):
        orientation_block = 0 if common_orientation else block
        permutation, coordinate_sign, block_seed = orientation(rank, orientation_block)
        bases: list[np.ndarray] = []
        start = block * BASES_PER_BLOCK
        for a in range(start, start + BASES_PER_BLOCK):
            # Rows are the w-indexed vectors H[w,x] c_a[x]/sqrt(1024).
            basis = hadamard * signs[a][None, :]
            basis = basis[:, permutation] * coordinate_sign[None, :]
            points = np.asarray(scale * basis, dtype=np.float32)
            bases.extend((points, -points))
        block_points = np.ascontiguousarray(np.concatenate(bases, axis=0), dtype=np.float32)
        if block_points.shape != (PATHS_PER_BLOCK, WIDTH):
            raise AssertionError("point block schema changed")
        blocks.append(block_points)
        orientation_receipts.append(
            {
                "block": block,
                "seed": block_seed,
                "permutation_sha256": hashlib.sha256(permutation.tobytes()).hexdigest(),
                "coordinate_sign_sha256": hashlib.sha256(coordinate_sign.tobytes()).hexdigest(),
                "orientation": "one common signed coordinate permutation per 8-MUB block",
                "common_across_blocks": bool(common_orientation),
            }
        )
    points = np.ascontiguousarray(np.concatenate(blocks, axis=0), dtype=np.float32)
    norms = np.linalg.norm(points[[0, PATHS_PER_BLOCK]], axis=1)
    if not np.allclose(norms, radius, rtol=2e-6, atol=2e-6):
        raise AssertionError(f"radial normalization failed: {norms} != {radius}")
    return points, {
        "expected_chi_1024": radius,
        "sphere_unit_normalization": "Hadamard rows divided by sqrt(1024)=32",
        "implemented_coordinate_scale": float(scale),
        "paths_per_basis": PATHS_PER_BASE,
        "bases_per_block": BASES_PER_BLOCK,
        "paths_per_block": PATHS_PER_BLOCK,
        "total_paths": TOTAL_PATHS,
        "antipodes": "explicit +u,-u for every basis row",
        "orientations": orientation_receipts,
    }


def exact_h1_h2_snap(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Exact h1 mean and fixed-radius h2 preactivation variance."""
    w0 = weights[0].to(torch.float64)
    w1 = weights[1].to(torch.float64)
    norms = torch.linalg.vector_norm(w0, dim=0)
    mu1 = norms / math.sqrt(2.0 * math.pi)
    gram = w0.T @ w0
    denominator = torch.clamp(norms[:, None] * norms[None, :], min=2.0**-50)
    rho = torch.clamp(gram / denominator, -1.0, 1.0)
    gaussian_second = denominator * (
        torch.sqrt(torch.clamp(1.0 - rho * rho, min=0.0))
        + (math.pi - torch.acos(rho)) * rho
    ) / (2.0 * math.pi)
    radial_factor = expected_chi(WIDTH) ** 2 / WIDTH
    sphere_covariance = radial_factor * gaussian_second - mu1[:, None] * mu1[None, :]
    exact_var2 = torch.sum((sphere_covariance @ w1) * w1, dim=0)
    exact_var2 = torch.clamp(exact_var2, min=2.0**-40)
    return (
        mu1.to(torch.float32),
        exact_var2.to(torch.float32),
        {
            "fixed_radius_second_moment_factor": radial_factor,
            "one_minus_radial_factor": 1.0 - radial_factor,
            "exact_h1_mean_min": float(mu1.min()),
            "exact_h1_mean_max": float(mu1.max()),
            "exact_h2_preactivation_variance_min": float(exact_var2.min()),
            "exact_h2_preactivation_variance_max": float(exact_var2.max()),
        },
    )


def propagate(
    points: np.ndarray,
    weights_array: np.ndarray,
    *,
    all_layer_snap_host: np.ndarray | None = None,
    all_layer_snap_sigma: np.ndarray | None = None,
    hermite_cv_host: np.ndarray | None = None,
    hermite_cv_raw: np.ndarray | None = None,
    hermite_cv_sigma: np.ndarray | None = None,
    retain_basis_readouts: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Run both blocks through h16 and apply the exact Phase-1 snaps."""
    torch.set_num_threads(12)
    weights = torch.from_numpy(np.ascontiguousarray(weights_array, dtype=np.float32))
    values = torch.from_numpy(points)
    mean_blocks = torch.empty((BLOCKS, DEPTH, WIDTH), dtype=torch.float32)
    gate_blocks = torch.empty_like(mean_blocks)
    h1_raw_mean = torch.empty((BLOCKS, WIDTH), dtype=torch.float32)
    h2_empirical_var = torch.empty((BLOCKS, WIDTH), dtype=torch.float32)
    h2_scale = torch.empty((BLOCKS, WIDTH), dtype=torch.float32)
    pre_snap_empirical_mean = torch.full(
        (BLOCKS, DEPTH, WIDTH), torch.nan, dtype=torch.float32
    )
    pre_snap_empirical_var = torch.full_like(pre_snap_empirical_mean, torch.nan)
    pre_snap_scale = torch.full_like(pre_snap_empirical_mean, torch.nan)
    pre_snap_target_mean = torch.full((DEPTH, WIDTH), torch.nan, dtype=torch.float32)
    pre_snap_target_var = torch.full_like(pre_snap_target_mean, torch.nan)
    cv_d2_mean_blocks = torch.full(
        (BLOCKS, DEPTH, WIDTH), torch.nan, dtype=torch.float32
    )
    cv_d4_mean_blocks = torch.full_like(cv_d2_mean_blocks, torch.nan)
    cv_d6_mean_blocks = torch.full_like(cv_d2_mean_blocks, torch.nan)
    cv_d8_mean_blocks = torch.full_like(cv_d2_mean_blocks, torch.nan)
    cv_d12_mean_blocks = torch.full_like(cv_d2_mean_blocks, torch.nan)
    cv_degree_correction_blocks = torch.full(
        (13, BLOCKS, DEPTH, WIDTH), torch.nan, dtype=torch.float32
    )
    mean_bases = torch.full(
        (TOTAL_BASES, DEPTH, WIDTH), torch.nan, dtype=torch.float32
    )
    gate_bases = torch.full_like(mean_bases, torch.nan)
    cv_d12_mean_bases = torch.full_like(mean_bases, torch.nan)
    timings: list[dict[str, float | int]] = []

    if (all_layer_snap_host is None) != (all_layer_snap_sigma is None):
        raise ValueError("all-layer snap host and sigma must be supplied together")
    all_layer_snap = all_layer_snap_host is not None
    if all_layer_snap:
        assert all_layer_snap_host is not None and all_layer_snap_sigma is not None
        if all_layer_snap_host.shape != (DEPTH, WIDTH):
            raise AssertionError("all-layer snap host schema changed")
        if all_layer_snap_sigma.shape != (DEPTH, WIDTH):
            raise AssertionError("all-layer snap sigma schema changed")
        snap_host = torch.from_numpy(
            np.ascontiguousarray(all_layer_snap_host, dtype=np.float32)
        )
        snap_sigma = torch.from_numpy(
            np.ascontiguousarray(all_layer_snap_sigma, dtype=np.float32)
        )
    else:
        snap_host = torch.empty(0)
        snap_sigma = torch.empty(0)
    cv_arguments = (hermite_cv_host, hermite_cv_raw, hermite_cv_sigma)
    hermite_cv = all(value is not None for value in cv_arguments)
    if any(value is not None for value in cv_arguments) and not hermite_cv:
        raise ValueError("Hermite-CV host/raw/sigma must be supplied together")
    if hermite_cv:
        assert hermite_cv_host is not None
        assert hermite_cv_raw is not None
        assert hermite_cv_sigma is not None
        cv_host = torch.from_numpy(
            np.ascontiguousarray(hermite_cv_host, dtype=np.float32)
        )
        cv_raw = torch.from_numpy(np.ascontiguousarray(hermite_cv_raw, dtype=np.float32))
        cv_sigma = torch.from_numpy(
            np.ascontiguousarray(hermite_cv_sigma, dtype=np.float32)
        )
    else:
        cv_host = torch.empty(0)
        cv_raw = torch.empty(0)
        cv_sigma = torch.empty(0)

    def record_hermite_cv(preactivation: torch.Tensor, layer: int) -> None:
        # Only q9:q15 is needed to form the decisive q10:q15 local suffix.
        if not hermite_cv or layer < 9:
            return
        if layer == 0:
            pre_mean = torch.zeros(WIDTH, dtype=torch.float32)
        else:
            pre_mean = cv_host[layer - 1] @ weights[layer]
        sigma_layer = cv_sigma[layer]
        u = (preactivation - pre_mean) / sigma_layer
        alpha = cv_raw[layer, :, 3]
        cdf = cv_raw[layer, :, 5]
        pdf = cv_raw[layer, :, 6]
        gaussian_mean = cv_raw[layer, :, 7] * sigma_layer
        gamma3 = cv_raw[layer, :, 10]
        alpha2 = alpha * alpha
        c1_he6 = sigma_layer * (
            alpha2 * alpha2 - 6.0 * alpha2 + 3.0
        ) * pdf
        positive = torch.relu(preactivation)
        # Exact probabilists' Hermite projection through D.  For d>=2,
        # c_d=E[ReLU(mu+sigma U) He_d(U)]
        #    =sigma*phi(alpha)*He_{d-2}(-alpha).
        alpha_hermites = [torch.ones_like(alpha), -alpha]
        for order in range(2, 11):
            alpha_hermites.append(
                (-alpha) * alpha_hermites[-1]
                - float(order - 1) * alpha_hermites[-2]
            )
        he_previous = torch.ones_like(u)
        he_current = u
        polynomial = gaussian_mean + sigma_layer * cdf * u
        analytic_degree3 = (
            sigma_layer * pdf * alpha_hermites[1] / 6.0
        ) * gamma3
        analytic_degree6 = c1_he6 * gamma3 * gamma3 / 72.0
        analytic_degree4 = (
            cv_host[layer] - gaussian_mean - analytic_degree3 - analytic_degree6
        )
        analytic_by_degree = {
            0: gaussian_mean,
            1: torch.zeros_like(gaussian_mean),
            2: torch.zeros_like(gaussian_mean),
            3: analytic_degree3,
            4: analytic_degree4,
            5: torch.zeros_like(gaussian_mean),
            6: analytic_degree6,
        }
        # The constant degree-zero correction is exactly zero.  Degree one is
        # recorded separately because the general d>=2 coefficient law differs.
        cv_degree_correction_blocks[0, :, layer] = 0.0
        degree1_term = sigma_layer * cdf * u
        for block in range(BLOCKS):
            block_slice = slice(
                block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK
            )
            cv_degree_correction_blocks[1, block, layer] = -torch.mean(
                degree1_term[block_slice], dim=0
            )
        destinations = {
            2: cv_d2_mean_blocks,
            4: cv_d4_mean_blocks,
            6: cv_d6_mean_blocks,
            8: cv_d8_mean_blocks,
            12: cv_d12_mean_blocks,
        }
        for degree in range(2, 13):
            he_next = u * he_current - float(degree - 1) * he_previous
            coefficient = (
                sigma_layer
                * pdf
                * alpha_hermites[degree - 2]
                / float(math.factorial(degree))
            )
            polynomial = polynomial + coefficient * he_next
            he_previous, he_current = he_current, he_next
            analytic_term = analytic_by_degree.get(
                degree, torch.zeros_like(gaussian_mean)
            )
            for block in range(BLOCKS):
                block_slice = slice(
                    block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK
                )
                cv_degree_correction_blocks[degree, block, layer] = (
                    analytic_term
                    - torch.mean((coefficient * he_next)[block_slice], dim=0)
                )
            if degree not in destinations:
                continue
            if degree == 2:
                analytic = gaussian_mean
            elif degree == 4:
                analytic = cv_host[layer] - c1_he6 * gamma3 * gamma3 / 72.0
            else:
                analytic = cv_host[layer]
            destination = destinations[degree]
            for block in range(BLOCKS):
                block_slice = slice(
                    block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK
                )
                destination[block, layer] = analytic + torch.mean(
                    positive[block_slice] - polynomial[block_slice], dim=0
                )
            if retain_basis_readouts and degree == 12:
                for basis in range(TOTAL_BASES):
                    basis_slice = slice(
                        basis * PATHS_PER_BASE, (basis + 1) * PATHS_PER_BASE
                    )
                    cv_d12_mean_bases[basis, layer] = analytic + torch.mean(
                        positive[basis_slice] - polynomial[basis_slice], dim=0
                    )

    def record_basis_activations(
        activations: torch.Tensor, gates: torch.Tensor, layer: int
    ) -> None:
        if not retain_basis_readouts:
            return
        for basis in range(TOTAL_BASES):
            basis_slice = slice(
                basis * PATHS_PER_BASE, (basis + 1) * PATHS_PER_BASE
            )
            mean_bases[basis, layer] = torch.mean(
                activations[basis_slice], dim=0
            )
            gate_bases[basis, layer] = torch.mean(
                gates[basis_slice].to(torch.float32), dim=0
            )

    started = time.perf_counter()
    exact_mu1, exact_var2, snap_receipt = exact_h1_h2_snap(weights)
    snap_receipt["exact_snap_seconds"] = time.perf_counter() - started

    with torch.no_grad():
        layer_started = time.perf_counter()
        preactivation = values @ weights[0]
        record_hermite_cv(preactivation, 0)
        for block in range(BLOCKS):
            block_slice = slice(block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK)
            block_pre = preactivation[block_slice]
            pre_snap_empirical_mean[block, 0] = torch.mean(block_pre, dim=0)
            pre_snap_empirical_var[block, 0] = torch.var(
                block_pre, dim=0, correction=0
            )
        gate = preactivation > 0.0
        values = torch.relu_(preactivation)
        for block in range(BLOCKS):
            block_slice = slice(block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK)
            block_values = values[block_slice]
            h1_raw_mean[block] = torch.mean(block_values, dim=0)
            mean_blocks[block, 0] = exact_mu1
            gate_blocks[block, 0] = torch.mean(gate[block_slice].to(torch.float32), dim=0)
            block_values.add_(exact_mu1 - h1_raw_mean[block])
        record_basis_activations(values, gate, 0)
        timings.append({"layer_one_based": 1, "seconds": time.perf_counter() - layer_started})

        layer_started = time.perf_counter()
        preactivation = values @ weights[1]
        exact_mean2 = exact_mu1 @ weights[1]
        pre_snap_target_mean[1] = exact_mean2
        pre_snap_target_var[1] = exact_var2
        for block in range(BLOCKS):
            block_slice = slice(block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK)
            centered = preactivation[block_slice] - exact_mean2
            empirical = torch.mean(centered * centered, dim=0)
            scale = torch.sqrt(exact_var2 / torch.clamp(empirical, min=2.0**-40))
            preactivation[block_slice] = exact_mean2 + centered * scale
            h2_empirical_var[block] = empirical
            h2_scale[block] = scale
            pre_snap_empirical_mean[block, 1] = torch.mean(
                preactivation[block_slice], dim=0
            )
            pre_snap_empirical_var[block, 1] = empirical
            pre_snap_scale[block, 1] = scale
        record_hermite_cv(preactivation, 1)
        gate = preactivation > 0.0
        values = torch.relu_(preactivation)
        for block in range(BLOCKS):
            block_slice = slice(block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK)
            mean_blocks[block, 1] = torch.mean(values[block_slice], dim=0)
            gate_blocks[block, 1] = torch.mean(gate[block_slice].to(torch.float32), dim=0)
        record_basis_activations(values, gate, 1)
        timings.append({"layer_one_based": 2, "seconds": time.perf_counter() - layer_started})

        for layer in range(2, DEPTH):
            layer_started = time.perf_counter()
            preactivation = values @ weights[layer]
            if all_layer_snap:
                target_mean64 = (
                    snap_host[layer - 1].to(torch.float64)
                    @ weights[layer].to(torch.float64)
                )
                target_mean = target_mean64.to(torch.float32)
                radial_factor = expected_chi(WIDTH) ** 2 / WIDTH
                sigma64 = snap_sigma[layer].to(torch.float64)
                target_var64 = radial_factor * (
                    sigma64 * sigma64 + target_mean64 * target_mean64
                ) - target_mean64 * target_mean64
                target_var = torch.clamp(target_var64, min=2.0**-40).to(torch.float32)
                pre_snap_target_mean[layer] = target_mean
                pre_snap_target_var[layer] = target_var
                for block in range(BLOCKS):
                    block_slice = slice(
                        block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK
                    )
                    block_pre = preactivation[block_slice]
                    empirical_mean = torch.mean(block_pre, dim=0)
                    centered = block_pre - empirical_mean
                    empirical_var = torch.mean(centered * centered, dim=0)
                    scale = torch.sqrt(
                        target_var / torch.clamp(empirical_var, min=2.0**-40)
                    )
                    preactivation[block_slice] = target_mean + centered * scale
                    pre_snap_empirical_mean[block, layer] = empirical_mean
                    pre_snap_empirical_var[block, layer] = empirical_var
                    pre_snap_scale[block, layer] = scale
            record_hermite_cv(preactivation, layer)
            gate = preactivation > 0.0
            values = torch.relu_(preactivation)
            for block in range(BLOCKS):
                block_slice = slice(block * PATHS_PER_BLOCK, (block + 1) * PATHS_PER_BLOCK)
                mean_blocks[block, layer] = torch.mean(values[block_slice], dim=0)
                gate_blocks[block, layer] = torch.mean(
                    gate[block_slice].to(torch.float32), dim=0
                )
            record_basis_activations(values, gate, layer)
            timings.append(
                {"layer_one_based": layer + 1, "seconds": time.perf_counter() - layer_started}
            )

    arrays = {
        "mean_blocks": mean_blocks.numpy(),
        "combined_mean": torch.mean(mean_blocks, dim=0).numpy(),
        "gate_blocks": gate_blocks.numpy(),
        "combined_gate": torch.mean(gate_blocks, dim=0).numpy(),
        "h1_raw_mean_blocks": h1_raw_mean.numpy(),
        "h1_exact_mean": exact_mu1.numpy(),
        "h2_empirical_var_blocks": h2_empirical_var.numpy(),
        "h2_exact_var": exact_var2.numpy(),
        "h2_scale_blocks": h2_scale.numpy(),
        "pre_snap_empirical_mean_blocks": pre_snap_empirical_mean.numpy(),
        "pre_snap_empirical_var_blocks": pre_snap_empirical_var.numpy(),
        "pre_snap_scale_blocks": pre_snap_scale.numpy(),
        "pre_snap_target_mean": pre_snap_target_mean.numpy(),
        "pre_snap_target_var": pre_snap_target_var.numpy(),
        "cv_d2_mean_blocks": cv_d2_mean_blocks.numpy(),
        "cv_d4_mean_blocks": cv_d4_mean_blocks.numpy(),
        "cv_d6_mean_blocks": cv_d6_mean_blocks.numpy(),
        "cv_d8_mean_blocks": cv_d8_mean_blocks.numpy(),
        "cv_d12_mean_blocks": cv_d12_mean_blocks.numpy(),
        "cv_degree_correction_blocks": cv_degree_correction_blocks.numpy(),
    }
    if retain_basis_readouts:
        arrays.update(
            {
                "mean_bases": mean_bases.numpy(),
                "gate_bases": gate_bases.numpy(),
                "cv_d12_mean_bases": cv_d12_mean_bases.numpy(),
            }
        )
    receipt: dict[str, object] = {
        "snap": snap_receipt,
        "layer_timings": timings,
        "total_propagation_seconds": sum(float(row["seconds"]) for row in timings),
        "full_depth_winograd_core_flops": WINOGRAD_FULL_DEPTH_CORE,
        "paired_mean_readout_flops_all_layers": DEPTH * MEAN_READOUT_PER_LAYER,
        "paired_gate_readout_flops_all_layers": DEPTH * GATE_READOUT_PER_LAYER,
        "core_plus_mean_readout_flops": WINOGRAD_FULL_DEPTH_CORE + DEPTH * MEAN_READOUT_PER_LAYER,
        "core_plus_mean_and_gate_readout_flops": (
            WINOGRAD_FULL_DEPTH_CORE + DEPTH * (MEAN_READOUT_PER_LAYER + GATE_READOUT_PER_LAYER)
        ),
        "all_layer_mean_variance_snap": all_layer_snap,
        "late_layer_hermite_cv": hermite_cv,
        "late_layer_hermite_cv_zero_based_layers": [9, 15] if hermite_cv else None,
        "retained_basis_readouts": bool(retain_basis_readouts),
        "basis_count": TOTAL_BASES if retain_basis_readouts else 0,
    }
    return arrays, receipt


def load_scalar_module():
    spec = importlib.util.spec_from_file_location(
        "exposure_scalar_runner", SCALAR_ROOT / "run_experiment.py"
    )
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen scalar runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    x = left.to(torch.float64).flatten()
    y = right.to(torch.float64).flatten()
    x = x - torch.mean(x)
    y = y - torch.mean(y)
    denominator = torch.sqrt(torch.sum(x * x) * torch.sum(y * y))
    if float(denominator) == 0.0:
        return 0.0
    return float(torch.sum(x * y) / denominator)


def mse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.mean((left.to(torch.float64) - right.to(torch.float64)) ** 2))


def score_prediction(prediction: torch.Tensor, truth_error: torch.Tensor) -> dict[str, object]:
    layer_mse = torch.mean(
        (prediction.to(torch.float64) - truth_error.to(torch.float64)) ** 2, dim=1
    )
    return {
        "final_mse": float(layer_mse[-1]),
        "mean_layer_mse": float(torch.mean(layer_mse)),
        "max_layer_mse": float(torch.max(layer_mse)),
        "layer_mse": [float(value) for value in layer_mse],
    }


def scalar_model_and_features(feature_dir: Path):
    scalar = load_scalar_module()
    checkpoint = torch.load(SCALAR_ROOT / "MODEL.pt", map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    centers = state_dict["centers"].numpy()
    scales = state_dict["scales"].numpy()
    layer_rms = state_dict["layer_rms"].numpy()
    model = scalar.LocalInnovationTransport(centers, scales, layer_rms)
    model.load_state_dict(state_dict)
    model.eval()
    raw = torch.from_numpy(
        np.ascontiguousarray(np.load(feature_dir / "cp1792_raw_locals.npy"), dtype=np.float32)
    )
    sigma = torch.from_numpy(
        np.ascontiguousarray(np.load(feature_dir / "cp1792_sigma.npy"), dtype=np.float32)
    )
    return model, raw, sigma


def scalar_rollouts(
    model,
    raw: torch.Tensor,
    sigma: torch.Tensor,
    weights: torch.Tensor,
    observations: torch.Tensor,
    gates: torch.Tensor,
    truth_error: torch.Tensor,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Frozen scalar capacity tests for state and local-source observation."""
    gain = model.transport_gain().detach()
    ybar = torch.mean(observations, dim=0)
    host_phi = raw[:, :, 5]

    def one_step(state: torch.Tensor, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer == FIRST_CORRECTED_LAYER:
            message = torch.zeros_like(state)
        else:
            message = host_phi[layer] * (state @ weights[layer])
        local = model.innovation(raw[layer], sigma[layer], message, layer)
        transport = gain[layer] * message
        return transport, local

    with torch.no_grad():
        autonomous = torch.zeros((DEPTH, WIDTH), dtype=torch.float32)
        teacher = torch.zeros_like(autonomous)
        state = torch.zeros(WIDTH, dtype=torch.float32)
        for layer in range(FIRST_CORRECTED_LAYER, DEPTH):
            transport, local = one_step(state, layer)
            state = transport + local
            autonomous[layer] = state
            teacher_prior = (
                torch.zeros_like(state)
                if layer == FIRST_CORRECTED_LAYER
                else truth_error[layer - 1]
            )
            transport, local = one_step(teacher_prior, layer)
            teacher[layer] = transport + local

        state_grid: dict[str, torch.Tensor] = {}
        for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
            state = torch.zeros(WIDTH, dtype=torch.float32)
            output = torch.zeros((DEPTH, WIDTH), dtype=torch.float32)
            for layer in range(DEPTH):
                if layer < FIRST_CORRECTED_LAYER:
                    prior_prediction = torch.zeros_like(state)
                else:
                    transport, local = one_step(state, layer)
                    prior_prediction = transport + local
                state = prior_prediction + beta * (ybar[layer] - prior_prediction)
                output[layer] = state
            state_grid[f"state_beta_{beta:.2f}"] = output

        wiener_state = torch.zeros((DEPTH, WIDTH), dtype=torch.float32)
        state = torch.zeros(WIDTH, dtype=torch.float32)
        state_wiener_rows: list[dict[str, float | int]] = []
        for layer in range(DEPTH):
            if layer < FIRST_CORRECTED_LAYER:
                prior_prediction = torch.zeros_like(state)
            else:
                transport, local = one_step(state, layer)
                prior_prediction = transport + local
            residual0 = observations[0, layer] - prior_prediction
            residual1 = observations[1, layer] - prior_prediction
            noise = torch.mean((observations[0, layer] - observations[1, layer]) ** 2) / 4.0
            signal = torch.clamp(torch.mean(residual0 * residual1), min=0.0)
            beta = signal / (signal + noise + 2.0**-50)
            state = prior_prediction + beta * (ybar[layer] - prior_prediction)
            wiener_state[layer] = state
            state_wiener_rows.append(
                {
                    "layer_one_based": layer + 1,
                    "signal_cross_mse": float(signal),
                    "mean_block_noise_mse": float(noise),
                    "beta": float(beta),
                }
            )

        # A local innovation is owned exactly once.  It subtracts transport of
        # the same block's prior observation before replacing/blending only the
        # learned source at the current layer.
        local_host = torch.zeros_like(observations)
        local_empirical = torch.zeros_like(observations)
        true_local = torch.zeros((DEPTH, WIDTH), dtype=torch.float32)
        for layer in range(FIRST_CORRECTED_LAYER, DEPTH):
            for block in range(BLOCKS):
                if layer == FIRST_CORRECTED_LAYER:
                    host_transport = torch.zeros(WIDTH)
                    empirical_transport = torch.zeros(WIDTH)
                else:
                    projected = observations[block, layer - 1] @ weights[layer]
                    host_transport = gain[layer] * host_phi[layer] * projected
                    empirical_transport = gain[layer] * gates[block, layer] * projected
                local_host[block, layer] = observations[block, layer] - host_transport
                local_empirical[block, layer] = observations[block, layer] - empirical_transport
            if layer == FIRST_CORRECTED_LAYER:
                true_transport = torch.zeros(WIDTH)
            else:
                true_transport = (
                    gain[layer]
                    * host_phi[layer]
                    * (truth_error[layer - 1] @ weights[layer])
                )
            true_local[layer] = truth_error[layer] - true_transport

        local_grid: dict[str, torch.Tensor] = {}
        for source_name, local_observation in (
            ("host_gate", local_host),
            ("empirical_gate", local_empirical),
        ):
            local_bar = torch.mean(local_observation, dim=0)
            for beta in (0.0, 0.25, 0.5, 0.75, 1.0):
                state = torch.zeros(WIDTH, dtype=torch.float32)
                output = torch.zeros((DEPTH, WIDTH), dtype=torch.float32)
                for layer in range(FIRST_CORRECTED_LAYER, DEPTH):
                    transport, learned_local = one_step(state, layer)
                    source = learned_local + beta * (local_bar[layer] - learned_local)
                    state = transport + source
                    output[layer] = state
                local_grid[f"local_{source_name}_beta_{beta:.2f}"] = output

        local_wiener = torch.zeros((DEPTH, WIDTH), dtype=torch.float32)
        state = torch.zeros(WIDTH, dtype=torch.float32)
        local_wiener_rows: list[dict[str, float | int]] = []
        for layer in range(FIRST_CORRECTED_LAYER, DEPTH):
            transport, learned_local = one_step(state, layer)
            residual0 = local_host[0, layer] - learned_local
            residual1 = local_host[1, layer] - learned_local
            noise = torch.mean((local_host[0, layer] - local_host[1, layer]) ** 2) / 4.0
            signal = torch.clamp(torch.mean(residual0 * residual1), min=0.0)
            beta = signal / (signal + noise + 2.0**-50)
            source = learned_local + beta * (
                torch.mean(local_host[:, layer], dim=0) - learned_local
            )
            state = transport + source
            local_wiener[layer] = state
            local_wiener_rows.append(
                {
                    "layer_one_based": layer + 1,
                    "signal_cross_mse": float(signal),
                    "mean_block_noise_mse": float(noise),
                    "beta": float(beta),
                }
            )

    diagnostics: dict[str, object] = {
        "checkpoint_scope": (
            "Frozen 6,688-parameter scalar CP1792-feature checkpoint; capacity "
            "discriminator only, not deployment-native four-state J"
        ),
        "autonomous": score_prediction(autonomous, truth_error),
        "teacher_forced": score_prediction(teacher, truth_error),
        "state_wiener": score_prediction(wiener_state, truth_error),
        "state_wiener_layers": state_wiener_rows,
        "local_host_gate_wiener": score_prediction(local_wiener, truth_error),
        "local_host_gate_wiener_layers": local_wiener_rows,
        "state_beta_grid": {
            name: score_prediction(value, truth_error) for name, value in state_grid.items()
        },
        "local_beta_grid": {
            name: score_prediction(value, truth_error) for name, value in local_grid.items()
        },
        "layer_observation": [],
        "layer_local_innovation": [],
    }
    for layer in range(DEPTH):
        noise = torch.mean((observations[0, layer] - observations[1, layer]) ** 2) / 4.0
        combined = ybar[layer]
        diagnostics["layer_observation"].append(
            {
                "layer_one_based": layer + 1,
                "block_a_truth_mse": mse(observations[0, layer], truth_error[layer]),
                "block_b_truth_mse": mse(observations[1, layer], truth_error[layer]),
                "combined_truth_mse": mse(combined, truth_error[layer]),
                "combined_truth_correlation": correlation(combined, truth_error[layer]),
                "mean_block_noise_mse": float(noise),
                "common_bias_proxy_mse": mse(combined, truth_error[layer]),
            }
        )
        if layer >= FIRST_CORRECTED_LAYER:
            local_bar = torch.mean(local_host[:, layer], dim=0)
            empirical_bar = torch.mean(local_empirical[:, layer], dim=0)
            local_noise = torch.mean(
                (local_host[0, layer] - local_host[1, layer]) ** 2
            ) / 4.0
            diagnostics["layer_local_innovation"].append(
                {
                    "layer_one_based": layer + 1,
                    "host_gate_combined_truth_mse": mse(local_bar, true_local[layer]),
                    "host_gate_combined_truth_correlation": correlation(
                        local_bar, true_local[layer]
                    ),
                    "empirical_gate_combined_vs_host_truth_mse": mse(
                        empirical_bar, true_local[layer]
                    ),
                    "empirical_gate_combined_vs_host_truth_correlation": correlation(
                        empirical_bar, true_local[layer]
                    ),
                    "host_gate_mean_block_noise_mse": float(local_noise),
                    "true_local_mse": float(torch.mean(true_local[layer] ** 2)),
                }
            )

    arrays = {
        "observation_residual_blocks": observations.numpy(),
        "observation_residual_combined": ybar.numpy(),
        "scalar_autonomous_state": autonomous.numpy(),
        "scalar_teacher_state": teacher.numpy(),
        "scalar_state_wiener": wiener_state.numpy(),
        "scalar_local_host_gate": local_host.numpy(),
        "scalar_local_empirical_gate": local_empirical.numpy(),
        "scalar_true_local_host_gate": true_local.numpy(),
        "scalar_local_host_gate_wiener": local_wiener.numpy(),
    }
    for name, value in state_grid.items():
        arrays[f"scalar_{name}"] = value.numpy()
    for name, value in local_grid.items():
        arrays[f"scalar_{name}"] = value.numpy()
    return diagnostics, arrays


def load_native_j_module():
    loader_dir = P2 / "public_full688_local_innovation_transport_v1"
    sys.path.insert(0, str(loader_dir))
    spec = importlib.util.spec_from_file_location("phase1_snap_native_j", NATIVE_J_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load native-J source")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def native_j_suffix_diagnostics(
    mean_blocks: np.ndarray,
    host: np.ndarray,
    raw_array: np.ndarray,
    sigma_array: np.ndarray,
    weights_array: np.ndarray,
    target: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    """Exact physical q10:q15 replay of the frozen deployment-native J bank."""
    j = load_native_j_module()
    checkpoint = torch.load(NATIVE_J_MODEL, map_location="cpu", weights_only=True)
    state_dict = checkpoint["state_dict"]
    model = j.TransportEnergyMultiStateResidual(
        state_dict["centers"].numpy(),
        state_dict["scales"].numpy(),
        state_dict["layer_rms"].numpy(),
    )
    model.load_state_dict(state_dict)
    model.eval()
    raw = torch.from_numpy(np.ascontiguousarray(raw_array, dtype=np.float32))
    sigma = torch.from_numpy(np.ascontiguousarray(sigma_array, dtype=np.float32))
    weights = torch.from_numpy(np.ascontiguousarray(weights_array, dtype=np.float32))
    observations = torch.from_numpy(
        np.ascontiguousarray(mean_blocks - host[None, :, :], dtype=np.float32)
    )
    truth_error = torch.from_numpy(np.ascontiguousarray(target - host, dtype=np.float32))
    phi = raw[:, :, 5]
    with torch.no_grad():
        autonomous_batched, state_history, _ = model(
            raw[None], sigma[None], weights[None], return_states=True
        )
        autonomous = autonomous_batched[0]
        full_states = state_history[0]
        q_bank = torch.zeros_like(autonomous)
        q_truth = torch.zeros_like(autonomous)
        q_blocks = torch.zeros_like(observations)
        q_bank[FIRST_CORRECTED_LAYER] = autonomous[FIRST_CORRECTED_LAYER]
        q_truth[FIRST_CORRECTED_LAYER] = truth_error[FIRST_CORRECTED_LAYER]
        q_blocks[:, FIRST_CORRECTED_LAYER] = observations[:, FIRST_CORRECTED_LAYER]
        for layer in range(FIRST_CORRECTED_LAYER + 1, DEPTH):
            q_bank[layer] = autonomous[layer] - phi[layer] * (
                autonomous[layer - 1] @ weights[layer]
            )
            q_truth[layer] = truth_error[layer] - phi[layer] * (
                truth_error[layer - 1] @ weights[layer]
            )
            for block in range(BLOCKS):
                q_blocks[block, layer] = observations[block, layer] - phi[layer] * (
                    observations[block, layer - 1] @ weights[layer]
                )

        suffix_start = 10  # zero-based q10:q15, the proven six-local oracle window.

        def replay(q_values: torch.Tensor) -> torch.Tensor:
            state = autonomous[suffix_start - 1].clone()
            for layer in range(suffix_start, DEPTH):
                state = phi[layer] * (state @ weights[layer]) + q_values[layer]
            return state

        bank_replay = replay(q_bank)
        truth_oracle = replay(q_truth)
        block_replays = torch.stack([replay(q_blocks[block]) for block in range(BLOCKS)])
        endpoint_actions = block_replays - bank_replay
        endpoint_average = torch.mean(endpoint_actions, dim=0)
        endpoint_truth = truth_oracle - bank_replay
        endpoint_noise = torch.mean((endpoint_actions[0] - endpoint_actions[1]) ** 2) / 4.0
        endpoint_signal = torch.clamp(
            torch.mean(endpoint_actions[0] * endpoint_actions[1]), min=0.0
        )
        split_beta = endpoint_signal / (endpoint_signal + endpoint_noise + 2.0**-50)
        truth_beta = torch.clamp(
            torch.sum(endpoint_average * (truth_error[-1] - bank_replay))
            / torch.clamp(torch.sum(endpoint_average * endpoint_average), min=2.0**-50),
            0.0,
            1.0,
        )

        def final_mse_for_beta(beta: torch.Tensor | float) -> float:
            return mse(bank_replay + float(beta) * endpoint_average, truth_error[-1])

        per_layer: list[dict[str, object]] = []
        for layer in range(suffix_start, DEPTH):
            q_average = torch.mean(q_blocks[:, layer], dim=0)
            q_noise = torch.mean((q_blocks[0, layer] - q_blocks[1, layer]) ** 2) / 4.0
            residual0 = q_blocks[0, layer] - q_bank[layer]
            residual1 = q_blocks[1, layer] - q_bank[layer]
            q_signal = torch.clamp(torch.mean(residual0 * residual1), min=0.0)
            q_beta = q_signal / (q_signal + q_noise + 2.0**-50)
            per_layer.append(
                {
                    "layer_zero_based": layer,
                    "layer_one_based": layer + 1,
                    "combined_q_truth_mse": mse(q_average, q_truth[layer]),
                    "combined_q_truth_correlation": correlation(q_average, q_truth[layer]),
                    "split_average_noise_mse": float(q_noise),
                    "cross_signal_around_bank_mse": float(q_signal),
                    "split_wiener_beta": float(q_beta),
                }
            )

    result = {
        "scope": "exact frozen deployment-native four-state J, physical-primary q10:q15 suffix",
        "model_sha256": sha256(NATIVE_J_MODEL),
        "suffix_zero_based": [suffix_start, DEPTH - 1],
        "bank_replay_byte_max_abs": float(torch.max(torch.abs(bank_replay - autonomous[-1]))),
        "bank_final_mse": mse(bank_replay, truth_error[-1]),
        "exact_truth_suffix_oracle_final_mse": mse(truth_oracle, truth_error[-1]),
        "sample_beta_one_final_mse": final_mse_for_beta(1.0),
        "endpoint_measurement_error_mse": mse(endpoint_average, endpoint_truth),
        "endpoint_split_average_noise_mse": float(endpoint_noise),
        "endpoint_cross_signal_around_bank_mse": float(endpoint_signal),
        "endpoint_split_wiener_beta": float(split_beta),
        "endpoint_split_wiener_final_mse": final_mse_for_beta(split_beta),
        "endpoint_truth_optimal_beta": float(truth_beta),
        "endpoint_truth_optimal_final_mse": final_mse_for_beta(truth_beta),
        "endpoint_average_truth_correlation": correlation(endpoint_average, endpoint_truth),
        "required_endpoint_measurement_error_mse": 2.075e-9,
        "required_gate_pass": bool(
            mse(endpoint_average, endpoint_truth) < 2.075e-9
            and correlation(endpoint_average, endpoint_truth) > 0.0
        ),
        "per_layer_q10_q15": per_layer,
    }
    arrays = {
        "native_j_autonomous": autonomous.numpy(),
        "native_j_full_states": full_states.numpy(),
        "native_j_q_bank": q_bank.numpy(),
        "native_j_q_truth": q_truth.numpy(),
        "native_j_q_blocks": q_blocks.numpy(),
        "native_j_endpoint_actions": endpoint_actions.numpy(),
        "native_j_endpoint_average": endpoint_average.numpy(),
        "native_j_endpoint_truth": endpoint_truth.numpy(),
    }
    return result, arrays


def native_j_basis_actions(
    mean_bases: np.ndarray,
    host: np.ndarray,
    phi: np.ndarray,
    weights_array: np.ndarray,
    autonomous: np.ndarray,
    q_bank: np.ndarray,
) -> dict[str, np.ndarray]:
    """Replay each retained basis through the frozen physical q10:q15 suffix."""
    means = torch.from_numpy(np.ascontiguousarray(mean_bases, dtype=np.float32))
    host_tensor = torch.from_numpy(np.ascontiguousarray(host, dtype=np.float32))
    phi_tensor = torch.from_numpy(np.ascontiguousarray(phi, dtype=np.float32))
    weights = torch.from_numpy(np.ascontiguousarray(weights_array, dtype=np.float32))
    autonomous_tensor = torch.from_numpy(
        np.ascontiguousarray(autonomous, dtype=np.float32)
    )
    q_bank_tensor = torch.from_numpy(np.ascontiguousarray(q_bank, dtype=np.float32))
    observations = means - host_tensor[None, :, :]
    q_bases = torch.zeros_like(observations)
    with torch.no_grad():
        for layer in range(FIRST_CORRECTED_LAYER + 1, DEPTH):
            q_bases[:, layer] = observations[:, layer] - phi_tensor[layer] * (
                observations[:, layer - 1] @ weights[layer]
            )

        def replay(q_values: torch.Tensor) -> torch.Tensor:
            state = autonomous_tensor[9].expand(q_values.shape[0], -1).clone()
            for layer in range(10, DEPTH):
                state = phi_tensor[layer] * (state @ weights[layer]) + q_values[:, layer]
            return state

        bank_state = autonomous_tensor[9].clone()
        for layer in range(10, DEPTH):
            bank_state = (
                phi_tensor[layer] * (bank_state @ weights[layer]) + q_bank_tensor[layer]
            )
        endpoints = replay(q_bases)
    return {
        "q_bases": q_bases.numpy(),
        "endpoint_basis_actions": (endpoints - bank_state[None, :]).numpy(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--open-target", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--reuse-tape", action="store_true")
    parser.add_argument("--all-layer-snap", action="store_true")
    parser.add_argument("--hermite-cv", action="store_true")
    parser.add_argument("--basis-readouts", action="store_true")
    parser.add_argument("--full-kerdock-seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rank = int(args.rank)
    feature_dir = FEATURE_ROOT / f"{rank:04d}"
    run_dir = RUN_ROOT / f"row_{rank:04d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    selected_kerdock_indices = None
    full_kerdock = args.full_kerdock_seed is not None
    if full_kerdock:
        from standard_real_kerdock import (
            chirp_signs as standard_chirp_signs,
            standard_kerdock_matrices,
            verify_kerdock_set,
        )

        matrices = standard_kerdock_matrices()
        full_proof = verify_kerdock_set(matrices)
        selection_rng = np.random.default_rng(
            int(args.full_kerdock_seed) ^ (rank * 0x9E3779B1)
        )
        selected_kerdock_indices = selection_rng.choice(
            matrices.shape[0], size=TOTAL_BASES, replace=False
        ).astype(np.int32)
        chirp_signs = standard_chirp_signs(matrices, selected_kerdock_indices)
        mub_receipt = verify_mub_chirps(
            chirp_signs, include_cross_block_pairs=True
        )
        mub_receipt.update(
            {
                "construction": "H_1024 diag((-1)^Q_M)/32",
                "coordinates": "v in GF(2)^10",
                "chirp": "(-1)^(v^T upper(M_s) v)",
                "primitive_polynomial_binary": "0b1000010001",
                "block_a_field_elements": None,
                "block_b_field_elements": None,
                "pairs_checked_within_blocks": 120,
                "family": "standard-real-kerdock-512",
                "full_set_proof": full_proof,
                "selection_seed": int(args.full_kerdock_seed),
                "selected_indices": selected_kerdock_indices.tolist(),
                "split_indices": [
                    selected_kerdock_indices[:BASES_PER_BLOCK].tolist(),
                    selected_kerdock_indices[BASES_PER_BLOCK:].tolist(),
                ],
            }
        )
    else:
        chirp_signs = chirps()
        mub_receipt = verify_mub_chirps(chirp_signs)
    if args.verify_only:
        print(json.dumps(mub_receipt, indent=2, sort_keys=True))
        return

    weights_path = feature_dir / "weights.npy"
    weights_array = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights_array.shape != (DEPTH, WIDTH, WIDTH) or weights_array.dtype != np.float32:
        raise AssertionError("weight schema changed")
    native_row_dir = NATIVE_ROW_ROOT / f"row_{rank:04d}"
    native_host = None
    native_raw = None
    native_sigma = None
    if native_row_dir.is_dir():
        native_host = np.ascontiguousarray(
            np.load(native_row_dir / "host.npy", allow_pickle=False), dtype=np.float32
        )
        native_raw = np.ascontiguousarray(
            np.load(native_row_dir / "raw20.npy", allow_pickle=False), dtype=np.float32
        )
        native_sigma = np.ascontiguousarray(
            np.load(native_row_dir / "sigma.npy", allow_pickle=False), dtype=np.float32
        )
    if args.all_layer_snap and native_host is None:
        raise FileNotFoundError(f"native Q24 snap tape unavailable: {native_row_dir}")

    mode_parts = ["ALL_LAYER_SNAP" if args.all_layer_snap else "BASE"]
    if args.hermite_cv:
        mode_parts.append("HERMITE_CV_D12_DEGREES")
    if full_kerdock:
        mode_parts.insert(0, f"FULL_KERDOCK_SEED{int(args.full_kerdock_seed)}")
    if args.basis_readouts:
        mode_parts.append("BASIS_READOUTS")
    mode_tag = "_".join(mode_parts)
    tape_path = run_dir / f"KERDOCK_TAPE_{mode_tag}.npz"
    legacy_tape = run_dir / "KERDOCK_TAPE.npz"
    if not args.all_layer_snap and not tape_path.exists() and legacy_tape.exists():
        tape_path = legacy_tape
    if args.reuse_tape and tape_path.is_file():
        with np.load(tape_path, allow_pickle=False) as loaded:
            tape_arrays = {name: loaded[name] for name in loaded.files}
        point_receipt: dict[str, object] = {"reused": True}
        propagation_receipt: dict[str, object] = {"reused": True}
    else:
        points, point_receipt = build_points(
            rank, chirp_signs, common_orientation=full_kerdock
        )
        if selected_kerdock_indices is not None:
            point_receipt["selected_kerdock_indices"] = selected_kerdock_indices.tolist()
        tape_arrays, propagation_receipt = propagate(
            points,
            weights_array,
            all_layer_snap_host=native_host if args.all_layer_snap else None,
            all_layer_snap_sigma=native_sigma if args.all_layer_snap else None,
            hermite_cv_host=native_host if args.hermite_cv else None,
            hermite_cv_raw=native_raw if args.hermite_cv else None,
            hermite_cv_sigma=native_sigma if args.hermite_cv else None,
            retain_basis_readouts=args.basis_readouts,
        )
        np.savez_compressed(tape_path, **tape_arrays)

    manifest = json.loads((feature_dir / "row_manifest.json").read_text(encoding="utf-8"))
    mlp_id = int(manifest["mlp_id"])
    scalar = load_scalar_module()
    if args.all_layer_snap:
        assert native_host is not None
        host_path = native_row_dir / "host.npy"
        host_source = "deployment-native-q24"
        host_old_max = None
        host = native_host
    else:
        host_path, host_source, host_old_max = scalar.selected_host(rank, mlp_id)
        host = np.ascontiguousarray(np.load(host_path), dtype=np.float32)
    if host.shape != (DEPTH, WIDTH):
        raise AssertionError("host schema changed")
    observations = np.ascontiguousarray(tape_arrays["mean_blocks"] - host[None, :, :])

    result: dict[str, object] = {
        "schema": "phase1-kerdock-hidden-snap-discriminator-v1",
        "role": "observer_fit",
        "role_rank": rank,
        "mlp_id": mlp_id,
        "target_opened": bool(args.open_target),
        "sampling_mode": mode_tag,
        "mub_proof": mub_receipt,
        "point_design": point_receipt,
        "propagation": propagation_receipt,
        "host": {
            "source": host_source,
            "path": str(host_path.relative_to(P2)),
            "sha256": sha256(host_path),
            "old_host_max_mse_gate_value": host_old_max,
        },
        "weights_sha256": sha256(weights_path),
        "artifacts": {
            "tape": tape_path.name,
            "tape_sha256": sha256(tape_path),
            "mean_blocks_shape": list(tape_arrays["mean_blocks"].shape),
            "gate_blocks_shape": list(tape_arrays["gate_blocks"].shape),
            "mean_bases_shape": (
                list(tape_arrays["mean_bases"].shape)
                if "mean_bases" in tape_arrays
                else None
            ),
        },
    }

    diagnostic_arrays = {
        "host": host,
        "observation_residual_blocks": observations,
        "observation_residual_combined": np.mean(observations, axis=0, dtype=np.float32),
    }
    if args.open_target:
        ids, target_prefix = scalar.target_prefix(rank + 1)
        if int(ids[rank]) != mlp_id:
            raise AssertionError("target rank/MLP mismatch")
        target = np.ascontiguousarray(target_prefix[rank], dtype=np.float32)
        truth_error = torch.from_numpy(target - host)
        model, raw, sigma = scalar_model_and_features(feature_dir)
        scalar_diagnostics, scalar_arrays = scalar_rollouts(
            model,
            raw,
            sigma,
            torch.from_numpy(np.ascontiguousarray(weights_array, dtype=np.float32)),
            torch.from_numpy(observations),
            torch.from_numpy(tape_arrays["gate_blocks"]),
            truth_error,
        )
        result["scalar_capacity_discriminator"] = scalar_diagnostics
        result["direct_combined_sample_mean"] = score_prediction(
            torch.from_numpy(tape_arrays["combined_mean"] - host), truth_error
        )
        result["host_only"] = score_prediction(torch.zeros_like(truth_error), truth_error)
        diagnostic_arrays["target"] = target
        diagnostic_arrays["truth_error"] = truth_error.numpy()
        diagnostic_arrays.update(scalar_arrays)
        if native_raw is not None and native_sigma is not None:
            native_diagnostics, native_arrays = native_j_suffix_diagnostics(
                tape_arrays["mean_blocks"],
                host,
                native_raw,
                native_sigma,
                weights_array,
                target,
            )
            result["native_j_suffix_discriminator"] = native_diagnostics
            diagnostic_arrays.update(native_arrays)
            if args.basis_readouts:
                raw_basis = native_j_basis_actions(
                    tape_arrays["mean_bases"],
                    host,
                    native_raw[:, :, 5],
                    weights_array,
                    native_arrays["native_j_autonomous"],
                    native_arrays["native_j_q_bank"],
                )
                diagnostic_arrays["native_j_q_bases"] = raw_basis["q_bases"]
                diagnostic_arrays["native_j_endpoint_basis_actions"] = raw_basis[
                    "endpoint_basis_actions"
                ]
            if args.hermite_cv:
                result["native_j_hermite_cv_suffix_discriminators"] = {}
                for degree in (2, 4, 6, 8, 12):
                    cv_key = f"cv_d{degree}_mean_blocks"
                    cv_means = np.array(tape_arrays["mean_blocks"], copy=True)
                    cv_means[:, 9:] = tape_arrays[cv_key][:, 9:]
                    cv_result, cv_arrays = native_j_suffix_diagnostics(
                        cv_means,
                        host,
                        native_raw,
                        native_sigma,
                        weights_array,
                        target,
                    )
                    result["native_j_hermite_cv_suffix_discriminators"][f"D{degree}"] = cv_result
                    for name, value in cv_arrays.items():
                        diagnostic_arrays[f"cv_d{degree}_{name}"] = value
                if args.basis_readouts:
                    cv_d12_basis_means = np.array(
                        tape_arrays["mean_bases"], copy=True
                    )
                    cv_d12_basis_means[:, 9:] = tape_arrays[
                        "cv_d12_mean_bases"
                    ][:, 9:]
                    cv_d12_basis = native_j_basis_actions(
                        cv_d12_basis_means,
                        host,
                        native_raw[:, :, 5],
                        weights_array,
                        native_arrays["native_j_autonomous"],
                        native_arrays["native_j_q_bank"],
                    )
                    diagnostic_arrays["cv_d12_native_j_q_bases"] = cv_d12_basis[
                        "q_bases"
                    ]
                    diagnostic_arrays[
                        "cv_d12_native_j_endpoint_basis_actions"
                    ] = cv_d12_basis["endpoint_basis_actions"]

    diagnostics_path = run_dir / f"DIAGNOSTIC_ARRAYS_{mode_tag}.npz"
    np.savez_compressed(diagnostics_path, **diagnostic_arrays)
    result["artifacts"]["diagnostics"] = diagnostics_path.name
    result["artifacts"]["diagnostics_sha256"] = sha256(diagnostics_path)
    result_path = run_dir / f"RESULT_{mode_tag}.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(result_path), "summary": result.get("scalar_capacity_discriminator")}, indent=2))


if __name__ == "__main__":
    main()
