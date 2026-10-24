#!/usr/bin/env python3
"""Read-only verifier for the late-layer Hermite-CV terminal audit."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
RECEIPT_PATH = HERE / "AUDIT_RECEIPT.json"


def find_repo() -> Path:
    for candidate in (HERE, *HERE.parents):
        if (candidate / ".git").exists():
            return candidate
    raise AssertionError("repository root not found")


REPO = find_repo()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def close(actual: float, expected: float, *, atol: float = 2e-13, rtol: float = 2e-12) -> None:
    if not math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol):
        raise AssertionError(f"numeric mismatch: actual={actual!r}, expected={expected!r}")


def coefficients_through(alpha: np.ndarray, pdf: np.ndarray, maximum: int) -> dict[int, np.ndarray]:
    """Stable normalized coefficients c_d/scale for d >= 2."""
    result: dict[int, np.ndarray] = {2: pdf / math.sqrt(2.0)}
    if maximum == 2:
        return result
    x = -alpha
    result[3] = x * pdf / math.sqrt(6.0)
    for degree in range(4, maximum + 1):
        n = degree - 2
        result[degree] = (
            x * result[degree - 1] / math.sqrt(n + 2.0)
            - (n - 1.0) * result[degree - 2] / math.sqrt((n + 1.0) * (n + 2.0))
        )
    return result


def stable_variance_and_residuals(
    alpha: np.ndarray, sigma: np.ndarray, degrees: list[int]
) -> tuple[float, dict[int, float]]:
    t = np.abs(alpha)
    pdf = np.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi)
    q = np.fromiter(
        (0.5 * math.erfc(float(value) / math.sqrt(2.0)) for value in t),
        dtype=np.float64,
        count=t.size,
    )
    mills_remainder = pdf - t * q
    residual_one = (
        (1.0 + t * t) * q
        - t * pdf
        - mills_remainder * mills_remainder
        - q * q
    )
    if float(np.min(residual_one)) < -1e-18:
        raise AssertionError("stable residual-one identity became negative")
    b1 = np.where(alpha >= 0.0, 1.0 - q, q)
    variance = b1 * b1 + residual_one
    coeff = coefficients_through(alpha, pdf, max(degrees))
    sigma2 = sigma * sigma
    total = float(np.sum(sigma2 * variance, dtype=np.float64))
    residuals: dict[int, float] = {}
    for degree in degrees:
        remaining = residual_one.copy()
        for order in range(2, degree + 1):
            remaining -= coeff[order] * coeff[order]
        if float(np.min(remaining)) < -2e-15:
            raise AssertionError(f"negative degree-{degree} residual")
        residuals[degree] = float(
            np.sum(sigma2 * np.maximum(remaining, 0.0), dtype=np.float64)
        )
    return total, residuals


def native_directory(base: Path, rank: int) -> Path:
    if rank == 35:
        return base / "runs/native_meter_rank0035_v1"
    return base / f"runs/native_rows_v1/row_{rank:04d}"


def main() -> None:
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    assert receipt["schema"] == "p2r0f3c-late-layer-hermite-cv-audit-v1"
    assert receipt["status"] == "CLOSED_CURRENT_FORM__EXTERNAL_CERTIFIED_MOMENT_ORACLE_ONLY"

    for relative, expected in receipt["source_sha256"].items():
        actual = sha256(REPO / relative)
        if actual != expected:
            raise AssertionError(f"source changed: {relative}: {actual} != {expected}")
    for name, expected in receipt["artifact_sha256"].items():
        actual = sha256(HERE / name)
        if actual != expected:
            raise AssertionError(f"artifact changed: {name}: {actual} != {expected}")
    if any(int(value) != 0 for value in receipt["access"].values()):
        raise AssertionError("nonzero access declaration")

    # Exact a=0 theory.
    phi = 1.0 / math.sqrt(2.0 * math.pi)
    variance_zero = 0.5 - phi * phi
    hermites = [1.0, 0.0]
    for n in range(2, 31):
        hermites.append(-(n - 1.0) * hermites[n - 2])
    for degree_text, expected in receipt["gaussian_theory"]["capture_at_a0"].items():
        degree = int(degree_text)
        captured = 0.25
        for order in range(2, degree + 1):
            coefficient = phi * hermites[order - 2] / math.sqrt(math.factorial(order))
            captured += coefficient * coefficient
        close(captured / variance_zero, expected)

    variance_range = receipt["gaussian_theory"]["public_variance_range"]
    ideal_n = receipt["gaussian_theory"]["ideal_free_feature_paths"]
    required = [1.0 - 1e-8 * ideal_n / value for value in variance_range]
    for actual, expected in zip(required, receipt["gaussian_theory"]["required_capture_at_ideal_n"]):
        close(actual, expected)
    assert receipt["gaussian_theory"]["capture_at_a0"]["6"] < required[1]
    assert receipt["gaussian_theory"]["capture_at_a0"]["8"] > required[1]

    # Frozen 200-row target-free native roster and census.
    native_relative = receipt["native_source"]["base"]
    native_base = REPO / native_relative
    degrees = [int(value) for value in receipt["native_census"]["degrees"]]
    roster: list[dict[str, object]] = []
    aggregate_total = 0.0
    aggregate_residual = {degree: 0.0 for degree in degrees}
    all_alpha: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []
    row_residual_eight: list[float] = []
    row_capture_eight: list[float] = []

    for rank in range(35, 235):
        directory = native_directory(native_base, rank)
        result_path = directory / ("RESULT.json" if rank == 35 else "result.json")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        raw_path = directory / "raw20.npy"
        sigma_path = directory / "sigma.npy"
        raw_file_hash = sha256(raw_path)
        sigma_file_hash = sha256(sigma_path)
        raw = np.load(raw_path, allow_pickle=False)
        sigma_full = np.load(sigma_path, allow_pickle=False)
        if raw.dtype != np.dtype("<f4") or raw.shape != (16, 1024, 20):
            raise AssertionError(f"raw schema changed at rank {rank}")
        if sigma_full.dtype != np.dtype("<f4") or sigma_full.shape != (16, 1024):
            raise AssertionError(f"sigma schema changed at rank {rank}")
        if rank == 35:
            expected = receipt["native_source"]["rank35_file_sha256"]
            assert raw_file_hash == expected["raw20.npy"]
            assert sigma_file_hash == expected["sigma.npy"]
            assert array_sha256(raw) == result["native_tape"]["raw20_array_sha256"]
            assert array_sha256(sigma_full) == result["native_tape"]["sigma_array_sha256"]
        else:
            assert raw_file_hash == result["files"]["raw20.npy"]["sha256"]
            assert sigma_file_hash == result["files"]["sigma.npy"]["sha256"]
            assert array_sha256(raw) == result["files"]["raw20.npy"]["array_sha256"]
            assert array_sha256(sigma_full) == result["files"]["sigma.npy"]["array_sha256"]
        roster.append(
            {
                "rank": rank,
                "result_sha256": sha256(result_path),
                "raw20_sha256": raw_file_hash,
                "sigma_sha256": sigma_file_hash,
            }
        )
        alpha = np.asarray(raw[15, :, 3], dtype=np.float64)
        sigma = np.asarray(sigma_full[15], dtype=np.float64)
        total, residuals = stable_variance_and_residuals(alpha, sigma, degrees)
        aggregate_total += total
        for degree in degrees:
            aggregate_residual[degree] += residuals[degree]
        row_residual_eight.append(residuals[8] / 1024.0)
        row_capture_eight.append(1.0 - residuals[8] / total)
        all_alpha.append(alpha)
        all_sigma.append(sigma)

    encoded_roster = json.dumps(roster, sort_keys=True, separators=(",", ":")).encode("utf-8")
    roster_digest = hashlib.sha256(encoded_roster).hexdigest()
    assert roster_digest == receipt["native_source"]["canonical_roster_sha256"]
    alpha_all = np.concatenate(all_alpha)
    sigma_all = np.concatenate(all_sigma)
    units = int(alpha_all.size)
    assert units == receipt["native_census"]["lanes"] == 204800
    close(float(np.min(alpha_all)), receipt["native_census"]["alpha_min"])
    close(float(np.max(alpha_all)), receipt["native_census"]["alpha_max"])
    close(float(np.median(alpha_all)), receipt["native_census"]["alpha_median"])
    close(float(np.sqrt(np.mean(alpha_all * alpha_all))), receipt["native_census"]["alpha_rms"])
    close(float(np.min(sigma_all)), receipt["native_census"]["sigma_min"])
    close(float(np.max(sigma_all)), receipt["native_census"]["sigma_max"])
    close(float(np.mean(sigma_all)), receipt["native_census"]["sigma_mean"])
    close(aggregate_total / units, receipt["native_census"]["mean_variance"])
    for degree in degrees:
        expected = receipt["native_census"]["by_degree"][str(degree)]
        mean_residual = aggregate_residual[degree] / units
        capture = 1.0 - aggregate_residual[degree] / aggregate_total
        close(mean_residual, expected["mean_residual_variance"])
        close(capture, expected["aggregate_capture"])
    close(min(row_residual_eight), receipt["native_census"]["degree8_row_residual_min"])
    close(max(row_residual_eight), receipt["native_census"]["degree8_row_residual_max"])
    close(min(row_capture_eight), receipt["native_census"]["degree8_row_capture_min"])
    close(max(row_capture_eight), receipt["native_census"]["degree8_row_capture_max"])

    # Exact cost and bias arithmetic.
    cost = receipt["cost"]
    budget = 2**41
    assert cost["budget"] == budget
    degree = cost["degree"]
    width = cost["width"]
    feature_per_path = (5 * degree + 1) * width
    assert feature_per_path == cost["feature_flops_per_path"] == 41984
    setup = 128 * width
    assert setup == cost["coefficient_and_supplied_moment_setup_flops"]
    path = 2**25
    assert path == cost["dense_path_flops"]
    assert (budget - setup) // (path + feature_per_path) == cost["free_oracle_path_ceiling"]
    paired = cost["measured_paired_64000_flops"]
    paired_augmented = paired + 64000 * feature_per_path + setup
    assert budget - paired_augmented == cost["measured_paired_64000_augmented_margin"]
    h10 = cost["existing_h10_provider_flops"]
    assert paired_augmented + h10 - budget == cost["h10_plus_measured_64000_over_budget"]
    historical = cost["historical_noeigh_static_host_flops"]
    historical_paths = (budget - historical - setup) // (path + feature_per_path)
    assert historical_paths == cost["historical_noeigh_path_ceiling"]
    illustrative = cost["illustrative_1p877t_host_flops"]
    illustrative_paths = (budget - illustrative - setup) // (path + feature_per_path)
    assert illustrative_paths == cost["illustrative_1p877t_path_ceiling"]

    mean_residual_eight = aggregate_residual[8] / units
    for paths_text, expected in receipt["bias_ceiling_from_native_d8"].items():
        paths = int(paths_text)
        sampling = mean_residual_eight / paths
        ceiling = math.sqrt(max(0.0, 1e-8 - sampling))
        close(sampling, expected["sampling_mse"])
        close(ceiling, expected["rms_combined_bias_ceiling"])

    # Same-sample cancellation and independent equal-cost split hard stop.
    split = receipt["same_cost_moment_estimation_hard_stop"]
    vr = float(split["example_residual_variance"])
    vs = float(split["example_control_variance"])
    plain = (vr + vs) / float(split["example_total_paths"])
    optimum = (math.sqrt(vr) + math.sqrt(vs)) ** 2 / float(split["example_total_paths"])
    assert optimum > plain
    close(optimum / plain, split["optimal_split_variance_ratio_to_plain"])

    # Bind the exact host semantics used by the moment-authority conclusion.
    host_source = (REPO / receipt["host_capability_audit"]["source_path"]).read_text(encoding="utf-8")
    for needle in receipt["host_capability_audit"]["required_source_needles"]:
        if needle not in host_source:
            raise AssertionError(f"host semantic needle missing: {needle}")
    for forbidden in receipt["host_capability_audit"]["absent_state_names"]:
        if forbidden in host_source:
            raise AssertionError(f"unexpected high-moment state appeared: {forbidden}")

    print("PASS p2r0f3c late-layer Hermite-CV audit")
    print(f"native_roster_sha256={roster_digest}")
    print(f"native_D8_capture={1.0 - aggregate_residual[8] / aggregate_total:.15f}")
    print("status=CLOSED_CURRENT_FORM__EXTERNAL_CERTIFIED_MOMENT_ORACLE_ONLY")


if __name__ == "__main__":
    main()
