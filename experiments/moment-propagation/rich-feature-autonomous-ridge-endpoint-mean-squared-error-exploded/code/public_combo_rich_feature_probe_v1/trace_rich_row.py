#!/usr/bin/env python3
"""Extract compact per-unit internal features from the exact combo estimator.

The hooks are read-only: each original estimator function is called first and
the returned state is summarized with ordinary NumPy for this offline public
development experiment.  No target is opened here.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import warnings

import numpy as np


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "fullbudget_k21_k4_closure_q24_manual_rho_hybrid_floor_v1/estimator.py"
ROWS = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit"
)
COMBO = P2 / "public_observer_fit_candidate_truth_score_v1/runs/manual_rho_hybrid_floor"
RUNS = HERE / "rich_rows"
FLOP_BUDGET = 2**41
WIDTH = 1024
DEPTH = 16

FEATURE_NAMES = (
    "pre_alpha",
    "pre_log_sigma",
    "pre_std_k3",
    "pre_std_k4",
    "post_std_k3",
    "post_std_k4",
    "k21_row_rms_std",
    "k21_col_rms_std",
    "k21_row_mean_std",
    "k31_row_rms_std",
    "k31_col_rms_std",
    "k22_row_rms_std",
    "k22_col_rms_std",
    "correlation_row_rms",
    "correlation_row_mean",
    "pairing_alpha",
    "cubic_star_row_rms_std",
    "quartic_star_row_rms_std",
    "repair_row_rms_corr",
    "repair_row_max_corr",
    "repair_near_floor_unit",
    "repair_exact_floor_unit",
    "repair_near_floor_count",
    "repair_exact_floor_count",
    "repair_log10_min_variance",
    "repair_min_eigenvalue",
    "repair_negative_eigen_count",
    "stieltjes_margin_a_relative",
    "stieltjes_margin_b_relative",
    "stieltjes_invalid_a",
    "stieltjes_invalid_b",
    "stieltjes_invalid_count",
    "cp3_p_row_rms",
    "cp3_q_row_rms",
    "cp3_rank_fraction",
    "cp4_u_row_rms",
    "cp4_diag_std",
    "cp4_rank_fraction",
    "gaussian_gate",
    "gaussian_birth_scaled",
)


def load_module(path: Path):
    tag = f"combo_rich_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def rms_incident(matrix, left_power, right_power, sigma):
    matrix = np.asarray(matrix, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-150)
    right_weight = sigma ** (-2.0 * right_power)
    left_weight = sigma ** (-2.0 * left_power)
    row = np.einsum("ij,j,ij->i", matrix, right_weight, matrix, optimize=False)
    col = np.einsum("ij,i,ij->j", matrix, left_weight, matrix, optimize=False)
    row = np.sqrt(np.maximum(row / WIDTH, 0.0)) / (sigma**left_power)
    col = np.sqrt(np.maximum(col / WIDTH, 0.0)) / (sigma**right_power)
    return row, col


def install_hooks(module):
    channels = {name: index for index, name in enumerate(FEATURE_NAMES)}
    rich = np.zeros((DEPTH, WIDTH, len(FEATURE_NAMES)), dtype=np.float32)
    calls = {"univariate": 0, "connected": 0, "repair": 0, "stars": 0, "cp3": 0, "cp4": 0}
    eig = []

    def put(layer, name, value):
        array = np.asarray(value, dtype=np.float64)
        # A nearly dead marginal can make a dimensionless diagnostic enormous.
        # Retain its sign/order signal without allowing an offline float32 tape
        # conversion to overflow; downstream fitting standardizes and clips.
        array = np.nan_to_num(array, nan=0.0, posinf=1.0e30, neginf=-1.0e30)
        array = np.clip(array, -1.0e30, 1.0e30)
        if array.ndim == 0:
            rich[layer, :, channels[name]] = np.float32(array)
        else:
            rich[layer, :, channels[name]] = np.asarray(array, dtype=np.float32)

    original_eigh = module.fnp.linalg.eigh

    def observed_eigh(*args, **kwargs):
        value = original_eigh(*args, **kwargs)
        eigenvalues = np.asarray(value[0], dtype=np.float64)
        eig.append((float(eigenvalues[0]), int(np.count_nonzero(eigenvalues < 0.0))))
        return value

    module.fnp.linalg.eigh = observed_eigh

    original_univariate = module._univariate_edgeworth

    def observed_univariate(*args, **kwargs):
        value = original_univariate(*args, **kwargs)
        layer = calls["univariate"] + 1
        calls["univariate"] += 1
        sigma = np.maximum(np.asarray(args[2], dtype=np.float64), 1e-150)
        alpha = np.asarray(args[3], dtype=np.float64)
        pdf = np.asarray(args[4], dtype=np.float64)
        k3 = np.asarray(args[5], dtype=np.float64)
        k4 = np.asarray(args[6], dtype=np.float64)
        raw1, raw2, raw3, raw4 = (np.asarray(item, dtype=np.float64) for item in value)
        variance = raw2 - raw1 * raw1
        a_left = raw1 * raw3
        a_right = raw2 * raw2
        margin_a = a_left - a_right
        centered21 = raw3 - raw1 * raw2
        b_left = (raw4 - a_right) * variance
        b_right = centered21 * centered21
        margin_b = b_left - b_right
        put(layer, "pre_alpha", alpha)
        put(layer, "pre_log_sigma", np.log(sigma))
        put(layer, "pre_std_k3", k3 / sigma**3)
        put(layer, "pre_std_k4", k4 / sigma**4)
        put(layer, "stieltjes_margin_a_relative", margin_a / np.maximum(np.abs(a_left) + np.abs(a_right), 1e-300))
        put(layer, "stieltjes_margin_b_relative", margin_b / np.maximum(np.abs(b_left) + np.abs(b_right), 1e-300))
        invalid_a = margin_a < 0.0
        invalid_b = margin_b < 0.0
        put(layer, "stieltjes_invalid_a", invalid_a)
        put(layer, "stieltjes_invalid_b", invalid_b)
        put(layer, "stieltjes_invalid_count", int(np.count_nonzero(invalid_a | invalid_b)))
        put(layer, "gaussian_gate", np.asarray(args[1][0], dtype=np.float64))
        put(layer, "gaussian_birth_scaled", pdf)
        return value

    module._univariate_edgeworth = observed_univariate

    original_connected = module._connected_planes

    def observed_connected(*args, **kwargs):
        value = original_connected(*args, **kwargs)
        layer = calls["connected"]
        calls["connected"] += 1
        covariance = np.asarray(args[1], dtype=np.float64)
        variance = np.maximum(np.asarray(args[9], dtype=np.float64), 1e-300)
        sigma = np.sqrt(variance)
        k21, k4, k31, k22 = value[:4]
        k21 = np.asarray(k21, dtype=np.float64)
        k4 = np.asarray(k4, dtype=np.float64)
        k31 = np.asarray(k31, dtype=np.float64)
        k22 = np.asarray(k22, dtype=np.float64)
        row, col = rms_incident(k21, 2, 1, sigma)
        put(layer, "post_std_k3", np.diag(k21) / sigma**3)
        put(layer, "post_std_k4", k4 / sigma**4)
        put(layer, "k21_row_rms_std", row)
        put(layer, "k21_col_rms_std", col)
        put(layer, "k21_row_mean_std", (k21 @ (1.0 / sigma)) / WIDTH / sigma**2)
        row, col = rms_incident(k31, 3, 1, sigma)
        put(layer, "k31_row_rms_std", row)
        put(layer, "k31_col_rms_std", col)
        row, col = rms_incident(k22, 2, 2, sigma)
        put(layer, "k22_row_rms_std", row)
        put(layer, "k22_col_rms_std", col)
        inv = 1.0 / sigma
        corr = covariance * inv[:, None] * inv[None, :]
        put(layer, "correlation_row_rms", np.sqrt(np.mean(corr * corr, axis=1)))
        put(layer, "correlation_row_mean", np.mean(corr, axis=1))
        return value

    module._connected_planes = observed_connected

    original_repair = module._repair_covariance

    def observed_repair(*args, **kwargs):
        layer = calls["repair"] + 1
        calls["repair"] += 1
        raw = np.asarray(args[0], dtype=np.float64).copy()
        variance = np.asarray(args[1], dtype=np.float64).copy()
        np.fill_diagonal(raw, variance)
        before_eig = len(eig)
        value = original_repair(*args, **kwargs)
        repaired = np.asarray(value, dtype=np.float64)
        sigma = np.sqrt(np.maximum(variance, 1e-300))
        scale = sigma[:, None] * sigma[None, :]
        delta = (repaired - raw) / np.maximum(scale, 1e-300)
        put(layer, "repair_row_rms_corr", np.sqrt(np.mean(delta * delta, axis=1)))
        put(layer, "repair_row_max_corr", np.max(np.abs(delta), axis=1))
        put(layer, "repair_near_floor_unit", variance <= 2.0 * module._VAR_FLOOR)
        put(layer, "repair_exact_floor_unit", variance == module._VAR_FLOOR)
        put(layer, "repair_near_floor_count", int(np.count_nonzero(variance <= 2.0 * module._VAR_FLOOR)))
        put(layer, "repair_exact_floor_count", int(np.count_nonzero(variance == module._VAR_FLOOR)))
        put(layer, "repair_log10_min_variance", np.log10(max(float(np.min(variance)), 1e-300)))
        if len(eig) != before_eig + 1:
            raise AssertionError("repair did not issue exactly one eigendecomposition")
        put(layer, "repair_min_eigenvalue", eig[-1][0])
        put(layer, "repair_negative_eigen_count", eig[-1][1])
        return value

    module._repair_covariance = observed_repair

    original_stars = module._reset_stars

    def observed_stars(*args, **kwargs):
        value = original_stars(*args, **kwargs)
        layer = calls["stars"]
        calls["stars"] += 1
        variance = np.maximum(np.asarray(args[6], dtype=np.float64), 1e-300)
        sigma = np.sqrt(variance)
        cubic = np.asarray(value[0], dtype=np.float64)
        quartic = np.asarray(value[1], dtype=np.float64)
        row, _ = rms_incident(cubic, 2, 1, sigma)
        qrow, _ = rms_incident(quartic, 3, 1, sigma)
        put(layer, "pairing_alpha", value[2])
        put(layer, "cubic_star_row_rms_std", row)
        put(layer, "quartic_star_row_rms_std", qrow)
        return value

    module._reset_stars = observed_stars

    original_cp3 = module._cp_grow

    def observed_cp3(*args, **kwargs):
        value = original_cp3(*args, **kwargs)
        layer = calls["cp3"]
        calls["cp3"] += 1
        rank = int(value[0].shape[0])
        if rank:
            p = np.asarray(value[1], dtype=np.float32)
            q = np.asarray(value[2], dtype=np.float32)
            put(layer, "cp3_p_row_rms", np.sqrt(np.einsum("ir,ir->i", p, p, optimize=False) / rank))
            put(layer, "cp3_q_row_rms", np.sqrt(np.einsum("ir,ir->i", q, q, optimize=False) / rank))
        put(layer, "cp3_rank_fraction", rank / module._CP3_CAP)
        return value

    module._cp_grow = observed_cp3

    original_cp4 = module._cp4_reset

    def observed_cp4(*args, **kwargs):
        value = original_cp4(*args, **kwargs)
        layer = calls["cp4"]
        calls["cp4"] += 1
        rank = int(value[0].shape[0])
        if rank:
            lam = np.asarray(value[0], dtype=np.float64)
            u = np.asarray(value[1], dtype=np.float32)
            put(layer, "cp4_u_row_rms", np.sqrt(np.einsum("ir,ir->i", u, u, optimize=False) / rank))
            diag = np.einsum("ir,ir,ir,ir,r->i", u, u, u, u, lam, optimize=False)
            target = np.asarray(args[4], dtype=np.float64)
            scale = np.maximum(np.sqrt(np.maximum(np.abs(target), 1e-300)) ** 4, 1e-300)
            put(layer, "cp4_diag_std", diag / scale)
        put(layer, "cp4_rank_fraction", rank / module._CP4_CAP)
        return value

    module._cp4_reset = observed_cp4
    return rich, calls


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-rank", required=True, type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    row_dir = ROWS / f"{args.role_rank:04d}"
    output_dir = RUNS / f"row_{args.role_rank:04d}"
    if (output_dir / "rich_features.npy").is_file() and not args.overwrite:
        print(json.dumps({"rank": args.role_rank, "status": "EXISTS"}), flush=True)
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((row_dir / "row_manifest.json").read_text())
    weights = np.load(row_dir / "weights.npy", mmap_mode="r", allow_pickle=False)

    import flopscope as flops
    from whestbench.domain import MLP

    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=[weights[layer] for layer in range(DEPTH)],
        seed=int(manifest["mlp_seed"]),
        name=f"combo-rich-{args.role_rank}",
    )
    module = load_module(SOURCE)
    rich, calls = install_hooks(module)
    started = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as budget:
            output = np.asarray(module.Estimator().predict(mlp, FLOP_BUDGET)).copy()
    wall = time.perf_counter() - started
    expected = np.load(COMBO / f"row_{args.role_rank:04d}/q24_output.npy", allow_pickle=False)
    if not np.array_equal(output, expected):
        raise AssertionError("hooks changed exact combo output bytes")
    if calls != {"univariate": 15, "connected": 15, "repair": 14, "stars": 15, "cp3": 15, "cp4": 15}:
        raise AssertionError(f"unexpected hook calls: {calls}")
    if not np.isfinite(rich).all():
        raise FloatingPointError("rich feature tape is nonfinite")
    np.save(output_dir / "rich_features.npy", rich, allow_pickle=False)
    result = {
        "role": "observer_fit",
        "role_rank": args.role_rank,
        "mlp_id": int(manifest["mlp_id"]),
        "feature_names": list(FEATURE_NAMES),
        "shape": list(rich.shape),
        "dtype": str(rich.dtype),
        "estimator_sha256": "5e0fccfe8569b43aa85f77997c1dd890c5dd1173f114167e6fb50f7e1d85e5b5",
        "output_byte_parity": True,
        "flops_used": int(budget.flops_used),
        "wall_seconds": wall,
        "calls": calls,
    }
    (output_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"rank": args.role_rank, "status": "OK", "wall": wall}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
