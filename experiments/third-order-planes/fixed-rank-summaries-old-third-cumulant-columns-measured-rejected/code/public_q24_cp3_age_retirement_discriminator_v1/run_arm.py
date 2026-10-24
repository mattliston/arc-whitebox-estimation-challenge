#!/usr/bin/env python3
"""Run one frozen Q24 CP3 chronological age-retirement arm.

The prototype deliberately uses fresh arrays at a retirement boundary.  It is
an accuracy discriminator, not the final low-allocation production spelling.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import sys
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = P2 / "p2r0c1d9_cp3_arena_depth1_full_estimator_v1/estimator.py"
FEATURE_ROW = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3/rows/observer_fit/0000"
)
DIAGNOSTIC = (
    P2
    / "public_phase1_kerdock_hidden_snap_v1/runs/row_0000"
    / "DIAGNOSTIC_ARRAYS_BASE.npz"
)
RUNS = HERE / "runs"
WIDTH = 1024
DEPTH = 16
COHORT = 1024
FULL_CAP = 16384
FLOP_BUDGET = 2**41
THREADS = 2


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


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def load_module(arm: str):
    name = f"q24_cp3_age_{arm}_{os.getpid()}"
    spec = importlib.util.spec_from_file_location(name, SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def install_age_retirement(module, window: int) -> dict[str, object]:
    """Patch only the CP read/post/grow seam for chronological retirement."""
    original_read = module._cp_read
    original_post = module._cp_post_read
    original_grow = module._cp_grow
    original_native_features = module._native_raw_features
    state: dict[str, object] = {
        "pending": None,
        "retirement_events": 0,
        "retired_columns": 0,
        "read_rank_sum": 0,
        "retire_rank_sum": 0,
        "keep_rank_sum": 0,
        "entering_ranks": [],
        "current_reset_parity_checks": 0,
        "current_reset_parity_max_abs": 0.0,
        "current_reset_parity_rms": 0.0,
    }
    cap = int(window) * COHORT

    def read(cp_lambda, cp_p, cp_q, product_a, product_b, arena_views):
        rank = int(cp_lambda.shape[0])
        state["entering_ranks"].append(rank)
        state["read_rank_sum"] += rank
        if rank < cap:
            state["pending"] = None
            return original_read(
                cp_lambda, cp_p, cp_q, product_a, product_b, arena_views
            )

        retire = rank - (cap - COHORT)
        if retire <= 0 or retire % COHORT != 0:
            raise AssertionError((window, rank, retire))
        keep_lambda = cp_lambda[retire:]
        keep_p = cp_p[:, retire:]
        keep_q = cp_q[:, retire:]
        retire_lambda = cp_lambda[:retire]
        retire_p = cp_p[:, :retire]
        retire_q = cp_q[:, :retire]

        retired_pre = original_read(
            retire_lambda,
            retire_p,
            retire_q,
            product_a,
            product_b,
            arena_views,
        )
        # The first read reuses the arena outputs, so materialize it before the
        # retained suffix overwrites the same fixed scratch views.
        retired_pre = module.fnp.asarray(retired_pre + 0.0, dtype=module.fnp.float64)
        kept_pre = original_read(
            keep_lambda,
            keep_p,
            keep_q,
            product_a,
            product_b,
            arena_views,
        )
        kept_pre = module.fnp.asarray(kept_pre + 0.0, dtype=module.fnp.float64)
        split_total = module.fnp.asarray(
            retired_pre + kept_pre, dtype=module.fnp.float64
        )
        if int(state["current_reset_parity_checks"]) == 0:
            unsplit = original_read(
                cp_lambda, cp_p, cp_q, product_a, product_b, arena_views
            )
            delta = np.asarray(split_total, dtype=np.float64) - np.asarray(
                unsplit, dtype=np.float64
            )
            state["current_reset_parity_checks"] = 1
            state["current_reset_parity_max_abs"] = float(np.max(np.abs(delta)))
            state["current_reset_parity_rms"] = float(np.sqrt(np.mean(delta * delta)))
        state["pending"] = (keep_lambda, keep_p, keep_q, kept_pre)
        state["retirement_events"] += 1
        state["retired_columns"] += retire
        state["retire_rank_sum"] += retire
        state["keep_rank_sum"] += int(keep_lambda.shape[0])
        return split_total

    def post(pre_k21, gate, pre_covariance, pre_variance, birth_lambda):
        pending = state["pending"]
        if pending is None:
            return original_post(
                pre_k21, gate, pre_covariance, pre_variance, birth_lambda
            )
        return original_post(
            pending[3], gate, pre_covariance, pre_variance, birth_lambda
        )

    def grow(
        cp_lambda,
        pushed_p,
        pushed_q,
        gate32,
        pre_covariance,
        birth_lambda,
        identity,
        active_p_bank,
        active_q_bank,
        lambda_bank,
    ):
        pending = state["pending"]
        if pending is None:
            return original_grow(
                cp_lambda,
                pushed_p,
                pushed_q,
                gate32,
                pre_covariance,
                birth_lambda,
                identity,
                active_p_bank,
                active_q_bank,
                lambda_bank,
            )

        keep_lambda, keep_p, keep_q, _kept_pre = pending
        covariance32 = module.fnp.asarray(pre_covariance, dtype=module.fnp.float32)
        birth_lambda32 = module.fnp.asarray(birth_lambda, dtype=module.fnp.float32)
        inherited_p = gate32[:, None] * keep_p
        inherited_q = gate32[:, None] * keep_q
        birth_p = gate32[:, None] * covariance32
        birth_q = module.fnp.asarray(identity, dtype=module.fnp.float32)
        new_lambda = module.fnp.concatenate((keep_lambda, birth_lambda32))
        new_p = module.fnp.concatenate((inherited_p, birth_p), axis=1)
        new_q = module.fnp.concatenate((inherited_q, birth_q), axis=1)
        if int(new_lambda.shape[0]) != cap:
            raise AssertionError((window, new_lambda.shape))
        state["pending"] = None
        return new_lambda, new_p, new_q

    module._cp_read = read
    module._cp_post_read = post
    module._cp_grow = grow

    def native_features(
        layer0,
        width,
        cp_rank,
        coefficients,
        gaussian_raw,
        sigma,
        alpha,
        pdf,
        pre_covariance,
        pre_variance,
        k3,
        reported,
        edge_summaries,
    ):
        # Preserve the frozen checkpoint's logical full-birth feature.  The
        # smaller physical rank is separately receipted by the read wrapper.
        logical_rank = min(int(layer0) * COHORT, FULL_CAP - COHORT)
        return original_native_features(
            layer0,
            width,
            logical_rank,
            coefficients,
            gaussian_raw,
            sigma,
            alpha,
            pdf,
            pre_covariance,
            pre_variance,
            k3,
            reported,
            edge_summaries,
        )

    module._native_raw_features = native_features
    return state


def mse(value: np.ndarray) -> float:
    return float(np.mean(np.asarray(value, dtype=np.float64) ** 2))


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).reshape(-1)
    b = np.asarray(right, dtype=np.float64).reshape(-1)
    a -= a.mean()
    b -= b.mean()
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def local_q_metrics(
    target: np.ndarray,
    host: np.ndarray,
    correction: np.ndarray,
    raw: np.ndarray,
    weights: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    phi = np.asarray(raw[:, :, 5], dtype=np.float32)
    weights32 = np.asarray(weights, dtype=np.float32)
    truth_error = np.asarray(target - host, dtype=np.float32)
    autonomous = np.asarray(correction, dtype=np.float32)
    q_truth = np.zeros_like(truth_error)
    q_bank = np.zeros_like(autonomous)
    q_truth[2] = truth_error[2]
    q_bank[2] = autonomous[2]
    for layer in range(3, DEPTH):
        q_truth[layer] = truth_error[layer] - phi[layer] * (
            truth_error[layer - 1] @ weights32[layer]
        )
        q_bank[layer] = autonomous[layer] - phi[layer] * (
            autonomous[layer - 1] @ weights32[layer]
        )

    state_bank = autonomous[9].copy()
    state_truth = autonomous[9].copy()
    for layer in range(10, DEPTH):
        state_bank = phi[layer] * (state_bank @ weights32[layer]) + q_bank[layer]
        state_truth = phi[layer] * (state_truth @ weights32[layer]) + q_truth[layer]

    layers = []
    for layer in range(10, DEPTH):
        gap = q_truth[layer] - q_bank[layer]
        layers.append(
            {
                "layer_zero_based": layer,
                "q_truth_mse": mse(q_truth[layer]),
                "q_bank_mse": mse(q_bank[layer]),
                "q_gap_mse": mse(gap),
                "q_bank_truth_correlation": correlation(q_bank[layer], q_truth[layer]),
            }
        )
    metrics = {
        "layers10_15": layers,
        "bank_replay_max_abs": float(
            np.max(np.abs(state_bank.astype(np.float64) - autonomous[-1]))
        ),
        "truth_suffix_oracle_final_mse": mse(truth_error[-1] - state_truth),
        "q10_15_gap_mean_mse": float(
            np.mean([entry["q_gap_mse"] for entry in layers])
        ),
    }
    return metrics, {"q_truth": q_truth, "q_bank": q_bank}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("full", "w1", "w2", "w3", "w4", "w6"), required=True)
    args = parser.parse_args()
    arm = str(args.arm)
    window = None if arm == "full" else int(arm[1:])
    RUNS.mkdir(parents=True, exist_ok=True)
    output_npz = RUNS / f"rank0000_{arm}.npz"
    output_json = RUNS / f"rank0000_{arm}.json"
    if output_npz.exists() or output_json.exists():
        raise FileExistsError((output_npz, output_json))

    manifest = json.loads((FEATURE_ROW / "row_manifest.json").read_text())
    weights_path = FEATURE_ROW / "weights.npy"
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.float32:
        raise AssertionError((weights.shape, weights.dtype))
    with np.load(DIAGNOSTIC, allow_pickle=False) as diagnostic:
        target = np.ascontiguousarray(diagnostic["target"], dtype=np.float32)
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=[weights[layer] for layer in range(DEPTH)],
        seed=int(derive_estimator_seed(int(manifest["mlp_seed"]))),
        name=f"q24-cp3-age-retirement-rank0-{arm}",
    )
    mlp.validate()
    module = load_module(arm)
    retirement_state = None
    if window is not None:
        retirement_state = install_age_retirement(module, window)

    started = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        with flops.BudgetContext(
            flop_budget=FLOP_BUDGET,
            wall_time_limit_s=120.0,
            quiet=True,
            namespace=f"q24_cp3_age_{arm}",
        ) as context:
            host_value, raw_value, sigma_value = module._predict(
                mlp, return_native_tape=True
            )
            correction_value = module.transport_correction(
                raw_value, sigma_value, mlp.weights, module._TRANSPORT_PAYLOAD
            )
            corrected_value = module.fnp.asarray(
                module.fnp.maximum(host_value + correction_value, 0.0),
                dtype=module.fnp.float32,
            )
    wall_seconds = time.perf_counter() - started
    host = np.ascontiguousarray(np.asarray(host_value), dtype=np.float32)
    raw = np.ascontiguousarray(np.asarray(raw_value), dtype=np.float32)
    sigma = np.ascontiguousarray(np.asarray(sigma_value), dtype=np.float32)
    correction = np.ascontiguousarray(np.asarray(correction_value), dtype=np.float32)
    corrected = np.ascontiguousarray(np.asarray(corrected_value), dtype=np.float32)
    q_metrics, q_arrays = local_q_metrics(
        target, host, correction, raw, np.asarray(weights)
    )
    host_error = target.astype(np.float64) - host.astype(np.float64)
    corrected_error = target.astype(np.float64) - corrected.astype(np.float64)
    summary = context.summary_dict(by_namespace=False)
    metrics = {
        "host_final_mse": mse(host_error[-1]),
        "native_j_final_mse": mse(corrected_error[-1]),
        "host_layer_mse": [mse(row) for row in host_error],
        "native_j_layer_mse": [mse(row) for row in corrected_error],
        "local_q": q_metrics,
    }
    state_receipt = {
        key: value
        for key, value in (retirement_state or {}).items()
        if key != "pending"
    }
    receipt = {
        "schema": "q24-native-cp3-age-retirement-rank0-arm-v1",
        "status": "COMPLETE",
        "role": "observer_fit",
        "role_rank": 0,
        "mlp_id": int(manifest["mlp_id"]),
        "arm": arm,
        "window_cohorts": window,
        "window_columns": None if window is None else window * COHORT,
        "mechanism": (
            "full D1 reference"
            if window is None
            else "read full carried CP at current reset; retire oldest cohort after reset; retained post plus exact birth defines cubic-star residual"
        ),
        "metrics": metrics,
        "prototype_resources": {
            "literal_flops": int(summary["flops_used"]),
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": resident_bytes(),
            "note": "Prototype split-read/fresh-array bill, not optimized production bill.",
        },
        "retirement_state": state_receipt,
        "bindings": {
            "source": str(SOURCE.relative_to(P2)),
            "source_sha256": sha256(SOURCE),
            "weights_sha256": sha256(weights_path),
            "diagnostic_sha256": sha256(DIAGNOSTIC),
            "runner_sha256": sha256(Path(__file__)),
        },
        "access": {
            "observer_fit_target_artifacts_opened": [str(DIAGNOSTIC.relative_to(P2))],
            "observer_selection": False,
            "network": False,
            "training": False,
            "fitting": False,
            "submission": False,
        },
        "array_hashes": {
            "host": array_sha256(host),
            "native_raw": array_sha256(raw),
            "native_sigma": array_sha256(sigma),
            "native_j_correction": array_sha256(correction),
            "native_j_output": array_sha256(corrected),
        },
    }
    np.savez_compressed(
        output_npz,
        host=host,
        native_raw=raw,
        native_sigma=sigma,
        native_j_correction=correction,
        native_j_output=corrected,
        q_truth=q_arrays["q_truth"],
        q_bank=q_arrays["q_bank"],
    )
    receipt["artifact"] = output_npz.name
    receipt["artifact_sha256"] = sha256(output_npz)
    output_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    del module, host_value, raw_value, sigma_value, correction_value, corrected_value
    gc.collect()


if __name__ == "__main__":
    main()
