#!/usr/bin/env python3
"""Run one frozen Q24 CP3 fixed-old-summary arm at physical rank 2048."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
AGE_DIR = P2 / "public_q24_cp3_age_retirement_discriminator_v1"
AGE_RUNNER = AGE_DIR / "run_arm.py"
RUNS = HERE / "runs"
WIDTH = 1024
DEPTH = 16
COHORT = 1024
CAP = 2048
OLD_QUOTA = CAP - COHORT
FULL_CAP = 16384
FLOP_BUDGET = 2**41
THREADS = 2


def load_age_helpers():
    spec = importlib.util.spec_from_file_location("q24_cp3_age_helpers", AGE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(AGE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_selector(module, selector: str) -> dict[str, object]:
    """Patch the CP3 seam, retaining recent birth plus a selected old summary."""
    original_read = module._cp_read
    original_post = module._cp_post_read
    original_grow = module._cp_grow
    original_native_features = module._native_raw_features
    state: dict[str, object] = {
        "pending": None,
        "birth_labels": np.zeros(0, dtype=np.int16),
        "birth_step": 0,
        "selection_events": 0,
        "selected_old_columns": 0,
        "dropped_old_columns": 0,
        "entering_ranks": [],
        "old_energy_capture": [],
        "selected_birth_histograms": [],
        "first_split_parity_checks": 0,
        "first_split_parity_max_abs": 0.0,
        "first_split_parity_rms": 0.0,
    }

    def read(cp_lambda, cp_p, cp_q, product_a, product_b, arena_views):
        rank = int(cp_lambda.shape[0])
        state["entering_ranks"].append(rank)
        total = original_read(
            cp_lambda, cp_p, cp_q, product_a, product_b, arena_views
        )
        if rank < CAP:
            state["pending"] = None
        else:
            labels = np.asarray(state["birth_labels"], dtype=np.int16)
            if labels.shape != (rank,):
                raise AssertionError((labels.shape, rank))
            state["pending"] = {
                "cp_lambda": cp_lambda,
                "pushed_p": cp_p,
                "pushed_q": cp_q,
                "full_pre": module.fnp.asarray(total + 0.0, dtype=module.fnp.float64),
                "labels": labels,
                "arena": (product_a, product_b, arena_views),
            }
        return total

    def post(pre_k21, gate, pre_covariance, pre_variance, birth_lambda):
        pending = state["pending"]
        if pending is None:
            return original_post(
                pre_k21, gate, pre_covariance, pre_variance, birth_lambda
            )
        cp_lambda = pending["cp_lambda"]
        pushed_p = pending["pushed_p"]
        pushed_q = pending["pushed_q"]
        rank = int(cp_lambda.shape[0])
        if rank != CAP:
            raise AssertionError((selector, rank, CAP))

        gate32 = module.fnp.asarray(gate, dtype=module.fnp.float32)
        inherited_p = gate32[:, None] * pushed_p
        inherited_q = gate32[:, None] * pushed_q
        norm_p2 = module.fnp.sum(inherited_p * inherited_p, axis=0)
        norm_q2 = module.fnp.sum(inherited_q * inherited_q, axis=0)
        dot_pq = module.fnp.sum(inherited_p * inherited_q, axis=0)
        lam2 = cp_lambda * cp_lambda
        energy = lam2 * (
            3.0 * norm_p2 * norm_p2 * norm_q2
            + 6.0 * norm_p2 * dot_pq * dot_pq
        )
        if selector == "prefix2":
            keep = module.fnp.arange(OLD_QUOTA, dtype=module.fnp.int64)
        elif selector == "energy2":
            keep = module.fnp.argsort(-energy, stable=True)[:OLD_QUOTA]
        else:
            raise AssertionError(selector)
        keep_np = np.asarray(keep, dtype=np.int64)
        drop_mask = np.ones(rank, dtype=bool)
        drop_mask[keep_np] = False
        drop = np.flatnonzero(drop_mask)

        product_a, product_b, arena_views = pending["arena"]
        kept_pre = original_read(
            cp_lambda[keep],
            pushed_p[:, keep],
            pushed_q[:, keep],
            product_a,
            product_b,
            arena_views,
        )
        kept_pre = module.fnp.asarray(kept_pre + 0.0, dtype=module.fnp.float64)
        if int(state["first_split_parity_checks"]) == 0:
            dropped_pre = original_read(
                cp_lambda[drop],
                pushed_p[:, drop],
                pushed_q[:, drop],
                product_a,
                product_b,
                arena_views,
            )
            delta = (
                np.asarray(kept_pre, dtype=np.float64)
                + np.asarray(dropped_pre, dtype=np.float64)
                - np.asarray(pending["full_pre"], dtype=np.float64)
            )
            state["first_split_parity_checks"] = 1
            state["first_split_parity_max_abs"] = float(np.max(np.abs(delta)))
            state["first_split_parity_rms"] = float(np.sqrt(np.mean(delta * delta)))

        energy64 = np.asarray(energy, dtype=np.float64)
        total_energy = float(np.sum(energy64))
        captured = float(np.sum(energy64[keep_np]))
        state["old_energy_capture"].append(
            1.0 if total_energy <= 0.0 else captured / total_energy
        )
        labels = np.asarray(pending["labels"], dtype=np.int16)[keep_np]
        unique, counts = np.unique(labels, return_counts=True)
        state["selected_birth_histograms"].append(
            {str(int(key)): int(value) for key, value in zip(unique, counts)}
        )
        state["selection_events"] += 1
        state["selected_old_columns"] += int(keep_np.size)
        state["dropped_old_columns"] += int(drop.size)
        pending.update(
            {
                "keep_lambda": cp_lambda[keep],
                "keep_p": pushed_p[:, keep],
                "keep_q": pushed_q[:, keep],
                "keep_labels": labels,
                "kept_pre": kept_pre,
            }
        )
        return original_post(
            kept_pre, gate, pre_covariance, pre_variance, birth_lambda
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
        birth_step = int(state["birth_step"])
        if pending is None:
            result = original_grow(
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
            previous = np.asarray(state["birth_labels"], dtype=np.int16)
            born = int(result[0].shape[0]) - int(previous.size)
            state["birth_labels"] = np.concatenate(
                (previous, np.full(born, birth_step, dtype=np.int16))
            )
            state["birth_step"] = birth_step + 1
            return result

        keep_lambda = pending["keep_lambda"]
        keep_p = pending["keep_p"]
        keep_q = pending["keep_q"]
        covariance32 = module.fnp.asarray(pre_covariance, dtype=module.fnp.float32)
        birth_lambda32 = module.fnp.asarray(birth_lambda, dtype=module.fnp.float32)
        inherited_p = gate32[:, None] * keep_p
        inherited_q = gate32[:, None] * keep_q
        birth_p = gate32[:, None] * covariance32
        birth_q = module.fnp.asarray(identity, dtype=module.fnp.float32)
        new_lambda = module.fnp.concatenate((keep_lambda, birth_lambda32))
        new_p = module.fnp.concatenate((inherited_p, birth_p), axis=1)
        new_q = module.fnp.concatenate((inherited_q, birth_q), axis=1)
        if int(new_lambda.shape[0]) != CAP:
            raise AssertionError((selector, new_lambda.shape))
        keep_labels = np.asarray(pending["keep_labels"], dtype=np.int16)
        state["birth_labels"] = np.concatenate(
            (keep_labels, np.full(COHORT, birth_step, dtype=np.int16))
        )
        state["birth_step"] = birth_step + 1
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=("prefix2", "energy2"), required=True)
    args = parser.parse_args()
    arm = str(args.arm)
    RUNS.mkdir(parents=True, exist_ok=True)
    output_npz = RUNS / f"rank0000_{arm}.npz"
    output_json = RUNS / f"rank0000_{arm}.json"
    if output_npz.exists() or output_json.exists():
        raise FileExistsError((output_npz, output_json))

    age = load_age_helpers()
    manifest = json.loads((age.FEATURE_ROW / "row_manifest.json").read_text())
    weights_path = age.FEATURE_ROW / "weights.npy"
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    with np.load(age.DIAGNOSTIC, allow_pickle=False) as diagnostic:
        target = np.ascontiguousarray(diagnostic["target"], dtype=np.float32)
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=[weights[layer] for layer in range(DEPTH)],
        seed=int(derive_estimator_seed(int(manifest["mlp_seed"]))),
        name=f"q24-cp3-fixed-summary-rank0-{arm}",
    )
    mlp.validate()
    module = age.load_module(arm)
    selector_state = install_selector(module, arm)

    started = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        with flops.BudgetContext(
            flop_budget=FLOP_BUDGET,
            wall_time_limit_s=120.0,
            quiet=True,
            namespace=f"q24_cp3_fixed_summary_{arm}",
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
    q_metrics, q_arrays = age.local_q_metrics(
        target, host, correction, raw, np.asarray(weights)
    )
    host_error = target.astype(np.float64) - host.astype(np.float64)
    corrected_error = target.astype(np.float64) - corrected.astype(np.float64)
    summary = context.summary_dict(by_namespace=False)
    state_receipt = {
        key: value
        for key, value in selector_state.items()
        if key not in ("pending", "birth_labels")
    }
    state_receipt["final_birth_histogram"] = {
        str(int(key)): int(value)
        for key, value in zip(
            *np.unique(np.asarray(selector_state["birth_labels"]), return_counts=True)
        )
    }
    receipt = {
        "schema": "q24-native-cp3-fixed-summary-rank0-arm-v1",
        "status": "COMPLETE",
        "role": "observer_fit",
        "role_rank": 0,
        "mlp_id": int(manifest["mlp_id"]),
        "arm": arm,
        "physical_cap_columns": CAP,
        "forced_recent_birth_columns": COHORT,
        "old_summary_columns": OLD_QUOTA,
        "metrics": {
            "host_final_mse": age.mse(host_error[-1]),
            "native_j_final_mse": age.mse(corrected_error[-1]),
            "host_layer_mse": [age.mse(row) for row in host_error],
            "native_j_layer_mse": [age.mse(row) for row in corrected_error],
            "local_q": q_metrics,
        },
        "selector_state": state_receipt,
        "prototype_resources": {
            "literal_flops": int(summary["flops_used"]),
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": age.resident_bytes(),
            "note": "Prototype includes an extra selected-subset read; optimized production bill is separate.",
        },
        "bindings": {
            "source": str(age.SOURCE.relative_to(P2)),
            "source_sha256": age.sha256(age.SOURCE),
            "age_helper_sha256": age.sha256(AGE_RUNNER),
            "runner_sha256": age.sha256(Path(__file__)),
            "weights_sha256": age.sha256(weights_path),
            "diagnostic_sha256": age.sha256(age.DIAGNOSTIC),
        },
        "access": {
            "observer_fit_target_artifacts_opened": [str(age.DIAGNOSTIC.relative_to(P2))],
            "observer_selection": False,
            "network": False,
            "training": False,
            "fitting": False,
            "submission": False,
        },
        "array_hashes": {
            "host": age.array_sha256(host),
            "native_raw": age.array_sha256(raw),
            "native_sigma": age.array_sha256(sigma),
            "native_j_correction": age.array_sha256(correction),
            "native_j_output": age.array_sha256(corrected),
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
    receipt["artifact_sha256"] = age.sha256(output_npz)
    output_json.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    del module, host_value, raw_value, sigma_value, correction_value, corrected_value
    gc.collect()


if __name__ == "__main__":
    main()
