#!/usr/bin/env python3
"""Run the target-blind R=16 suffix-action CP3 summary arm at rank 2048."""

from __future__ import annotations

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
BASE_RUNNER = HERE / "run_arm.py"
RUNS = HERE / "runs"
WIDTH = 1024
DEPTH = 16
COHORT = 1024
CAP = 2048
OLD_QUOTA = 1024
FULL_CAP = 16384
PROBES = 16
PROBE_SEED = 20260826
FLOP_BUDGET = 2**41
THREADS = 2


def load_base():
    spec = importlib.util.spec_from_file_location("q24_cp3_fixed_summary_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(BASE_RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install_action_selector(module, weights: np.ndarray) -> dict[str, object]:
    original_read = module._cp_read
    original_post = module._cp_post_read
    original_grow = module._cp_grow
    original_reset = module._full_relu_reset
    original_native_features = module._native_raw_features
    rng = np.random.default_rng(PROBE_SEED)
    endpoint_probes = rng.integers(0, 2, size=(WIDTH, PROBES), dtype=np.int8)
    endpoint_probes = np.asarray(2 * endpoint_probes - 1, dtype=np.float32)
    state: dict[str, object] = {
        "pending": None,
        "birth_labels": np.zeros(0, dtype=np.int16),
        "birth_step": 0,
        "live_rho3": None,
        "suffix_adjoints": None,
        "selection_events": 0,
        "selected_old_columns": 0,
        "dropped_old_columns": 0,
        "entering_ranks": [],
        "action_score_capture": [],
        "energy_score_capture": [],
        "action_energy_topk_overlap": [],
        "selected_birth_histograms": [],
        "first_split_parity_checks": 0,
        "first_split_parity_max_abs": 0.0,
        "first_split_parity_rms": 0.0,
    }

    def reset(*args, **kwargs):
        result = original_reset(*args, **kwargs)
        pre_k21 = args[2]
        coefficients = args[6]
        sigma = args[8]
        k3 = module.fnp.diag(pre_k21)
        inv_sigma = 1.0 / sigma
        inv_sigma3 = inv_sigma * inv_sigma * inv_sigma
        inv_sigma6 = inv_sigma3 * inv_sigma3
        c1 = coefficients[0]
        state["live_rho3"] = module.fnp.asarray(
            c1[3] * inv_sigma3 / 6.0
            + c1[6] * k3 * inv_sigma6 / 36.0,
            dtype=module.fnp.float32,
        )
        return result

    def suffix_adjoints():
        cached = state["suffix_adjoints"]
        if cached is not None:
            return cached
        # A uniform expected ReLU gate is a layerwise scalar, hence it cannot
        # change within-layer ranking and is omitted to avoid underflow.
        adjoints = [None] * DEPTH
        adjoints[-1] = module.fnp.asarray(endpoint_probes, dtype=module.fnp.float32)
        for layer in range(DEPTH - 2, -1, -1):
            weight = module.fnp.asarray(weights[layer + 1], dtype=module.fnp.float32)
            adjoints[layer] = weight @ adjoints[layer + 1]
        state["suffix_adjoints"] = adjoints
        return adjoints

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
        rho3 = state["live_rho3"]
        if rho3 is None:
            raise AssertionError("full reset did not expose live rho3")
        layer = int(state["birth_step"])
        cp_lambda = pending["cp_lambda"]
        pushed_p = pending["pushed_p"]
        pushed_q = pending["pushed_q"]
        rank = int(cp_lambda.shape[0])
        if rank != CAP or not (2 <= layer < DEPTH - 1):
            raise AssertionError((rank, layer))

        marginal = (
            3.0
            * pushed_p
            * pushed_p
            * pushed_q
            * cp_lambda[None, :]
        )
        weighted = rho3[:, None] * marginal
        projected = module.fnp.transpose(suffix_adjoints()[layer]) @ weighted
        action_score = module.fnp.sum(projected * projected, axis=0) / float(PROBES)
        action_order = module.fnp.argsort(-action_score, stable=True)
        keep = action_order[:OLD_QUOTA]
        keep_np = np.asarray(keep, dtype=np.int64)
        drop_mask = np.ones(rank, dtype=bool)
        drop_mask[keep_np] = False
        drop = np.flatnonzero(drop_mask)

        gate32 = module.fnp.asarray(gate, dtype=module.fnp.float32)
        inherited_p = gate32[:, None] * pushed_p
        inherited_q = gate32[:, None] * pushed_q
        norm_p2 = module.fnp.sum(inherited_p * inherited_p, axis=0)
        norm_q2 = module.fnp.sum(inherited_q * inherited_q, axis=0)
        dot_pq = module.fnp.sum(inherited_p * inherited_q, axis=0)
        lam2 = cp_lambda * cp_lambda
        energy_score = lam2 * (
            3.0 * norm_p2 * norm_p2 * norm_q2
            + 6.0 * norm_p2 * dot_pq * dot_pq
        )
        energy_order = np.asarray(
            module.fnp.argsort(-energy_score, stable=True)[:OLD_QUOTA],
            dtype=np.int64,
        )

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

        action64 = np.asarray(action_score, dtype=np.float64)
        energy64 = np.asarray(energy_score, dtype=np.float64)
        action_total = float(np.sum(action64))
        energy_total = float(np.sum(energy64))
        state["action_score_capture"].append(
            1.0 if action_total <= 0.0 else float(np.sum(action64[keep_np])) / action_total
        )
        state["energy_score_capture"].append(
            1.0 if energy_total <= 0.0 else float(np.sum(energy64[keep_np])) / energy_total
        )
        state["action_energy_topk_overlap"].append(
            float(np.intersect1d(keep_np, energy_order).size) / float(OLD_QUOTA)
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
        new_lambda = module.fnp.concatenate((keep_lambda, birth_lambda32))
        new_p = module.fnp.concatenate(
            (gate32[:, None] * keep_p, gate32[:, None] * covariance32), axis=1
        )
        new_q = module.fnp.concatenate(
            (gate32[:, None] * keep_q, module.fnp.asarray(identity, dtype=module.fnp.float32)),
            axis=1,
        )
        if int(new_lambda.shape[0]) != CAP:
            raise AssertionError(new_lambda.shape)
        state["birth_labels"] = np.concatenate(
            (
                np.asarray(pending["keep_labels"], dtype=np.int16),
                np.full(COHORT, birth_step, dtype=np.int16),
            )
        )
        state["birth_step"] = birth_step + 1
        state["pending"] = None
        return new_lambda, new_p, new_q

    module._full_relu_reset = reset
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
    output_npz = RUNS / "rank0000_action2.npz"
    output_json = RUNS / "rank0000_action2.json"
    if output_npz.exists() or output_json.exists():
        raise FileExistsError((output_npz, output_json))
    RUNS.mkdir(parents=True, exist_ok=True)
    base = load_base()
    age = base.load_age_helpers()
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
        name="q24-cp3-fixed-summary-rank0-action2",
    )
    mlp.validate()
    module = age.load_module("action2")
    selector_state = install_action_selector(module, weights)
    started = time.perf_counter()
    with threadpool_limits(limits=THREADS):
        with flops.BudgetContext(
            flop_budget=FLOP_BUDGET,
            wall_time_limit_s=120.0,
            quiet=True,
            namespace="q24_cp3_fixed_summary_action2",
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
    state_receipt = {
        key: value
        for key, value in selector_state.items()
        if key not in ("pending", "birth_labels", "live_rho3", "suffix_adjoints")
    }
    state_receipt["final_birth_histogram"] = {
        str(int(key)): int(value)
        for key, value in zip(
            *np.unique(np.asarray(selector_state["birth_labels"]), return_counts=True)
        )
    }
    receipt = {
        "schema": "q24-native-cp3-fixed-summary-action-rank0-v1",
        "status": "COMPLETE",
        "role": "observer_fit",
        "role_rank": 0,
        "mlp_id": int(manifest["mlp_id"]),
        "arm": "action2",
        "physical_cap_columns": CAP,
        "forced_recent_birth_columns": COHORT,
        "old_summary_columns": OLD_QUOTA,
        "selector": {
            "endpoint_probes": PROBES,
            "probe_seed": PROBE_SEED,
            "suffix": "weight-only reverse products; uniform expected gate omitted as rank-invariant scalar",
            "local_sensitivity": "rho3=c1[3]/(6 sigma^3)+c1[6] k3/(36 sigma^6)",
            "score": "mean_r <u_l,r, rho3 * 3 lambda p^2 q>^2",
        },
        "metrics": {
            "host_final_mse": age.mse(host_error[-1]),
            "native_j_final_mse": age.mse(corrected_error[-1]),
            "host_layer_mse": [age.mse(row) for row in host_error],
            "native_j_layer_mse": [age.mse(row) for row in corrected_error],
            "local_q": q_metrics,
        },
        "selector_state": state_receipt,
        "prototype_resources": {
            "literal_flops": int(context.summary_dict(by_namespace=False)["flops_used"]),
            "wall_seconds": wall_seconds,
            "peak_rss_bytes": age.resident_bytes(),
            "note": "Includes R=16 suffix adjoints, action projections, and one extra selected-subset read.",
        },
        "bindings": {
            "source": str(age.SOURCE.relative_to(HERE.parent)),
            "source_sha256": age.sha256(age.SOURCE),
            "age_helper_sha256": age.sha256(age.AGE_DIR / "run_arm.py") if hasattr(age, "AGE_DIR") else age.sha256(base.AGE_RUNNER),
            "base_runner_sha256": age.sha256(BASE_RUNNER),
            "runner_sha256": age.sha256(Path(__file__)),
            "weights_sha256": age.sha256(weights_path),
            "diagnostic_sha256": age.sha256(age.DIAGNOSTIC),
        },
        "access": {
            "observer_fit_target_artifacts_opened": [str(age.DIAGNOSTIC.relative_to(HERE.parent))],
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
