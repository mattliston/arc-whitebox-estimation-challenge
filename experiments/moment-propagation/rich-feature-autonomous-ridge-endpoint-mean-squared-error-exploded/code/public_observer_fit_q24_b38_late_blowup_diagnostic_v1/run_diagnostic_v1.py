#!/usr/bin/env python3
"""Trace one public observer-fit row through the exact submitted Q24 b38.

The estimator source is imported unchanged.  Read-only monkeypatch observers
record the values crossing its existing recursive seams and return the exact
objects produced by the original functions.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
Q24 = P2 / "fullbudget_k21_k4_closure_q24_v1/estimator.py"
STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
STORE_MANIFEST = STORE / "manifest.json"
TARGET_BUNDLE = P2 / "data_gate/sealed_role_target_bundle"
TARGET_MANIFEST = TARGET_BUNDLE / "manifest.json"
TARGET_PARQUET = TARGET_BUNDLE / "observer_fit.parquet"
TARGET_AUTHORIZATION = (
    P2 / "public_observer_fit_truth_scan_v1/TARGET_ACCESS_AUTHORIZATION.json"
)
RUNS = HERE / "runs"

Q24_SHA256 = "b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52"
STORE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
TARGET_MANIFEST_SHA256 = "2c0ea0bb09d0c86d46ca07fdde3563c4cda046a1fff152cdff26fde965e1f7e6"
TARGET_PARQUET_SHA256 = "dcc759bd0eb7df3544bec8ea8a7b788a6915eefa8d79fb7e3858b88bfd4ad9c7"
FLOP_BUDGET = 2**41
TWO_U = 2.220446049250313e-16


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(np, value) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def finite_summary(np, value) -> dict:
    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "finite_count": int(finite_values.size),
        "nonfinite_count": int(array.size - finite_values.size),
        "minimum_finite": None if not finite_values.size else float(np.min(finite_values)),
        "maximum_finite": None if not finite_values.size else float(np.max(finite_values)),
        "maximum_absolute_finite": (
            None if not finite_values.size else float(np.max(np.abs(finite_values)))
        ),
    }


def covariance_summary(np, value) -> dict:
    array = np.asarray(value)
    diagonal = np.diag(array)
    result = finite_summary(np, array)
    result.update(
        {
            "diagonal_minimum_finite": (
                None if not np.isfinite(diagonal).any() else float(np.min(diagonal[np.isfinite(diagonal)]))
            ),
            "diagonal_maximum_finite": (
                None if not np.isfinite(diagonal).any() else float(np.max(diagonal[np.isfinite(diagonal)]))
            ),
            "diagonal_nonfinite_count": int(np.count_nonzero(~np.isfinite(diagonal))),
            "diagonal_negative_count": int(np.count_nonzero(diagonal < 0.0)),
        }
    )
    return result


def upward(np, value):
    return np.nextafter(value, np.inf)


def add_sub_error(np, error_x, error_y, result):
    inherited = upward(np, error_x + error_y)
    local = upward(np, TWO_U * np.abs(result))
    return upward(np, inherited + local)


def multiply_error(np, x, error_x, y, error_y, result):
    left = upward(np, np.abs(x) * error_y)
    right = upward(np, np.abs(y) * error_x)
    inherited = upward(np, left + right)
    cross = upward(np, error_x * error_y)
    inherited = upward(np, inherited + cross)
    local = upward(np, TWO_U * np.abs(result))
    return upward(np, inherited + local)


def certified_moment_diagnostic(np, raw1, raw2, raw3, raw4) -> dict:
    """Pure-NumPy copy of the successor's conservative fp64 A/B certificate."""
    zero = np.zeros_like(raw1)
    raw1_squared = raw1 * raw1
    error_raw1_squared = multiply_error(
        np, raw1, zero, raw1, zero, raw1_squared
    )
    variance = raw2 - raw1_squared
    error_variance = add_sub_error(np, zero, error_raw1_squared, variance)

    raw2_squared = raw2 * raw2
    error_raw2_squared = multiply_error(
        np, raw2, zero, raw2, zero, raw2_squared
    )
    a_left = raw1 * raw3
    error_a_left = multiply_error(np, raw1, zero, raw3, zero, a_left)
    margin_a = a_left - raw2_squared
    error_margin_a = add_sub_error(
        np, error_a_left, error_raw2_squared, margin_a
    )

    raw1_raw2 = raw1 * raw2
    error_raw1_raw2 = multiply_error(
        np, raw1, zero, raw2, zero, raw1_raw2
    )
    centered_21 = raw3 - raw1_raw2
    error_centered_21 = add_sub_error(np, zero, error_raw1_raw2, centered_21)
    raw4_minus_raw2_squared = raw4 - raw2_squared
    error_raw4_minus = add_sub_error(
        np, zero, error_raw2_squared, raw4_minus_raw2_squared
    )
    b_left = raw4_minus_raw2_squared * variance
    error_b_left = multiply_error(
        np,
        raw4_minus_raw2_squared,
        error_raw4_minus,
        variance,
        error_variance,
        b_left,
    )
    centered_21_squared = centered_21 * centered_21
    error_centered_21_squared = multiply_error(
        np,
        centered_21,
        error_centered_21,
        centered_21,
        error_centered_21,
        centered_21_squared,
    )
    margin_b = b_left - centered_21_squared
    error_margin_b = add_sub_error(
        np, error_b_left, error_centered_21_squared, margin_b
    )
    invalid_a = margin_a < -error_margin_a
    invalid_b = margin_b < -error_margin_b
    return {
        "variance": variance,
        "margin_a": margin_a,
        "margin_b": margin_b,
        "error_margin_a": error_margin_a,
        "error_margin_b": error_margin_b,
        "invalid_a": invalid_a,
        "invalid_b": invalid_b,
        "invalid": invalid_a | invalid_b,
    }


def moment_record(np, layer: int, raw_values, source: str, **extra) -> dict:
    raw = tuple(np.asarray(value, dtype=np.float64).copy() for value in raw_values)
    certificate = certified_moment_diagnostic(np, *raw)
    invalid = certificate["invalid"]
    record = {
        "layer": layer,
        "source": source,
        "raw": raw,
        "certificate": certificate,
        "raw_summaries": [finite_summary(np, value) for value in raw],
        "represented_variance": finite_summary(np, certificate["variance"]),
        "represented_variance_negative_count": int(
            np.count_nonzero(certificate["variance"] < 0.0)
        ),
        "certified_invalid_a_count": int(np.count_nonzero(certificate["invalid_a"])),
        "certified_invalid_b_count": int(np.count_nonzero(certificate["invalid_b"])),
        "certified_invalid_union_count": int(np.count_nonzero(invalid)),
        "nonfinite_raw_value_count": int(
            sum(np.count_nonzero(~np.isfinite(value)) for value in raw)
        ),
    }
    record.update(extra)
    return record


def install_observers(np, module) -> dict:
    trace = {
        "moments": {},
        "covariance_repairs": {},
        "full_resets": {},
        "star_resets": {},
        "cp3": {},
        "cp4": {},
    }
    original_initialize = module._initialize_h1
    original_univariate = module._univariate_edgeworth
    original_repair = module._repair_covariance
    original_full_reset = module._full_relu_reset
    original_reset_stars = module._reset_stars
    original_cp_grow = module._cp_grow
    original_cp4_reset = module._cp4_reset

    def observed_initialize(*args, **kwargs):
        value = original_initialize(*args, **kwargs)
        variance = np.maximum(np.asarray(args[1], dtype=np.float64), module._VAR_FLOOR)
        sigma = np.sqrt(variance)
        raw = (
            sigma * module._INV_SQRT_2PI,
            0.5 * variance,
            2.0 * sigma * variance * module._INV_SQRT_2PI,
            1.5 * variance * variance,
        )
        trace["moments"][0] = moment_record(
            np,
            0,
            raw,
            "exact_zero_mean_gaussian_relu_h1",
            pre_covariance=covariance_summary(np, args[0]),
            post_covariance=covariance_summary(np, value[1]),
        )
        return value

    def observed_univariate(*args, **kwargs):
        value = original_univariate(*args, **kwargs)
        layer = 1 + len([index for index in trace["moments"] if index > 0])
        trace["moments"][layer] = moment_record(
            np,
            layer,
            value,
            "submitted_q24_univariate_edgeworth",
            input_pre_variance=finite_summary(np, args[7]),
            input_k3=finite_summary(np, args[5]),
            input_k4=finite_summary(np, args[6]),
        )
        return value

    def observed_repair(*args, **kwargs):
        layer = len(trace["covariance_repairs"]) + 1
        raw_covariance = np.asarray(args[0]).copy()
        requested_variance = np.asarray(args[1]).copy()
        value = original_repair(*args, **kwargs)
        adjusted_raw = raw_covariance.copy()
        np.fill_diagonal(adjusted_raw, requested_variance)
        repaired = np.asarray(value)
        finite_delta = repaired - adjusted_raw
        trace["covariance_repairs"][layer] = {
            "layer": layer,
            "raw_postactivation_covariance": covariance_summary(np, adjusted_raw),
            "requested_variance": finite_summary(np, requested_variance),
            "repaired_covariance": covariance_summary(np, repaired),
            "maximum_absolute_repair": (
                None
                if not np.isfinite(finite_delta).any()
                else float(np.max(np.abs(finite_delta[np.isfinite(finite_delta)])))
            ),
            "repaired_diagonal_matches_requested_bytes": bool(
                np.array_equal(np.diag(repaired), requested_variance)
            ),
        }
        return value

    def observed_full_reset(*args, **kwargs):
        value = original_full_reset(*args, **kwargs)
        layer = len(trace["full_resets"]) + 1
        trace["full_resets"][layer] = {
            "layer": layer,
            "preactivation_mean": finite_summary(np, args[0]),
            "preactivation_covariance": covariance_summary(np, args[1]),
            "postactivation_mean": finite_summary(np, value[0]),
            "postactivation_covariance": covariance_summary(np, value[1]),
            "repaired_variance": finite_summary(np, value[9]),
            "connected_k21": finite_summary(np, value[2]),
            "connected_k4": finite_summary(np, value[3]),
            "connected_k31": finite_summary(np, value[4]),
            "connected_k22": finite_summary(np, value[5]),
        }
        return value

    def observed_reset_stars(*args, **kwargs):
        value = original_reset_stars(*args, **kwargs)
        layer = len(trace["star_resets"])
        alpha = np.asarray(value[2])
        trace["star_resets"][layer] = {
            "layer": layer,
            "pairing_alpha": None if not np.isfinite(alpha).all() else float(alpha),
            "pairing_alpha_nonfinite_count": int(np.count_nonzero(~np.isfinite(alpha))),
            "selected_covariance": covariance_summary(np, args[5]),
            "selected_variance": finite_summary(np, args[6]),
            "cubic_star": finite_summary(np, value[0]),
            "quartic_star": finite_summary(np, value[1]),
        }
        return value

    def observed_cp_grow(*args, **kwargs):
        value = original_cp_grow(*args, **kwargs)
        layer = len(trace["cp3"])
        pre_rank = int(args[0].shape[0])
        post_rank = int(value[0].shape[0])
        trace["cp3"][layer] = {
            "layer": layer,
            "carrier": "full_birth_cp3",
            "pre_rank": pre_rank,
            "birth_count": post_rank - pre_rank,
            "post_rank": post_rank,
            "cap": int(module._CP3_CAP),
            "mode": "APPEND_FULL_BIRTH" if post_rank > pre_rank else "AT_CAP",
        }
        return value

    def observed_cp4_reset(*args, **kwargs):
        value = original_cp4_reset(*args, **kwargs)
        layer = len(trace["cp4"])
        pre_rank = int(args[0].shape[0])
        width = int(args[2].shape[0])
        candidate_rank = pre_rank + width
        post_rank = int(value[0].shape[0])
        trace["cp4"][layer] = {
            "layer": layer,
            "carrier": "symmetric_cp4_coordinate_birth",
            "pre_rank": pre_rank,
            "candidate_rank": candidate_rank,
            "post_rank": post_rank,
            "dropped_count": candidate_rank - post_rank,
            "cap": int(module._CP4_CAP),
            "mode": (
                "APPEND_WITHOUT_TRUNCATION"
                if candidate_rank <= int(module._CP4_CAP)
                else "TOP_ENERGY_TRUNCATION"
            ),
        }
        return value

    module._initialize_h1 = observed_initialize
    module._univariate_edgeworth = observed_univariate
    module._repair_covariance = observed_repair
    module._full_relu_reset = observed_full_reset
    module._reset_stars = observed_reset_stars
    module._cp_grow = observed_cp_grow
    module._cp4_reset = observed_cp4_reset
    return trace


def load_target(np, role_rank: int):
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(TARGET_PARQUET)
    if parquet.schema_arrow.names != ["mlp_id", "all_layer_means"]:
        raise RuntimeError("target projection schema changed")
    for index, batch in enumerate(
        parquet.iter_batches(
            batch_size=1,
            columns=["mlp_id", "all_layer_means"],
            use_threads=False,
        )
    ):
        if index != role_rank:
            continue
        mlp_id = int(batch.column(0)[0].as_py())
        storage = batch.column(1)
        storage = getattr(storage, "storage", storage)
        while not pa.types.is_primitive(storage.type):
            storage = storage.flatten()
        target = storage.to_numpy(zero_copy_only=True).reshape(16, 1024)
        if target.dtype != np.float32 or not np.isfinite(target).all():
            raise RuntimeError("target geometry or finiteness changed")
        return mlp_id, np.ascontiguousarray(target).copy()
    raise RuntimeError("observer-fit target role rank not found")


def reconstruct_prediction(np, trace):
    if set(trace["moments"]) != set(range(16)):
        return None, None
    fp64 = np.stack([trace["moments"][layer]["raw"][0] for layer in range(16)])
    return np.ascontiguousarray(fp64), np.ascontiguousarray(fp64, dtype=np.float32)


def first_layer(layers, predicate):
    return next((layer for layer in sorted(layers) if predicate(layers[layer])), None)


def finite_float(np, value):
    scalar = float(value)
    return scalar if np.isfinite(scalar) else None


def json_layer_record(record: dict) -> dict:
    return {
        key: value
        for key, value in record.items()
        if key not in {"raw", "certificate"}
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role-rank", required=True, type=int)
    parser.add_argument("--late-mse-threshold", default=5e-7, type=float)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not 0 <= args.role_rank < 688:
        raise ValueError("role rank outside observer_fit")
    if not args.late_mse_threshold > 0.0:
        raise ValueError("late MSE threshold must be positive")

    bindings = {
        Q24: Q24_SHA256,
        STORE_MANIFEST: STORE_MANIFEST_SHA256,
        TARGET_MANIFEST: TARGET_MANIFEST_SHA256,
        TARGET_PARQUET: TARGET_PARQUET_SHA256,
    }
    for path, expected in bindings.items():
        if sha256_file(path) != expected:
            raise RuntimeError(f"frozen binding mismatch: {path}")
    authorization = json.loads(TARGET_AUTHORIZATION.read_text())
    if not (
        authorization.get("authorized") is True
        and authorization.get("allowed_roles") == ["observer_fit"]
        and authorization.get("allowed_columns") == ["mlp_id", "all_layer_means"]
        and authorization.get("raw_full_dataset_access") == "forbidden"
    ):
        raise PermissionError("observer-fit target authorization is absent or changed")

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else RUNS / f"row_{args.role_rank:04d}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    import numpy as np
    import flopscope as flops
    from whestbench.domain import MLP

    target_mlp_id, target = load_target(np, args.role_rank)
    store_manifest = json.loads(STORE_MANIFEST.read_text())
    entry = store_manifest["roles"]["observer_fit"]["row_manifests"][args.role_rank]
    row_manifest_path = STORE / entry["path"]
    if sha256_file(row_manifest_path) != entry["sha256"]:
        raise RuntimeError("weight row manifest hash mismatch")
    row_manifest = json.loads(row_manifest_path.read_text())
    if int(row_manifest["role_rank"]) != args.role_rank:
        raise RuntimeError("weight role rank mismatch")
    if int(row_manifest["mlp_id"]) != target_mlp_id:
        raise RuntimeError("target/weight MLP identity mismatch")
    weights_path = row_manifest_path.parent / "weights.npy"
    weights = np.load(weights_path, allow_pickle=False, mmap_mode="r")
    if weights.shape != (16, 1024, 1024) or weights.dtype != np.float32:
        raise RuntimeError("weight geometry changed")
    if sha256_file(weights_path) != row_manifest["files"]["weights.npy"]["sha256"]:
        raise RuntimeError("weight file hash mismatch")

    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=int(row_manifest["mlp_seed"]),
        name=f"public-observer-fit-diagnostic-rank-{args.role_rank}",
    )
    mlp.validate()
    module = load_module(f"q24_b38_diagnostic_{args.role_rank}_{time.time_ns()}", Q24)
    trace = install_observers(np, module)

    exception = None
    output = None
    started = time.perf_counter()
    try:
        with flops.BudgetContext(flop_budget=FLOP_BUDGET, quiet=True) as budget:
            output = module.Estimator().predict(mlp, FLOP_BUDGET)
    except BaseException as caught:
        exception = caught
    wall_seconds = time.perf_counter() - started

    source_output = None if output is None else np.ascontiguousarray(np.asarray(output))
    mean_fp64, traced_output = reconstruct_prediction(np, trace)
    if source_output is not None:
        if source_output.shape != (16, 1024) or source_output.dtype != np.float32:
            raise RuntimeError("submitted output geometry changed")
        if traced_output is None or not np.array_equal(source_output, traced_output):
            raise RuntimeError("observer trace does not reconstruct submitted output bytes")
    prediction = source_output if source_output is not None else traced_output

    metrics = None
    if prediction is not None:
        delta = prediction.astype(np.float64) - target.astype(np.float64)
        layer_mse = np.mean(delta * delta, axis=1)
        layer_max_abs = np.max(np.abs(delta), axis=1)
        metrics = [
            {
                "layer": layer,
                "mse": finite_float(np, layer_mse[layer]),
                "rms": finite_float(np, np.sqrt(layer_mse[layer])),
                "maximum_absolute_error": finite_float(np, layer_max_abs[layer]),
            }
            for layer in range(16)
        ]
    else:
        layer_mse = None

    first_certified_invalid_reset = first_layer(
        {k: v for k, v in trace["moments"].items() if 1 <= k <= 14},
        lambda record: record["certified_invalid_union_count"] > 0,
    )
    first_negative_variance_reset = first_layer(
        {k: v for k, v in trace["moments"].items() if 1 <= k <= 14},
        lambda record: record["represented_variance_negative_count"] > 0,
    )
    first_nonfinite_moment_layer = first_layer(
        trace["moments"], lambda record: record["nonfinite_raw_value_count"] > 0
    )
    first_target_mse_layer = (
        None
        if metrics is None
        else next(
            (
                row["layer"]
                for row in metrics
                if row["mse"] is not None
                and row["mse"] > args.late_mse_threshold
            ),
            None,
        )
    )

    np.save(output_dir / "target.npy", target, allow_pickle=False)
    if prediction is not None:
        np.save(output_dir / "submitted_prediction.npy", prediction, allow_pickle=False)
    if mean_fp64 is not None:
        np.save(output_dir / "mean_trajectory_fp64.npy", mean_fp64, allow_pickle=False)
    if layer_mse is not None:
        np.save(output_dir / "layer_mse.npy", layer_mse, allow_pickle=False)
    if trace["moments"]:
        ordered_layers = sorted(trace["moments"])
        np.savez(
            output_dir / "moment_trajectory.npz",
            layer=np.asarray(ordered_layers, dtype=np.int16),
            raw=np.stack([trace["moments"][i]["raw"] for i in ordered_layers]),
            variance=np.stack(
                [trace["moments"][i]["certificate"]["variance"] for i in ordered_layers]
            ),
            margin_a=np.stack(
                [trace["moments"][i]["certificate"]["margin_a"] for i in ordered_layers]
            ),
            margin_b=np.stack(
                [trace["moments"][i]["certificate"]["margin_b"] for i in ordered_layers]
            ),
            error_margin_a=np.stack(
                [trace["moments"][i]["certificate"]["error_margin_a"] for i in ordered_layers]
            ),
            error_margin_b=np.stack(
                [trace["moments"][i]["certificate"]["error_margin_b"] for i in ordered_layers]
            ),
            invalid_a=np.stack(
                [trace["moments"][i]["certificate"]["invalid_a"] for i in ordered_layers]
            ),
            invalid_b=np.stack(
                [trace["moments"][i]["certificate"]["invalid_b"] for i in ordered_layers]
            ),
        )

    result = {
        "schema": "phase2-public-observer-fit-q24-b38-late-blowup-diagnostic-v1",
        "role": "observer_fit",
        "role_rank": args.role_rank,
        "mlp_id": target_mlp_id,
        "mlp_seed": int(row_manifest["mlp_seed"]),
        "weights_raw_sha256": row_manifest["weights_raw_sha256"],
        "submitted_q24_sha256": Q24_SHA256,
        "outcome": "EXCEPTION" if exception is not None else "FINITE",
        "exception": (
            None
            if exception is None
            else {"type": type(exception).__name__, "message": str(exception)}
        ),
        "wall_seconds": wall_seconds,
        "flops_used": None if "budget" not in locals() else int(budget.flops_used),
        "source_output_sha256": (
            None if source_output is None else array_sha256(np, source_output)
        ),
        "trace_reconstructed_output_sha256": (
            None if traced_output is None else array_sha256(np, traced_output)
        ),
        "observer_preserved_reported_output_bytes": (
            None
            if source_output is None
            else bool(np.array_equal(source_output, traced_output))
        ),
        "late_mse_threshold": args.late_mse_threshold,
        "layer_metrics": metrics,
        "onsets": {
            "first_certified_invalid_recursive_reset": first_certified_invalid_reset,
            "first_negative_represented_variance_recursive_reset": first_negative_variance_reset,
            "first_nonfinite_moment_layer": first_nonfinite_moment_layer,
            "first_target_mse_above_threshold": first_target_mse_layer,
        },
        "moment_layers": [
            json_layer_record(trace["moments"][layer])
            for layer in sorted(trace["moments"])
        ],
        "covariance_repairs": [
            trace["covariance_repairs"][layer]
            for layer in sorted(trace["covariance_repairs"])
        ],
        "full_resets": [
            trace["full_resets"][layer] for layer in sorted(trace["full_resets"])
        ],
        "star_resets": [
            trace["star_resets"][layer] for layer in sorted(trace["star_resets"])
        ],
        "cp3_carrier": [trace["cp3"][layer] for layer in sorted(trace["cp3"])],
        "cp4_carrier": [trace["cp4"][layer] for layer in sorted(trace["cp4"])],
        "scope": {
            "public_columns_decoded": ["mlp_id", "all_layer_means"],
            "board_names_scores_joins": 0,
            "withheld_roles": 0,
            "estimator_source_mutations": 0,
        },
        "artifacts": {
            "target": "target.npy",
            "submitted_prediction": None if prediction is None else "submitted_prediction.npy",
            "mean_trajectory_fp64": None if mean_fp64 is None else "mean_trajectory_fp64.npy",
            "layer_mse": None if layer_mse is None else "layer_mse.npy",
            "moment_trajectory": None if not trace["moments"] else "moment_trajectory.npz",
        },
    }
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "role_rank": args.role_rank,
                "mlp_id": target_mlp_id,
                "outcome": result["outcome"],
                "final_mse": None if metrics is None else metrics[-1]["mse"],
                "onsets": result["onsets"],
                "result": str(result_path),
            },
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if exception is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
