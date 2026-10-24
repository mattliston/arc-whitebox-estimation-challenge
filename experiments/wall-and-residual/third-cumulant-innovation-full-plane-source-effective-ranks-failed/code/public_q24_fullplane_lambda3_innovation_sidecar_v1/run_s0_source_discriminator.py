#!/usr/bin/env python3
"""Frozen target-blind S0 discriminator for the lambda3 innovation plane.

The default action is a standard-library-only static audit.  Numerical
execution is impossible unless an independently created S0_RELEASE.json pins
this runner, the derivation, and the preregistration byte for byte.  The
execution path reads one sealed observer_fit row (weights and row metadata
only), runs a separately freed base-parity call, and runs the audited lambda3
source with a split reducer.  It never opens a target, score, host cache,
observer-selection row, or high-host prediction.
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
from typing import Any


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
DERIVATION = HERE / "DERIVATION.md"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
RUNNER = HERE / "run_s0_source_discriminator.py"
RELEASE = HERE / "S0_RELEASE.json"
BASE_SOURCE = P2 / "public_condrepair_deployment_card_v1/estimator.py"
LAMBDA3_SOURCE = P2 / "public_lambda3_shell_probe_v1/instrumented_estimator_x.py"
STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)

BASE_SHA256 = "24ed2325e4db0c0287a222189a986c2d6462f80832c717827c27b783a6655ee2"
LAMBDA3_SHA256 = "a2712a3f6ba3d5b085c371305a06547e94b292d5a79a77954b032b0fe5627d07"
ALLOWED_RANKS = (35, 201, 643)
CAPTURE_LAYERS = (4, 9, 14)
PHASE2_BUDGET = 2**41
SOURCE_ADDED_FLOP_CAP = 170_000_000_000
PROCESS_RSS_CAP = 8 * 1024**3
ETA_MIN = 1.0e-6
EFFECTIVE_RANK_MIN = 128.0
EFFECTIVE_RANK_MEDIAN_MIN = 256.0
ORTHOGONAL_ENERGY_MIN = 0.50
ACTION_MIN = 0.005
ACTION_LAYERS_MIN = 2


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def load_module(tag: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def static_receipt() -> dict[str, Any]:
    prereg = json.loads(PREREGISTRATION.read_text())
    authority = prereg["authority"]
    checks = {
        "derivation_exists": DERIVATION.is_file(),
        "preregistration_json_valid": True,
        "base_source_sha256": sha256_file(BASE_SOURCE) == BASE_SHA256,
        "lambda3_source_sha256": sha256_file(LAMBDA3_SOURCE) == LAMBDA3_SHA256,
        "prereg_base_binding": authority["base_source_sha256"] == BASE_SHA256,
        "prereg_lambda3_binding": authority["lambda3_source_sha256"] == LAMBDA3_SHA256,
        "targets_forbidden": prereg["scope"]["targets_allowed"] is False,
        "training_forbidden": prereg["scope"]["training_allowed"] is False,
        "high_host_forbidden": prereg["scope"]["high_host_output_allowed"] is False,
        "release_absent_by_default": not RELEASE.exists(),
    }
    return {
        "schema": "q24-fullplane-lambda3-innovation-s0-static-v1",
        "status": "STATIC_READY_HOLD_NO_RELEASE" if all(checks.values()) else "STATIC_HOLD",
        "checks": checks,
        "sha256": {
            "derivation": sha256_file(DERIVATION),
            "preregistration": sha256_file(PREREGISTRATION),
            "runner": sha256_file(RUNNER),
            "base_source": sha256_file(BASE_SOURCE),
            "lambda3_source": sha256_file(LAMBDA3_SOURCE),
        },
        "allowed_role_ranks": list(ALLOWED_RANKS),
        "capture_post_layers": list(CAPTURE_LAYERS),
        "release_path": str(RELEASE),
        "release_present": RELEASE.exists(),
        "numerical_work_performed": False,
        "targets_read": 0,
    }


def require_release(rank: int) -> dict[str, Any]:
    if not RELEASE.is_file():
        raise RuntimeError("S0_RELEASE.json is absent; numerical execution is not authorized")
    release = json.loads(RELEASE.read_text())
    expected_hashes = {
        "derivation_sha256": sha256_file(DERIVATION),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "runner_sha256": sha256_file(RUNNER),
    }
    if release.get("status") != "RELEASE_S0_SOURCE_ONLY":
        raise RuntimeError("release status is not RELEASE_S0_SOURCE_ONLY")
    for key, value in expected_hashes.items():
        if release.get(key) != value:
            raise RuntimeError(f"release hash mismatch: {key}")
    allowed = release.get("allowed_role_ranks")
    if allowed != [rank]:
        raise RuntimeError(f"release must authorize exactly the singleton role rank {rank}")
    forbidden_authorities = (
        "targets_allowed",
        "training_allowed",
        "high_host_output_allowed",
        "observer_selection_allowed",
    )
    if any(release.get(key, False) for key in forbidden_authorities):
        raise RuntimeError("release attempts to broaden the source-only authority")
    prerequisite_ranks = {35: (), 201: (35,), 643: (35, 201)}[rank]
    prerequisite_hashes = release.get("prerequisite_result_sha256", {})
    required_names = {f"S0_RANK{prior:04d}_RESULT.json" for prior in prerequisite_ranks}
    if set(prerequisite_hashes) != required_names:
        raise RuntimeError("release does not pin the exact prerequisite result set")
    for prior in prerequisite_ranks:
        name = f"S0_RANK{prior:04d}_RESULT.json"
        path = HERE / name
        if not path.is_file() or sha256_file(path) != prerequisite_hashes[name]:
            raise RuntimeError(f"missing or hash-mismatched prerequisite {name}")
        receipt = json.loads(path.read_text())
        expected_decision = (
            "GO_TARGET_BLIND_REPLICATION"
            if prior == 35
            else "PASS_TARGET_BLIND_REPLICATION_ROW"
        )
        if (
            receipt.get("status") != "COMPLETE_TARGET_BLIND_SOURCE_ONLY"
            or receipt.get("decision") != expected_decision
            or receipt.get("all_gates_pass") is not True
            or receipt.get("role") != "observer_fit"
            or int(receipt.get("role_rank", -1)) != prior
            or receipt.get("firewall", {}).get("targets_read") != 0
        ):
            raise RuntimeError(f"prerequisite {name} is not a passing source-only receipt")
    return release


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def load_sealed_row(rank: int) -> tuple[dict[str, Any], Any, dict[str, str]]:
    import numpy as np

    manifest_path = STORE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise RuntimeError("manifest role-rank mismatch")
    row_path = STORE / entry["path"]
    if sha256_file(row_path) != entry["sha256"]:
        raise RuntimeError("sealed row-manifest hash mismatch")
    row = json.loads(row_path.read_text())
    if row.get("role") != "observer_fit" or int(row.get("role_rank", -1)) != rank:
        raise RuntimeError("row is not the requested observer_fit rank")
    if set(row.get("decoded_columns", ())) != {"mlp_id", "mlp_name", "mlp_seed", "weights"}:
        raise RuntimeError("sealed row decoded-column firewall changed")
    weights_path = row_path.parent / "weights.npy"
    expected_weight_hash = row["files"]["weights.npy"]["sha256"]
    if sha256_file(weights_path) != expected_weight_hash:
        raise RuntimeError("sealed weight-file hash mismatch")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.shape != (16, 1024, 1024) or weights.dtype != np.dtype("<f4"):
        raise RuntimeError("sealed weights have unexpected shape or dtype")
    hashes = {
        "store_manifest_sha256": sha256_file(manifest_path),
        "row_manifest_sha256": sha256_file(row_path),
        "weights_file_sha256": expected_weight_hash,
    }
    return row, weights, hashes


def make_mlp(weights: Any, row: dict[str, Any], rank: int) -> Any:
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed

    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[index] for index in range(16)],
        seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
        name=f"s0-observer-fit-{rank:04d}",
    )
    mlp.validate()
    return mlp


def run_base(module: Any, weights: Any, row: dict[str, Any], rank: int) -> tuple[tuple[Any, Any, Any], int, float]:
    import flopscope as flops
    import numpy as np
    from threadpoolctl import threadpool_limits

    mlp = make_mlp(weights, row, rank)
    started = time.perf_counter()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(
            flop_budget=PHASE2_BUDGET,
            wall_time_limit_s=300.0,
            quiet=True,
        ) as meter:
            remote = module.predict_with_native_tape(mlp)
    local = tuple(np.array(value, copy=True) for value in remote)
    flops_used = int(meter.flops_used)
    del remote, mlp
    gc.collect()
    return local, flops_used, time.perf_counter() - started


def install_split_capture(module: Any) -> tuple[dict[str, Any], Any, Any]:
    """Patch only the pair reducer and wrap reset to copy fixed scratch."""

    fnp = module.fnp
    original_four = module._four_pair_edgeworth
    original_reset = module._full_relu_reset
    state: dict[str, Any] = {
        "current_layer": None,
        "reset_calls": 0,
        "captures": {},
    }

    def split_four(context: Any, weights: Any, scratch: Any) -> tuple[Any, Any, Any, Any]:
        weighted_kernels = module._g_four_all(context)
        fnp.multiply(weights[None, :, :, :], weighted_kernels, out=weighted_kernels)
        result, group_sum = scratch
        fnp.copyto(result, 0.0)
        # This is exactly the frozen base association: 0:3, 3:6, ..., 15:17.
        for start in range(0, 17, 3):
            stop = min(start + 3, 17)
            fnp.sum(weighted_kernels[:, start:stop], axis=1, out=group_sum)
            fnp.add(result, group_sum, out=result)

        layer = state["current_layer"]
        if layer in CAPTURE_LAYERS:
            if layer in state["captures"]:
                raise RuntimeError(f"duplicate reducer capture at h{layer}")
            base_copy = fnp.copy(result)
            shell = fnp.zeros_like(result)
            # Fixed shell-only three-channel association; no averaging or
            # equal-RMS normalization is introduced.
            for start in range(17, 35, 3):
                stop = min(start + 3, 35)
                fnp.sum(weighted_kernels[:, start:stop], axis=1, out=group_sum)
                fnp.add(shell, group_sum, out=shell)
            state["captures"][layer] = {
                "base_moments": base_copy,
                "shell_moments": shell,
            }
        # The high total is deliberately neither returned nor persisted.
        return result[0], result[1], result[2], result[3]

    def wrapped_reset(*args: Any, **kwargs: Any) -> Any:
        layer = 2 + int(state["reset_calls"])
        state["current_layer"] = layer
        try:
            answer = original_reset(*args, **kwargs)
        finally:
            state["current_layer"] = None
        if layer in CAPTURE_LAYERS:
            capture = state["captures"].get(layer)
            if capture is None:
                raise RuntimeError(f"missing reducer capture at h{layer}")
            # args[8] is the live preactivation sigma; answer[1] is the
            # repaired base postactivation covariance.  Explicit copies are
            # mandatory because both underlying arenas are reused.
            capture["pre_sigma"] = fnp.copy(args[8])
            capture["post_covariance"] = fnp.copy(answer[1])
        state["reset_calls"] = int(state["reset_calls"]) + 1
        return answer

    module._four_pair_edgeworth = split_four
    module._full_relu_reset = wrapped_reset
    return state, original_four, original_reset


def run_source(
    module: Any,
    weights: Any,
    row: dict[str, Any],
    rank: int,
) -> tuple[tuple[Any, Any, Any], dict[int, dict[str, Any]], int, float]:
    import flopscope as flops
    import numpy as np
    from threadpoolctl import threadpool_limits

    state, original_four, original_reset = install_split_capture(module)
    mlp = make_mlp(weights, row, rank)
    remote = None
    started = time.perf_counter()
    try:
        with threadpool_limits(limits=2):
            with flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=300.0,
                quiet=True,
            ) as meter:
                remote = module.predict_with_native_tape(mlp)
        local = tuple(np.array(value, copy=True) for value in remote)
        captures: dict[int, dict[str, Any]] = {}
        if int(state["reset_calls"]) != 14 or set(state["captures"]) != set(CAPTURE_LAYERS):
            raise RuntimeError("unexpected reset count or capture-layer set")
        for layer in CAPTURE_LAYERS:
            captures[layer] = {
                name: np.array(value, dtype=np.float64, copy=True)
                for name, value in state["captures"][layer].items()
            }
        flops_used = int(meter.flops_used)
    finally:
        module._four_pair_edgeworth = original_four
        module._full_relu_reset = original_reset
    del remote, mlp
    state["captures"].clear()
    gc.collect()
    return local, captures, flops_used, time.perf_counter() - started


def same_bytes(left: Any, right: Any) -> bool:
    import numpy as np

    return (
        left.shape == right.shape
        and left.dtype == right.dtype
        and np.array_equal(left.view(np.uint8), right.view(np.uint8))
    )


def offdiag(array: Any) -> Any:
    import numpy as np

    result = np.array(array, dtype=np.float64, copy=True)
    np.fill_diagonal(result, 0.0)
    return result


def standardized_planes(raw: Any, sigma: Any) -> list[Any]:
    import numpy as np

    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1.0e-30)
    si = sigma[:, None]
    sj = sigma[None, :]
    left11 = raw[0] / (si * sj)
    z11 = offdiag(0.5 * (left11 + left11.T))
    left21 = raw[1] / (si * si * sj)
    z21 = offdiag(0.5 * (left21 + left21.T))
    left31 = raw[2] / (si * si * si * sj)
    z31 = offdiag(0.5 * (left31 + left31.T))
    left22 = raw[3] / (si * si * sj * sj)
    z22 = offdiag(0.5 * (left22 + left22.T))
    return [z11, z21, z31, z22]


def double_center(array: Any) -> Any:
    import numpy as np

    row_mean = np.mean(array, axis=1, keepdims=True, dtype=np.float64)
    column_mean = np.mean(array, axis=0, keepdims=True, dtype=np.float64)
    grand_mean = float(np.mean(array, dtype=np.float64))
    return array - row_mean - column_mean + grand_mean


def layer_metrics(layer: int, capture: dict[str, Any], next_weight: Any) -> dict[str, Any]:
    import numpy as np

    shell = standardized_planes(capture["shell_moments"], capture["pre_sigma"])
    base = standardized_planes(capture["base_moments"], capture["pre_sigma"])
    shell_centered = [double_center(value) for value in shell]
    base_centered = [double_center(value) for value in base]

    shell_energy = float(sum(np.vdot(value, value).real for value in shell_centered))
    base_energy = float(sum(np.vdot(value, value).real for value in base_centered))
    eta = float(np.sqrt(shell_energy / base_energy)) if base_energy > 0.0 else float("nan")

    # The horizontal-stack singular spectrum is computed through its exactly
    # equivalent Gram sum, avoiding a materialized 1024x4096 stack.
    gram = np.zeros((1024, 1024), dtype=np.float64)
    for value in shell_centered:
        gram += value @ value.T
    gram = 0.5 * (gram + gram.T)
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    spectral_energy = float(np.sum(eigenvalues, dtype=np.float64))
    if spectral_energy > 0.0:
        probabilities = eigenvalues[eigenvalues > 0.0] / spectral_energy
        effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
        orthogonal_energy = float(np.sum(eigenvalues[:-16]) / spectral_energy)
    else:
        effective_rank = float("nan")
        orthogonal_energy = float("nan")

    covariance = np.asarray(capture["post_covariance"], dtype=np.float64)
    weight = np.asarray(next_weight, dtype=np.float64)
    tau = np.sqrt(np.maximum(np.diag(covariance), 1.0e-30))
    covariance_weight = covariance @ weight
    next_variance = np.einsum("ij,ij->j", weight, covariance_weight, optimize=False)
    sigma_next = np.sqrt(np.maximum(next_variance, 1.0e-30))
    scale_weight = (tau[:, None] * weight) / sigma_next[None, :]
    action_numerator = 0.0
    action_denominator = 0.0
    for value in shell:
        pushed = value @ scale_weight
        diagonal_action = np.einsum("ij,ij->j", scale_weight, pushed, optimize=False)
        action_numerator += float(np.vdot(diagonal_action, diagonal_action).real)
        action_denominator += float(np.vdot(value, value).real)
    action = (
        float(np.sqrt(action_numerator / action_denominator))
        if action_denominator > 0.0
        else float("nan")
    )

    plane_diagnostics = []
    for index, (shell_plane, base_plane) in enumerate(zip(shell, base)):
        shell_rms = float(np.sqrt(np.mean(shell_plane * shell_plane, dtype=np.float64)))
        base_rms = float(np.sqrt(np.mean(base_plane * base_plane, dtype=np.float64)))
        plane_diagnostics.append(
            {
                "moment": ("M11", "M21", "M31", "M22")[index],
                "shell_rms": shell_rms,
                "base_rms": base_rms,
                "shell_to_base_rms_ratio": shell_rms / base_rms if base_rms > 0.0 else None,
            }
        )

    all_arrays = shell + base + [covariance, capture["pre_sigma"], scale_weight]
    return {
        "post_layer": layer,
        "plane_diagnostics_not_gates": plane_diagnostics,
        "stacked_shell_energy": shell_energy,
        "stacked_base_energy": base_energy,
        "stacked_shell_to_base_energy_ratio_sqrt": eta,
        "spectral_energy": spectral_energy,
        "entropy_effective_rank": effective_rank,
        "best_rank16_orthogonal_energy_fraction": orthogonal_energy,
        "quadratic_action_numerator": action_numerator,
        "quadratic_action_denominator": action_denominator,
        "next_weight_quadratic_action_ratio": action,
        "diagonal_max_abs": float(max(np.max(np.abs(np.diag(value))) for value in shell)),
        "symmetry_max_abs": float(max(np.max(np.abs(value - value.T)) for value in shell)),
        "all_values_finite": bool(all(np.isfinite(value).all() for value in all_arrays)),
        "all_hard_denominators_finite_strictly_positive": bool(
            np.isfinite(base_energy)
            and base_energy > 0.0
            and np.isfinite(spectral_energy)
            and spectral_energy > 0.0
            and np.isfinite(action_denominator)
            and action_denominator > 0.0
        ),
    }


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    import math

    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def execute(rank: int) -> dict[str, Any]:
    import flopscope as flops
    import numpy as np

    if rank not in ALLOWED_RANKS:
        raise RuntimeError(f"rank must be one of {ALLOWED_RANKS}")
    release = require_release(rank)
    output_path = HERE / f"S0_RANK{rank:04d}_RESULT.json"
    if output_path.exists():
        raise RuntimeError(f"refusing retry/overwrite of {output_path.name}")
    flops.configure(symmetry_warnings=False)
    row, weights, store_hashes = load_sealed_row(rank)
    base = load_module(f"s0_base_{rank}", BASE_SOURCE)
    source = load_module(f"s0_lambda3_{rank}", LAMBDA3_SOURCE)

    base_values, base_flops, base_wall = run_base(base, weights, row, rank)
    del base
    sys.modules.pop(f"s0_base_{rank}", None)
    gc.collect()
    source_values, captures, source_flops, source_wall = run_source(
        source, weights, row, rank
    )

    parity = [same_bytes(left, right) for left, right in zip(base_values, source_values)]
    layer_rows = [
        layer_metrics(layer, captures[layer], weights[layer])
        for layer in CAPTURE_LAYERS
    ]
    effective_ranks = [row_["entropy_effective_rank"] for row_ in layer_rows]
    action_values = [row_["next_weight_quadratic_action_ratio"] for row_ in layer_rows]
    peak = peak_rss_bytes()
    added_flops = source_flops - base_flops
    gates = {
        "base_source_host_native_raw_sigma_byte_identical": bool(all(parity)),
        "capture_layer_set_exact": set(captures) == set(CAPTURE_LAYERS),
        "source_added_flops_at_most_170b": 0 <= added_flops <= SOURCE_ADDED_FLOP_CAP,
        "source_total_flops_below_phase2_budget": source_flops < PHASE2_BUDGET,
        "process_peak_rss_below_8gib": peak < PROCESS_RSS_CAP,
        "all_values_finite": bool(
            all(np.isfinite(value).all() for value in base_values + source_values)
            and all(row_["all_values_finite"] for row_ in layer_rows)
        ),
        "all_hard_denominators_finite_strictly_positive": all(
            row_["all_hard_denominators_finite_strictly_positive"] for row_ in layer_rows
        ),
        "standardized_diagonal_exactly_zero": all(
            row_["diagonal_max_abs"] == 0.0 for row_ in layer_rows
        ),
        "standardized_symmetry_exactly_zero": all(
            row_["symmetry_max_abs"] == 0.0 for row_ in layer_rows
        ),
        "stacked_energy_eta_at_least_1e_6_each_layer": all(
            row_["stacked_shell_to_base_energy_ratio_sqrt"] >= ETA_MIN for row_ in layer_rows
        ),
        "entropy_effective_rank_at_least_128_each_layer": all(
            value >= EFFECTIVE_RANK_MIN for value in effective_ranks
        ),
        "entropy_effective_rank_median_at_least_256": bool(
            np.median(effective_ranks) >= EFFECTIVE_RANK_MEDIAN_MIN
        ),
        "best_rank16_orthogonal_energy_at_least_half_each_layer": all(
            row_["best_rank16_orthogonal_energy_fraction"] >= ORTHOGONAL_ENERGY_MIN
            for row_ in layer_rows
        ),
        "quadratic_action_finite_strictly_positive_all_layers": all(
            np.isfinite(value) and value > 0.0 for value in action_values
        ),
        "quadratic_action_at_least_0_005_on_two_of_three_layers": sum(
            value >= ACTION_MIN for value in action_values
        ) >= ACTION_LAYERS_MIN,
        "targets_read_exactly_zero": True,
        "observer_selection_opened_exactly_zero": True,
        "high_host_output_produced_exactly_zero": True,
    }
    passed = all(gates.values())
    decision = (
        "GO_TARGET_BLIND_REPLICATION"
        if passed and rank == 35
        else "PASS_TARGET_BLIND_REPLICATION_ROW"
        if passed and rank == 201
        else "GO_SOURCE_INFORMATION"
        if passed and rank == 643
        else "CLOSE_SOURCE_COLLAPSE"
    )
    result = {
        "schema": "q24-fullplane-lambda3-innovation-s0-result-v1",
        "status": "COMPLETE_TARGET_BLIND_SOURCE_ONLY",
        "decision": decision,
        "role": "observer_fit",
        "role_rank": rank,
        "mlp_id": int(row["mlp_id"]),
        "capture_post_layers": list(CAPTURE_LAYERS),
        "source_association": {
            "host_return": "three-channel groups over channels 0:17, exactly frozen base",
            "captured_shell": "three-channel groups over channels 17:35",
            "high_total_returned_or_persisted": False,
        },
        "parity": {
            "host": parity[0],
            "native_raw20": parity[1],
            "native_sigma": parity[2],
            "base_sha256": [array_sha256(value) for value in base_values],
            "split_source_sha256": [array_sha256(value) for value in source_values],
        },
        "meter": {
            "phase2_budget": PHASE2_BUDGET,
            "base_flops": base_flops,
            "source_flops": source_flops,
            "source_added_flops": added_flops,
            "source_added_flop_cap": SOURCE_ADDED_FLOP_CAP,
            "base_wall_seconds": base_wall,
            "source_wall_seconds": source_wall,
        },
        "rss": {
            "process_peak_bytes": peak,
            "process_cap_bytes": PROCESS_RSS_CAP,
            "diagnostic_may_materialize_4x35_fp64": True,
            "proves_production_512mib_increment": False,
        },
        "layers": layer_rows,
        "gates": gates,
        "all_gates_pass": passed,
        "firewall": {
            "targets_read": 0,
            "observer_selection_opened": 0,
            "training_runs": 0,
            "high_host_outputs": 0,
            "full_oof_runs": 0,
            "packages": 0,
            "submissions": 0,
            "rank35_pass_authorizes_targets": False,
        },
        "authority": {
            **store_hashes,
            "base_source_sha256": sha256_file(BASE_SOURCE),
            "lambda3_source_sha256": sha256_file(LAMBDA3_SOURCE),
            "derivation_sha256": sha256_file(DERIVATION),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "runner_sha256": sha256_file(RUNNER),
            "release_sha256": sha256_file(RELEASE),
            "release_review": release.get("review"),
            "validated_release_payload": release,
        },
    }
    atomic_json(output_path, result)
    return {"output": str(output_path), "decision": decision, "all_gates_pass": passed}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rank", type=int, default=35)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(static_receipt(), indent=2, sort_keys=True))
        return
    print(json.dumps(execute(args.rank), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
