#!/usr/bin/env python3
"""One-use, target-free small-width source probe for frozen Q24 CS3B."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from whestbench.domain import MLP


sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SOURCE = HERE / "estimator.py"
SOURCE_SHA256 = "49060ee223137d98fc2b43d94e4d5e5d99290c7289bccd3043e3e078775b0aaa"
SOURCE_HOLD = HERE / "SOURCE_HOLD_V1.json"
SOURCE_HOLD_SHA256 = "ac29611a20280de259c7ab2f970d865b09ca1c417d1ac09d64aeab5a89d23bb5"
PARENT = ROOT / "fullbudget_k21_k4_closure_q24_connected_shutdown_v2" / "estimator.py"
PARENT_SHA256 = "4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779"
FIXTURE_HELPER = ROOT / "q24_two_point_moment_repair_probe_v1" / "run_probe.py"
FIXTURE_HELPER_SHA256 = "2c37c363155ac2b709de00a0d7a949a33a43106592e0df88daa7509bf2468b2d"
PREREGISTER = HERE / "SOURCE_PROBE_PREREGISTER_V1.md"
PREREGISTER_SHA256 = "588b7a56721194947cd46924ac1dda81b72009c503545564361b0161b7dfeec3"
STATIC_BINDING = HERE / "SOURCE_PROBE_STATIC_BINDING_V1.json"
STATIC_BINDING_SHA256 = "9858e4720fe24911c7ae4e83701efb8d281e094e484598c623b62146a4a3658e"
STATIC_AUDIT = (
    ROOT / "q24_cs3b_hybrid_exact_lowrank_pairing_covariance_design_v1"
    / "INDEPENDENT_STATIC_SOURCE_AUDIT_CS3B_V1.json"
)
STATIC_AUDIT_SHA256 = "27e5d66752520ebf287f7bc362330489849816b3d8cec0a1c7628aa931ca6c6c"
RELEASE = HERE / "SOURCE_PROBE_RELEASE_V1.json"
RELEASE_AUDIT = HERE / "INDEPENDENT_SOURCE_PROBE_RELEASE_AUDIT_CS3B_V1.json"
OUTPUT_DIR = HERE / "runs" / "source_probe_v1"
CLAIM = OUTPUT_DIR / "O_EXCL_CLAIM.json"
OUTPUT = OUTPUT_DIR / "results.json"
PROBE_BUDGET = 10**15
SEED = 20260824

EXPECTED_STATIC_STATUS = (
    "PASS_INDEPENDENT_STATIC_SOURCE_AUDIT__SOURCE_PROBE_RESULT_NULL__NO_EXECUTION_AUTHORITY"
)
EXPECTED_RELEASE_STATUS = (
    "AUTHORIZED_ONE_USE_TARGET_FREE_SMALLWIDTH_SOURCE_PROBE__NO_FULLWIDTH_OR_METER"
)
EXPECTED_RELEASE_AUDIT_STATUS = (
    "PASS_INDEPENDENT_SOURCE_PROBE_RUNNER_AND_RELEASE_AUDIT__ONE_USE_EXECUTION_PERMITTED"
)
EXPECTED_CERTIFICATE_LEDGER = {
    "nextafter": 57,
    "abs": 23,
    "multiply": 35,
    "add": 28,
    "subtract": 5,
    "negative": 2,
    "less": 2,
    "bitwise_or": 1,
    "zeros_like": 1,
}
CERTIFICATE_COMMON_NAMES = (
    "zero_error",
    "raw1_squared",
    "error_raw1_squared",
    "variance",
    "error_variance",
    "raw2_squared",
    "error_raw2_squared",
    "a_left",
    "error_a_left",
    "margin_a",
    "error_margin_a",
    "raw1_raw2",
    "error_raw1_raw2",
    "centered_21",
    "error_centered_21",
    "raw4_minus_raw2_squared",
    "error_raw4_minus_raw2_squared",
    "b_left",
    "error_b_left",
    "centered_21_squared",
    "error_centered_21_squared",
    "margin_b",
    "error_margin_b",
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest_array(value) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    hasher = hashlib.sha256()
    hasher.update(array.dtype.str.encode("ascii"))
    hasher.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    hasher.update(memoryview(array).cast("B"))
    return hasher.hexdigest()


def load(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load frozen module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def operation_rows(budget) -> dict[str, dict[str, int]]:
    return {
        name: {
            "calls": int(row.get("calls", 0)),
            "flops": int(row.get("flop_cost", 0)),
        }
        for name, row in sorted(
            budget.summary_dict(by_namespace=False).get("operations", {}).items()
        )
    }


def operation_count(budget) -> int:
    return sum(row["calls"] for row in operation_rows(budget).values())


def seal_json_o_excl(path: Path, value: object) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def preflight() -> tuple[dict, str, str]:
    expected = {
        SOURCE: SOURCE_SHA256,
        SOURCE_HOLD: SOURCE_HOLD_SHA256,
        PARENT: PARENT_SHA256,
        FIXTURE_HELPER: FIXTURE_HELPER_SHA256,
        PREREGISTER: PREREGISTER_SHA256,
        STATIC_BINDING: STATIC_BINDING_SHA256,
        STATIC_AUDIT: STATIC_AUDIT_SHA256,
    }
    mismatches = {
        str(path): {"expected": value, "observed": digest(path)}
        for path, value in expected.items()
        if digest(path) != value
    }
    if mismatches:
        raise RuntimeError(f"immutable input hash mismatch: {mismatches}")
    static = json.loads(STATIC_AUDIT.read_text())
    if static.get("status") != EXPECTED_STATIC_STATUS:
        raise RuntimeError("independent static source audit is not exact PASS")
    release = json.loads(RELEASE.read_text())
    release_sha = digest(RELEASE)
    if release.get("status") != EXPECTED_RELEASE_STATUS:
        raise RuntimeError("one-use source-probe release status mismatch")
    bindings = release.get("bindings", {})
    required = {
        "candidate_estimator_sha256": SOURCE_SHA256,
        "candidate_source_hold_sha256": SOURCE_HOLD_SHA256,
        "source_probe_preregister_sha256": PREREGISTER_SHA256,
        "source_probe_static_binding_sha256": STATIC_BINDING_SHA256,
        "independent_static_source_audit_sha256": STATIC_AUDIT_SHA256,
        "runner_sha256": digest(Path(__file__)),
        "output_path": "runs/source_probe_v1/results.json",
        "claim_path": "runs/source_probe_v1/O_EXCL_CLAIM.json",
    }
    if any(bindings.get(key) != value for key, value in required.items()):
        raise RuntimeError("one-use release binding mismatch")
    release_audit = json.loads(RELEASE_AUDIT.read_text())
    release_audit_sha = digest(RELEASE_AUDIT)
    if release_audit.get("status") != EXPECTED_RELEASE_AUDIT_STATUS:
        raise RuntimeError("independent runner/release audit status mismatch")
    audit_bindings = release_audit.get("bindings", {})
    audit_required = {
        "candidate_estimator_sha256": SOURCE_SHA256,
        "candidate_source_hold_sha256": SOURCE_HOLD_SHA256,
        "source_probe_preregister_sha256": PREREGISTER_SHA256,
        "source_probe_static_binding_sha256": STATIC_BINDING_SHA256,
        "independent_static_source_audit_sha256": STATIC_AUDIT_SHA256,
        "runner_sha256": digest(Path(__file__)),
        "author_release_sha256": release_sha,
        "output_path": "runs/source_probe_v1/results.json",
        "claim_path": "runs/source_probe_v1/O_EXCL_CLAIM.json",
    }
    if any(audit_bindings.get(key) != value for key, value in audit_required.items()):
        raise RuntimeError("independent runner/release audit binding mismatch")
    if OUTPUT_DIR.exists() or CLAIM.exists() or OUTPUT.exists():
        raise FileExistsError(f"immutable output namespace already exists: {OUTPUT_DIR}")
    return release, release_sha, release_audit_sha


def make_claim(release_sha: str, release_audit_sha: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    claim = {
        "schema": "q24-cs3b-source-probe-o-excl-claim-v1",
        "claimed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "CLAIMED_BEFORE_CANDIDATE_IMPORT_OR_FIXTURE_CONSTRUCTION",
        "bindings": {
            "candidate_estimator_sha256": SOURCE_SHA256,
            "candidate_source_hold_sha256": SOURCE_HOLD_SHA256,
            "preregister_sha256": PREREGISTER_SHA256,
            "static_binding_sha256": STATIC_BINDING_SHA256,
            "independent_static_audit_sha256": STATIC_AUDIT_SHA256,
            "runner_sha256": digest(Path(__file__)),
            "release_sha256": release_sha,
            "independent_runner_release_audit_sha256": release_audit_sha,
            "result_path": "results.json",
        },
        "candidate_imports_before_claim": 0,
        "fixture_constructions_before_claim": 0,
    }
    seal_json_o_excl(CLAIM, claim)


def certificate_roster() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    boundary = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
    rows.append(tuple(boundary))
    for index in range(4):
        for direction in (-np.inf, np.inf):
            neighbor = boundary.copy()
            neighbor[index] = np.nextafter(neighbor[index], direction)
            rows.append(tuple(neighbor))
    rows.extend(
        [
            (0.0, 0.0, 0.0, 0.0),
            (-0.0, 0.0, -0.0, 0.0),
            (0.0, -0.0, 0.0, -0.0),
            (-0.0, -0.0, -0.0, -0.0),
        ]
    )
    tiny = np.nextafter(np.float64(0.0), np.float64(np.inf))
    rows.extend(
        [
            (tiny, tiny, tiny, tiny),
            (-tiny, tiny, -tiny, tiny),
            (tiny, -tiny, tiny, -tiny),
        ]
    )
    for mean in (2.0**-200, 2.0**-40, 2.0**40, 1.0e30):
        rows.append((mean, 2.0 * mean**2, 4.0 * mean**3, 8.0 * mean**4))
    rng = np.random.default_rng(SEED)
    while len(rows) < 9 + 4 + 3 + 4 + 256:
        coefficients = rng.standard_normal(4)
        exponents = rng.integers(-120, 121, size=4)
        row = coefficients * np.exp2(exponents)
        if bool(np.isfinite(row).all()):
            rows.append(tuple(float(value) for value in row))
    matrix = np.asarray(rows, dtype=np.float64)
    return tuple(matrix[:, index] for index in range(4))


def capture_certificate(module, values) -> dict:
    captured: dict[str, np.ndarray] = {}
    code = module._certified_invalid_marginals.__code__

    def tracer(frame, event, _argument):
        if frame.f_code is code and event == "return":
            for name in CERTIFICATE_COMMON_NAMES:
                if name in frame.f_locals:
                    captured[name] = np.asarray(frame.f_locals[name]).copy()
        return tracer

    tracked = tuple(fnp.asarray(value, dtype=fnp.float64) for value in values)
    previous = sys.gettrace()
    sys.settrace(tracer)
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            invalid = module._certified_invalid_marginals(*tracked)
    finally:
        sys.settrace(previous)
    operations = operation_rows(budget)
    ledger = {name: operations.get(name, {}).get("calls", 0)
              for name in EXPECTED_CERTIFICATE_LEDGER}
    return {
        "invalid": np.asarray(invalid).copy(),
        "invalid_sha256": digest_array(invalid),
        "named": captured,
        "named_sha256": {name: digest_array(value) for name, value in captured.items()},
        "ledger": ledger,
        "total_calls": operation_count(budget),
        "flops": int(budget.flops_used),
        "operations": operations,
    }


def certificate_probe(parent_module, candidate_module) -> dict:
    values = certificate_roster()
    parent = capture_certificate(parent_module, values)
    candidate = capture_certificate(candidate_module, values)
    common_names_exact = set(parent["named"]) == set(CERTIFICATE_COMMON_NAMES)
    common_names_exact = common_names_exact and set(candidate["named"]) == set(
        CERTIFICATE_COMMON_NAMES
    )
    intermediate_equal = common_names_exact and all(
        parent["named_sha256"][name] == candidate["named_sha256"][name]
        for name in CERTIFICATE_COMMON_NAMES
    )
    mask_equal = parent["invalid_sha256"] == candidate["invalid_sha256"]
    ledger_gate = (
        parent["ledger"] == EXPECTED_CERTIFICATE_LEDGER
        and candidate["ledger"] == EXPECTED_CERTIFICATE_LEDGER
        and parent["total_calls"] == 154
        and candidate["total_calls"] == 154
    )
    return {
        "roster_size": int(values[0].shape[0]),
        "input_sha256": [digest_array(value) for value in values],
        "parent": {key: value for key, value in parent.items() if key != "named"},
        "candidate": {key: value for key, value in candidate.items() if key != "named"},
        "common_named_intermediate_set_exact": common_names_exact,
        "named_intermediate_bytes_equal": intermediate_equal,
        "invalid_mask_bytes_equal": mask_equal,
        "exact_154_call_ledgers": ledger_gate,
        "pass": bool(common_names_exact and intermediate_equal and mask_equal and ledger_gate),
    }


def lowrank_cost(rank: int, width: int) -> int:
    return (
        16 * rank * width * width
        + 4 * rank * rank * width
        + 8 * width * width
        + 20 * rank * width
        + 8 * rank * rank
        + 2 * width
    )


def dense_cost(width: int) -> int:
    return 4 * width**3 + 8 * width**2


def selected_mode(rank: int, width: int) -> str:
    return "LOWRANK64" if lowrank_cost(rank, width) <= dense_cost(width) else "DENSE32"


def nonsymmetric_arrays(width: int, rank: int):
    rng = np.random.default_rng(np.random.SeedSequence([SEED, width, rank]))
    covariance = rng.standard_normal((width, width), dtype=np.float64)
    covariance += np.diag(np.linspace(0.25, 1.25, width, dtype=np.float64))
    weight = rng.standard_normal((width, width), dtype=np.float64) / math.sqrt(width)
    invalid = np.zeros(width, dtype=bool)
    order = rng.permutation(width)
    invalid[order[:rank]] = True
    return covariance, weight, invalid


def gamma_value(count: int, unit: float) -> float:
    product = count * unit
    return product / (1.0 - product)


def lowrank_bound(covariance, weight, selected_covariance, rank: int):
    width = covariance.shape[0]
    gamma = gamma_value(32 * width + 16 * rank + 16, 2.0**-53)
    scale = (
        (np.abs(weight).T @ np.abs(covariance)) @ np.abs(weight)
        + (np.abs(weight).T @ np.abs(selected_covariance)) @ np.abs(weight)
        + 1.0
    )
    return gamma * scale


def dense_bound(stored32, weight32, first32):
    width = stored32.shape[0]
    weight64 = weight32.astype(np.float64)
    stored64 = stored32.astype(np.float64)
    first64 = first32.astype(np.float64)
    gamma = gamma_value(width, 2.0**-24)
    absolute_weight = np.abs(weight64)
    first_error = gamma * (absolute_weight.T @ np.abs(stored64))
    reference_scale = (absolute_weight.T @ np.abs(stored64)) @ absolute_weight
    return (
        first_error @ absolute_weight
        + gamma * (np.abs(first64) @ absolute_weight)
        + 64.0 * 2.0**-52 * (reference_scale + 1.0)
    )


def localized_resource_row(width: int, rank: int) -> dict:
    covariance_np, weight_np, invalid_np = nonsymmetric_arrays(width, rank)
    valid_np = ~invalid_np
    valid_plane_np = valid_np[:, None] & valid_np[None, :]
    covariance = fnp.asarray(covariance_np, dtype=fnp.float64)
    weight64 = fnp.asarray(weight_np, dtype=fnp.float64)
    weight32 = fnp.asarray(weight_np, dtype=fnp.float32)
    weight32_t = fnp.transpose(weight32)
    valid = fnp.asarray(valid_np, dtype=fnp.bool_)
    valid_plane = fnp.asarray(valid_plane_np, dtype=fnp.bool_)
    pre_left_np = weight_np.T @ covariance_np
    pre_covariance_np = pre_left_np @ weight_np
    pre_left = fnp.asarray(pre_left_np, dtype=fnp.float64)
    pre_covariance = fnp.asarray(pre_covariance_np, dtype=fnp.float64)
    mode = selected_mode(rank, width)
    with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
        localized_invalid = fnp.invert(valid)
        indices = fnp.flatnonzero(localized_invalid)
        if mode == "LOWRANK64":
            current = fnp.where(valid_plane, covariance, 0.0)
            _current_variance = fnp.diag(current)
            u = weight64[indices, :]
            crows = covariance[indices, :]
            g = crows[:, indices]
            a = pre_left[:, indices]
            b = crows @ weight64
            u_t = fnp.transpose(u)
            left_correction = u_t @ b
            right_correction = a @ u
            h = g @ u
            joint_correction = u_t @ h
            pushed = ((pre_covariance - left_correction) - right_correction) + joint_correction
            _pushed_variance = fnp.diag(pushed)
            first32_np = None
        else:
            stored32 = fnp.where(
                valid_plane, fnp.asarray(covariance, dtype=fnp.float32), 0.0
            )
            current = fnp.asarray(stored32, dtype=fnp.float64)
            _current_variance = fnp.diag(current)
            first32 = weight32_t @ stored32
            pushed32 = first32 @ weight32
            pushed = fnp.asarray(pushed32, dtype=fnp.float64)
            _pushed_variance = fnp.diag(pushed)
            first32_np = np.asarray(first32).copy()
    pushed_np = np.asarray(pushed).copy()
    current_np = np.asarray(current).copy()
    expected_flops = lowrank_cost(rank, width) if mode == "LOWRANK64" else dense_cost(width) + 2 * width
    expected_calls = 18 if mode == "LOWRANK64" else 10
    row = {
        "width": width,
        "rank": rank,
        "mode": mode,
        "invalid_index_sha256": digest_array(indices),
        "flops": int(budget.flops_used),
        "expected_flops": expected_flops,
        "calls": operation_count(budget),
        "expected_calls": expected_calls,
        "resource_exact": int(budget.flops_used) == expected_flops
        and operation_count(budget) == expected_calls,
        "operations": operation_rows(budget),
    }
    if mode == "LOWRANK64":
        reference = (weight_np.T @ current_np) @ weight_np
        bound = lowrank_bound(covariance_np, weight_np, current_np, rank)
        difference = np.abs(pushed_np - reference)
        row.update(
            {
                "direct_reference_sha256": digest_array(reference),
                "candidate_sha256": digest_array(pushed_np),
                "maximum_absolute_difference": float(np.max(difference)),
                "maximum_bound": float(np.max(bound)),
                "bound_violation_count": int(np.count_nonzero(difference > bound)),
            }
        )
    else:
        stored32_np = current_np.astype(np.float32)
        weight32_np = weight_np.astype(np.float32)
        reference = (weight32_np.T.astype(np.float64) @ stored32_np.astype(np.float64)) @ weight32_np.astype(np.float64)
        bound = dense_bound(stored32_np, weight32_np, first32_np)
        difference = np.abs(pushed_np - reference)
        row.update(
            {
                "stored_fp32_reference_sha256": digest_array(reference),
                "candidate_sha256": digest_array(pushed_np),
                "maximum_absolute_difference": float(np.max(difference)),
                "maximum_bound": float(np.max(bound)),
                "bound_violation_count": int(np.count_nonzero(difference > bound)),
            }
        )
    return row


def signed_permutation_probe() -> dict:
    width = 6
    rows = []
    permutation = np.asarray([2, 5, 1, 4, 0, 3])
    signs = np.asarray([1.0, -1.0, 1.0, 1.0, -1.0, -1.0])
    weight = np.zeros((width, width), dtype=np.float64)
    weight[np.arange(width), permutation] = signs
    grid = np.arange(width * width, dtype=np.float64).reshape(width, width)
    covariance = (grid + 2.0 * grid.T + 1.0) / 16.0
    for rank in range(1, width + 1):
        if selected_mode(rank, width) != "LOWRANK64":
            continue
        invalid = np.zeros(width, dtype=bool)
        invalid[np.arange(rank)] = True
        selected = np.where((~invalid)[:, None] & (~invalid)[None, :], covariance, 0.0)
        left = weight.T @ covariance
        q = left @ weight
        indices = np.flatnonzero(invalid)
        u = weight[indices, :]
        crows = covariance[indices, :]
        result = ((q - u.T @ (crows @ weight)) - left[:, indices] @ u) + u.T @ ((crows[:, indices]) @ u)
        mapped = permutation[indices]
        violation = int(np.count_nonzero(result[mapped, :])) + int(np.count_nonzero(result[:, mapped]))
        rows.append({"rank": rank, "mapped_indices": mapped.tolist(), "exact_zero_violation_count": violation, "result_sha256": digest_array(result), "direct_sha256": digest_array(weight.T @ selected @ weight)})
    return {"rows": rows, "all_exact_mapped_zeros": bool(rows) and all(row["exact_zero_violation_count"] == 0 for row in rows)}


def resource_and_algebra_probe() -> dict:
    rows = [localized_resource_row(width, rank)
            for width in (4, 5, 6, 8)
            for rank in range(1, width + 1)]
    witnesses = [
        (5, 1, "DENSE32"),
        (6, 1, "LOWRANK64"),
        (128, 29, "LOWRANK64"),
        (128, 30, "DENSE32"),
        (1024, 241, "LOWRANK64"),
        (1024, 242, "DENSE32"),
    ]
    witness_rows = [
        {
            "width": width,
            "rank": rank,
            "expected": expected,
            "observed": selected_mode(rank, width),
            "lowrank_cost": lowrank_cost(rank, width),
            "dense_cost": dense_cost(width),
            "pass": selected_mode(rank, width) == expected,
        }
        for width, rank, expected in witnesses
    ]
    signed = signed_permutation_probe()
    return {
        "rows": rows,
        "integer_witnesses": witness_rows,
        "signed_permutation": signed,
        "both_modes_exercised": {row["mode"] for row in rows} == {"LOWRANK64", "DENSE32"},
        "all_resource_exact": all(row["resource_exact"] for row in rows),
        "all_forward_bounds": all(row["bound_violation_count"] == 0 for row in rows),
        "all_integer_witnesses": all(row["pass"] for row in witness_rows),
        "pass": all(row["resource_exact"] and row["bound_violation_count"] == 0 for row in rows)
        and all(row["pass"] for row in witness_rows)
        and signed["all_exact_mapped_zeros"],
    }


def incident_count(vector_or_plane, invalid, plane: bool) -> int:
    array = np.asarray(vector_or_plane)
    if plane:
        invalid_area = invalid[:, None] | invalid[None, :]
        return int(np.count_nonzero(array[invalid_area]))
    return int(np.count_nonzero(array[invalid]))


def off_diagonal(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    np.fill_diagonal(result, 0.0)
    return result


def reconstruction_bound(width: int, scale: float) -> float:
    return gamma_value(64 * width, 2.0**-24) * (1.0 + scale)


def snapshot(value) -> np.ndarray:
    return np.asarray(value).copy()


def scale_comparison(parent: np.ndarray, child: np.ndarray) -> dict:
    parent64 = parent.astype(np.float64)
    delta = child.astype(np.float64) - parent64
    parent_rms = float(np.sqrt(np.mean(parent64 * parent64)))
    disagreement_rms = float(np.sqrt(np.mean(delta * delta)))
    parent_max = float(np.max(np.abs(parent64)))
    disagreement_max = float(np.max(np.abs(delta)))
    return {
        "all_layer_mse": float(np.mean(delta * delta)),
        "final_layer_mse": float(np.mean(delta[-1] * delta[-1])),
        "maximum_absolute_difference": disagreement_max,
        "parent_rms": parent_rms,
        "disagreement_rms": disagreement_rms,
        "parent_maximum_absolute_value": parent_max,
        "rms_scale_gate": disagreement_rms <= parent_rms if parent_rms > 0.0 else disagreement_rms == 0.0,
        "maximum_scale_gate": disagreement_max <= parent_max if parent_max > 0.0 else disagreement_max == 0.0,
    }


def plain_run(path: Path, tag: str, label: str, weights) -> dict:
    module = load(tag, path)
    mlp = MLP(width=int(weights[0].shape[0]), depth=16, weights=weights, seed=0, name=label)
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            output = module._predict(mlp)
        array = snapshot(output)
        return {
            "outcome": "FINITE" if bool(np.isfinite(array).all()) else "NONFINITE",
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "output_sha256": digest_array(array),
            "flops": int(budget.flops_used),
            "calls": operation_count(budget),
            "_array": array,
        }
    except Exception as exc:
        return {"outcome": "EXCEPTION", "exception_type": type(exc).__name__, "exception_message": str(exc), "_array": None}


def candidate_run(label: str, weights, forced_one_invalid: bool = False) -> dict:
    module = load(f"q24_cs3b_{label}_{time.time_ns()}", SOURCE)
    holder: dict[str, object] = {
        "reset": None,
        "cp_post": None,
        "provider": None,
        "pending": None,
        "push_index": 0,
    }
    reset_records: list[dict] = []
    push_records: list[dict] = []
    original_certificate = module._certified_invalid_marginals
    original_full_reset = module._full_relu_reset
    original_cp_post = module._cp_post_read
    original_cp4_reset = module._cp4_reset
    original_reset_stars = module._reset_stars
    original_push = module._push_quartic
    original_push_diagonal = module._push_quartic_diagonal

    def observed_certificate(*args, **kwargs):
        if forced_one_invalid:
            mask = np.zeros(int(args[0].shape[0]), dtype=bool)
            mask[0] = True
            return fnp.asarray(mask, dtype=fnp.bool_)
        return original_certificate(*args, **kwargs)

    def observed_full_reset(*args, **kwargs):
        value = original_full_reset(*args, **kwargs)
        holder["reset"] = {
            "covariance": snapshot(value[1]),
            "valid": snapshot(value[12]).astype(bool),
            "valid_plane": snapshot(value[13]).astype(bool),
        }
        return value

    def observed_cp_post(*args, **kwargs):
        value = original_cp_post(*args, **kwargs)
        holder["cp_post"] = snapshot(value)
        return value

    def observed_cp4_reset(*args, **kwargs):
        value = original_cp4_reset(*args, **kwargs)
        holder["provider"] = tuple(snapshot(item) for item in value[2])
        return value

    def capture_push(args, value, final: bool):
        index = int(holder["push_index"])
        holder["push_index"] = index + 1
        selected = snapshot(args[3])
        selected_variance = snapshot(args[4])
        pending = holder.get("pending")
        record = {
            "layer_index": index + 1,
            "final_diagonal": final,
            "selected_covariance_sha256": digest_array(selected),
            "selected_variance_sha256": digest_array(selected_variance),
            "diag_bytes_equal": digest_array(np.diag(selected)) == digest_array(selected_variance),
            "mode": "NOMINAL" if pending is None else pending["mode"],
        }
        if pending is not None:
            weight64 = np.asarray(weights[index + 1], dtype=np.float64)
            covariance = pending["covariance"]
            invalid = pending["invalid"]
            mode = pending["mode"]
            if mode == "LOWRANK64":
                indices = np.flatnonzero(invalid)
                left = weight64.T @ covariance
                q = left @ weight64
                u = weight64[indices, :]
                crows = covariance[indices, :]
                expected = ((q - u.T @ (crows @ weight64)) - left[:, indices] @ u) + u.T @ (crows[:, indices] @ u)
                direct = weight64.T @ np.where((~invalid)[:, None] & (~invalid)[None, :], covariance, 0.0) @ weight64
                bound = lowrank_bound(covariance, weight64, np.where((~invalid)[:, None] & (~invalid)[None, :], covariance, 0.0), int(np.count_nonzero(invalid)))
                record["source_spelling_bytes_equal"] = digest_array(selected) == digest_array(expected)
                record["direct_bound_violation_count"] = int(np.count_nonzero(np.abs(selected - direct) > bound))
            elif mode == "DENSE32":
                weight32 = np.asarray(weights[index + 1], dtype=np.float32)
                stored32 = pending["stored32"]
                first32 = weight32.T @ stored32
                expected32 = first32 @ weight32
                expected = expected32.astype(np.float64)
                reference = (weight32.T.astype(np.float64) @ stored32.astype(np.float64)) @ weight32.astype(np.float64)
                bound = dense_bound(stored32, weight32, first32)
                record["source_spelling_bytes_equal"] = digest_array(selected) == digest_array(expected)
                record["direct_bound_violation_count"] = int(np.count_nonzero(np.abs(selected - reference) > bound))
            else:
                expected = (weight64.T @ covariance) @ weight64
                record["source_spelling_bytes_equal"] = digest_array(selected) == digest_array(expected)
                record["direct_bound_violation_count"] = 0
            if np.all(np.sum(np.abs(weight64) != 0.0, axis=0) == 1) and np.all(np.sum(np.abs(weight64) != 0.0, axis=1) == 1) and np.all(np.isin(weight64[weight64 != 0.0], (-1.0, 1.0))):
                mapped = np.argmax(np.abs(weight64[np.flatnonzero(invalid), :]), axis=1)
                record["mapped_exact_zero_violation_count"] = int(np.count_nonzero(selected[mapped, :])) + int(np.count_nonzero(selected[:, mapped]))
        push_records.append(record)

    def observed_push(*args, **kwargs):
        value = original_push(*args, **kwargs)
        capture_push(args, value, False)
        return value

    def observed_push_diagonal(*args, **kwargs):
        value = original_push_diagonal(*args, **kwargs)
        capture_push(args, value, True)
        return value

    def observed_reset_stars(*args, **kwargs):
        value = original_reset_stars(*args, **kwargs)
        state = holder["reset"]
        if state is None or holder["cp_post"] is None or holder["provider"] is None:
            raise RuntimeError("observer ordering invariant failed")
        valid = state["valid"]
        invalid = ~valid
        valid_plane = state["valid_plane"]
        width = int(valid.shape[0])
        rank = int(np.count_nonzero(invalid))
        mode = "NOMINAL" if rank == 0 else selected_mode(rank, width)
        covariance = state["covariance"]
        selected = snapshot(args[5])
        selected_variance = snapshot(args[6])
        if mode == "NOMINAL":
            expected_current = covariance
            stored32 = None
        elif mode == "LOWRANK64":
            expected_current = np.where(valid_plane, covariance, 0.0)
            stored32 = None
        else:
            stored32 = np.where(valid_plane, covariance.astype(np.float32), np.float32(0.0))
            expected_current = stored32.astype(np.float64)
        k21 = snapshot(args[0])
        cp_post = snapshot(args[1])
        k4 = snapshot(args[2])
        k31 = snapshot(args[3])
        k22 = snapshot(args[4])
        provider_d, provider_a, provider_b = holder["provider"]
        cubic, quartic, alpha_value = (snapshot(item) for item in value)
        alpha = float(np.asarray(alpha_value))
        pairing_diagonal = 3.0 * alpha * selected_variance * selected_variance
        pairing_k31 = 3.0 * alpha * selected_variance[:, None] * selected
        pairing_plane = off_diagonal(np.outer(selected_variance, selected_variance) + 2.0 * selected * selected)
        reconstructed_k21 = cubic.astype(np.float64) + cp_post
        reconstructed_k4 = np.diag(quartic).astype(np.float64) + pairing_diagonal + provider_d
        reconstructed_k31 = off_diagonal(quartic.astype(np.float64)) + off_diagonal(pairing_k31) + provider_a
        adjusted_k22 = 0.5 * ((k22 - provider_b) + (k22 - provider_b).T)
        residual_alpha = off_diagonal(adjusted_k22 - alpha * pairing_plane)
        residual_zero = off_diagonal(adjusted_k22)
        scale = max(float(np.max(np.abs(k21))), float(np.max(np.abs(k4))), float(np.max(np.abs(k31))), 1.0)
        tolerance = reconstruction_bound(width, scale)
        target_zero_count = incident_count(k21, invalid, True) + incident_count(k4, invalid, False) + incident_count(k31, invalid, True) + incident_count(k22, invalid, True)
        carrier_zero_count = incident_count(cp_post, invalid, True) + incident_count(provider_d, invalid, False) + incident_count(provider_a, invalid, True) + incident_count(provider_b, invalid, True) + incident_count(cubic, invalid, True) + incident_count(quartic, invalid, True)
        aggregate_zero_count = incident_count(reconstructed_k21, invalid, True) + incident_count(reconstructed_k4, invalid, False) + incident_count(reconstructed_k31, invalid, True)
        reconstruction_error = max(float(np.max(np.abs(reconstructed_k21 - k21))), float(np.max(np.abs(reconstructed_k4 - k4))), float(np.max(np.abs(reconstructed_k31 - k31))))
        valid_values_alpha = residual_alpha[valid_plane]
        valid_values_zero = residual_zero[valid_plane]
        norm_alpha = float(np.linalg.norm(valid_values_alpha))
        norm_zero = float(np.linalg.norm(valid_values_zero))
        reset_records.append(
            {
                "invalid_count": rank,
                "invalid_mask_sha256": digest_array(invalid),
                "mode": mode,
                "selected_current_bytes_equal": digest_array(selected) == digest_array(expected_current),
                "selected_current_invalid_nonzero_count": incident_count(selected, invalid, True),
                "selected_variance_diag_bytes_equal": digest_array(np.diag(selected)) == digest_array(selected_variance),
                "target_incident_nonzero_count": target_zero_count,
                "carrier_and_star_incident_nonzero_count": carrier_zero_count,
                "aggregate_incident_nonzero_count": aggregate_zero_count,
                "maximum_reconstruction_error": reconstruction_error,
                "reconstruction_bound": tolerance,
                "reconstruction_bound_gate": reconstruction_error <= tolerance,
                "valid_valid_k22_residual_norm": norm_alpha,
                "alpha_zero_valid_valid_k22_residual_norm": norm_zero,
                "valid_valid_residual_gate": norm_alpha <= norm_zero + tolerance,
                "pairing_alpha": alpha,
                "pairing_alpha_bytes": digest_array(alpha_value),
            }
        )
        holder["pending"] = {
            "mode": mode,
            "covariance": covariance,
            "invalid": invalid,
            "stored32": stored32,
        }
        return value

    module._certified_invalid_marginals = observed_certificate
    module._full_relu_reset = observed_full_reset
    module._cp_post_read = observed_cp_post
    module._cp4_reset = observed_cp4_reset
    module._reset_stars = observed_reset_stars
    module._push_quartic = observed_push
    module._push_quartic_diagonal = observed_push_diagonal
    width = int(weights[0].shape[0])
    mlp = MLP(width=width, depth=16, weights=weights, seed=0, name=label)
    try:
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            output = module._predict(mlp)
        array = snapshot(output)
        outcome = "FINITE" if bool(np.isfinite(array).all()) else "NONFINITE"
        error = None
    except Exception as exc:
        array = None
        outcome = "EXCEPTION"
        error = {"type": type(exc).__name__, "message": str(exc)}
        budget = None
    modes = Counter(record["mode"] for record in reset_records)
    mode_mask_payload = [{"mode": record["mode"], "mask": record["invalid_mask_sha256"]} for record in reset_records]
    gate_records = reset_records + push_records
    result = {
        "outcome": outcome,
        "error": error,
        "shape": None if array is None else list(array.shape),
        "dtype": None if array is None else str(array.dtype),
        "output_sha256": None if array is None else digest_array(array),
        "flops": None if budget is None else int(budget.flops_used),
        "calls": None if budget is None else operation_count(budget),
        "reset_count": len(reset_records),
        "mode_counts": dict(sorted(modes.items())),
        "mode_mask_sha256": hashlib.sha256(json.dumps(mode_mask_payload, sort_keys=True).encode()).hexdigest(),
        "activation_count": sum(record["invalid_count"] for record in reset_records),
        "all_current_selected_bytes": all(record["selected_current_bytes_equal"] and record["selected_variance_diag_bytes_equal"] for record in reset_records),
        "all_exact_incident_zeros": all(record["target_incident_nonzero_count"] == 0 and record["carrier_and_star_incident_nonzero_count"] == 0 and record["aggregate_incident_nonzero_count"] == 0 and record["selected_current_invalid_nonzero_count"] == 0 for record in reset_records),
        "all_reconstruction_bounds": all(record["reconstruction_bound_gate"] and record["valid_valid_residual_gate"] for record in reset_records),
        "all_push_selected_law": all(record["diag_bytes_equal"] and record.get("source_spelling_bytes_equal", True) and record.get("direct_bound_violation_count", 0) == 0 and record.get("mapped_exact_zero_violation_count", 0) == 0 for record in push_records),
        "reset_records": reset_records,
        "push_records": push_records,
        "_array": array,
    }
    return result


def fixture_probe(helper) -> dict:
    rows = []
    for fixture_index, (label, weights) in enumerate(helper.fixture_rows()):
        parent = plain_run(PARENT, f"q24_cs3b_parent_{fixture_index}", label, weights)
        first = candidate_run(f"{label}_run1", weights)
        second = candidate_run(f"{label}_run2", weights)
        parent_array = parent.pop("_array")
        first_array = first.pop("_array")
        second_array = second.pop("_array")
        comparison = None
        if parent_array is not None and first_array is not None:
            comparison = scale_comparison(parent_array, first_array)
        deterministic = (
            first["output_sha256"] == second["output_sha256"]
            and first["mode_mask_sha256"] == second["mode_mask_sha256"]
        )
        no_activation_parity = (
            first["activation_count"] != 0
            or first["output_sha256"] == parent.get("output_sha256")
        )
        rows.append(
            {
                "fixture": label,
                "parent": parent,
                "candidate_run1": first,
                "candidate_run2": second,
                "finite_pair_disagreement": comparison,
                "two_run_determinism": deterministic,
                "no_activation_byte_parity": no_activation_parity,
            }
        )
    natural_lowrank = sum(row["candidate_run1"]["mode_counts"].get("LOWRANK64", 0) for row in rows)
    no_regression = all(not (row["parent"]["outcome"] == "FINITE" and row["candidate_run1"]["outcome"] != "FINITE") for row in rows)
    deterministic = all(row["two_run_determinism"] for row in rows)
    parity = all(row["no_activation_byte_parity"] for row in rows)
    scale = all(row["finite_pair_disagreement"] is None or (row["finite_pair_disagreement"]["rms_scale_gate"] and row["finite_pair_disagreement"]["maximum_scale_gate"]) for row in rows)
    carrier = all(row["candidate_run1"]["all_current_selected_bytes"] and row["candidate_run1"]["all_exact_incident_zeros"] and row["candidate_run1"]["all_reconstruction_bounds"] and row["candidate_run1"]["all_push_selected_law"] for row in rows)
    return {
        "rows": rows,
        "fixture_count": len(rows),
        "natural_lowrank_activation_count": natural_lowrank,
        "no_parent_finite_regression": no_regression,
        "two_run_determinism": deterministic,
        "no_activation_byte_parity": parity,
        "finite_pair_parent_scale_gate": scale,
        "carrier_and_reconstruction_gates": carrier,
        "pass": len(rows) == 10 and natural_lowrank > 0 and no_regression and deterministic and parity and scale and carrier,
    }


def forced_dense_probe() -> dict:
    width = 5
    eye = np.eye(width, dtype=np.float32)
    weights = [eye.copy() for _ in range(16)]
    result = candidate_run("forced_dense_width5_r1", weights, forced_one_invalid=True)
    result.pop("_array", None)
    return {
        "result": result,
        "expected_lowrank_cost": lowrank_cost(1, width),
        "expected_dense_cost": dense_cost(width),
        "pass": result["outcome"] == "FINITE"
        and result["reset_count"] == 14
        and result["mode_counts"].get("DENSE32", 0) == 14
        and result["mode_counts"].get("LOWRANK64", 0) == 0
        and result["all_current_selected_bytes"]
        and result["all_exact_incident_zeros"]
        and result["all_push_selected_law"],
    }


def null_face_probe(candidate_module) -> dict:
    width = 4
    identity = fnp.asarray(np.eye(width, dtype=bool), dtype=fnp.bool_)
    zero_plane = fnp.asarray(np.zeros((width, width)), dtype=fnp.float64)
    zero_vector = fnp.asarray(np.zeros(width), dtype=fnp.float64)
    k21 = zero_plane
    cp_post = zero_plane
    k4_np = np.asarray([2.0, 3.0, 4.0, 5.0])
    k31_np = np.arange(width * width, dtype=np.float64).reshape(width, width) / 8.0
    np.fill_diagonal(k31_np, k4_np)
    k22_np = np.arange(width * width, dtype=np.float64).reshape(width, width) / 16.0
    np.fill_diagonal(k22_np, k4_np)
    k4 = fnp.asarray(k4_np, dtype=fnp.float64)
    k31 = fnp.asarray(k31_np, dtype=fnp.float64)
    k22 = fnp.asarray(k22_np, dtype=fnp.float64)
    provider = (zero_vector, zero_plane, zero_plane)

    def run(covariance_np):
        covariance = fnp.asarray(covariance_np, dtype=fnp.float64)
        variance = fnp.diag(covariance)
        with flops.BudgetContext(flop_budget=PROBE_BUDGET, quiet=True) as budget:
            cubic, quartic, alpha = candidate_module._reset_stars(
                k21, cp_post, k4, k31, k22, covariance, variance,
                covariance * covariance, fnp.outer(variance, variance),
                provider, identity,
            )
        return snapshot(cubic), snapshot(quartic), snapshot(alpha), budget

    null_covariance = np.diag([1.0, 0.0, 0.0, 0.0]).astype(np.float64)
    ordinary_covariance = np.eye(width, dtype=np.float64)
    _null_cubic, null_quartic, null_alpha, null_budget = run(null_covariance)
    _ordinary_cubic, _ordinary_quartic, ordinary_alpha, ordinary_budget = run(ordinary_covariance)
    null_variance = np.diag(null_covariance)
    reconstructed_k4 = np.diag(null_quartic).astype(np.float64) + 3.0 * float(null_alpha) * null_variance**2
    reconstructed_k31 = off_diagonal(null_quartic.astype(np.float64)) + off_diagonal(3.0 * float(null_alpha) * null_variance[:, None] * null_covariance)
    tolerance = reconstruction_bound(width, float(np.max(np.abs(k31_np))))
    null_bytes = np.asarray(null_alpha, dtype=np.float64).tobytes()
    positive_zero = null_bytes == np.asarray(0.0, dtype=np.float64).tobytes()
    null_ops = operation_rows(null_budget)
    ordinary_ops = operation_rows(ordinary_budget)
    division_gate = null_ops.get("true_divide", {}).get("calls", 0) == 1 and ordinary_ops.get("true_divide", {}).get("calls", 0) == 1
    ledger_gate = operation_count(null_budget) == operation_count(ordinary_budget) and int(null_budget.flops_used) == int(ordinary_budget.flops_used)
    k22_residual = off_diagonal(0.5 * (k22_np + k22_np.T))
    reconstruction_error = max(float(np.max(np.abs(reconstructed_k4 - k4_np))), float(np.max(np.abs(reconstructed_k31 - off_diagonal(k31_np)))))
    return {
        "null_alpha_sha256": digest_array(null_alpha),
        "ordinary_alpha": float(ordinary_alpha),
        "positive_zero_alpha": positive_zero,
        "null_true_divide_calls": null_ops.get("true_divide", {}).get("calls", 0),
        "ordinary_true_divide_calls": ordinary_ops.get("true_divide", {}).get("calls", 0),
        "operation_ledger_equal": ledger_gate,
        "maximum_k4_k31_reconstruction_error": reconstruction_error,
        "reconstruction_bound": tolerance,
        "k22_residual_norm": float(np.linalg.norm(k22_residual)),
        "k22_residual_sha256": digest_array(k22_residual),
        "operations_null": null_ops,
        "operations_ordinary": ordinary_ops,
        "pass": positive_zero and division_gate and ledger_gate and reconstruction_error <= tolerance,
    }


def no_bytecode_debris() -> bool:
    for base in (HERE, STATIC_AUDIT.parent):
        for path in base.rglob("*"):
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
                return False
    return True


def run_probe() -> dict:
    flops.configure(symmetry_warnings=False)
    parent_module = load("q24_cs3b_certificate_parent", PARENT)
    candidate_module = load("q24_cs3b_certificate_candidate", SOURCE)
    helper = load("q24_cs3b_fixture_helper", FIXTURE_HELPER)
    certificate = certificate_probe(parent_module, candidate_module)
    resource = resource_and_algebra_probe()
    fixtures = fixture_probe(helper)
    forced_dense = forced_dense_probe()
    null_face = null_face_probe(candidate_module)
    debris_gate = no_bytecode_debris()
    passed = bool(
        certificate["pass"]
        and resource["pass"]
        and fixtures["pass"]
        and forced_dense["pass"]
        and null_face["pass"]
        and debris_gate
    )
    return {
        "schema": "q24-cs3b-target-free-smallwidth-source-probe-v1",
        "status": "PASS_SOURCE_PROBE__FULLWIDTH_METER_ACCURACY_BOARD_HOLD" if passed else "REJECT_SOURCE_PROBE",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bindings": {
            "candidate_estimator_sha256": SOURCE_SHA256,
            "candidate_source_hold_sha256": SOURCE_HOLD_SHA256,
            "parent_estimator_sha256": PARENT_SHA256,
            "fixture_helper_sha256": FIXTURE_HELPER_SHA256,
            "source_probe_preregister_sha256": PREREGISTER_SHA256,
            "source_probe_static_binding_sha256": STATIC_BINDING_SHA256,
            "independent_static_source_audit_sha256": STATIC_AUDIT_SHA256,
            "runner_sha256": digest(Path(__file__)),
            "release_sha256": digest(RELEASE),
            "independent_runner_release_audit_sha256": digest(RELEASE_AUDIT),
            "o_excl_claim_sha256": digest(CLAIM),
        },
        "certificate_equivalence": certificate,
        "resource_selector_and_nonsymmetric_algebra": resource,
        "ten_frozen_fixtures": fixtures,
        "forced_dense_control": forced_dense,
        "exact_null_control": null_face,
        "gates": {
            "certificate": certificate["pass"],
            "resource_and_algebra": resource["pass"],
            "ten_fixtures": fixtures["pass"],
            "forced_dense": forced_dense["pass"],
            "exact_null": null_face["pass"],
            "no_pycache_pyc_pyo": debris_gate,
            "source_probe_pass": passed,
        },
        "scope": {
            "candidate_private_predict_runs_only": true,
            "maximum_estimator_width": 128,
            "width1024_estimator_runs": 0,
            "literal_meter_runs": 0,
            "dataset_rows_targets_teachers_scores_private_state": 0,
            "packages": 0,
            "submissions": 0,
            "network_or_credentials": 0,
            "canonical_graph_edits": 0,
        },
    }


def main() -> int:
    release_sha = "UNREAD"
    try:
        _release, release_sha, release_audit_sha = preflight()
        make_claim(release_sha, release_audit_sha)
    except Exception as exc:
        print(f"preflight refused source probe: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3
    try:
        result = run_probe()
    except Exception as exc:
        result = {
            "schema": "q24-cs3b-target-free-smallwidth-source-probe-v1",
            "status": "REJECT_SOURCE_PROBE__UNCAUGHT_REQUIRED_CONTROL_EXCEPTION",
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bindings": {
                "candidate_estimator_sha256": SOURCE_SHA256,
                "candidate_source_hold_sha256": SOURCE_HOLD_SHA256,
                "source_probe_preregister_sha256": PREREGISTER_SHA256,
                "source_probe_static_binding_sha256": STATIC_BINDING_SHA256,
                "independent_static_source_audit_sha256": STATIC_AUDIT_SHA256,
                "runner_sha256": digest(Path(__file__)),
                "release_sha256": release_sha,
                "independent_runner_release_audit_sha256": digest(RELEASE_AUDIT),
                "o_excl_claim_sha256": digest(CLAIM),
            },
            "exception": {"type": type(exc).__name__, "message": str(exc)},
            "scope": {
                "width1024_estimator_runs": 0,
                "literal_meter_runs": 0,
                "dataset_rows_targets_teachers_scores_private_state": 0,
                "packages_submissions_network_graph_edits": 0,
            },
        }
    seal_json_o_excl(OUTPUT, result)
    print(json.dumps({"status": result["status"], "output": str(OUTPUT), "sha256": digest(OUTPUT)}, indent=2))
    return 0 if result["status"].startswith("PASS_SOURCE_PROBE") else 2


if __name__ == "__main__":
    raise SystemExit(main())
