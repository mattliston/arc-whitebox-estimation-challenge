#!/usr/bin/env python3
"""One-shot tiny exact-runtime discriminator for frozen P2R0C1D7R1.

This imports only the isolated staged kernel.  It never imports or calls an
estimator and opens no fixture, exact-shape operand, target, score, name,
board, common driver, production, submission, or network resource.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import resource
import statistics
import sys
import traceback

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D7 = P2 / "p2r0c1d7_cp3_staged_factorized_einsum_design_v1"
SOURCE = HERE / "staged_winograd_kernel.py"
MANIFEST = HERE / "ACCOUNTING_AND_RESOURCE_MANIFEST.json"
DESIGN = HERE / "DESIGN.md"
STATIC_VERIFIER = HERE / "verify_design_static.py"
D5_KERNEL_SMOKE = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1/KERNEL_SMOKE_RESULT.json"
D5_KERNEL_RUNNER = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1/run_kernel_smoke.py"
DUAL_LAUNCH_AUDIT = D7 / "INDEPENDENT_D7_DUAL_LAUNCH_FAILURE_AND_R1_ELIGIBILITY_AUDIT.md"
ATTEMPT_001_SEAL = D7 / "TINY_RUNTIME_LAUNCH_SEAL_ATTEMPT_001.json"
ATTEMPT_001_TRANSCRIPT = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_001_TRANSCRIPT.txt"
ATTEMPT_001_FAILURE = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_001_FAILURE.json"
ATTEMPT_002_SEAL = D7 / "TINY_RUNTIME_LAUNCH_SEAL.json"
ATTEMPT_002_TRANSCRIPT = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_002_TRANSCRIPT.txt"
ATTEMPT_002_FAILURE = D7 / "TINY_RUNTIME_LAUNCH_ATTEMPT_002_FAILURE.json"
SEAL = HERE / "D7R1_TINY_RUNTIME_LAUNCH_SEAL.json"
ATTEMPT_MARKER = HERE / "D7R1_TINY_ATTEMPT_MARKER.json"
BIND_PREASSERT = HERE / "D7R1_BIND_INVENTORY_PREASSERT.json"
EXCEPTION_RECEIPT = HERE / "D7R1_RUNTIME_EXCEPTION_RECEIPT.json"
RESULT = HERE / "D7R1_TINY_RUNTIME_RESULT.json"

EXPECTED_SOURCE = "81d14f2ca983c2820ab078d15a3a8a861427ac5a8b1b5e82828300256648bbe9"
EXPECTED_MANIFEST = "94ad1b2e89a58417bf91cb4fd844ac1d02674b58a2738b0218bf7cc63eb145ec"
EXPECTED_DESIGN = "3ccdefd80d69817f1ad8519124f2f0def2f03e95a5c96385d921edfdc97b1b2e"
EXPECTED_STATIC_VERIFIER = "27a067afcee05558f2d066bfc7ee594752dce8fa62a5c7166353b97ebed9a816"
EXPECTED_D5_KERNEL_SMOKE = "a50ca69aae9e4b39fca8d3de9014bcc7688af63a1f5ced6aab1f94de957235c9"
EXPECTED_D5_KERNEL_RUNNER = "44940602fc2f6fdb336911821de024bddb94da021a01ddf0ba9570e251bfb996"
EXPECTED_RECOVERY_AUTHORITIES = {
    DUAL_LAUNCH_AUDIT: "98db28ab6c80c5fd012fc569ed123cfbcf48d33fea7ec16a380b04be53560998",
    ATTEMPT_001_SEAL: "8aeb6adc651a8292dea39be53f6d3c01f66c0a4a9e82be257e05bc9253fb31fa",
    ATTEMPT_001_TRANSCRIPT: "0e244c00291c6a915f2129082df5bed2510f7ffae4df00c49a96086207281417",
    ATTEMPT_001_FAILURE: "5bdcc2530ff208843de0dc169bbc8d8b17746e75f12a1829b626531d24d0bbff",
    ATTEMPT_002_SEAL: "72c7bc9e264f63de2f6bcbe5129b927ed006bfe41b4cbbb98e7282c35c379602",
    ATTEMPT_002_TRANSCRIPT: "4ac804fd6b724871362fb274216ca640d0e19ea1806b01c512d3707540b949f2",
    ATTEMPT_002_FAILURE: "6084d8f0e7f36be2c82ac5c67cf2ccb0b0dc3de719ee171adee6083587f33bcf",
}
LAUNCH_OWNER = "/root"
ATTEMPT_ACQUIRED = False

MAX_SHAPE = (64, 64, 64)
GAPPED_SHAPE = (64, 32, 48)
ROSTER_CALLS = 58
REPETITIONS = 8
THREADS = 2
RESIDUAL_DELTA_LIMIT = 0.010
NORMALIZED_MAX_ABSOLUTE_VS_FLOAT64_LIMIT = 1.0112751933930056e-4
EXPECTED_CALLS = {
    "copyto": 58,
    "einsum": 696,
    "matmul": 58,
    "reshape": 174,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def claim_attempt(launch_seal_sha256: str) -> dict[str, object]:
    """Atomically consume the sole D7R1 attempt before kernel import."""
    global ATTEMPT_ACQUIRED
    payload = {
        "schema": "p2r0c1d7r1-tiny-attempt-marker-v1",
        "status": "CLAIMED_PERMANENTLY",
        "node": "P2R0C1D7R1",
        "launch_owner": LAUNCH_OWNER,
        "launch_seal_sha256": launch_seal_sha256,
        "runner_sha256": sha256(Path(__file__)),
        "pid": os.getpid(),
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    try:
        descriptor = os.open(
            ATTEMPT_MARKER,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o444,
        )
    except FileExistsError as error:
        raise PermissionError("D7R1 tiny attempt already consumed") from error
    ATTEMPT_ACQUIRED = True
    try:
        written = 0
        while written < len(encoded):
            count = os.write(descriptor, encoded[written:])
            if count <= 0:
                raise OSError("short write while persisting D7R1 attempt marker")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return payload


def persist_exception_receipt() -> None:
    if not ATTEMPT_ACQUIRED or EXCEPTION_RECEIPT.exists():
        return
    payload = {
        "schema": "p2r0c1d7r1-runtime-exception-receipt-v1",
        "status": "TERMINAL_EXCEPTION_AFTER_ATOMIC_ATTEMPT_CLAIM",
        "node": "P2R0C1D7R1",
        "attempt_marker_sha256": (
            sha256(ATTEMPT_MARKER) if ATTEMPT_MARKER.is_file() else None
        ),
        "bind_preassert_sha256": (
            sha256(BIND_PREASSERT) if BIND_PREASSERT.is_file() else None
        ),
        "result_sha256": sha256(RESULT) if RESULT.is_file() else None,
        "traceback": traceback.format_exc(),
        "estimator_calls": 0,
        "exact_shape_calls": 0,
        "common_driver_calls": 0,
        "network": False,
    }
    write_once(EXCEPTION_RECEIPT, payload)


def load_kernel():
    spec = importlib.util.spec_from_file_location("p2r0c1d7r1_tiny_kernel", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d7(m: int, k: int, n: int) -> int:
    a, b, c = m * k, k * n, m * n
    return (
        7**4 * mm(m // 16, k // 16, n // 16)
        + sum(
            49 * 7**level * (a + b) // 4 ** (level + 1)
            for level in range(4)
        )
        + sum(
            52 * 7**level * c // 4 ** (level + 1)
            for level in range(4)
        )
        + a + b + 2 * c
    )


def inventory(meter) -> dict[str, object]:
    calls = Counter(record.op_name for record in meter.op_log)
    costs = Counter()
    for record in meter.op_log:
        costs[record.op_name] += int(record.flop_cost)
    return {
        "flops": int(meter.flops_used),
        "calls": {name: int(value) for name, value in sorted(calls.items())},
        "costs": {name: int(value) for name, value in sorted(costs.items())},
    }


def scaled_inputs(shape: tuple[int, int, int], seed: int):
    m, k, n = shape
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((m, k)).astype(np.float32)
    right = rng.standard_normal((k, n)).astype(np.float32)
    np.multiply(right, np.float32(1.0 / math.sqrt(k)), out=right)
    return left, right


def drift(observed: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    delta = observed.astype(np.float64) - reference.astype(np.float64)
    rms = float(np.sqrt(np.mean(delta * delta, dtype=np.float64)))
    maximum_absolute = float(np.max(np.abs(delta)))
    reference64 = reference.astype(np.float64)
    reference_rms = float(np.sqrt(np.mean(reference64 * reference64, dtype=np.float64)))
    if not reference_rms > 0.0:
        raise AssertionError("zero reference RMS cannot define normalized drift")
    return {
        "rms": rms,
        "maximum_absolute": maximum_absolute,
        "reference_rms": reference_rms,
        "normalized_rms": rms / reference_rms,
        "normalized_maximum_absolute": maximum_absolute / reference_rms,
    }


def measured_arm(call) -> tuple[float, dict[str, object]]:
    with flops.BudgetContext(flop_budget=2_000_000_000, quiet=True) as meter:
        for _ in range(ROSTER_CALLS):
            call()
    return float(meter.summary_dict()["residual_wall_time_s"]), inventory(meter)


def verify_launch_seal() -> dict[str, object]:
    for path in (ATTEMPT_MARKER, BIND_PREASSERT, EXCEPTION_RECEIPT, RESULT):
        if path.exists():
            raise FileExistsError(path)
    if not SEAL.is_file():
        raise PermissionError("tiny discriminator is not sealed")
    seal = json.loads(SEAL.read_text())
    if seal.get("status") != "AUTHORIZED_ONE_SHOT_D7R1_TINY_RUNTIME_ONLY":
        raise PermissionError("tiny discriminator seal is not authorized")
    if seal.get("launch_owner") != LAUNCH_OWNER:
        raise PermissionError("D7R1 launch owner mismatch")
    if seal.get("authorized_runs") != 1:
        raise PermissionError("D7R1 seal must authorize exactly one run")
    if seal.get("exact_shape_authorized") is not False:
        raise PermissionError("D7R1 seal cannot authorize exact shape")
    for relative, expected in seal["sha256"].items():
        path = HERE / relative
        if sha256(path) != expected:
            raise AssertionError((relative, sha256(path), expected))
    return seal


def main() -> None:
    seal = verify_launch_seal()
    marker = claim_attempt(sha256(SEAL))
    bindings = {
        "source_sha256": sha256(SOURCE),
        "manifest_sha256": sha256(MANIFEST),
        "design_sha256": sha256(DESIGN),
        "static_verifier_sha256": sha256(STATIC_VERIFIER),
        "d5_kernel_smoke_authority_sha256": sha256(D5_KERNEL_SMOKE),
        "d5_kernel_smoke_runner_sha256": sha256(D5_KERNEL_RUNNER),
        "dual_launch_audit_sha256": sha256(DUAL_LAUNCH_AUDIT),
        "attempt_001_seal_sha256": sha256(ATTEMPT_001_SEAL),
        "attempt_001_transcript_sha256": sha256(ATTEMPT_001_TRANSCRIPT),
        "attempt_001_failure_sha256": sha256(ATTEMPT_001_FAILURE),
        "attempt_002_seal_sha256": sha256(ATTEMPT_002_SEAL),
        "attempt_002_transcript_sha256": sha256(ATTEMPT_002_TRANSCRIPT),
        "attempt_002_failure_sha256": sha256(ATTEMPT_002_FAILURE),
        "runner_sha256": sha256(Path(__file__)),
        "launch_seal_sha256": sha256(SEAL),
        "attempt_marker_sha256": sha256(ATTEMPT_MARKER),
    }
    assert bindings["source_sha256"] == EXPECTED_SOURCE
    assert bindings["manifest_sha256"] == EXPECTED_MANIFEST
    assert bindings["design_sha256"] == EXPECTED_DESIGN
    assert bindings["static_verifier_sha256"] == EXPECTED_STATIC_VERIFIER
    assert bindings["d5_kernel_smoke_authority_sha256"] == EXPECTED_D5_KERNEL_SMOKE
    assert bindings["d5_kernel_smoke_runner_sha256"] == EXPECTED_D5_KERNEL_RUNNER
    for authority, expected in EXPECTED_RECOVERY_AUTHORITIES.items():
        assert sha256(authority) == expected, (authority, sha256(authority), expected)
    assert getattr(flops, "__version__", None) == "0.12.0+np2.4.6"
    assert np.__version__ == "2.4.6"

    baseline_rss = resident_bytes()
    kernel = load_kernel()
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True) as bind_meter:
        coefficients = kernel.bind_coefficients()
    bind_inventory = inventory(bind_meter)
    expected_bind_inventory = {
        "flops": 168,
        "calls": {"asarray": 3},
        "costs": {"asarray": 168},
    }
    write_once(BIND_PREASSERT, {
        "schema": "p2r0c1d7r1-bind-inventory-preassert-v1",
        "status": "PERSISTED_BEFORE_BIND_ASSERTION",
        "node": "P2R0C1D7R1",
        "attempt_marker_sha256": sha256(ATTEMPT_MARKER),
        "launch_seal_sha256": sha256(SEAL),
        "runner_sha256": sha256(Path(__file__)),
        "observed": bind_inventory,
        "expected": expected_bind_inventory,
    })
    bindings["bind_preassert_sha256"] = sha256(BIND_PREASSERT)
    assert bind_inventory == expected_bind_inventory

    left_np, right_np = scaled_inputs(MAX_SHAPE, 7_260_001)
    left, right = fnp.asarray(left_np), fnp.asarray(right_np)
    candidate_out = fnp.empty(MAX_SHAPE[::2], dtype=fnp.float32)
    direct_out = fnp.empty(MAX_SHAPE[::2], dtype=fnp.float32)
    candidate_call = lambda: kernel.staged_winograd(
        left, right, candidate_out, coefficients
    )
    direct_call = lambda: fnp.matmul(left, right, out=direct_out)

    # One separately metered gapped-stride semantic and accounting case.
    gm, gk, gn = GAPPED_SHAPE
    gap_left_np, gap_right_np = scaled_inputs(GAPPED_SHAPE, 7_260_002)
    gap_left, gap_right = fnp.asarray(gap_left_np), fnp.asarray(gap_right_np)
    gap_base = fnp.empty((gm, 64), dtype=fnp.float32)
    gap_out = gap_base[:, :gn]
    np.asarray(gap_base)[:, gn:] = np.float32(913.0)
    gap_guard_before = np.asarray(gap_base)[:, gn:].copy()
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as gap_meter:
        gap_returned = kernel.staged_winograd(
            gap_left, gap_right, gap_out, coefficients
        )
    gap_observed = np.asarray(gap_out).copy()
    gap_direct_float32 = gap_left_np @ gap_right_np
    gap_reference_float64 = (
        gap_left_np.astype(np.float64) @ gap_right_np.astype(np.float64)
    )
    gap_drift_vs_direct_float32 = drift(gap_observed, gap_direct_float32)
    gap_drift_vs_float64 = drift(gap_observed, gap_reference_float64)
    gap_inventory = inventory(gap_meter)

    with threadpool_limits(limits=THREADS):
        # Warm paths, wrappers, BLAS, and allocator before alternated timing.
        with flops.BudgetContext(flop_budget=100_000_000, quiet=True):
            candidate_call()
            direct_call()
        warm_candidate = np.asarray(candidate_out).copy()
        warm_direct = np.asarray(direct_out).copy()
        maximum_drift_vs_direct_float32 = drift(warm_candidate, warm_direct)
        maximum_reference_float64 = (
            left_np.astype(np.float64) @ right_np.astype(np.float64)
        )
        maximum_drift_vs_float64 = drift(
            warm_candidate, maximum_reference_float64
        )
        warm_hash = hashlib.sha256(warm_candidate.tobytes()).hexdigest()

        candidate_residuals: list[float] = []
        direct_residuals: list[float] = []
        deltas: list[float] = []
        candidate_inventories: list[dict[str, object]] = []
        direct_inventories: list[dict[str, object]] = []
        for repetition in range(REPETITIONS):
            if repetition % 2 == 0:
                candidate_residual, candidate_inventory = measured_arm(candidate_call)
                direct_residual, direct_inventory = measured_arm(direct_call)
            else:
                direct_residual, direct_inventory = measured_arm(direct_call)
                candidate_residual, candidate_inventory = measured_arm(candidate_call)
            candidate_residuals.append(candidate_residual)
            direct_residuals.append(direct_residual)
            deltas.append(candidate_residual - direct_residual)
            candidate_inventories.append(candidate_inventory)
            direct_inventories.append(direct_inventory)

    final_hash = hashlib.sha256(np.asarray(candidate_out).tobytes()).hexdigest()
    peak_rss = resident_bytes()
    expected_candidate_flops = d7(*MAX_SHAPE) * ROSTER_CALLS
    expected_direct_flops = mm(*MAX_SHAPE) * ROSTER_CALLS
    median_candidate = statistics.median(candidate_residuals)
    median_direct = statistics.median(direct_residuals)
    median_delta = statistics.median(deltas)
    checks = {
        "seal_authority_tiny_only": seal["exact_shape_authorized"] is False,
        "source_and_design_hashes_exact": bindings["source_sha256"] == EXPECTED_SOURCE,
        "coefficient_binding_exact": bind_inventory == {
            "flops": 168,
            "calls": {"asarray": 3},
            "costs": {"asarray": 168},
        },
        "bind_inventory_persisted_before_assertion": BIND_PREASSERT.is_file(),
        "atomic_attempt_marker_persisted": ATTEMPT_MARKER.is_file(),
        "gapped_returned_original_out": gap_returned is gap_out,
        "gapped_out_is_strided": not bool(np.asarray(gap_out).flags.c_contiguous),
        "gapped_bit_view_shares_memory": bool(np.shares_memory(
            np.asarray(gap_out),
            np.reshape(np.asarray(gap_out), (2, 2, 2, 2, 4, 2, 2, 2, 2, 3)),
        )),
        "gapped_guard_unchanged": np.array_equal(
            np.asarray(gap_base)[:, gn:], gap_guard_before
        ),
        "gapped_inventory_exact": gap_inventory["calls"] == {
            "copyto": 1, "einsum": 12, "matmul": 1, "reshape": 3,
        } and gap_inventory["flops"] == d7(*GAPPED_SHAPE),
        "gapped_finite_float32": (
            str(gap_observed.dtype) == "float32"
            and bool(np.isfinite(gap_observed).all())
        ),
        "gapped_normalized_maximum_absolute_vs_float64_within_authority": (
            gap_drift_vs_float64["normalized_maximum_absolute"]
            <= NORMALIZED_MAX_ABSOLUTE_VS_FLOAT64_LIMIT
        ),
        "maximum_shape_finite_float32": (
            str(warm_candidate.dtype) == "float32"
            and bool(np.isfinite(warm_candidate).all())
        ),
        "maximum_shape_normalized_maximum_absolute_vs_float64_within_authority": (
            maximum_drift_vs_float64["normalized_maximum_absolute"]
            <= NORMALIZED_MAX_ABSOLUTE_VS_FLOAT64_LIMIT
        ),
        "warm_output_repeat_byte_identical": warm_hash == final_hash,
        "candidate_roster_bills_and_inventories_exact": all(
            item["calls"] == EXPECTED_CALLS
            and item["flops"] == expected_candidate_flops
            for item in candidate_inventories
        ),
        "direct_roster_bills_and_inventories_exact": all(
            item["calls"] == {"matmul": 58}
            and item["flops"] == expected_direct_flops
            for item in direct_inventories
        ),
        "all_timings_finite": all(np.isfinite(
            candidate_residuals + direct_residuals + deltas
        )),
        "median_residual_delta_at_most_10ms": median_delta <= RESIDUAL_DELTA_LIMIT,
    }
    status = (
        "PASS_D7R1_TINY_RUNTIME_ADMISSION_PREPARATION_ONLY"
        if all(checks.values())
        else "FAIL_D7R1_TINY_RUNTIME_TERMINAL_CLOSE"
    )
    payload = {
        "schema": "p2r0c1d7r1-tiny-runtime-discriminator-v1",
        "status": status,
        "node": "P2R0C1D7R1",
        "bindings": bindings,
        "attempt_marker": marker,
        "environment": {
            "python_executable": sys.executable,
            "python_version": platform.python_version(),
            "flopscope_version": getattr(flops, "__version__", "unknown"),
            "numpy_version": np.__version__,
            "threads": THREADS,
        },
        "shape": list(MAX_SHAPE),
        "gapped_shape": list(GAPPED_SHAPE),
        "roster_calls_per_arm": ROSTER_CALLS,
        "alternated_repetitions": REPETITIONS,
        "candidate_residual_wall_seconds": candidate_residuals,
        "direct_residual_wall_seconds": direct_residuals,
        "candidate_minus_direct_residual_seconds": deltas,
        "median_candidate_residual_wall_seconds": median_candidate,
        "median_direct_residual_wall_seconds": median_direct,
        "median_candidate_minus_direct_residual_seconds": median_delta,
        "residual_delta_gate_seconds": RESIDUAL_DELTA_LIMIT,
        "residual_gate_pass": median_delta <= RESIDUAL_DELTA_LIMIT,
        "coefficient_binding_meter": bind_inventory,
        "gapped_meter": gap_inventory,
        "gapped_drift_vs_direct_float32": gap_drift_vs_direct_float32,
        "gapped_drift_vs_float64": gap_drift_vs_float64,
        "maximum_shape_drift_vs_direct_float32": maximum_drift_vs_direct_float32,
        "maximum_shape_drift_vs_float64": maximum_drift_vs_float64,
        "kernel_normalized_maximum_absolute_delta_vs_float64_gate": (
            NORMALIZED_MAX_ABSOLUTE_VS_FLOAT64_LIMIT
        ),
        "final_network_tolerances_applied": False,
        "expected_candidate_roster_flops": expected_candidate_flops,
        "expected_direct_roster_flops": expected_direct_flops,
        "expected_candidate_roster_calls": EXPECTED_CALLS,
        "candidate_roster_meter": candidate_inventories[0],
        "direct_roster_meter": direct_inventories[0],
        "rss": {
            "baseline_bytes": baseline_rss,
            "peak_bytes": peak_rss,
            "increment_bytes": peak_rss - baseline_rss,
            "scope": "tiny process high-water only; exact-shape RSS remains unmeasured",
        },
        "checks": checks,
        "estimator_imported": False,
        "estimator_calls": 0,
        "fixture_calls": 0,
        "exact_shape_calls": 0,
        "common_driver_calls": 0,
        "targets_scores_names_board_opened": False,
        "production": False,
        "submission": False,
        "network": False,
        "disposition": (
            "AUTHORIZE_DESIGN_PREPARATION_ONLY_NO_EXACT_SHAPE"
            if status == "PASS_D7R1_TINY_RUNTIME_ADMISSION_PREPARATION_ONLY"
            else "CLOSE_D7R1_RECOVERY_FORM_NO_RETRY"
        ),
    }
    write_once(RESULT, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if status != "PASS_D7R1_TINY_RUNTIME_ADMISSION_PREPARATION_ONLY":
        raise RuntimeError(status)


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        persist_exception_receipt()
        raise
