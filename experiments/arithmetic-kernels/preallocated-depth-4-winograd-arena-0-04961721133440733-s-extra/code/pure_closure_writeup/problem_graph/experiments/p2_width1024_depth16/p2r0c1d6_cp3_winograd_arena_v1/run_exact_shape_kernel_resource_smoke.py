#!/usr/bin/env python3
"""One-shot target-free exact-shape P2R0C1D6 kernel/resource smoke.

PREPARATION ONLY: the checked-in draft manifest is deliberately unsealed, so
this program refuses to import the candidate or allocate the arena.  A future
independent audit must first freeze all hashes and create the one-use seal.

The eventual smoke uses deterministic synthetic matrices for the exact 30
push plus 28 read channel roster.  It opens no estimator fixture, MLP, target,
score, name, board, production, submission, or network path.
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import sys
import time
import traceback

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "estimator.py"
MANIFEST = HERE / "EXACT_SHAPE_SMOKE_DRAFT_MANIFEST.json"
SEAL = HERE / "EXACT_SHAPE_SMOKE_SEAL.json"
SCHEMA = HERE / "EXACT_SHAPE_SMOKE_TERMINAL_RECEIPT_SCHEMA.json"
RESULT = HERE / "EXACT_SHAPE_SMOKE_RESULT.json"

SOURCE_SHA256 = "e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58"
THREADS = 2
LEVELS = 4
ARENA_ELEMENTS = 844_165_120
PHASE2_BUDGET = 2**41
PER_KERNEL_WALL_LIMIT_S = 120.0
MAX_RSS_INCREMENT_BYTES = 3_913_531_392
MAX_ABSOLUTE_RSS_BYTES = 7 * 1024**3
MAX_RESIDUAL_DELTA_S = 0.010
MAX_POOLED_RMS_DELTA = 1.714909904361275e-6
MAX_ABSOLUTE_DELTA = 8.574549521806375e-6
EXPECTED_BIND_RESHAPE_CALLS = 52
EXPECTED_BIND_RESHAPE_FLOPS = 1_919_598_592
EXPECTED_KERNEL_INVENTORY_PER_CHANNEL = {
    "add": 28,
    "copyto": 56,
    "matmul": 1,
    "subtract": 32,
}
EXPECTED_ROSTER_INVENTORY = {
    key: value * 58
    for key, value in EXPECTED_KERNEL_INVENTORY_PER_CHANNEL.items()
}
EXPECTED_D6_ROSTER_FLOPS = 591_553_830_912
EXPECTED_DIRECT_ROSTER_FLOPS = 966_086_623_232
EXPECTED_D5_CALLS = (
    1_320_876_655_991,
    1_359_519_832_584,
    1_320_873_503_232,
)
EXPECTED_INCUMBENT_CALLS = (
    1_692_484_904_311,
    1_731_128_080_904,
    1_692_481_751_552,
)
EXPECTED_D6_CALLS = (
    1_319_635_780_983,
    1_356_359_358_984,
    1_317_713_029_632,
)
EXPECTED_INCUMBENT_MINUS_D6 = (
    372_849_123_328,
    374_768_721_920,
    374_768_721_920,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def array_sha256_streamed(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    for row in value:
        contiguous = np.ascontiguousarray(row)
        digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def roster() -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    ordinal = 0
    for layer in range(1, 16):
        rank = 1024 * layer
        for channel in range(2):
            rows.append({
                "ordinal": ordinal,
                "phase": "push",
                "layer": layer,
                "rank": rank,
                "channel": channel,
                "m": 1024,
                "k": 1024,
                "n": rank,
                "seed": 6_000_000 + ordinal,
            })
            ordinal += 1
    for layer in range(1, 15):
        rank = 1024 * layer
        for channel in range(2):
            rows.append({
                "ordinal": ordinal,
                "phase": "read",
                "layer": layer,
                "rank": rank,
                "channel": channel,
                "m": 1024,
                "k": rank,
                "n": 1024,
                "seed": 6_000_000 + ordinal,
            })
            ordinal += 1
    assert len(rows) == 58
    return rows


def mm_flops(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d6_kernel_flops(m: int, k: int, n: int) -> int:
    leaf = 7**4 * mm_flops(m // 16, k // 16, n // 16)
    transform = sum(
        7**level * 11 * (m * k + k * n) // 4 ** (level + 1)
        for level in range(4)
    )
    reconstruct = sum(
        7**level * 7 * (m * n) // 4 ** (level + 1)
        for level in range(4)
    )
    return leaf + transform + reconstruct


def operation_calls(summary: dict[str, object]) -> dict[str, int]:
    operations = summary["operations"]
    return {
        name: int(item["calls"])
        for name, item in operations.items()
        if int(item["calls"]) != 0
    }


def operation_flop_sum(summary: dict[str, object]) -> int:
    return sum(
        int(item["flop_cost"])
        for item in summary["operations"].values()
    )


def load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("p2r0c1d6_exact_shape_smoke", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_seal() -> tuple[dict[str, object], dict[str, object]]:
    # Deliberately first: the draft tree has no seal and must stop before the
    # candidate import, RNG construction, BudgetContext, or arena allocation.
    if not SEAL.is_file():
        raise PermissionError("exact-shape smoke has no launch seal")
    seal = json.loads(SEAL.read_text())
    if seal.get("status") != "FROZEN_ONE_SHOT_EXACT_SHAPE_SMOKE_AUTHORIZED":
        raise PermissionError("exact-shape smoke seal is not authorized")
    for relative, expected in seal["sha256"].items():
        path = (HERE / relative).resolve()
        if sha256(path) != expected:
            raise AssertionError((relative, sha256(path), expected))
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("status") != "FROZEN_BEFORE_ONE_SHOT_EXACT_SHAPE_SMOKE":
        raise PermissionError("exact-shape manifest remains draft/hold")
    prerequisites = manifest["prerequisites"]
    if prerequisites["independent_implementation_audit_status"] != "PASS":
        raise PermissionError("independent implementation audit did not pass")
    if prerequisites["tiny_exact_runtime_smoke_status"] != "PASS":
        raise PermissionError("tiny exact-runtime smoke did not pass")
    if float(prerequisites["tiny_residual_delta_seconds"]) > MAX_RESIDUAL_DELTA_S:
        raise PermissionError("tiny residual delta exceeds frozen 10ms gate")
    if RESULT.exists():
        raise FileExistsError("one-shot exact-shape result already exists")
    return seal, manifest


def deterministic_inputs(entry: dict[str, int | str]) -> tuple[np.ndarray, np.ndarray]:
    m, k, n = int(entry["m"]), int(entry["k"]), int(entry["n"])
    generator = np.random.Generator(np.random.PCG64(int(entry["seed"])))
    left = generator.standard_normal((m, k), dtype=np.float32)
    right = generator.standard_normal((k, n), dtype=np.float32)
    np.multiply(left, np.float32(1.0 / math.sqrt(k)), out=left)
    return left, right


def meter_call(namespace: str, function) -> tuple[object, dict[str, object], float]:
    context = flops.BudgetContext(
        flop_budget=PHASE2_BUDGET,
        wall_time_limit_s=PER_KERNEL_WALL_LIMIT_S,
        quiet=True,
        namespace=namespace,
    )
    started = time.perf_counter()
    with context:
        value = function()
    outer = time.perf_counter() - started
    return value, context.summary_dict(by_namespace=False), outer


def tiled_drift(candidate: np.ndarray, direct: np.ndarray) -> dict[str, float]:
    assert candidate.shape == direct.shape
    sum_squared = 0.0
    count = 0
    maximum = 0.0
    for start in range(0, candidate.shape[1], 64):
        stop = min(start + 64, candidate.shape[1])
        delta = (
            candidate[:, start:stop].astype(np.float64)
            - direct[:, start:stop].astype(np.float64)
        )
        sum_squared += float(np.sum(delta * delta, dtype=np.float64))
        count += int(delta.size)
        maximum = max(maximum, float(np.max(np.abs(delta))))
    return {
        "sum_squared": sum_squared,
        "count": count,
        "rms": math.sqrt(sum_squared / count),
        "maximum_absolute": maximum,
    }


def base_result(
    status: str,
    reason: str | None,
    bindings: dict[str, str],
    partial: dict[str, object],
) -> dict[str, object]:
    return {
        "schema": "p2r0c1d6-exact-shape-smoke-result-v1",
        "status": status,
        "node": "P2R0C1D6",
        "terminal_reason": reason,
        "call_roster": roster(),
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "estimator_fixtures_opened": [],
        "estimator_predict_calls": 0,
        "training": False,
        "production": False,
        "submission": False,
        "network": False,
        "bindings": bindings,
        **partial,
    }


def finish_once(
    status: str,
    reason: str | None,
    bindings: dict[str, str],
    partial: dict[str, object],
) -> None:
    payload = base_result(status, reason, bindings, partial)
    write_json_once(RESULT, payload)
    print(json.dumps({"status": status, "reason": reason}, sort_keys=True), flush=True)


def run_smoke(seal: dict[str, object], manifest: dict[str, object]) -> None:
    bindings = {
        "seal_sha256": sha256(SEAL),
        "manifest_sha256": sha256(MANIFEST),
        "runner_sha256": sha256(Path(__file__)),
        "terminal_schema_sha256": sha256(SCHEMA),
        "candidate_source_sha256": sha256(SOURCE),
        "independent_implementation_audit_sha256":
            seal["sha256"][manifest["prerequisites"]["independent_implementation_audit_path"]],
        "tiny_exact_runtime_smoke_sha256":
            seal["sha256"][manifest["prerequisites"]["tiny_exact_runtime_smoke_path"]],
    }
    partial: dict[str, object] = {
        "cold_binding": {},
        "cold_trials": [],
        "warm_trials": [],
        "aggregate": {},
    }
    try:
        if sha256(SOURCE) != SOURCE_SHA256:
            raise AssertionError("candidate source hash changed")
        module = load_candidate(SOURCE)
        baseline_rss = resident_bytes()

        # Empty allocations cost zero; every persistent view reshape is metered
        # once.  Push destinations reproduce the production cap stride; the
        # Direct outputs are separately allocated, contiguous exact-shape
        # buffers outside the measured direct call.
        def cold_bind():
            arena = fnp.empty(ARENA_ELEMENTS, dtype=fnp.float32)
            push_destinations = fnp.empty((2, 1024, 16384), dtype=fnp.float32)
            push_views = module._cp3_bind_arena_views(
                arena, module._CP3_PUSH_ARENA_VIEW_SPECS
            )
            read_views = module._cp3_bind_arena_views(
                arena, module._CP3_READ_ARENA_VIEW_SPECS
            )
            return arena, push_destinations, push_views, read_views

        bound, bind_summary, bind_outer = meter_call("d6_exact_cold_bind", cold_bind)
        arena, push_destinations, push_views, read_views = bound
        bind_calls = operation_calls(bind_summary)
        bind_gates = {
            "reshape_calls_exact": bind_calls.get("reshape", 0) == EXPECTED_BIND_RESHAPE_CALLS,
            "reshape_flops_exact": int(bind_summary["operations"]["reshape"]["flop_cost"])
                == EXPECTED_BIND_RESHAPE_FLOPS,
            "literal_flops_exact": int(bind_summary["flops_used"])
                == EXPECTED_BIND_RESHAPE_FLOPS,
            "literal_operation_sum_equals_total": operation_flop_sum(bind_summary)
                == int(bind_summary["flops_used"]),
            "only_empty_and_reshape_operations": set(bind_calls) <= {"empty", "reshape"},
            "outer_wall_below_limit": bind_outer < PER_KERNEL_WALL_LIMIT_S,
        }
        bind_gates["conjunctive_pass"] = all(bind_gates.values())
        partial["cold_binding"] = {
            "literal_meter": bind_summary,
            "literal_operation_sum": operation_flop_sum(bind_summary),
            "exact_outer_wall_seconds": bind_outer,
            "operation_calls": bind_calls,
            "gates": bind_gates,
        }
        if not bind_gates["conjunctive_pass"]:
            raise RuntimeError("cold binding gate failure")

        pooled_sum_squared = 0.0
        pooled_count = 0
        pooled_maximum = 0.0
        cold_hashes: list[str] = []
        cold_candidate_residual = float(bind_summary["residual_wall_time_s"])
        cold_direct_residual = 0.0
        cold_candidate_flops = 0
        cold_direct_flops = 0
        cold_inventory = {key: 0 for key in EXPECTED_ROSTER_INVENTORY}

        with threadpool_limits(limits=THREADS):
            for entry in roster():
                ordinal = int(entry["ordinal"])
                m, k, n = int(entry["m"]), int(entry["k"]), int(entry["n"])
                channel = int(entry["channel"])
                left, right = deterministic_inputs(entry)
                if entry["phase"] == "push":
                    views = push_views
                    candidate_destination = push_destinations[channel, :, :n]
                else:
                    views = read_views
                    candidate_destination = read_views[24 + channel]

                candidate_value, candidate_summary, candidate_outer = meter_call(
                    f"d6_exact_cold_{ordinal}",
                    lambda: module._cp3_winograd_arena(
                        left, right, candidate_destination, views
                    ),
                )
                candidate_array = np.asarray(candidate_value)
                candidate_hash = array_sha256_streamed(candidate_array)
                direct_view, direct_setup_summary, _direct_setup_outer = meter_call(
                    f"direct_output_setup_{ordinal}",
                    lambda: fnp.empty((m, n), dtype=fnp.float32),
                )
                if (
                    int(direct_setup_summary["flops_used"]) != 0
                    or operation_calls(direct_setup_summary) != {"empty": 1}
                ):
                    raise RuntimeError(
                        f"direct output setup changed at ordinal {ordinal}"
                    )
                direct_value, direct_summary, direct_outer = meter_call(
                    f"direct_exact_{ordinal}",
                    lambda: fnp.matmul(left, right, out=direct_view),
                )
                direct_array = np.asarray(direct_value)
                drift = tiled_drift(candidate_array, direct_array)
                pooled_sum_squared += drift["sum_squared"]
                pooled_count += int(drift["count"])
                pooled_maximum = max(pooled_maximum, drift["maximum_absolute"])
                cold_hashes.append(candidate_hash)

                candidate_calls = operation_calls(candidate_summary)
                expected_kernel_flops = d6_kernel_flops(m, k, n)
                direct_flops = mm_flops(m, k, n)
                trial_gates = {
                    "candidate_shape": tuple(candidate_array.shape) == (m, n),
                    "candidate_dtype_float32": candidate_array.dtype == np.float32,
                    "candidate_finite": bool(np.isfinite(candidate_array).all()),
                    "direct_shape": tuple(direct_array.shape) == (m, n),
                    "direct_dtype_float32": direct_array.dtype == np.float32,
                    "direct_finite": bool(np.isfinite(direct_array).all()),
                    "candidate_bill_exact": int(candidate_summary["flops_used"])
                        == expected_kernel_flops,
                    "candidate_operation_sum_equals_total":
                        operation_flop_sum(candidate_summary)
                        == int(candidate_summary["flops_used"]),
                    "candidate_inventory_exact": candidate_calls
                        == EXPECTED_KERNEL_INVENTORY_PER_CHANNEL,
                    "direct_bill_exact": int(direct_summary["flops_used"])
                        == direct_flops,
                    "direct_inventory_exact": operation_calls(direct_summary)
                        == {"matmul": 1},
                    "candidate_outer_wall_below_limit":
                        candidate_outer < PER_KERNEL_WALL_LIMIT_S,
                    "direct_outer_wall_below_limit": direct_outer < PER_KERNEL_WALL_LIMIT_S,
                }
                trial_gates["conjunctive_pass"] = all(trial_gates.values())
                trial = {
                    **entry,
                    "candidate_literal_meter": candidate_summary,
                    "candidate_literal_operation_sum": operation_flop_sum(candidate_summary),
                    "candidate_exact_outer_wall_seconds": candidate_outer,
                    "candidate_output_sha256": candidate_hash,
                    "direct_literal_meter": direct_summary,
                    "direct_literal_operation_sum": operation_flop_sum(direct_summary),
                    "direct_exact_outer_wall_seconds": direct_outer,
                    "direct_output_sha256": array_sha256_streamed(direct_array),
                    "drift_vs_direct_float32": drift,
                    "gates": trial_gates,
                }
                partial["cold_trials"].append(trial)
                if not trial_gates["conjunctive_pass"]:
                    raise RuntimeError(f"cold roster gate failure at ordinal {ordinal}")
                cold_candidate_residual += float(candidate_summary["residual_wall_time_s"])
                cold_direct_residual += float(direct_summary["residual_wall_time_s"])
                cold_candidate_flops += int(candidate_summary["flops_used"])
                cold_direct_flops += int(direct_summary["flops_used"])
                for name in cold_inventory:
                    cold_inventory[name] += candidate_calls.get(name, 0)
                del left, right, candidate_array, direct_array
                gc.collect()

            warm_candidate_residual = 0.0
            warm_candidate_flops = 0
            warm_inventory = {key: 0 for key in EXPECTED_ROSTER_INVENTORY}
            for entry in roster():
                ordinal = int(entry["ordinal"])
                m, k, n = int(entry["m"]), int(entry["k"]), int(entry["n"])
                channel = int(entry["channel"])
                left, right = deterministic_inputs(entry)
                if entry["phase"] == "push":
                    views = push_views
                    destination = push_destinations[channel, :, :n]
                else:
                    views = read_views
                    destination = read_views[24 + channel]
                value, summary, outer = meter_call(
                    f"d6_exact_warm_{ordinal}",
                    lambda: module._cp3_winograd_arena(left, right, destination, views),
                )
                array = np.asarray(value)
                output_hash = array_sha256_streamed(array)
                calls = operation_calls(summary)
                gates = {
                    "bill_exact": int(summary["flops_used"])
                        == d6_kernel_flops(m, k, n),
                    "operation_sum_equals_total": operation_flop_sum(summary)
                        == int(summary["flops_used"]),
                    "inventory_exact": calls == EXPECTED_KERNEL_INVENTORY_PER_CHANNEL,
                    "repeat_byte_identical": output_hash == cold_hashes[ordinal],
                    "dtype_float32": array.dtype == np.float32,
                    "shape": tuple(array.shape) == (m, n),
                    "finite": bool(np.isfinite(array).all()),
                    "outer_wall_below_limit": outer < PER_KERNEL_WALL_LIMIT_S,
                }
                gates["conjunctive_pass"] = all(gates.values())
                partial["warm_trials"].append({
                    **entry,
                    "candidate_literal_meter": summary,
                    "candidate_literal_operation_sum": operation_flop_sum(summary),
                    "candidate_exact_outer_wall_seconds": outer,
                    "candidate_output_sha256": output_hash,
                    "gates": gates,
                })
                if not gates["conjunctive_pass"]:
                    raise RuntimeError(f"warm roster gate failure at ordinal {ordinal}")
                warm_candidate_residual += float(summary["residual_wall_time_s"])
                warm_candidate_flops += int(summary["flops_used"])
                for name in warm_inventory:
                    warm_inventory[name] += calls.get(name, 0)
                del left, right, array
                gc.collect()

        peak_rss = resident_bytes()
        rss_increment = peak_rss - baseline_rss
        pooled_rms = math.sqrt(pooled_sum_squared / pooled_count)
        cold_residual_delta = cold_candidate_residual - cold_direct_residual
        warm_residual_delta = warm_candidate_residual - cold_direct_residual
        measured_cold_call = (
            EXPECTED_D5_CALLS[0]
            - 594_693_332_992
            - 20_971_520
            + cold_candidate_flops
            + int(bind_summary["flops_used"])
        )
        measured_warm_calls = [
            EXPECTED_D5_CALLS[index]
            - 594_693_332_992
            - 20_971_520
            + warm_candidate_flops
            for index in (1, 2)
        ]
        measured_calls = [measured_cold_call, *measured_warm_calls]
        measured_deltas = [
            incumbent - candidate
            for incumbent, candidate in zip(EXPECTED_INCUMBENT_CALLS, measured_calls)
        ]
        aggregate_gates = {
            "cold_roster_complete": len(partial["cold_trials"]) == 58,
            "warm_roster_complete": len(partial["warm_trials"]) == 58,
            "cold_roster_flops_exact": cold_candidate_flops == EXPECTED_D6_ROSTER_FLOPS,
            "warm_roster_flops_exact": warm_candidate_flops == EXPECTED_D6_ROSTER_FLOPS,
            "direct_roster_flops_exact": cold_direct_flops == EXPECTED_DIRECT_ROSTER_FLOPS,
            "cold_inventory_exact": cold_inventory == EXPECTED_ROSTER_INVENTORY,
            "warm_inventory_exact": warm_inventory == EXPECTED_ROSTER_INVENTORY,
            "cold_residual_delta_at_most_10ms": cold_residual_delta <= MAX_RESIDUAL_DELTA_S,
            "warm_residual_delta_at_most_10ms": warm_residual_delta <= MAX_RESIDUAL_DELTA_S,
            "pooled_rms_drift_gate": pooled_rms <= MAX_POOLED_RMS_DELTA,
            "maximum_absolute_drift_gate": pooled_maximum <= MAX_ABSOLUTE_DELTA,
            "rss_increment_gate": rss_increment <= MAX_RSS_INCREMENT_BYTES,
            "absolute_rss_strict_gate": peak_rss < MAX_ABSOLUTE_RSS_BYTES,
            "measured_full_calls_exact": measured_calls == list(EXPECTED_D6_CALLS),
            "measured_saving_gate": all(
                value >= 371_608_248_320 for value in measured_deltas
            ),
            "candidate_calls_no_greater_than_d5": all(
                candidate <= d5 for candidate, d5 in zip(measured_calls, EXPECTED_D5_CALLS)
            ),
        }
        aggregate_gates["conjunctive_pass"] = all(aggregate_gates.values())
        partial["aggregate"] = {
            "baseline_process_rss_bytes": baseline_rss,
            "peak_process_rss_bytes": peak_rss,
            "process_rss_increment_bytes": rss_increment,
            "cold_candidate_residual_wall_seconds": cold_candidate_residual,
            "warm_candidate_residual_wall_seconds": warm_candidate_residual,
            "direct_residual_wall_seconds": cold_direct_residual,
            "cold_candidate_minus_direct_residual_seconds": cold_residual_delta,
            "warm_candidate_minus_direct_residual_seconds": warm_residual_delta,
            "cold_candidate_roster_flops": cold_candidate_flops,
            "warm_candidate_roster_flops": warm_candidate_flops,
            "direct_roster_flops": cold_direct_flops,
            "cold_candidate_inventory": cold_inventory,
            "warm_candidate_inventory": warm_inventory,
            "pooled_rms_delta_vs_direct_float32": pooled_rms,
            "maximum_absolute_delta_vs_direct_float32": pooled_maximum,
            "measured_full_candidate_calls_35_279_35": measured_calls,
            "measured_incumbent_minus_candidate": measured_deltas,
            "gates": aggregate_gates,
        }
        if not aggregate_gates["conjunctive_pass"]:
            raise RuntimeError("aggregate exact-shape resource gate failure")
        finish_once("PASS_ONE_SHOT_EXACT_SHAPE_SMOKE", None, bindings, partial)
    except BaseException as exc:
        if not RESULT.exists():
            partial["exception"] = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
            finish_once(
                "FAIL_TERMINAL_NO_RETRY",
                f"{type(exc).__name__}: {exc}",
                bindings,
                partial,
            )
        raise


def main() -> None:
    seal, manifest = verify_seal()
    run_smoke(seal, manifest)


if __name__ == "__main__":
    main()
