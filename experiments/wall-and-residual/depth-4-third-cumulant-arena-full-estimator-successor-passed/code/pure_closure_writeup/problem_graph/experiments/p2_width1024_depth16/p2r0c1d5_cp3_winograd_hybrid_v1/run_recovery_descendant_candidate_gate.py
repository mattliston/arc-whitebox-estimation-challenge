#!/usr/bin/env python3
"""One-shot zero-reference-call P2R0C1D5R1 candidate gate."""

from __future__ import annotations

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


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "estimator.py"
MANIFEST = HERE / "RECOVERY_DESCENDANT_MANIFEST.json"
SEAL = HERE / "RECOVERY_DESCENDANT_LAUNCH_SEAL.json"
REFERENCE_RESULT = HERE / "RECOVERED_REFERENCE_EVIDENCE.json"
RESULT = HERE / "RECOVERY_DESCENDANT_RESULT.json"
REFERENCE_OUTPUTS = (
    HERE / "reference_rank35_first.npy",
    HERE / "reference_rank279.npy",
    HERE / "reference_rank35_repeat.npy",
)
CANDIDATE_OUTPUTS = (
    HERE / "recovery_candidate_rank35_first.npy",
    HERE / "recovery_candidate_rank279.npy",
    HERE / "recovery_candidate_rank35_repeat.npy",
)
SOURCE_SHA256 = "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5"
CALL_ORDER = (35, 279, 35)
EXPECTED_FLOPS = (1_320_876_655_991, 1_359_519_832_584, 1_320_873_503_232)
EXPECTED_REFERENCE_FLOPS = (1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552)
EXACT_SAVING = 371_608_248_320
PHASE2_BUDGET = 2**41
THREADS = 2
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_WALL_SECONDS = 120.0
MAX_RESIDUAL_SECONDS = 0.4
MAX_DEPLOYMENT_RESIDUAL_SECONDS = 0.385
MAX_POOLED_FINAL_RMS_DELTA = 1.714909904361275e-6
MAX_FINAL_ABS_DELTA = 8.574549521806375e-6


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


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_array_once(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)


def verify_seal() -> tuple[dict[str, object], dict[str, object]]:
    seal = json.loads(SEAL.read_text())
    if seal.get("status") != "FROZEN_RECOVERY_DESCENDANT_AUTHORIZED":
        raise AssertionError("recovery-descendant launch seal status changed")
    for relative, expected in seal["sha256"].items():
        observed = sha256((HERE / relative).resolve())
        if observed != expected:
            raise AssertionError((relative, observed, expected))
    manifest = json.loads(MANIFEST.read_text())
    if tuple(manifest["call_order"]) != CALL_ORDER:
        raise AssertionError("candidate call order changed")
    if tuple(manifest["expected_candidate_flops"]) != EXPECTED_FLOPS:
        raise AssertionError("candidate bill changed")
    if int(manifest["amended_exact_net_saving"]) != EXACT_SAVING:
        raise AssertionError("candidate saving changed")
    return seal, manifest


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("p2r0c1d5r1_candidate_gate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fixture(role_rank: int, manifest: dict[str, object]) -> MLP:
    entry = manifest["fixtures"][str(role_rank)]
    path = (HERE / str(entry["weights_path"])).resolve()
    if sha256(path) != entry["weights_sha256"]:
        raise AssertionError(f"fixture weights changed: {role_rank}")
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError(f"fixture schema changed: {role_rank}")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=0,
        name="sealed-weight-fixture",
    )
    mlp.validate()
    return mlp


def load_references(seal: dict[str, object]) -> tuple[dict[str, object], list[np.ndarray]]:
    reference_result = json.loads(REFERENCE_RESULT.read_text())
    if reference_result.get("status") != "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE":
        raise AssertionError("recovered reference evidence status changed")
    if reference_result.get("original_protocol_status") != "FAIL_TERMINAL_NO_RETRY":
        raise AssertionError("original terminal status changed")
    if reference_result.get("original_process_exit_code") != 1:
        raise AssertionError("original process exit changed")
    if reference_result.get("original_candidate_calls") != 0:
        raise AssertionError("candidate was already called")
    if reference_result.get("recovery_reference_estimator_calls") != 0:
        raise AssertionError("recovery made a reference call")
    if reference_result.get("candidate_calls_at_recovery") != 0:
        raise AssertionError("candidate was already called at recovery")
    if tuple(reference_result.get("call_order", ())) != CALL_ORDER:
        raise AssertionError("recovered reference call order changed")
    if reference_result.get("access_declarations") != {
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "training": False,
        "network": False,
    }:
        raise AssertionError("recovered reference scope declaration changed")
    if sha256(REFERENCE_RESULT) != seal["sha256"]["RECOVERED_REFERENCE_EVIDENCE.json"]:
        raise AssertionError("recovered reference evidence changed")
    bindings = reference_result["bindings"]
    if bindings["COMMON_DRIVER_SEAL.json"] != seal["sha256"]["COMMON_DRIVER_SEAL.json"]:
        raise AssertionError("original seal binding changed")
    if bindings["run_reference_capture.py"] != seal["sha256"]["run_reference_capture.py"]:
        raise AssertionError("reference runner binding changed")
    if bindings["../public_noeigh_card_v1/estimator.py"] != seal["sha256"]["../public_noeigh_card_v1/estimator.py"]:
        raise AssertionError("reference source binding changed")
    if len(reference_result["trials"]) != 3:
        raise AssertionError("reference roster incomplete")
    arrays = []
    for ordinal, path in enumerate(REFERENCE_OUTPUTS):
        trial = reference_result["trials"][ordinal]
        if trial["ordinal"] != ordinal or trial["role_rank"] != CALL_ORDER[ordinal]:
            raise AssertionError(f"reference roster changed: {ordinal}")
        if trial["output_file"] != path.name:
            raise AssertionError(f"reference output path changed: {ordinal}")
        if trial["output_file_sha256"] != sha256(path):
            raise AssertionError(f"reference output file changed: {ordinal}")
        if int(trial["literal_meter"]["flops_used"]) != EXPECTED_REFERENCE_FLOPS[ordinal]:
            raise AssertionError(f"reference bill changed: {ordinal}")
        if int(trial["literal_operation_sum"]) != EXPECTED_REFERENCE_FLOPS[ordinal]:
            raise AssertionError(f"reference operation sum changed: {ordinal}")
        if not trial["gates"]["conjunctive_pass"] or not all(trial["gates"].values()):
            raise AssertionError(f"reference gate changed: {ordinal}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (16, 1024):
            raise AssertionError(f"reference schema changed: {ordinal}")
        if not np.isfinite(array).all():
            raise AssertionError(f"reference finiteness changed: {ordinal}")
        if array_sha256(array) != trial["output_array_sha256"]:
            raise AssertionError(f"reference array changed: {ordinal}")
        arrays.append(np.ascontiguousarray(array, dtype=np.float32))
    if array_sha256(arrays[0]) != array_sha256(arrays[2]):
        raise AssertionError("reference rank35 repeat mismatch")
    return reference_result, arrays


def layer_drift(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    rms = np.sqrt(np.mean(delta * delta, axis=1))
    maximum = np.max(np.abs(delta), axis=1)
    return {
        "per_layer_rms": [float(value) for value in rms],
        "per_layer_maximum_absolute": [float(value) for value in maximum],
        "worst_layer_by_rms": int(np.argmax(rms)),
        "worst_layer_by_maximum_absolute": int(np.argmax(maximum)),
        "final_layer_rms": float(rms[-1]),
        "final_layer_maximum_absolute": float(maximum[-1]),
    }


def build_result(
    status: str,
    trials: list[dict[str, object]],
    aggregate: dict[str, object],
    reason: str | None,
    bindings: dict[str, str],
) -> dict[str, object]:
    return {
        "schema": "p2r0c1d5r1-target-free-recovery-descendant-gate-v1",
        "status": status,
        "node": "P2R0C1D5R1",
        "original_protocol_status": "FAIL_TERMINAL_NO_RETRY",
        "reference_estimator_calls_in_descendant": 0,
        "call_order": list(CALL_ORDER),
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "training": False,
        "network": False,
        "trials": trials,
        "aggregate": aggregate,
        "terminal_reason": reason,
        "bindings": bindings,
    }


def finish(
    status: str,
    trials: list[dict[str, object]],
    aggregate: dict[str, object],
    reason: str | None = None,
) -> None:
    bindings = {
        "recovery_descendant_launch_seal_sha256": sha256(SEAL),
        "recovery_descendant_manifest_sha256": sha256(MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "candidate_source_sha256": sha256(SOURCE),
        "recovered_reference_evidence_sha256": sha256(REFERENCE_RESULT),
    }
    result = build_result(status, trials, aggregate, reason, bindings)
    write_json_once(RESULT, result)
    print(json.dumps({"status": status, "reason": reason, "aggregate": aggregate}, sort_keys=True), flush=True)


def main() -> None:
    seal, manifest = verify_seal()
    if sha256(SOURCE) != SOURCE_SHA256:
        raise AssertionError("candidate source changed")
    if RESULT.exists() or any(path.exists() for path in CANDIDATE_OUTPUTS):
        raise FileExistsError("one-shot candidate artifact already exists")
    reference_result, references = load_references(seal)
    fixtures = {
        rank: load_fixture(rank, manifest) for rank in sorted(set(CALL_ORDER))
    }
    module = load_module(SOURCE)
    estimator = module.Estimator()
    trials: list[dict[str, object]] = []
    first_hash: str | None = None
    with threadpool_limits(limits=THREADS):
        for ordinal, role_rank in enumerate(CALL_ORDER):
            context = flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=MAX_WALL_SECONDS,
                quiet=True,
                namespace=f"p2r0c1d5r1_candidate_{ordinal}",
            )
            started = time.perf_counter()
            with context:
                value = estimator.predict(fixtures[role_rank], PHASE2_BUDGET)
            outer_wall = time.perf_counter() - started
            summary = context.summary_dict(by_namespace=False)
            raw_output = np.asarray(value)
            raw_schema_ok = (
                raw_output.dtype == np.float32
                and raw_output.shape == (16, 1024)
            )
            output = np.ascontiguousarray(raw_output)
            output_hash = array_sha256(output)
            operation_sum = sum(
                int(item["flop_cost"]) for item in summary["operations"].values()
            )
            drift = layer_drift(output, references[ordinal])
            repeat_ok = ordinal != 2 or output_hash == first_hash
            rss = resident_bytes()
            measured_reference_flops = int(
                reference_result["trials"][ordinal]["literal_meter"]["flops_used"]
            )
            measured_candidate_flops = int(summary["flops_used"])
            measured_delta = measured_reference_flops - measured_candidate_flops
            gates = {
                "literal_flops_exact": measured_candidate_flops == EXPECTED_FLOPS[ordinal],
                "literal_operation_sum_equals_total": operation_sum == measured_candidate_flops,
                "flops_strictly_below_2pow41": measured_candidate_flops < PHASE2_BUDGET,
                "measured_exact_delta_vs_measured_incumbent": measured_delta == EXACT_SAVING,
                "outer_wall_strictly_below_120": outer_wall < MAX_WALL_SECONDS,
                "meter_wall_strictly_below_120": float(summary["wall_time_s"]) < MAX_WALL_SECONDS,
                "residual_strictly_below_0_4": float(summary["residual_wall_time_s"]) < MAX_RESIDUAL_SECONDS,
                "deployment_residual_at_most_0_385": float(summary["residual_wall_time_s"]) <= MAX_DEPLOYMENT_RESIDUAL_SECONDS,
                "rss_strictly_below_8_gib": rss < MAX_RSS_BYTES,
                "raw_output_float32_16x1024": raw_schema_ok,
                "output_finite": bool(np.isfinite(output).all()),
                "final_layer_maximum_absolute_delta": drift["final_layer_maximum_absolute"] <= MAX_FINAL_ABS_DELTA,
                "repeat_rank35_byte_identical": repeat_ok,
            }
            gates["conjunctive_call_pass"] = all(gates.values())
            write_array_once(CANDIDATE_OUTPUTS[ordinal], output)
            trial = {
                "ordinal": ordinal,
                "role_rank": role_rank,
                "literal_meter": summary,
                "literal_operation_sum": operation_sum,
                "measured_reference_flops": measured_reference_flops,
                "measured_candidate_flops": measured_candidate_flops,
                "measured_reference_minus_candidate_flops": measured_delta,
                "exact_outer_wall_seconds": outer_wall,
                "maximum_process_rss_bytes": rss,
                "output_array_sha256": output_hash,
                "output_file": CANDIDATE_OUTPUTS[ordinal].name,
                "output_file_sha256": sha256(CANDIDATE_OUTPUTS[ordinal]),
                "drift_vs_exact_incumbent": drift,
                "gates": gates,
            }
            trials.append(trial)
            print(json.dumps(trial, sort_keys=True), flush=True)
            if ordinal == 0:
                first_hash = output_hash
            if not gates["conjunctive_call_pass"]:
                finish("FAIL_TERMINAL", trials, {}, f"candidate call {ordinal} gate failure")
                raise RuntimeError("terminal candidate failure; no retry")
    all_final_deltas = [
        np.asarray(np.load(CANDIDATE_OUTPUTS[index], allow_pickle=False)[-1], dtype=np.float64)
        - np.asarray(references[index][-1], dtype=np.float64)
        for index in range(3)
    ]
    unique_final_deltas = [
        np.asarray(np.load(CANDIDATE_OUTPUTS[index], allow_pickle=False)[-1], dtype=np.float64)
        - np.asarray(references[index][-1], dtype=np.float64)
        for index in (0, 1)
    ]
    pooled_all_final_rms = float(np.sqrt(np.mean(np.concatenate(all_final_deltas) ** 2)))
    pooled_final_rms = float(np.sqrt(np.mean(np.concatenate(unique_final_deltas) ** 2)))
    maximum_final_abs = float(max(np.max(np.abs(delta)) for delta in all_final_deltas))
    per_call_final_rms = [
        float(trial["drift_vs_exact_incumbent"]["final_layer_rms"])
        for trial in trials
    ]
    per_call_final_max = [
        float(trial["drift_vs_exact_incumbent"]["final_layer_maximum_absolute"])
        for trial in trials
    ]
    worst_rms_ordinal = int(np.argmax(per_call_final_rms))
    worst_max_ordinal = int(np.argmax(per_call_final_max))
    aggregate = {
        "all_three_call_gates_pass": all(trial["gates"]["conjunctive_call_pass"] for trial in trials),
        "repeat_rank35_candidate_byte_identical": trials[0]["output_array_sha256"] == trials[2]["output_array_sha256"],
        "pooled_all_three_calls_final_layer_rms_delta": pooled_all_final_rms,
        "pooled_all_three_calls_final_rms_gate": pooled_all_final_rms <= MAX_POOLED_FINAL_RMS_DELTA,
        "pooled_unique_fixture_final_layer_rms_delta": pooled_final_rms,
        "pooled_final_rms_gate": pooled_final_rms <= MAX_POOLED_FINAL_RMS_DELTA,
        "maximum_all_three_calls_final_layer_absolute_delta": maximum_final_abs,
        "maximum_final_absolute_gate": maximum_final_abs <= MAX_FINAL_ABS_DELTA,
        "worst_role_rank_by_final_rms": CALL_ORDER[worst_rms_ordinal],
        "worst_ordinal_by_final_rms": worst_rms_ordinal,
        "worst_role_rank_by_final_maximum_absolute": CALL_ORDER[worst_max_ordinal],
        "worst_ordinal_by_final_maximum_absolute": worst_max_ordinal,
        "maximum_residual_wall_seconds": max(float(trial["literal_meter"]["residual_wall_time_s"]) for trial in trials),
        "deployment_residual_gate": max(float(trial["literal_meter"]["residual_wall_time_s"]) for trial in trials) <= MAX_DEPLOYMENT_RESIDUAL_SECONDS,
        "maximum_process_rss_bytes": max(int(trial["maximum_process_rss_bytes"]) for trial in trials),
        "rss_gate": max(int(trial["maximum_process_rss_bytes"]) for trial in trials) < MAX_RSS_BYTES,
    }
    aggregate["conjunctive_pass"] = all(
        value for key, value in aggregate.items() if key.endswith("_gate") or key.endswith("_pass") or key.endswith("_identical")
    )
    if not aggregate["conjunctive_pass"]:
        finish("FAIL_TERMINAL", trials, aggregate, "aggregate drift/repeat/resource gate failure")
        raise RuntimeError("terminal candidate aggregate failure; no retry")
    finish("PREREG_TARGET_FREE_PASS", trials, aggregate)


if __name__ == "__main__":
    main()
