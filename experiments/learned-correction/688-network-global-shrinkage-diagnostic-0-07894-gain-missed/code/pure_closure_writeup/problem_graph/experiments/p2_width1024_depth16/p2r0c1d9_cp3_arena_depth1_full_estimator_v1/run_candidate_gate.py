#!/usr/bin/env python3
"""Candidate-only [35,279,35] gate for the D1 arena full estimator.

The runner makes zero reference-estimator calls and opens no targets, scores,
names, board resources, or network paths.  It must be invoked at most once by
the root executor after independent review; this file does not authorize a run.
"""

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
P2 = HERE.parent
D5 = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1"
LADDER = P2 / "p2r0c1d7_cp3_arena_depth_ladder_v1"
INCUMBENT = P2 / "public_noeigh_card_v1" / "estimator.py"
SOURCE = HERE / "estimator.py"
BUILDER = HERE / "build_candidate.py"
REFERENCE_EVIDENCE = D5 / "RECOVERED_REFERENCE_EVIDENCE.json"
RESULT = HERE / "D1_CANDIDATE_RESULT.json"

REFERENCE_OUTPUTS = (
    D5 / "reference_rank35_first.npy",
    D5 / "reference_rank279.npy",
    D5 / "reference_rank35_repeat.npy",
)
CANDIDATE_OUTPUTS = (
    HERE / "candidate_rank35_first.npy",
    HERE / "candidate_rank279.npy",
    HERE / "candidate_rank35_repeat.npy",
)

CALL_ORDER = (35, 279, 35)
FIXTURES = {
    35: (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit/0035/weights.npy",
        "14563b956b4e326aee081263077c0dd1dcb86037a0b7870dc58a5e33865f47f2",
    ),
    279: (
        P2 / "learned_residual_observer_design/v4_feature_store_v3/"
        "sealed_target_blind_feature_store_v3/rows/observer_fit/0279/weights.npy",
        "46b1e740e32de76c0b5d4d855ac307e8ba1509497303456f2dfc42ece2f34298",
    ),
}

SOURCE_SHA256 = "f2ec257c891241b57c85803c70c73f365e075711e2b2cd214d9b58250762892e"
BUILDER_SHA256 = "ab49dfc7340d819fbf6df0f721d898161f6d54667b545eb7cc0b37be7055b09b"
INCUMBENT_SHA256 = "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
LADDER_SHA256 = "98b48f71588c1bb332fbea6608b4e5d32b7b433164eb276143640a2e2858ae19"
REFERENCE_EVIDENCE_SHA256 = (
    "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6"
)
REFERENCE_OUTPUT_SHA256 = (
    "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
    "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba",
    "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812",
)

EXPECTED_REFERENCE_FLOPS = (
    1_692_484_904_311,
    1_731_128_080_904,
    1_692_481_751_552,
)
EXPECTED_CANDIDATE_FLOPS = (
    1_574_162_015_607,
    1_612_602_817_032,
    1_573_956_487_680,
)
EXPECTED_REFERENCE_MINUS_CANDIDATE = (
    118_322_888_704,
    118_525_263_872,
    118_525_263_872,
)
EXACT_HOT_SAVING = 118_525_263_872
COLD_VIEW_BINDING_BILL = 202_375_168

PHASE2_BUDGET = 2**41
THREADS = 2
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_WALL_SECONDS = 120.0
MAX_RESIDUAL_SECONDS = 0.4
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
    digest.update(
        json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
    )
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


def verify_static_bindings() -> None:
    bindings = {
        SOURCE: SOURCE_SHA256,
        BUILDER: BUILDER_SHA256,
        INCUMBENT: INCUMBENT_SHA256,
        LADDER / "DEPTH_LADDER_RECEIPT.json": LADDER_SHA256,
        REFERENCE_EVIDENCE: REFERENCE_EVIDENCE_SHA256,
    }
    for path, expected in bindings.items():
        observed = sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))
    for path, expected in zip(REFERENCE_OUTPUTS, REFERENCE_OUTPUT_SHA256):
        observed = sha256(path)
        if observed != expected:
            raise AssertionError((str(path), observed, expected))
    if EXPECTED_REFERENCE_MINUS_CANDIDATE[0] + COLD_VIEW_BINDING_BILL != EXACT_HOT_SAVING:
        raise AssertionError("cold/hot accounting changed")
    if EXPECTED_REFERENCE_MINUS_CANDIDATE[1:] != (EXACT_HOT_SAVING,) * 2:
        raise AssertionError("hot accounting changed")


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("d1_arena_candidate_gate", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fixture(role_rank: int) -> MLP:
    path, expected = FIXTURES[role_rank]
    if sha256(path) != expected:
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


def load_references() -> tuple[dict[str, object], list[np.ndarray]]:
    evidence = json.loads(REFERENCE_EVIDENCE.read_text())
    if evidence.get("status") != "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE":
        raise AssertionError("recovered reference status changed")
    if tuple(evidence.get("call_order", ())) != CALL_ORDER:
        raise AssertionError("recovered call order changed")
    if evidence.get("original_candidate_calls") != 0:
        raise AssertionError("original candidate calls changed")
    if evidence.get("recovery_reference_estimator_calls") != 0:
        raise AssertionError("recovery reference calls changed")
    if evidence.get("candidate_calls_at_recovery") != 0:
        raise AssertionError("recovery candidate calls changed")
    if evidence.get("access_declarations") != {
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "training": False,
        "network": False,
    }:
        raise AssertionError("recovered access declaration changed")

    arrays: list[np.ndarray] = []
    for ordinal, path in enumerate(REFERENCE_OUTPUTS):
        trial = evidence["trials"][ordinal]
        if trial["ordinal"] != ordinal or trial["role_rank"] != CALL_ORDER[ordinal]:
            raise AssertionError(f"reference roster changed: {ordinal}")
        if trial["output_file_sha256"] != REFERENCE_OUTPUT_SHA256[ordinal]:
            raise AssertionError(f"reference file binding changed: {ordinal}")
        if int(trial["literal_meter"]["flops_used"]) != EXPECTED_REFERENCE_FLOPS[ordinal]:
            raise AssertionError(f"reference bill changed: {ordinal}")
        if int(trial["literal_operation_sum"]) != EXPECTED_REFERENCE_FLOPS[ordinal]:
            raise AssertionError(f"reference operation sum changed: {ordinal}")
        array = np.load(path, allow_pickle=False)
        if array.dtype != np.float32 or array.shape != (16, 1024):
            raise AssertionError(f"reference schema changed: {ordinal}")
        if not np.isfinite(array).all():
            raise AssertionError(f"reference finiteness changed: {ordinal}")
        if array_sha256(array) != trial["output_array_sha256"]:
            raise AssertionError(f"reference array changed: {ordinal}")
        arrays.append(np.ascontiguousarray(array, dtype=np.float32))
    if array_sha256(arrays[0]) != array_sha256(arrays[2]):
        raise AssertionError("reference repeat mismatch")
    return evidence, arrays


def layer_drift(candidate: np.ndarray, reference: np.ndarray) -> dict[str, object]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    rms = np.sqrt(np.mean(delta * delta, axis=1))
    maximum = np.max(np.abs(delta), axis=1)
    return {
        "per_layer_rms": [float(value) for value in rms],
        "per_layer_maximum_absolute": [float(value) for value in maximum],
        "final_layer_rms": float(rms[-1]),
        "final_layer_maximum_absolute": float(maximum[-1]),
    }


def finish(
    status: str,
    trials: list[dict[str, object]],
    aggregate: dict[str, object],
    reason: str | None = None,
) -> None:
    payload = {
        "schema": "p2r0c1-d1-arena-candidate-only-gate-v1",
        "status": status,
        "call_order": list(CALL_ORDER),
        "reference_estimator_calls": 0,
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
        "bindings": {
            "runner_sha256": sha256(Path(__file__)),
            "candidate_source_sha256": sha256(SOURCE),
            "builder_sha256": sha256(BUILDER),
            "incumbent_source_sha256": sha256(INCUMBENT),
            "depth_ladder_receipt_sha256": sha256(
                LADDER / "DEPTH_LADDER_RECEIPT.json"
            ),
            "recovered_reference_evidence_sha256": sha256(REFERENCE_EVIDENCE),
        },
    }
    write_json_once(RESULT, payload)
    print(
        json.dumps(
            {"status": status, "reason": reason, "aggregate": aggregate},
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    verify_static_bindings()
    if RESULT.exists() or any(path.exists() for path in CANDIDATE_OUTPUTS):
        raise FileExistsError("D1 candidate-only gate has already been attempted")
    reference_evidence, references = load_references()
    fixtures = {rank: load_fixture(rank) for rank in sorted(set(CALL_ORDER))}
    module = load_module(SOURCE)
    estimator = module.Estimator()
    trials: list[dict[str, object]] = []
    outputs: list[np.ndarray] = []
    first_hash: str | None = None

    with threadpool_limits(limits=THREADS):
        for ordinal, role_rank in enumerate(CALL_ORDER):
            context = flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=MAX_WALL_SECONDS,
                quiet=True,
                namespace=f"d1_arena_candidate_{ordinal}",
            )
            started = time.perf_counter()
            with context:
                value = estimator.predict(fixtures[role_rank], PHASE2_BUDGET)
            outer_wall = time.perf_counter() - started
            summary = context.summary_dict(by_namespace=False)
            raw = np.asarray(value)
            schema_ok = raw.dtype == np.float32 and raw.shape == (16, 1024)
            output = np.ascontiguousarray(raw)
            output_hash = array_sha256(output)
            operation_sum = sum(
                int(item["flop_cost"])
                for item in summary["operations"].values()
            )
            candidate_flops = int(summary["flops_used"])
            reference_flops = int(
                reference_evidence["trials"][ordinal]["literal_meter"]["flops_used"]
            )
            measured_saving = reference_flops - candidate_flops
            drift = layer_drift(output, references[ordinal])
            repeat_ok = ordinal != 2 or output_hash == first_hash
            rss = resident_bytes()
            gates = {
                "literal_flops_exact": candidate_flops
                == EXPECTED_CANDIDATE_FLOPS[ordinal],
                "literal_operation_sum_equals_total": operation_sum
                == candidate_flops,
                "reference_minus_candidate_exact": measured_saving
                == EXPECTED_REFERENCE_MINUS_CANDIDATE[ordinal],
                "hot_saving_exact_if_hot": ordinal == 0
                or measured_saving == EXACT_HOT_SAVING,
                "flops_strictly_below_2pow41": candidate_flops < PHASE2_BUDGET,
                "outer_wall_strictly_below_120": outer_wall < MAX_WALL_SECONDS,
                "meter_wall_strictly_below_120": float(summary["wall_time_s"])
                < MAX_WALL_SECONDS,
                "official_residual_strictly_below_0_4": float(
                    summary["residual_wall_time_s"]
                )
                < MAX_RESIDUAL_SECONDS,
                "rss_strictly_below_8_gib": rss < MAX_RSS_BYTES,
                "raw_output_float32_16x1024": schema_ok,
                "output_finite": bool(np.isfinite(output).all()),
                "final_layer_maximum_absolute_delta": drift[
                    "final_layer_maximum_absolute"
                ]
                <= MAX_FINAL_ABS_DELTA,
                "repeat_rank35_byte_identical": repeat_ok,
            }
            gates["conjunctive_call_pass"] = all(gates.values())
            write_array_once(CANDIDATE_OUTPUTS[ordinal], output)
            trial = {
                "ordinal": ordinal,
                "role_rank": role_rank,
                "literal_meter": summary,
                "literal_operation_sum": operation_sum,
                "reference_flops": reference_flops,
                "candidate_flops": candidate_flops,
                "reference_minus_candidate_flops": measured_saving,
                "exact_outer_wall_seconds": outer_wall,
                "maximum_process_rss_bytes": rss,
                "output_array_sha256": output_hash,
                "output_file": CANDIDATE_OUTPUTS[ordinal].name,
                "output_file_sha256": sha256(CANDIDATE_OUTPUTS[ordinal]),
                "drift_vs_incumbent": drift,
                "gates": gates,
            }
            trials.append(trial)
            outputs.append(output)
            print(json.dumps(trial, sort_keys=True), flush=True)
            if ordinal == 0:
                first_hash = output_hash
            if not gates["conjunctive_call_pass"]:
                finish(
                    "FAIL_TERMINAL",
                    trials,
                    {},
                    f"candidate call {ordinal} gate failure",
                )
                raise RuntimeError("terminal D1 candidate failure; no retry")

    all_deltas = [
        output[-1].astype(np.float64) - reference[-1].astype(np.float64)
        for output, reference in zip(outputs, references)
    ]
    unique_deltas = all_deltas[:2]
    pooled_all_rms = float(
        np.sqrt(np.mean(np.concatenate(all_deltas) ** 2))
    )
    pooled_unique_rms = float(
        np.sqrt(np.mean(np.concatenate(unique_deltas) ** 2))
    )
    maximum_abs = float(max(np.max(np.abs(delta)) for delta in all_deltas))
    aggregate = {
        "all_three_call_gates_pass": all(
            trial["gates"]["conjunctive_call_pass"] for trial in trials
        ),
        "repeat_rank35_candidate_byte_identical": trials[0][
            "output_array_sha256"
        ]
        == trials[2]["output_array_sha256"],
        "pooled_all_three_final_layer_rms_delta": pooled_all_rms,
        "pooled_all_three_final_rms_gate": pooled_all_rms
        <= MAX_POOLED_FINAL_RMS_DELTA,
        "pooled_unique_fixture_final_layer_rms_delta": pooled_unique_rms,
        "pooled_unique_fixture_final_rms_gate": pooled_unique_rms
        <= MAX_POOLED_FINAL_RMS_DELTA,
        "maximum_final_layer_absolute_delta": maximum_abs,
        "maximum_final_absolute_gate": maximum_abs <= MAX_FINAL_ABS_DELTA,
        "maximum_official_residual_wall_seconds": max(
            float(trial["literal_meter"]["residual_wall_time_s"])
            for trial in trials
        ),
        "official_residual_gate": max(
            float(trial["literal_meter"]["residual_wall_time_s"])
            for trial in trials
        )
        < MAX_RESIDUAL_SECONDS,
        "maximum_process_rss_bytes": max(
            int(trial["maximum_process_rss_bytes"]) for trial in trials
        ),
        "rss_gate": max(
            int(trial["maximum_process_rss_bytes"]) for trial in trials
        )
        < MAX_RSS_BYTES,
    }
    aggregate["conjunctive_pass"] = all(
        value
        for key, value in aggregate.items()
        if key.endswith("_gate")
        or key.endswith("_pass")
        or key.endswith("_identical")
    )
    if not aggregate["conjunctive_pass"]:
        finish(
            "FAIL_TERMINAL",
            trials,
            aggregate,
            "aggregate drift/repeat/resource gate failure",
        )
        raise RuntimeError("terminal D1 aggregate failure; no retry")
    finish("PASS_TARGET_FREE_CANDIDATE_GATE", trials, aggregate)


if __name__ == "__main__":
    main()
