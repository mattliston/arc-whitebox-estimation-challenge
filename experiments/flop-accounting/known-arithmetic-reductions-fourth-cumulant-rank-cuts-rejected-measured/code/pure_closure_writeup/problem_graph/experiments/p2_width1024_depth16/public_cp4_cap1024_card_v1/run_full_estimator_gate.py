#!/usr/bin/env python3
"""One-shot target-free full-estimator gate for the CP4-cap1024 card."""

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
SOURCE = HERE / "estimator.py"
INCUMBENT_SOURCE = P2 / "public_noeigh_card_v1/estimator.py"
RESULT = HERE / "FULL_ESTIMATOR_GATE_RESULT.json"
EXPECTED_SOURCE_SHA256 = "7a1f2e3f3a51ae3c44cc4717c56fd70d0f6bef11c4b2fabf27f3937de6900e0e"
EXPECTED_INCUMBENT_SOURCE_SHA256 = "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
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
REFERENCES = (
    (D5 / "reference_rank35_first.npy", "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812"),
    (D5 / "reference_rank279.npy", "f6970b298c3397b512635846c4fa75bb377ee6af0ca4ff0e6241db2dcd5015ba"),
    (D5 / "reference_rank35_repeat.npy", "391219e5ba7527a2b3e6926cd85a651f76749e33908110573192fb1bc0d13812"),
)
REFERENCE_EVIDENCE = D5 / "RECOVERED_REFERENCE_EVIDENCE.json"
EXPECTED_REFERENCE_EVIDENCE_SHA256 = "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6"
EXPECTED_INCUMBENT_FLOPS = (
    1_692_484_904_311,
    1_731_128_080_904,
    1_692_481_751_552,
)
PHASE2_BUDGET = 2**41
THREADS = 2
MAX_WALL_SECONDS = 120.0
MAX_RESIDUAL_SECONDS = 0.4
MAX_DEPLOYMENT_RESIDUAL_SECONDS = 0.385
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_FINAL_RMS_DELTA = 6.002184665264464e-6
MAX_FINAL_ABS_DELTA = 3.001092332632232e-5
INHERITED_FINAL_RMS_DELTA = 1.714909904361275e-6
INHERITED_FINAL_ABS_DELTA = 8.574549521806375e-6


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


def write_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def verify_source_lineage() -> None:
    if sha256(INCUMBENT_SOURCE) != EXPECTED_INCUMBENT_SOURCE_SHA256:
        raise AssertionError("incumbent source changed")
    if sha256(SOURCE) != EXPECTED_SOURCE_SHA256:
        raise AssertionError("candidate source changed")
    incumbent = INCUMBENT_SOURCE.read_bytes()
    needle = b"_CP4_CAP = 2048\n"
    if incumbent.count(needle) != 1:
        raise AssertionError("incumbent CP4 cap literal is not unique")
    expected_candidate = incumbent.replace(needle, b"_CP4_CAP = 1024\n", 1)
    if SOURCE.read_bytes() != expected_candidate:
        raise AssertionError("candidate is not the exact one-literal incumbent descendant")


def load_candidate():
    spec = importlib.util.spec_from_file_location("public_cp4_cap1024_candidate", SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_fixture(role_rank: int) -> MLP:
    path, expected = FIXTURES[role_rank]
    if sha256(path) != expected:
        raise AssertionError((path, sha256(path), expected))
    weights = np.load(path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError((role_rank, weights.dtype, weights.shape))
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=0,
        name="sealed-target-free-weight-fixture",
    )
    mlp.validate()
    return mlp


def load_references() -> list[np.ndarray]:
    if sha256(REFERENCE_EVIDENCE) != EXPECTED_REFERENCE_EVIDENCE_SHA256:
        raise AssertionError("reference evidence changed")
    evidence = json.loads(REFERENCE_EVIDENCE.read_text())
    assert evidence["status"] == "RECOVERED_REFERENCE_SCIENTIFIC_EVIDENCE"
    assert tuple(evidence["call_order"]) == CALL_ORDER
    assert evidence["recovery_reference_estimator_calls"] == 0
    arrays = []
    for ordinal, (path, expected) in enumerate(REFERENCES):
        assert sha256(path) == expected, (path, sha256(path), expected)
        array = np.load(path, allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (16, 1024)
        assert bool(np.isfinite(array).all())
        trial = evidence["trials"][ordinal]
        assert int(trial["literal_meter"]["flops_used"]) == EXPECTED_INCUMBENT_FLOPS[ordinal]
        assert array_sha256(array) == trial["output_array_sha256"]
        arrays.append(np.ascontiguousarray(array))
    assert array_sha256(arrays[0]) == array_sha256(arrays[2])
    return arrays


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


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    verify_source_lineage()
    references = load_references()
    fixtures = {rank: load_fixture(rank) for rank in sorted(set(CALL_ORDER))}
    baseline_rss = resident_bytes()
    module = load_candidate()
    estimator = module.Estimator()
    trials: list[dict[str, object]] = []
    final_layer_deltas: list[np.ndarray] = []
    first_hash: str | None = None
    with threadpool_limits(limits=THREADS):
        for ordinal, role_rank in enumerate(CALL_ORDER):
            context = flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=MAX_WALL_SECONDS,
                quiet=True,
                namespace=f"public_cp4_cap1024_full_{ordinal}",
            )
            started = time.perf_counter()
            with context:
                value = estimator.predict(fixtures[role_rank], PHASE2_BUDGET)
            outer_wall = time.perf_counter() - started
            summary = context.summary_dict(by_namespace=False)
            raw = np.asarray(value)
            output = np.ascontiguousarray(raw)
            output_hash = array_sha256(output)
            drift = layer_drift(output, references[ordinal])
            final_layer_deltas.append(
                output[-1].astype(np.float64) - references[ordinal][-1].astype(np.float64)
            )
            candidate_flops = int(summary["flops_used"])
            incumbent_flops = EXPECTED_INCUMBENT_FLOPS[ordinal]
            saving = incumbent_flops - candidate_flops
            operation_sum = sum(
                int(operation["flop_cost"])
                for operation in summary["operations"].values()
            )
            rss = resident_bytes()
            repeat_ok = ordinal != 2 or output_hash == first_hash
            gates = {
                "output_float32_16x1024": bool(
                    raw.dtype == np.float32 and raw.shape == (16, 1024)
                ),
                "output_finite": bool(np.isfinite(output).all()),
                "literal_operation_sum_matches_meter": operation_sum == candidate_flops,
                "flops_below_budget": candidate_flops < PHASE2_BUDGET,
                "measured_incumbent_minus_candidate_positive": saving > 0,
                "outer_wall_below_120s": outer_wall < MAX_WALL_SECONDS,
                "meter_wall_below_120s": float(summary["wall_time_s"]) < MAX_WALL_SECONDS,
                "residual_below_0_4s": float(summary["residual_wall_time_s"]) < MAX_RESIDUAL_SECONDS,
                "rss_below_8_gib": rss < MAX_RSS_BYTES,
                "final_layer_maximum_within_gate": bool(
                    drift["final_layer_maximum_absolute"] <= MAX_FINAL_ABS_DELTA
                ),
                "repeat_rank35_byte_identical": repeat_ok,
            }
            gates["conjunctive_call_pass"] = all(gates.values())
            diagnostics = {
                "deployment_residual_at_most_0_385s": float(
                    summary["residual_wall_time_s"]
                ) <= MAX_DEPLOYMENT_RESIDUAL_SECONDS,
                "stronger_inherited_final_maximum_gate": bool(
                    drift["final_layer_maximum_absolute"] <= INHERITED_FINAL_ABS_DELTA
                ),
            }
            trial = {
                "ordinal": ordinal,
                "role_rank": role_rank,
                "literal_meter": summary,
                "literal_operation_sum": operation_sum,
                "measured_incumbent_flops": incumbent_flops,
                "measured_candidate_flops": candidate_flops,
                "measured_incumbent_minus_candidate_flops": saving,
                "outer_wall_seconds": outer_wall,
                "maximum_process_rss_bytes": rss,
                "rss_increment_from_preimport_baseline_bytes": rss - baseline_rss,
                "output_array_sha256": output_hash,
                "drift_vs_established_incumbent_reference": drift,
                "gates": gates,
                "diagnostics": diagnostics,
            }
            trials.append(trial)
            print(json.dumps(trial, sort_keys=True), flush=True)
            if ordinal == 0:
                first_hash = output_hash

    per_call_final_rms = [
        float(trial["drift_vs_established_incumbent_reference"]["final_layer_rms"])
        for trial in trials
    ]
    per_call_final_max = [
        float(trial["drift_vs_established_incumbent_reference"]["final_layer_maximum_absolute"])
        for trial in trials
    ]
    pooled_all = np.concatenate(final_layer_deltas)
    pooled_unique = np.concatenate(final_layer_deltas[:2])
    pooled_all_rms = float(np.sqrt(np.mean(pooled_all * pooled_all)))
    pooled_unique_rms = float(np.sqrt(np.mean(pooled_unique * pooled_unique)))
    maximum_final_abs = max(per_call_final_max)
    maximum_residual = max(
        float(trial["literal_meter"]["residual_wall_time_s"]) for trial in trials
    )
    maximum_rss = max(int(trial["maximum_process_rss_bytes"]) for trial in trials)
    aggregate = {
        "all_call_gates_pass": all(trial["gates"]["conjunctive_call_pass"] for trial in trials),
        "repeat_rank35_byte_identical": trials[0]["output_array_sha256"] == trials[2]["output_array_sha256"],
        "measured_candidate_flops": [int(trial["measured_candidate_flops"]) for trial in trials],
        "measured_incumbent_minus_candidate_flops": [
            int(trial["measured_incumbent_minus_candidate_flops"]) for trial in trials
        ],
        "maximum_final_layer_rms_delta": max(per_call_final_rms),
        "pooled_all_three_calls_final_layer_rms_delta": pooled_all_rms,
        "pooled_all_three_calls_final_layer_rms_gate": pooled_all_rms <= MAX_FINAL_RMS_DELTA,
        "pooled_unique_fixture_final_layer_rms_delta": pooled_unique_rms,
        "pooled_unique_fixture_final_layer_rms_gate": pooled_unique_rms <= MAX_FINAL_RMS_DELTA,
        "maximum_final_layer_absolute_delta": maximum_final_abs,
        "maximum_final_absolute_gate": maximum_final_abs <= MAX_FINAL_ABS_DELTA,
        "stronger_inherited_pooled_all_three_rms": pooled_all_rms <= INHERITED_FINAL_RMS_DELTA,
        "stronger_inherited_pooled_unique_rms": pooled_unique_rms <= INHERITED_FINAL_RMS_DELTA,
        "stronger_inherited_final_absolute": maximum_final_abs <= INHERITED_FINAL_ABS_DELTA,
        "maximum_residual_wall_seconds": maximum_residual,
        "deployment_residual_at_most_0_385": maximum_residual <= MAX_DEPLOYMENT_RESIDUAL_SECONDS,
        "maximum_process_rss_bytes": maximum_rss,
        "rss_gate": maximum_rss < MAX_RSS_BYTES,
    }
    aggregate["conjunctive_pass"] = all(
        value
        for key, value in aggregate.items()
        if key.endswith("_gate") or key.endswith("_pass") or key.endswith("_identical")
    )
    payload = {
        "schema": "public-cp4-cap1024-full-estimator-target-free-gate-v1",
        "status": "PASS_FULL_ESTIMATOR_GATE" if aggregate["conjunctive_pass"] else "FAIL_FULL_ESTIMATOR_GATE",
        "node": "PUBLIC_CP4_CAP1024_CARD_V1",
        "call_order": list(CALL_ORDER),
        "threads": THREADS,
        "preimport_baseline_rss_bytes": baseline_rss,
        "trials": trials,
        "aggregate": aggregate,
        "gate_thresholds": {
            "phase2_flop_budget": PHASE2_BUDGET,
            "maximum_wall_seconds": MAX_WALL_SECONDS,
            "maximum_residual_wall_seconds_exclusive": MAX_RESIDUAL_SECONDS,
            "maximum_process_rss_bytes_exclusive": MAX_RSS_BYTES,
            "maximum_pooled_final_layer_rms_delta": MAX_FINAL_RMS_DELTA,
            "maximum_final_layer_absolute_delta": MAX_FINAL_ABS_DELTA,
        },
        "bindings": {
            "candidate_source_sha256": sha256(SOURCE),
            "incumbent_source_sha256": sha256(INCUMBENT_SOURCE),
            "source_lineage": "exact incumbent bytes with only _CP4_CAP = 2048 changed to 1024",
            "runner_sha256": sha256(Path(__file__)),
            "reference_evidence_sha256": sha256(REFERENCE_EVIDENCE),
            "reference_output_sha256": [sha256(path) for path, _ in REFERENCES],
            "fixture_weight_sha256": {str(rank): sha256(path) for rank, (path, _) in FIXTURES.items()},
        },
        "access_declarations": {
            "weights_only": True,
            "reference_estimator_calls": 0,
            "target_columns_opened": [],
            "scores_opened": [],
            "names_opened": [],
            "board_opened": False,
            "submission": False,
            "training": False,
            "network": False,
        },
    }
    write_once(RESULT, payload)
    print(json.dumps({"status": payload["status"], "aggregate": aggregate}, sort_keys=True), flush=True)
    if not aggregate["conjunctive_pass"]:
        raise RuntimeError(payload["status"])


if __name__ == "__main__":
    main()
