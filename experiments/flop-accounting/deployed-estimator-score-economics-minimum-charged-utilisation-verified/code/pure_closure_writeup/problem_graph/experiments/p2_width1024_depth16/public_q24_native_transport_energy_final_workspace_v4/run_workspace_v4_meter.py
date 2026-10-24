#!/usr/bin/env python3
"""Sealed target-free persistent-worker literal meter for H7B workspace v4."""

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
from whestbench.seeds import derive_estimator_seed


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = HERE / "estimator.py"
SOURCE_FREEZE = HERE / "SOURCE_FREEZE.json"
PREREGISTRATION = HERE / "METER_PREREGISTRATION.json"
RUN_MANIFEST = HERE / "METER_RUN_MANIFEST.json"
SEAL = HERE / "METER_SEAL.json"
RESULT = HERE / "METER_RESULT.json"
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)

SOURCE_SHA256 = "2fd2d40d4e30cc32269afb0113e244ee76c0a51005bce166c7625510f66a8a76"
FEATURE_MANIFEST_SHA256 = "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
EXPECTED_OUTPUTS = {
    35: "9f434f8f7f8eef8452560a13909026eaa397b4e85be9e92d7816e63cc9620047",
    279: "db57adf6c8e448ccf39e40ecefb4ba298fcc701536596987182fdfc550657aee",
}
EXPECTED_FLOPS = (2132366833051, 2123756968456, 2132363680256)
CALL_ORDER = (35, 279, 35)
PHASE2_BUDGET = 2**41
THREADS = 2
MAX_RSS_BYTES = 8 * (1 << 30)
MAX_WALL_SECONDS = 120.0
MAX_RESIDUAL_SECONDS = 0.4


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


def load_module(path: Path):
    specification = importlib.util.spec_from_file_location(
        "q24_native_all688_final_workspace_v4_meter", path
    )
    if specification is None or specification.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def verify_seal() -> dict[str, object]:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "frozen-before-workspace-v4-meter":
        raise AssertionError("workspace-v4 seal status changed")
    expected = {
        "source_sha256": sha256(SOURCE),
        "source_freeze_sha256": sha256(SOURCE_FREEZE),
        "preregistration_sha256": sha256(PREREGISTRATION),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "feature_manifest_sha256": sha256(FEATURE_STORE / "manifest.json"),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"workspace-v4 seal changed: {key}: {observed}")
    if expected["source_sha256"] != SOURCE_SHA256:
        raise AssertionError("workspace-v4 source literal changed")
    if expected["feature_manifest_sha256"] != FEATURE_MANIFEST_SHA256:
        raise AssertionError("workspace-v4 feature manifest literal changed")
    return seal


def load_fixture(role_rank: int, manifest: dict[str, object], run_manifest: dict[str, object]) -> MLP:
    entry = manifest["roles"]["observer_fit"]["row_manifests"][role_rank]
    declared = run_manifest["fixtures"][str(role_rank)]
    if int(entry["role_rank"]) != role_rank or int(entry["mlp_id"]) != int(declared["mlp_id"]):
        raise AssertionError(f"fixture identity changed: {role_rank}")
    row_manifest_path = FEATURE_STORE / str(entry["path"])
    if str(entry["path"]) != declared["row_manifest_path"]:
        raise AssertionError(f"fixture path changed: {role_rank}")
    if sha256(row_manifest_path) != declared["row_manifest_sha256"]:
        raise AssertionError(f"row manifest changed: {role_rank}")
    row = json.loads(row_manifest_path.read_text(encoding="utf-8"))
    if row.get("role") != "observer_fit" or int(row.get("role_rank", -1)) != role_rank:
        raise AssertionError(f"fixture role changed: {role_rank}")
    weights_path = row_manifest_path.parent / "weights.npy"
    if sha256(weights_path) != declared["weights_sha256"]:
        raise AssertionError(f"fixture weights changed: {role_rank}")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError(f"fixture weight schema changed: {role_rank}")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
        name=f"target-blind-workspace-v4-meter-rank-{role_rank}",
    )
    mlp.validate()
    return mlp


def main() -> None:
    seal = verify_seal()
    if RESULT.exists():
        raise FileExistsError(RESULT)
    manifest = json.loads((FEATURE_STORE / "manifest.json").read_text(encoding="utf-8"))
    run_manifest = json.loads(RUN_MANIFEST.read_text(encoding="utf-8"))
    fixtures = {
        role_rank: load_fixture(role_rank, manifest, run_manifest)
        for role_rank in sorted(set(CALL_ORDER))
    }
    module = load_module(SOURCE)
    estimator = module.Estimator()
    trials = []
    with threadpool_limits(limits=THREADS):
        for ordinal, role_rank in enumerate(CALL_ORDER):
            context = flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=MAX_WALL_SECONDS,
                quiet=True,
                namespace=f"q24_native_workspace_v4_trial_{ordinal}",
            )
            started = time.perf_counter()
            with context:
                value = estimator.predict(fixtures[role_rank], PHASE2_BUDGET)
            exact_wall = time.perf_counter() - started
            summary = context.summary_dict(by_namespace=False)
            output = np.ascontiguousarray(np.asarray(value), dtype=np.float32)
            output_hash = array_sha256(output)
            max_rss = resident_bytes()
            gates = {
                "flops_exact": int(summary["flops_used"]) == EXPECTED_FLOPS[ordinal],
                "flops_within_2pow41": int(summary["flops_used"]) <= PHASE2_BUDGET,
                "wall_below_120_seconds": float(summary["wall_time_s"]) < MAX_WALL_SECONDS,
                "residual_strictly_below_0_4_seconds": float(summary["residual_wall_time_s"]) < MAX_RESIDUAL_SECONDS,
                "rss_below_8_gib": max_rss < MAX_RSS_BYTES,
                "output_float32_16x1024": output.dtype == np.float32 and output.shape == (16, 1024),
                "output_finite": bool(np.isfinite(output).all()),
                "output_exact_byte_parity": output_hash == EXPECTED_OUTPUTS[role_rank],
            }
            gates["conjunctive_pass"] = all(gates.values())
            trial = {
                "ordinal": ordinal,
                "role_rank": role_rank,
                "literal_meter": summary,
                "exact_outer_wall_seconds": exact_wall,
                "maximum_process_rss_bytes": max_rss,
                "output_sha256": output_hash,
                "gates": gates,
            }
            trials.append(trial)
            print(json.dumps(trial, sort_keys=True), flush=True)
    aggregate = {
        "all_three_conjunctive_pass": all(t["gates"]["conjunctive_pass"] for t in trials),
        "all_three_residual_pass": all(t["gates"]["residual_strictly_below_0_4_seconds"] for t in trials),
        "repeat_rank35_exact": trials[0]["output_sha256"] == trials[2]["output_sha256"],
        "maximum_residual_wall_seconds": max(float(t["literal_meter"]["residual_wall_time_s"]) for t in trials),
    }
    aggregate["conjunctive_pass"] = all(aggregate.values())
    result = {
        "schema": "q24-native-all688-final-workspace-v4-meter-result-v1",
        "status": "PASS" if aggregate["conjunctive_pass"] else "FAIL",
        "node": "P2R0H7B_RESIDUAL_WORKSPACE_V4",
        "role": "observer_fit",
        "call_order": list(CALL_ORDER),
        "target_columns_opened": [],
        "observer_selection_opened": False,
        "name_or_board_join": False,
        "trial_results": trials,
        "aggregate": aggregate,
        "bindings": {
            "meter_seal_sha256": sha256(SEAL),
            "source_sha256": SOURCE_SHA256,
            "source_freeze_sha256": seal["source_freeze_sha256"],
            "preregistration_sha256": seal["preregistration_sha256"],
            "run_manifest_sha256": seal["run_manifest_sha256"],
            "program_sha256": seal["program_sha256"],
            "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
        },
    }
    write_json_once(RESULT, result)
    print(json.dumps({"status": result["status"], "aggregate": aggregate}, sort_keys=True), flush=True)
    if not aggregate["conjunctive_pass"]:
        raise RuntimeError("workspace-v4 sealed literal meter failed")


if __name__ == "__main__":
    main()
