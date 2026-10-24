#!/usr/bin/env python3
"""One-shot exact-incumbent, weights-only reference capture for P2R0C1D5."""

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
SOURCE = HERE.parent / "public_noeigh_card_v1" / "estimator.py"
MANIFEST = HERE / "COMMON_DRIVER_MANIFEST.json"
SEAL = HERE / "COMMON_DRIVER_SEAL.json"
RESULT = HERE / "REFERENCE_RESULT.json"
OUTPUT_PATHS = (
    HERE / "reference_rank35_first.npy",
    HERE / "reference_rank279.npy",
    HERE / "reference_rank35_repeat.npy",
)
SOURCE_SHA256 = "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7"
CALL_ORDER = (35, 279, 35)
EXPECTED_FLOPS = (1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552)
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
    if seal.get("status") != "FROZEN_BEFORE_FIRST_COMMON_DRIVER_CALL":
        raise AssertionError("common-driver seal status changed")
    for relative, expected in seal["sha256"].items():
        observed = sha256((HERE / relative).resolve())
        if observed != expected:
            raise AssertionError((relative, observed, expected))
    manifest = json.loads(MANIFEST.read_text())
    if manifest["status"] != "FROZEN_BEFORE_FIRST_COMMON_DRIVER_CALL":
        raise AssertionError("manifest status changed")
    if tuple(manifest["call_order"]) != CALL_ORDER:
        raise AssertionError("reference call order changed")
    if tuple(manifest["expected_incumbent_flops"]) != EXPECTED_FLOPS:
        raise AssertionError("reference bill changed")
    return seal, manifest


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("p2r0c1d5_exact_incumbent", path)
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


def finish(
    status: str,
    seal: dict[str, object],
    trials: list[dict[str, object]],
    reason: str | None = None,
) -> None:
    result = {
        "schema": "p2r0c1d5-exact-incumbent-reference-v1",
        "status": status,
        "call_order": list(CALL_ORDER),
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": false,
        "trials": trials,
        "terminal_reason": reason,
        "bindings": {
            "common_driver_seal_sha256": sha256(SEAL),
            "manifest_sha256": sha256(MANIFEST),
            "program_sha256": sha256(Path(__file__)),
            "source_sha256": sha256(SOURCE),
        },
    }
    write_json_once(RESULT, result)
    print(json.dumps({"status": status, "reason": reason}, sort_keys=True), flush=True)


def main() -> None:
    seal, manifest = verify_seal()
    if sha256(SOURCE) != SOURCE_SHA256:
        raise AssertionError("exact incumbent source changed")
    if RESULT.exists() or any(path.exists() for path in OUTPUT_PATHS):
        raise FileExistsError("one-shot reference artifact already exists")
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
                namespace=f"p2r0c1d5_reference_{ordinal}",
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
            repeat_ok = ordinal != 2 or output_hash == first_hash
            gates = {
                "literal_flops_exact": int(summary["flops_used"]) == EXPECTED_FLOPS[ordinal],
                "literal_operation_sum_equals_total": operation_sum == int(summary["flops_used"]),
                "flops_strictly_below_2pow41": int(summary["flops_used"]) < PHASE2_BUDGET,
                "outer_wall_strictly_below_120": outer_wall < MAX_WALL_SECONDS,
                "meter_wall_strictly_below_120": float(summary["wall_time_s"]) < MAX_WALL_SECONDS,
                "residual_strictly_below_0_4": float(summary["residual_wall_time_s"]) < MAX_RESIDUAL_SECONDS,
                "rss_strictly_below_8_gib": resident_bytes() < MAX_RSS_BYTES,
                "raw_output_float32_16x1024": raw_schema_ok,
                "output_finite": bool(np.isfinite(output).all()),
                "repeat_rank35_byte_identical": repeat_ok,
            }
            gates["conjunctive_pass"] = all(gates.values())
            write_array_once(OUTPUT_PATHS[ordinal], output)
            trial = {
                "ordinal": ordinal,
                "role_rank": role_rank,
                "literal_meter": summary,
                "literal_operation_sum": operation_sum,
                "exact_outer_wall_seconds": outer_wall,
                "maximum_process_rss_bytes": resident_bytes(),
                "output_array_sha256": output_hash,
                "output_file": OUTPUT_PATHS[ordinal].name,
                "output_file_sha256": sha256(OUTPUT_PATHS[ordinal]),
                "gates": gates,
            }
            trials.append(trial)
            print(json.dumps(trial, sort_keys=True), flush=True)
            if ordinal == 0:
                first_hash = output_hash
            if not gates["conjunctive_pass"]:
                finish("FAIL_TERMINAL", seal, trials, f"reference call {ordinal} gate failure")
                raise RuntimeError("terminal reference failure; no retry")
    finish("PASS_REFERENCE_CAPTURE", seal, trials)


if __name__ == "__main__":
    main()
