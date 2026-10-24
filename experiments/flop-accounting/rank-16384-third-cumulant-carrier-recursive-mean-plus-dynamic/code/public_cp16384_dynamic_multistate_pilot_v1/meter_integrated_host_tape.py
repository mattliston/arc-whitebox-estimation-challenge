#!/usr/bin/env python3
"""One target-blind literal meter and source-parity gate for D3."""

from __future__ import annotations

from decimal import Decimal, getcontext
import difflib
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
INSTRUMENTED = HERE / "instrumented_estimator.py"
SOURCE_HOST = P2 / "cp3_cap16384_mean_recurrence_ablation/estimator.py"
LANE_WORKER = P2 / "cp3_full_birth_cap_ladder/mini100_seal_v1/lane_worker.py"
DYNAMIC4 = P2 / "public_full688_dynamic_multistate_residual_cv_v1/run_experiment.py"
FEATURE_STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
FEATURE_MANIFEST = FEATURE_STORE / "manifest.json"
PREREGISTRATION = HERE / "METER_PREREGISTRATION.json"
RUN_MANIFEST = HERE / "METER_RUN_MANIFEST.json"
SEAL = HERE / "METER_SEAL.json"
OUTPUT = HERE / "runs/integrated_meter_rank0000_v1"

INSTRUMENTED_SHA256 = (
    "76a87a1d161db9583dd318971b2688079d1919579d7a900fa9ec1a0230318105"
)
SOURCE_HOST_SHA256 = (
    "361c87cba008987c117871bf9892beecce75911634798ed7ad3fa26fd8ee5814"
)
LANE_WORKER_SHA256 = (
    "c86baaf705316ca43f041909d3c10136bfc05422c8697d4200885f825669937b"
)
DYNAMIC4_SHA256 = (
    "8871ac10f3f122d9231652e3618c888bdc69b71cc0de44ffa00a79630258225d"
)
FEATURE_MANIFEST_SHA256 = (
    "9a43423982924bcc7a0c5b87288b19ff42452c90c9cc7b7f590e565d1b276f50"
)
PHASE2_BUDGET = 2**41
SOURCE_HOST_FLOPS = 620_090_560_514
DYNAMIC4_CORRECTION_UPPER = 920_637_440
RAW20_MATERIALIZATION_RESERVE = 268_435_456
POST_TAPE_CORRECTION_UPPER = (
    DYNAMIC4_CORRECTION_UPPER - RAW20_MATERIALIZATION_RESERVE
)
ADMISSION_CEILING_EXCLUSIVE = 666_987_623_541


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


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def write_json_once(path: Path, payload: dict[str, object]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def verify_seal() -> dict[str, object]:
    seal = json.loads(SEAL.read_text(encoding="utf-8"))
    if seal.get("status") != "frozen-before-literal-meter-no-targets":
        raise AssertionError("meter seal status changed")
    expected = {
        "preregistration_sha256": sha256(PREREGISTRATION),
        "run_manifest_sha256": sha256(RUN_MANIFEST),
        "program_sha256": sha256(Path(__file__)),
        "instrumented_source_sha256": sha256(INSTRUMENTED),
        "source_host_sha256": sha256(SOURCE_HOST),
        "lane_worker_sha256": sha256(LANE_WORKER),
        "dynamic4_source_sha256": sha256(DYNAMIC4),
        "feature_manifest_sha256": sha256(FEATURE_MANIFEST),
    }
    for key, observed in expected.items():
        if seal.get(key) != observed:
            raise AssertionError(f"meter binding changed: {key}: {observed}")
    return seal


def main() -> None:
    seal = verify_seal()
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    feature_manifest = json.loads(FEATURE_MANIFEST.read_text(encoding="utf-8"))
    entry = feature_manifest["roles"]["observer_fit"]["row_manifests"][0]
    if int(entry["role_rank"]) != 0:
        raise AssertionError("frozen meter row is no longer observer_fit rank 0")
    row_manifest_path = FEATURE_STORE / str(entry["path"])
    if sha256(row_manifest_path) != str(entry["sha256"]):
        raise AssertionError("meter row manifest hash changed")
    row_manifest = json.loads(row_manifest_path.read_text(encoding="utf-8"))
    weights_path = row_manifest_path.parent / "weights.npy"
    if sha256(weights_path) != row_manifest["files"]["weights.npy"]["sha256"]:
        raise AssertionError("meter weight hash changed")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError("meter weights schema changed")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=int(derive_estimator_seed(int(row_manifest["mlp_seed"]))),
        name="target-blind-observer-fit-rank-0000-d3-meter",
    )
    mlp.validate()

    instrumented = load_module("p2r0c1d3_instrumented_meter", INSTRUMENTED)
    started = time.perf_counter()
    with threadpool_limits(limits=4):
        context = flops.BudgetContext(
            flop_budget=PHASE2_BUDGET,
            wall_time_limit_s=120.0,
            quiet=True,
            namespace="p2r0c1d3_integrated_host_tape_rank0000",
        )
        with context:
            prediction_value, raw_value, sigma_value = instrumented.predict_with_tape(mlp)
    integrated_wall = time.perf_counter() - started
    summary = context.summary_dict(by_namespace=False)
    prediction = np.ascontiguousarray(np.asarray(prediction_value), dtype=np.dtype("<f4"))
    raw20 = np.ascontiguousarray(np.asarray(raw_value), dtype=np.dtype("<f4"))
    sigma = np.ascontiguousarray(np.asarray(sigma_value), dtype=np.dtype("<f4"))
    del prediction_value, raw_value, sigma_value, instrumented
    gc.collect()
    if prediction.shape != (16, 1024) or not np.isfinite(prediction).all():
        raise AssertionError("integrated prediction contract failed")
    if raw20.shape != (16, 1024, 20) or not np.isfinite(raw20).all():
        raise AssertionError("integrated raw20 contract failed")
    if sigma.shape != (16, 1024) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
        raise AssertionError("integrated sigma contract failed")
    expected_depth = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    if not np.array_equal(raw20[:, :, 0], np.broadcast_to(expected_depth[:, None], (16, 1024))):
        raise AssertionError("raw20 depth feature changed")
    expected_rank = np.arange(16, dtype=np.float32) / np.float32(16.0)
    if not np.array_equal(raw20[:, :, 1], np.broadcast_to(expected_rank[:, None], (16, 1024))):
        raise AssertionError("raw20 full-birth rank feature changed")
    if not np.array_equal(raw20[:, :, 2], np.ones((16, 1024), dtype=np.float32)):
        raise AssertionError("full-birth retained feature is not identically one")

    lane = load_module("p2r0c1d3_exact_source_lane", LANE_WORKER)
    scratch = OUTPUT.parent / ".d3_meter_source_scratch"
    scratch.mkdir(parents=True, exist_ok=False)
    try:
        source_result = lane.predict_lane(
            estimator_path=str(SOURCE_HOST),
            expected_estimator_sha256=SOURCE_HOST_SHA256,
            class_name="EstimatorRecursiveMeanCarry",
            label="p2r0c1d3_exact_source_parity_rank0000",
            expected_flops=SOURCE_HOST_FLOPS,
            weights_path=str(weights_path),
            mlp_name="target-blind-observer-fit-rank-0000-d3-source-parity",
            mlp_seed=int(row_manifest["mlp_seed"]),
            scratch_dir=str(scratch),
        )
    finally:
        scratch.rmdir()
    source_prediction = np.ascontiguousarray(
        source_result.pop("prediction"), dtype=np.dtype("<f4")
    )
    parity = bool(np.array_equal(prediction, source_prediction))
    if not parity:
        raise AssertionError("instrumented host is not byte-identical to exact source")

    source_text = SOURCE_HOST.read_text(encoding="utf-8").splitlines(keepends=True)
    instrumented_text = INSTRUMENTED.read_text(encoding="utf-8").splitlines(keepends=True)
    diff_text = "".join(
        difflib.unified_diff(
            source_text,
            instrumented_text,
            fromfile=f"source/{SOURCE_HOST.name}",
            tofile=f"instrumented/{INSTRUMENTED.name}",
        )
    )
    diff_sha256 = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    integrated_flops = int(summary["flops_used"])
    combined_upper = integrated_flops + POST_TAPE_CORRECTION_UPPER
    getcontext().prec = 50
    multiplier = Decimal(combined_upper) / Decimal(PHASE2_BUDGET)
    break_even_raw = Decimal("1e-8") / multiplier
    admitted = combined_upper < ADMISSION_CEILING_EXCLUSIVE
    if not admitted:
        raise RuntimeError(
            f"integrated D3 cost {combined_upper} fails ceiling {ADMISSION_CEILING_EXCLUSIVE}"
        )

    OUTPUT.mkdir(parents=True, exist_ok=False)
    np.save(OUTPUT / "host.npy", prediction, allow_pickle=False)
    np.save(OUTPUT / "raw20.npy", raw20, allow_pickle=False)
    np.save(OUTPUT / "sigma.npy", sigma, allow_pickle=False)
    (OUTPUT / "SOURCE_DIFF.patch").write_text(diff_text, encoding="utf-8")
    result: dict[str, object] = {
        "schema": "p2r0c1d3-integrated-cp16384-raw20-meter-v1",
        "status": "pass-admitted-before-target-fit",
        "role": "observer_fit",
        "role_rank": 0,
        "mlp_id": int(entry["mlp_id"]),
        "name_or_board_join": False,
        "target_roles_opened": [],
        "target_columns_opened": [],
        "source": {
            "instrumented_sha256": INSTRUMENTED_SHA256,
            "exact_host_sha256": SOURCE_HOST_SHA256,
            "unified_diff_sha256": diff_sha256,
            "unified_diff_lines": len(diff_text.splitlines()),
            "prediction_byte_parity": parity,
            "prediction_array_sha256": array_sha256(prediction),
            "source_prediction_array_sha256": array_sha256(source_prediction),
        },
        "tape": {
            "raw20_array_sha256": array_sha256(raw20),
            "sigma_array_sha256": array_sha256(sigma),
            "full_birth_rank_sequence": list(range(0, 16384, 1024)),
            "retained_fraction_identically_one": True,
        },
        "literal_integrated_meter": {
            "flops_used": integrated_flops,
            "flop_budget": int(summary["flop_budget"]),
            "flops_remaining": int(summary["flops_remaining"]),
            "wall_time_s": float(summary["wall_time_s"]),
            "residual_wall_time_s": float(summary["residual_wall_time_s"]),
            "host_tape_wall_time_s": integrated_wall,
            "operations": summary["operations"],
        },
        "source_host_meter": source_result["meter"],
        "cost_admission": {
            "dynamic4_original_correction_upper": DYNAMIC4_CORRECTION_UPPER,
            "raw20_materialization_already_in_literal_tape": RAW20_MATERIALIZATION_RESERVE,
            "post_tape_dynamic4_correction_upper": POST_TAPE_CORRECTION_UPPER,
            "integrated_host_tape_plus_correction_upper": combined_upper,
            "admission_ceiling_exclusive": ADMISSION_CEILING_EXCLUSIVE,
            "admitted": admitted,
            "phase2_budget": PHASE2_BUDGET,
            "score_multiplier": str(multiplier),
            "break_even_raw_mse_for_adjusted_below_1e_8": str(break_even_raw),
        },
        "bindings": {
            "meter_seal_sha256": sha256(SEAL),
            "preregistration_sha256": seal["preregistration_sha256"],
            "run_manifest_sha256": seal["run_manifest_sha256"],
            "program_sha256": seal["program_sha256"],
            "feature_manifest_sha256": FEATURE_MANIFEST_SHA256,
            "row_manifest_sha256": str(entry["sha256"]),
            "weights_file_sha256": row_manifest["files"]["weights.npy"]["sha256"],
            "dynamic4_source_sha256": DYNAMIC4_SHA256,
        },
    }
    write_json_once(OUTPUT / "RESULT.json", result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "integrated_flops": integrated_flops,
                "combined_upper": combined_upper,
                "break_even_raw": str(break_even_raw),
                "prediction_parity": parity,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
