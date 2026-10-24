#!/usr/bin/env python3
"""Generate one target-free CP4-cap output for the frozen sign screen."""

from __future__ import annotations

import argparse
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
PREREGISTRATION = HERE / "PREREGISTRATION.json"
PREREGISTRATION_SHA256 = "9d5bb3045cea6d170c906155889b11d00fa3153d5f97e939c5a609f66c6c88cb"
SOURCES = {
    "cap1792": (
        P2 / "public_cp4_cap1792_card_v1/estimator.py",
        "c0e68e11af37e118ee9ff906bd607280e4353cf6d378e22e41fffdfe726414c6",
    ),
    "cap1024": (
        P2 / "public_cp4_cap1024_card_v1/estimator.py",
        "7a1f2e3f3a51ae3c44cc4717c56fd70d0f6bef11c4b2fabf27f3937de6900e0e",
    ),
    "cap768": (
        HERE / "estimator_cap768.py",
        "33c2f01770c95b1490c611a5b6d62c5e0568cf49ee8722cf52381c07ed088e26",
    ),
}
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
PHASE2_BUDGET = 2**41
MAX_WALL_SECONDS = 120.0
MAX_RSS_BYTES = 8 * (1 << 30)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb", buffering=0) as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def resident_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform != "darwin":
        value *= 1024
    return value


def load_module(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(SOURCES), required=True)
    parser.add_argument("--rank", type=int, choices=tuple(FIXTURES), required=True)
    arguments = parser.parse_args()
    arm = arguments.arm
    role_rank = arguments.rank
    output = HERE / "outputs" / f"{arm}_rank{role_rank}"
    if output.exists():
        raise FileExistsError(output)
    if sha256(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise AssertionError("preregistration changed")
    source, source_hash = SOURCES[arm]
    if sha256(source) != source_hash:
        raise AssertionError("cap source changed")
    weights_path, weights_hash = FIXTURES[role_rank]
    if sha256(weights_path) != weights_hash:
        raise AssertionError("fixture changed")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.dtype != np.dtype("<f4") or weights.shape != (16, 1024, 1024):
        raise AssertionError("fixture schema changed")
    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[layer] for layer in range(16)],
        seed=0,
        name="sealed-target-free-weight-fixture",
    )
    mlp.validate()
    module = load_module(f"cp4_sign_{arm}_{role_rank}", source)
    estimator = module.Estimator()
    context = flops.BudgetContext(
        flop_budget=PHASE2_BUDGET,
        wall_time_limit_s=MAX_WALL_SECONDS,
        quiet=True,
        namespace=f"cp4_sign_{arm}_{role_rank}",
    )
    started = time.perf_counter()
    with threadpool_limits(limits=2), context:
        value = estimator.predict(mlp, PHASE2_BUDGET)
    outer_wall = time.perf_counter() - started
    summary = context.summary_dict(by_namespace=False)
    prediction = np.ascontiguousarray(np.asarray(value))
    operation_sum = sum(int(item["flop_cost"]) for item in summary["operations"].values())
    rss = resident_bytes()
    gates = {
        "float32_16x1024": bool(prediction.dtype == np.float32 and prediction.shape == (16, 1024)),
        "finite": bool(np.isfinite(prediction).all()),
        "operation_sum_equals_bill": operation_sum == int(summary["flops_used"]),
        "below_phase2_budget": int(summary["flops_used"]) < PHASE2_BUDGET,
        "outer_wall_below_120s": outer_wall < MAX_WALL_SECONDS,
        "meter_wall_below_120s": float(summary["wall_time_s"]) < MAX_WALL_SECONDS,
        "rss_below_8_gib": rss < MAX_RSS_BYTES,
    }
    if not all(gates.values()):
        raise RuntimeError({"arm": arm, "rank": role_rank, "gates": gates})
    temporary = HERE / "outputs" / f".{arm}_rank{role_rank}.{os.getpid()}.tmp"
    temporary.mkdir(parents=True, exist_ok=False)
    prediction_path = temporary / "prediction.npy"
    with prediction_path.open("xb") as handle:
        np.save(handle, prediction, allow_pickle=False)
    receipt = {
        "schema": "q24-cp4-cap-co-design-target-free-arm-v1",
        "status": "PASS_TARGET_FREE_ARM",
        "arm": arm,
        "role": "observer_fit",
        "role_rank": role_rank,
        "literal_meter": summary,
        "literal_operation_sum": operation_sum,
        "outer_wall_seconds": outer_wall,
        "maximum_process_rss_bytes": rss,
        "prediction_file_sha256": sha256(prediction_path),
        "prediction_array_sha256": array_sha256(prediction),
        "gates": gates,
        "bindings": {
            "preregistration_sha256": sha256(PREREGISTRATION),
            "source_sha256": sha256(source),
            "weights_sha256": sha256(weights_path),
            "runner_sha256": sha256(Path(__file__)),
        },
        "access_declarations": {
            "weights_only": True,
            "target_columns_opened": [],
            "scores_opened": [],
            "board_opened": False,
            "submission": False,
            "network": False,
        },
    }
    (temporary / "RECEIPT.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(output)
    print(json.dumps({"status": receipt["status"], "arm": arm, "rank": role_rank, "flops": summary["flops_used"]}, sort_keys=True))


if __name__ == "__main__":
    main()
