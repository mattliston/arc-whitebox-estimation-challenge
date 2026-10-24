#!/usr/bin/env python3
"""Spawn-child entry point for one frozen full-birth CP cap prediction."""

from __future__ import annotations

import hashlib
from pathlib import Path
import time
from typing import Any

import flopscope as flops
import numpy as np
from threadpoolctl import threadpool_limits
from whestbench import SetupContext
from whestbench.domain import MLP
from whestbench.loader import load_estimator_from_path
from whestbench.seeds import derive_estimator_seed


WIDTH = 1024
DEPTH = 16
FLOP_BUDGET = 2**41
WALL_TIME_LIMIT_S = 120.0
RESIDUAL_WALL_TIME_LIMIT_S = 0.4
MAX_BLAS_THREADS = 4
ALLOWED_ESTIMATOR_SHA256 = frozenset(
    (
        "035e3bd2fd08d3358f5424fc67f10281e5559e65fa9c6d6dce97791108b176dc",
        "361c87cba008987c117871bf9892beecce75911634798ed7ad3fa26fd8ee5814",
        "63809cea47419c3f2321ed76db86492a2394fa025d6d17837da9d639e1753e43",
    )
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def predict_lane(
    *,
    estimator_path: str,
    expected_estimator_sha256: str,
    class_name: str,
    label: str,
    expected_flops: int,
    weights_path: str,
    mlp_name: str,
    mlp_seed: int,
    scratch_dir: str,
) -> dict[str, Any]:
    """Load, meter, and return exactly one lane inside the killable child."""

    source_path = Path(estimator_path).resolve()
    if expected_estimator_sha256 not in ALLOWED_ESTIMATOR_SHA256:
        raise AssertionError("estimator hash is outside the frozen source allow-list")
    if _sha256_file(source_path) != expected_estimator_sha256:
        raise AssertionError("frozen estimator changed before child import")
    weights_file = Path(weights_path).resolve()
    weights = np.load(weights_file, mmap_mode="r", allow_pickle=False)
    if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.dtype("<f4"):
        raise AssertionError("scratch weight geometry changed")
    for layer in weights:
        if not np.isfinite(layer).all():
            raise AssertionError("scratch weights contain a nonfinite value")
    derived_seed = int(derive_estimator_seed(int(mlp_seed)))
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=[weights[layer] for layer in range(DEPTH)],
        seed=derived_seed,
        name=str(mlp_name),
    )
    mlp.validate()

    estimator: Any | None = None
    try:
        estimator, metadata = load_estimator_from_path(source_path, class_name=class_name)
        if metadata.class_name != class_name:
            raise AssertionError("spawn child loaded the wrong estimator class")
        estimator.setup(
            SetupContext(
                width=WIDTH,
                depth=DEPTH,
                flop_budget=FLOP_BUDGET,
                api_version="1.0",
                scratch_dir=str(Path(scratch_dir).resolve()),
                submission_dir=str(source_path.parent),
                seed=0,
            )
        )
        host_started = time.perf_counter()
        with threadpool_limits(limits=MAX_BLAS_THREADS):
            context = flops.BudgetContext(
                flop_budget=FLOP_BUDGET,
                wall_time_limit_s=WALL_TIME_LIMIT_S,
                quiet=True,
                namespace=label,
            )
            with context:
                returned = estimator.predict(mlp, FLOP_BUDGET)
        host_predict_wall_s = time.perf_counter() - host_started
        summary = context.summary_dict(by_namespace=False)
        flops_used = int(summary["flops_used"])
        if flops_used != int(expected_flops):
            raise AssertionError(
                f"exact FLOP bill drift: {label}: {flops_used} != {expected_flops}"
            )
        if flops_used > FLOP_BUDGET:
            raise AssertionError(f"lane crossed the full Phase-2 budget: {label}")
        if float(summary["wall_time_s"]) > WALL_TIME_LIMIT_S:
            raise AssertionError(f"lane wall-time limit exceeded: {label}")
        if float(summary["residual_wall_time_s"]) >= RESIDUAL_WALL_TIME_LIMIT_S:
            raise AssertionError(f"lane residual-time limit exceeded: {label}")
        prediction = np.ascontiguousarray(np.asarray(returned, dtype=np.dtype("<f4")))
        if prediction.shape != (DEPTH, WIDTH) or not np.isfinite(prediction).all():
            raise AssertionError(f"lane output contract failed: {label}")
        return {
            "prediction": prediction,
            "derived_estimator_seed": derived_seed,
            "meter": {
                "flop_budget": int(summary["flop_budget"]),
                "flops_used": flops_used,
                "flops_remaining": int(summary["flops_remaining"]),
                "integer_flop_margin_below_full_budget": FLOP_BUDGET - flops_used,
                "wall_time_s": float(summary["wall_time_s"]),
                "flopscope_backend_time_s": float(summary["flopscope_backend_time_s"]),
                "flopscope_overhead_time_s": float(summary["flopscope_overhead_time_s"]),
                "residual_wall_time_s": float(summary["residual_wall_time_s"]),
                "host_predict_wall_s": host_predict_wall_s,
                "operations": summary["operations"],
            },
        }
    finally:
        if estimator is not None:
            estimator.teardown()
