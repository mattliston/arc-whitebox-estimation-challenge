#!/usr/bin/env python3
"""Compare frozen V4 and Q24 on weights only; never read or compute targets.

``official`` reconstructs the exact width-1024/depth-16/seed-20260824
synthetic weight fixture shared by the two bound literal meter receipts.  It
will not run unless both receipts and both estimator sources rehash exactly.

``he`` runs fixed, smaller random-He source-law panels.  It calls each
module's internal analytic closure because the public Phase-2 estimator guard
correctly returns zeros away from width 1024/depth 16.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import flopscope as flops
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
V4_SOURCE = ROOT / "fullbudget_k21_k4_closure_v4" / "estimator.py"
Q24_SOURCE = HERE / "estimator.py"
V4_RECEIPT = (
    ROOT
    / "meter_protocol"
    / "runs"
    / "fb_ec4_v4_sparse_resetarena_cp3compact_n1_s8_seed20260824"
    / "receipt.json"
)
Q24_RECEIPT = (
    ROOT
    / "meter_protocol"
    / "runs"
    / "fb_ec4_q24_full_v1_b38cd865_n1_s8_seed20260824"
    / "receipt.json"
)

V4_SOURCE_SHA256 = (
    "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
)
Q24_SOURCE_SHA256 = (
    "b38cd865297c344626da3c4f414e1d4923a9efdc62147e3103f155b2034f7b52"
)
V4_RECEIPT_SHA256 = (
    "14d6eeb53d2f202241457ad32669f82ed6dc26107e572a061184aa2f8fec5017"
)
Q24_RECEIPT_SHA256 = (
    "d69fd338972386af2c45f120d2983d5036a9a52ae57845a01977658e4c21fd11"
)
OFFICIAL_AGGREGATE_WEIGHTS_SHA256 = (
    "4073521961bbac02491a2d1ba620c568fea73ed21bf052c73e42c807286a56ec"
)
OFFICIAL_SEED = 20260824
PRODUCTION_WIDTH = 1024
DEPTH = 16
FLOP_BUDGET = 2**41
WALL_LIMIT_S = 120.0
RESIDUAL_LIMIT_S = 0.4
MEMORY_LIMIT_BYTES = 8 * 1024**3
HE_PANELS = (
    (32, 20260825),
    (32, 20260826),
    (64, 20260825),
    (64, 20260826),
    (128, 20260825),
    (128, 20260826),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _array_digest(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(
        json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
    )
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _combine_named_digests(items: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for name, value in items:
        encoded_name = name.encode("utf-8")
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(value.encode("ascii"))
    return digest.hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_source_bindings() -> None:
    if _sha256(V4_SOURCE) != V4_SOURCE_SHA256:
        raise RuntimeError("frozen V4 source hash mismatch")
    if _sha256(Q24_SOURCE) != Q24_SOURCE_SHA256:
        raise RuntimeError("frozen Q24 source hash mismatch")


def _verify_meter_receipt(
    path: Path,
    expected_receipt_sha256: str,
    expected_estimator_sha256: str,
) -> dict[str, Any]:
    if _sha256(path) != expected_receipt_sha256:
        raise RuntimeError(f"meter receipt hash mismatch: {path}")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt.get("status") != "PASS_PHASE2_SUBPROCESS_INTERFACE_AND_METER__ACCURACY_UNSCORED":
        raise RuntimeError(f"meter receipt is not a technical pass: {path}")
    if receipt["estimator"]["sha256"] != expected_estimator_sha256:
        raise RuntimeError(f"meter receipt estimator binding mismatch: {path}")
    assertions = receipt["assertions"]
    required = (
        "whest_returncode_zero",
        "no_failed_mlps",
        "run_config_matches_invocation",
        "all_weights_reconstruct_exactly",
        "no_dataset_argument",
        "no_public_target_values_accessed",
        "sampled_solution_descendant_rss_below_memory_limit",
    )
    if not all(assertions.get(name) is True for name in required):
        raise RuntimeError(f"meter receipt assertion failure: {path}")
    fixture = receipt["deterministic_synthetic_fixture"]
    if (
        int(fixture["seed"]) != OFFICIAL_SEED
        or int(fixture["width"]) != PRODUCTION_WIDTH
        or int(fixture["depth"]) != DEPTH
        or int(fixture["n_mlps"]) != 1
        or fixture["aggregate_weights_sha256"]
        != OFFICIAL_AGGREGATE_WEIGHTS_SHA256
    ):
        raise RuntimeError(f"meter receipt fixture mismatch: {path}")
    measured = receipt["measured"]
    peak = int(measured["memory"]["sampler"]["peak_descendants_aggregate_rss_bytes"])
    if (
        int(measured["n_failed_mlps"]) != 0
        or measured["output_contract"]["all_values_finite"] is not True
        or int(measured["flops_used_per_mlp"][0]) > FLOP_BUDGET
        or float(measured["wall_time_s_per_mlp"][0]) > WALL_LIMIT_S
        or float(measured["residual_wall_time_s_per_mlp"][0]) > RESIDUAL_LIMIT_S
        or peak >= MEMORY_LIMIT_BYTES
    ):
        raise RuntimeError(f"meter receipt resource gate failure: {path}")
    return receipt


def _official_weight_fixture() -> tuple[MLP, str, list[str]]:
    streams = np.random.SeedSequence(OFFICIAL_SEED).spawn(3)
    rng = np.random.default_rng(streams[0])
    scale = float(np.sqrt(2.0 / PRODUCTION_WIDTH))
    weights = []
    layer_digests = []
    named_digests = []
    for layer in range(DEPTH):
        weight = (
            rng.standard_normal((PRODUCTION_WIDTH, PRODUCTION_WIDTH)) * scale
        ).astype(np.float32)
        digest = _array_digest(weight)
        layer_digests.append(digest)
        named_digests.append((f"mlp0/layer{layer}", digest))
        weights.append(weight)
    aggregate = _combine_named_digests(named_digests)
    if aggregate != OFFICIAL_AGGREGATE_WEIGHTS_SHA256:
        raise RuntimeError("recreated official synthetic weights do not match receipt")
    mlp = MLP(
        width=PRODUCTION_WIDTH,
        depth=DEPTH,
        weights=weights,
        seed=int(streams[2].generate_state(1)[0]),
        name="target-free-official-seed20260824",
    )
    mlp.validate()
    return mlp, aggregate, layer_digests


def _he_weight_fixture(width: int, seed: int) -> tuple[MLP, str, list[str]]:
    rng = np.random.default_rng(seed)
    scale = float(np.sqrt(2.0 / width))
    weights = []
    layer_digests = []
    named_digests = []
    for layer in range(DEPTH):
        weight = (rng.standard_normal((width, width)) * scale).astype(np.float32)
        digest = _array_digest(weight)
        layer_digests.append(digest)
        named_digests.append((f"layer{layer}", digest))
        weights.append(weight)
    mlp = MLP(
        width=width,
        depth=DEPTH,
        weights=weights,
        seed=seed,
        name=f"target-free-random-he-w{width}-seed{seed}",
    )
    mlp.validate()
    return mlp, _combine_named_digests(named_digests), layer_digests


def _run(
    module,
    mlp: MLP,
    production_entrypoint: bool,
    allow_failure: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    start = time.perf_counter()
    budget = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with flops.BudgetContext(
                flop_budget=FLOP_BUDGET,
                quiet=True,
                wall_time_limit_s=WALL_LIMIT_S,
            ) as budget:
                if production_entrypoint:
                    prediction = module.Estimator().predict(mlp, FLOP_BUDGET)
                else:
                    prediction = module._predict(mlp)
    except Exception as error:
        if not allow_failure:
            raise
        elapsed = time.perf_counter() - start
        operations = budget.summary_dict()["operations"] if budget is not None else {}
        return None, {
            "completed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "flops_used_before_failure": (
                int(budget.flops_used) if budget is not None else None
            ),
            "operation_calls_before_failure": int(
                sum(item["calls"] for item in operations.values())
            ),
            "direct_context_elapsed_seconds": float(elapsed),
            "all_finite": False,
        }
    elapsed = time.perf_counter() - start
    materialized = np.asarray(prediction).copy()
    del prediction
    operations = budget.summary_dict()["operations"]
    telemetry = {
        "completed": True,
        "flops_used": int(budget.flops_used),
        "operation_calls": int(sum(item["calls"] for item in operations.values())),
        "direct_context_elapsed_seconds": float(elapsed),
        "shape": list(materialized.shape),
        "dtype": str(materialized.dtype),
        "all_finite": bool(np.isfinite(materialized).all()),
        "prediction_sha256": _array_digest(materialized),
    }
    return materialized, telemetry


def _comparison(v4: np.ndarray, q24: np.ndarray) -> dict[str, Any]:
    if v4.shape != q24.shape or v4.ndim != 2 or v4.shape[0] != DEPTH:
        raise RuntimeError(f"prediction shape mismatch: V4={v4.shape}, Q24={q24.shape}")
    v4_finite = np.isfinite(v4)
    q24_finite = np.isfinite(q24)
    if not bool(v4_finite.all()) or not bool(q24_finite.all()):
        raise RuntimeError("nonfinite prediction in target-free comparison")
    v4_64 = np.asarray(v4, dtype=np.float64)
    q24_64 = np.asarray(q24, dtype=np.float64)
    delta = q24_64 - v4_64
    squared = delta * delta
    per_layer_mse = np.mean(squared, axis=1)
    per_layer_rms = np.sqrt(per_layer_mse)
    per_layer_max_abs = np.max(np.abs(delta), axis=1)
    reference_rms = np.sqrt(np.mean(v4_64 * v4_64, axis=1))
    relative_rms = np.divide(
        per_layer_rms,
        reference_rms,
        out=np.full_like(per_layer_rms, np.inf),
        where=reference_rms != 0.0,
    )
    bitwise = [bool(np.array_equal(v4[layer], q24[layer])) for layer in range(DEPTH)]
    changed = [index + 1 for index, same in enumerate(bitwise) if not same]
    return {
        "all_layer_prediction_disagreement_mse": float(np.mean(squared)),
        "all_layer_prediction_disagreement_rms": float(np.sqrt(np.mean(squared))),
        "all_layer_prediction_disagreement_max_abs": float(np.max(np.abs(delta))),
        "final_prediction_disagreement_mse": float(per_layer_mse[-1]),
        "final_prediction_disagreement_rms": float(per_layer_rms[-1]),
        "final_prediction_disagreement_max_abs": float(per_layer_max_abs[-1]),
        "first_non_bitwise_layer": changed[0] if changed else None,
        "bitwise_equal_all_layers": not changed,
        "per_layer": [
            {
                "layer": layer + 1,
                "mse": float(per_layer_mse[layer]),
                "rms": float(per_layer_rms[layer]),
                "max_abs": float(per_layer_max_abs[layer]),
                "v4_reference_rms": float(reference_rms[layer]),
                "relative_rms": float(relative_rms[layer]),
                "bitwise_equal": bitwise[layer],
                "v4_all_finite": bool(v4_finite[layer].all()),
                "q24_all_finite": bool(q24_finite[layer].all()),
            }
            for layer in range(DEPTH)
        ],
    }


def _run_pair(
    v4_module,
    q24_module,
    mlp: MLP,
    production: bool,
    allow_failure: bool = False,
) -> dict[str, Any]:
    v4, v4_telemetry = _run(v4_module, mlp, production, allow_failure)
    gc.collect()
    q24, q24_telemetry = _run(q24_module, mlp, production, allow_failure)
    result = {
        "v4": v4_telemetry,
        "q24": q24_telemetry,
        "disagreement": (
            _comparison(v4, q24)
            if v4 is not None and q24 is not None
            else None
        ),
    }
    del v4, q24
    gc.collect()
    return result


def _official(v4_module, q24_module) -> dict[str, Any]:
    v4_receipt = _verify_meter_receipt(
        V4_RECEIPT, V4_RECEIPT_SHA256, V4_SOURCE_SHA256
    )
    q24_receipt = _verify_meter_receipt(
        Q24_RECEIPT, Q24_RECEIPT_SHA256, Q24_SOURCE_SHA256
    )
    v4_weights = v4_receipt["deterministic_synthetic_fixture"]["aggregate_weights_sha256"]
    q24_weights = q24_receipt["deterministic_synthetic_fixture"]["aggregate_weights_sha256"]
    if v4_weights != q24_weights:
        raise RuntimeError("V4 and Q24 meter receipts do not bind the same weights")
    mlp, aggregate, layer_digests = _official_weight_fixture()
    return {
        "fixture": {
            "kind": "exact_whest_synthetic_weights_only",
            "seed": OFFICIAL_SEED,
            "width": PRODUCTION_WIDTH,
            "depth": DEPTH,
            "weight_generation": "float32(N(0,1) float64 * sqrt(2/width))",
            "aggregate_weights_sha256": aggregate,
            "layer_weight_sha256": layer_digests,
            "public_or_synthetic_target_values_read": 0,
            "truth_computed": False,
        },
        "meter_authority": {
            "v4_receipt_path": str(V4_RECEIPT.relative_to(ROOT)),
            "v4_receipt_sha256": V4_RECEIPT_SHA256,
            "q24_receipt_path": str(Q24_RECEIPT.relative_to(ROOT)),
            "q24_receipt_sha256": Q24_RECEIPT_SHA256,
            "both_technical_meter_pass": True,
        },
        "result": _run_pair(v4_module, q24_module, mlp, production=True),
    }


def _he(v4_module, q24_module) -> dict[str, Any]:
    panels = []
    for width, seed in HE_PANELS:
        mlp, aggregate, layer_digests = _he_weight_fixture(width, seed)
        panels.append(
            {
                "fixture": {
                    "kind": "fixed_random_he_weights_only",
                    "seed": seed,
                    "width": width,
                    "depth": DEPTH,
                    "weight_generation": "float32(N(0,1) float64 * sqrt(2/width))",
                    "aggregate_weights_sha256": aggregate,
                    "layer_weight_sha256": layer_digests,
                    "public_or_synthetic_target_values_read": 0,
                    "truth_computed": False,
                },
                "result": _run_pair(
                    v4_module,
                    q24_module,
                    mlp,
                    production=False,
                    allow_failure=True,
                ),
            }
        )
    return {"panels": panels}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("official", "he"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not sys.dont_write_bytecode:
        raise SystemExit("run with Python -B / PYTHONDONTWRITEBYTECODE=1")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt: {args.output}")
    _verify_source_bindings()
    v4_module = _load("frozen_v4_q24_comparator", V4_SOURCE)
    q24_module = _load("frozen_q24_comparator", Q24_SOURCE)
    body = (
        _official(v4_module, q24_module)
        if args.mode == "official"
        else _he(v4_module, q24_module)
    )
    receipt = {
        "schema": "fb-ec4-v4-q24-target-free-disagreement-v1",
        "status": "TECHNICAL_PASS__SCIENTIFIC_AND_BOARD_HOLD",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode,
        "sources": {
            "v4": {
                "path": str(V4_SOURCE.relative_to(ROOT)),
                "bytes": V4_SOURCE.stat().st_size,
                "sha256": V4_SOURCE_SHA256,
            },
            "q24": {
                "path": str(Q24_SOURCE.relative_to(ROOT)),
                "bytes": Q24_SOURCE.stat().st_size,
                "sha256": Q24_SOURCE_SHA256,
            },
        },
        **body,
        "scope": {
            "dataset_accessed": False,
            "public_target_values_accessed": 0,
            "synthetic_target_values_accessed": 0,
            "truth_computed": False,
            "fits": 0,
            "samples_used_by_estimator": 0,
            "submissions": 0,
            "graph_edits": 0,
            "interpretation": (
                "Weights-only prediction disagreement is target-free supporting evidence. "
                "It is not raw-MSE, accuracy, score, or board-wall evidence."
            ),
        },
    }
    payload = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, args.output)
    if not args.quiet:
        print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
