#!/usr/bin/env python3
"""Run the frozen width-1024 V4 moment replay with weights only, never targets."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any

import flopscope as flops
import numpy as np
from whestbench.domain import MLP


HERE = Path(__file__).resolve().parent
ESTIMATOR = HERE / "estimator.py"
MANIFEST = HERE / "SOURCE_MANIFEST.json"
V4_ESTIMATOR_SHA256 = (
    "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
)
V4_AUDIT_SHA256 = (
    "c9553e1596e8e61b29c862256487586d38f66c92a8389cd1541c5b1e9ffa5a9d"
)
WIDTH = 1024
DEPTH = 16
FLOP_BUDGET = 2**41
WALL_LIMIT_S = 120.0
RESIDUAL_LIMIT_S = 0.4
MEMORY_LIMIT_BYTES = 8 * 1024**3
EXPECTED_FINITE_PER_LAYER = 7 * WIDTH
CHANNELS = (
    "min_r1", "count_r1_lt_zero",
    "min_v", "count_v_lt_zero",
    "min_A", "count_A_lt_zero",
    "min_B", "count_B_lt_zero",
    "finite_count_r1_r2_r3_r4_v_A_B",
)

# Frozen after the tiny exact-price probes in verify_parity.py.
EXPECTED_INCREMENTAL_FLOPS = 1_032_640


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _atomic_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _verify_source_manifest() -> dict[str, Any]:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["parent"]["estimator_sha256"] != V4_ESTIMATOR_SHA256:
        raise RuntimeError("diagnostic manifest V4 estimator binding mismatch")
    if manifest["parent"]["independent_audit_sha256"] != V4_AUDIT_SHA256:
        raise RuntimeError("diagnostic manifest V4 audit binding mismatch")
    for binding in manifest["files"]:
        path = HERE / binding["path"]
        if _sha256(path) != binding["sha256"]:
            raise RuntimeError(f"diagnostic frozen-file hash mismatch: {path.name}")
    return manifest


def _verify_v4_meter_receipt(path: Path) -> tuple[dict[str, Any], int]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    if receipt["estimator"]["sha256"] != V4_ESTIMATOR_SHA256:
        raise RuntimeError("meter receipt does not bind the frozen V4 estimator")
    assertions = receipt["assertions"]
    required_assertions = (
        "whest_returncode_zero",
        "no_failed_mlps",
        "run_config_matches_invocation",
        "all_weights_reconstruct_exactly",
        "no_dataset_argument",
        "no_public_target_values_accessed",
        "sampled_solution_descendant_rss_below_memory_limit",
    )
    if not all(assertions.get(name) is True for name in required_assertions):
        raise RuntimeError("frozen V4 meter receipt does not pass every required assertion")
    measured = receipt["measured"]
    if measured["n_failed_mlps"] != 0:
        raise RuntimeError("frozen V4 meter recorded a failed MLP")
    if measured["output_contract"]["all_values_finite"] is not True:
        raise RuntimeError("frozen V4 meter output was nonfinite")
    base_flops = int(measured["flops_used_per_mlp"][0])
    if base_flops > FLOP_BUDGET:
        raise RuntimeError("frozen V4 meter exceeded the FLOP budget")
    if float(measured["wall_time_s_per_mlp"][0]) > WALL_LIMIT_S:
        raise RuntimeError("frozen V4 meter exceeded the 120-second wall cap")
    if float(measured["residual_wall_time_s_per_mlp"][0]) > RESIDUAL_LIMIT_S:
        raise RuntimeError("frozen V4 meter exceeded the 0.4-second residual cap")
    peak = int(
        measured["memory"]["sampler"]["peak_descendants_aggregate_rss_bytes"]
    )
    if peak >= MEMORY_LIMIT_BYTES:
        raise RuntimeError("frozen V4 meter exceeded the 8-GiB solution RSS cap")
    return receipt, base_flops


def _weight_fixture(seed: int) -> tuple[MLP, list[dict[str, Any]], str]:
    streams = np.random.SeedSequence(seed).spawn(3)
    weight_rng = np.random.default_rng(streams[0])
    scale = float(np.sqrt(2.0 / WIDTH))
    weights = []
    records = []
    aggregate = hashlib.sha256()
    for layer in range(DEPTH):
        weight = (
            weight_rng.standard_normal((WIDTH, WIDTH)) * scale
        ).astype(np.float32)
        contiguous = np.ascontiguousarray(weight)
        layer_hasher = hashlib.sha256()
        layer_hasher.update(contiguous.dtype.str.encode("ascii"))
        layer_hasher.update(
            json.dumps(list(contiguous.shape), separators=(",", ":")).encode("ascii")
        )
        layer_hasher.update(memoryview(contiguous).cast("B"))
        digest = layer_hasher.hexdigest()
        records.append(
            {
                "layer": layer,
                "sha256": digest,
                "shape": [WIDTH, WIDTH],
                "dtype": "float32",
            }
        )
        encoded = f"mlp0/layer{layer}".encode()
        aggregate.update(len(encoded).to_bytes(8, "big"))
        aggregate.update(encoded)
        aggregate.update(digest.encode())
        weights.append(weight)
    estimator_seed = int(streams[2].generate_state(1)[0])
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=weights,
        seed=estimator_seed,
        name="target-free-v4-moment-replay",
    )
    mlp.validate()
    return mlp, records, aggregate.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4-meter-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt: {args.output}")
    manifest = _verify_source_manifest()
    meter_receipt, base_flops = _verify_v4_meter_receipt(
        args.v4_meter_receipt.resolve()
    )
    if base_flops + EXPECTED_INCREMENTAL_FLOPS > FLOP_BUDGET:
        raise SystemExit("base V4 plus frozen diagnostic increment exceeds 2^41")
    mlp, weight_records, aggregate_weights = _weight_fixture(args.seed)
    fixture = meter_receipt["deterministic_synthetic_fixture"]
    if int(fixture["seed"]) != args.seed:
        raise SystemExit("V4 meter receipt seed differs from diagnostic seed")
    if fixture["aggregate_weights_sha256"] != aggregate_weights:
        raise SystemExit("weights-only replay does not match the metered V4 fixture")
    expected_estimator_seed = int(fixture["per_mlp"][0]["estimator_seed"])
    if mlp.seed != expected_estimator_seed:
        raise SystemExit("recreated estimator seed differs from the metered V4 fixture")
    module = _load("frozen_v4_moment_diagnostic_run", ESTIMATOR)
    with flops.BudgetContext(
        flop_budget=FLOP_BUDGET,
        quiet=True,
        namespace="v4_moment_diagnostic",
        wall_time_limit_s=WALL_LIMIT_S,
    ) as budget:
        diagnostic = module.MomentDiagnostic().predict(mlp, FLOP_BUDGET)
    summary = budget.summary_dict(by_namespace=True)
    values = np.asarray(diagnostic)
    if values.shape != (DEPTH, len(CHANNELS)):
        raise RuntimeError(f"unexpected diagnostic shape: {values.shape}")
    rows = []
    finite_gate = True
    exact_feasibility_gate = True
    for layer, row in enumerate(values, start=1):
        record = {name: float(value) for name, value in zip(CHANNELS, row, strict=True)}
        for count_name in (
            "count_r1_lt_zero", "count_v_lt_zero",
            "count_A_lt_zero", "count_B_lt_zero",
            "finite_count_r1_r2_r3_r4_v_A_B",
        ):
            if not np.isfinite(record[count_name]) or record[count_name] != round(record[count_name]):
                raise RuntimeError(f"layer {layer} has a nonintegral count channel")
            record[count_name] = int(round(record[count_name]))
        row_finite = (
            record["finite_count_r1_r2_r3_r4_v_A_B"] == EXPECTED_FINITE_PER_LAYER
        )
        row_feasible = row_finite and all(
            record[count_name] == 0
            for count_name in (
                "count_r1_lt_zero", "count_v_lt_zero",
                "count_A_lt_zero", "count_B_lt_zero",
            )
        ) and all(
            np.isfinite(record[min_name]) and record[min_name] >= 0.0
            for min_name in ("min_r1", "min_v", "min_A", "min_B")
        )
        finite_gate &= row_finite
        exact_feasibility_gate &= row_feasible
        rows.append({"layer": layer, **record, "row_exact_feasible": row_feasible})

    measured_flops = int(summary["flops_used"])
    measured_increment = measured_flops - base_flops
    exact_price_gate = measured_increment == EXPECTED_INCREMENTAL_FLOPS
    resource_gate = (
        measured_flops <= FLOP_BUDGET
        and float(summary["wall_time_s"]) <= WALL_LIMIT_S
        and float(summary["residual_wall_time_s"]) <= RESIDUAL_LIMIT_S
    )
    receipt = {
        "schema": "frozen-v4-target-free-actual-moment-replay-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PASS_NECESSARY_MARGINAL_MOMENT_FEASIBILITY"
            if finite_gate and exact_feasibility_gate and exact_price_gate and resource_gate
            else "SCIENTIFIC_HOLD__SEE_GATES"
        ),
        "source": {
            "diagnostic_estimator_sha256": _sha256(ESTIMATOR),
            "diagnostic_source_manifest_sha256": _sha256(MANIFEST),
            "v4_estimator_sha256": V4_ESTIMATOR_SHA256,
            "v4_independent_audit_sha256": V4_AUDIT_SHA256,
            "manifest_status": manifest["status"],
        },
        "v4_meter_receipt": {
            "path": str(args.v4_meter_receipt.resolve()),
            "sha256": _sha256(args.v4_meter_receipt.resolve()),
            "status": meter_receipt.get("status"),
            "base_flops": base_flops,
        },
        "scope": {
            "width": WIDTH,
            "depth": DEPTH,
            "seed": args.seed,
            "dataset_accesses": 0,
            "target_or_truth_accesses": 0,
            "score_or_fit_calls": 0,
            "sampling_estimator_calls": 0,
            "scalar_moment_repairs": 0,
            "weight_stream": "SeedSequence(seed).spawn(3)[0], official He-normal recipe",
        },
        "weights": {
            "aggregate_sha256": aggregate_weights,
            "layers": weight_records,
        },
        "channels": list(CHANNELS),
        "rows": rows,
        "gates": {
            "all_7x1024_values_finite_each_layer": finite_gate,
            "all_exact_necessary_margins_nonnegative": exact_feasibility_gate,
            "exact_frozen_incremental_price": exact_price_gate,
            "diagnostic_flops_wall_residual_within_phase2_caps": resource_gate,
        },
        "meter": summary,
        "billing": {
            "budget": FLOP_BUDGET,
            "base_v4_flops": base_flops,
            "expected_incremental_flops": EXPECTED_INCREMENTAL_FLOPS,
            "measured_diagnostic_total_flops": measured_flops,
            "measured_incremental_flops": measured_increment,
        },
        "claim_boundary": (
            "A pass proves only necessary univariate order-four moment feasibility "
            "on one target-free synthetic weight trajectory; it proves neither a "
            "joint law nor closure accuracy, MSE, or leaderboard performance."
        ),
        "environment": {
            "python": sys.version,
            "flopscope": importlib.metadata.version("flopscope"),
            "whestbench": importlib.metadata.version("whestbench"),
            "numpy_harness_only": importlib.metadata.version("numpy"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, receipt)
    print(json.dumps({
        "status": receipt["status"],
        "output": str(args.output),
        "gates": receipt["gates"],
        "billing": receipt["billing"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
