#!/usr/bin/env python3
"""Run frozen V4 normalized-moment diagnostic v2, weights only, never targets."""

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
MANIFEST = HERE / "SOURCE_MANIFEST_V2.json"
FREEZE_RECEIPT = HERE / "FREEZE_RECEIPT_V2.json"
V2_MANIFEST_SHA256 = (
    "439e7a2aeb43ac38841037ba73321110cdfd954e7b5958133b715749d838f722"
)
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
CHANNELS = (
    "min_relative_A", "worst_A_relative_error_bound",
    "worst_A_raw_margin", "worst_A_absolute_error_bound",
    "worst_A_index", "worst_A_scale",
    "worst_A_r1", "worst_A_r2", "worst_A_r3", "worst_A_r4",
    "count_A_CERT_NEG", "count_A_ZERO_COMPATIBLE",
    "count_A_CERT_POS", "count_A_scale_zero",
    "min_relative_B", "worst_B_relative_error_bound",
    "worst_B_raw_margin", "worst_B_absolute_error_bound",
    "worst_B_index", "worst_B_scale",
    "worst_B_r1", "worst_B_r2", "worst_B_r3", "worst_B_r4",
    "count_B_CERT_NEG", "count_B_ZERO_COMPATIBLE",
    "count_B_CERT_POS", "count_B_scale_zero",
)

# Frozen after the tiny exact-price probes in verify_parity.py.
EXPECTED_INCREMENTAL_FLOPS = 5_902_272
V1_HOLD_RECEIPT_SHA256 = (
    "93124980ee134d2b0bad4e6f00b46ced6a7329fcc356945c12736443f4723ee8"
)


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
    if _sha256(MANIFEST) != V2_MANIFEST_SHA256:
        raise RuntimeError("diagnostic v2 manifest hash mismatch")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["status"] != "FROZEN_V2_SOURCE_GATES_PASS__REPLAY_READY":
        raise RuntimeError("diagnostic v2 manifest status mismatch")
    frozen_v4 = manifest["parents"]["frozen_v4"]
    if frozen_v4["estimator"]["sha256"] != V4_ESTIMATOR_SHA256:
        raise RuntimeError("diagnostic manifest V4 estimator binding mismatch")
    if frozen_v4["independent_audit"]["sha256"] != V4_AUDIT_SHA256:
        raise RuntimeError("diagnostic manifest V4 audit binding mismatch")
    if (
        manifest["parents"]["v1_hold_result93124980"]["sha256"]
        != V1_HOLD_RECEIPT_SHA256
    ):
        raise RuntimeError("diagnostic manifest V1 HOLD binding mismatch")
    for binding in manifest["files"]:
        path = HERE / binding["path"]
        if (
            not path.is_file()
            or path.stat().st_size != int(binding["bytes"])
            or _sha256(path) != binding["sha256"]
        ):
            raise RuntimeError(f"diagnostic frozen-file hash mismatch: {path.name}")
    return manifest


def _verify_freeze_receipt(manifest: dict[str, Any]) -> dict[str, Any]:
    freeze = json.loads(FREEZE_RECEIPT.read_text(encoding="utf-8"))
    if freeze["schema"] != "fb-ec4-v4-normalized-moment-diagnostic-v2-freeze-v1":
        raise RuntimeError("diagnostic v2 freeze schema mismatch")
    if freeze["status"] != "PASS_SOURCE_STATIC_TINY_PARITY__REPLAY_AUTHORIZED":
        raise RuntimeError("diagnostic v2 freeze status mismatch")
    manifest_binding = freeze["bindings"]["v2_source_manifest"]
    if (
        manifest_binding["path"] != MANIFEST.name
        or manifest_binding["sha256"] != V2_MANIFEST_SHA256
        or int(manifest_binding["bytes"]) != MANIFEST.stat().st_size
    ):
        raise RuntimeError("diagnostic v2 freeze-to-manifest binding mismatch")
    expected_files = {
        binding["path"]: (int(binding["bytes"]), binding["sha256"])
        for binding in manifest["files"]
    }
    expected_files[Path(__file__).name] = (
        Path(__file__).stat().st_size,
        _sha256(Path(__file__)),
    )
    actual_files = {
        binding["path"]: (int(binding["bytes"]), binding["sha256"])
        for binding in freeze["bindings"]["v2_files"]
    }
    if actual_files != expected_files:
        raise RuntimeError("diagnostic v2 freeze file-set binding mismatch")
    for relative, (size, digest) in actual_files.items():
        path = HERE / relative
        if (
            not path.is_file()
            or path.stat().st_size != size
            or _sha256(path) != digest
        ):
            raise RuntimeError(f"diagnostic v2 freeze file mismatch: {relative}")
    if freeze["bindings"]["parents"] != manifest["parents"]:
        raise RuntimeError("diagnostic v2 freeze parent bindings differ from manifest")
    return freeze


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
    parser.add_argument("--v1-hold-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing receipt: {args.output}")
    manifest = _verify_source_manifest()
    freeze_receipt = _verify_freeze_receipt(manifest)
    v1_hold_path = args.v1_hold_receipt.resolve()
    if _sha256(v1_hold_path) != V1_HOLD_RECEIPT_SHA256:
        raise SystemExit("V1 HOLD receipt hash mismatch")
    v1_hold = json.loads(v1_hold_path.read_text(encoding="utf-8"))
    if v1_hold.get("status") != "SCIENTIFIC_HOLD__SEE_GATES":
        raise SystemExit("V1 result is not the bound scientific HOLD")
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
    if v1_hold["weights"]["aggregate_sha256"] != aggregate_weights:
        raise SystemExit("weights-only replay does not match the V1 HOLD fixture")
    if int(v1_hold["scope"]["seed"]) != args.seed:
        raise SystemExit("V1 HOLD seed differs from diagnostic v2 seed")
    expected_estimator_seed = int(fixture["per_mlp"][0]["estimator_seed"])
    if mlp.seed != expected_estimator_seed:
        raise SystemExit("recreated estimator seed differs from the metered V4 fixture")
    module = _load("frozen_v4_normalized_moment_diagnostic_v2_run", ESTIMATOR)
    with flops.BudgetContext(
        flop_budget=FLOP_BUDGET,
        quiet=True,
        namespace="v4_normalized_moment_diagnostic_v2",
        wall_time_limit_s=WALL_LIMIT_S,
    ) as budget:
        diagnostic = module.MomentDiagnostic().predict(mlp, FLOP_BUDGET)
    summary = budget.summary_dict(by_namespace=True)
    values = np.asarray(diagnostic)
    if values.shape != (DEPTH, len(CHANNELS)):
        raise RuntimeError(f"unexpected diagnostic shape: {values.shape}")
    rows = []
    finite_gate = True
    index_gate = True
    positive_scale_gate = True
    error_bound_gate = True
    class_count_gate = True
    any_cert_negative = False
    for layer, row in enumerate(values, start=1):
        record = {name: float(value) for name, value in zip(CHANNELS, row, strict=True)}
        row_finite = bool(np.isfinite(row).all())
        row_indices_valid = True
        for index_name in ("worst_A_index", "worst_B_index"):
            value = record[index_name]
            valid = np.isfinite(value) and value == round(value) and 0 <= value < WIDTH
            row_indices_valid &= bool(valid)
            if valid:
                record[index_name] = int(round(value))
        row_scales_positive = (
            np.isfinite(record["worst_A_scale"])
            and np.isfinite(record["worst_B_scale"])
            and record["worst_A_scale"] > 0.0
            and record["worst_B_scale"] > 0.0
        )
        row_error_bounds_valid = all(
            np.isfinite(record[name]) and record[name] >= 0.0
            for name in (
                "worst_A_relative_error_bound",
                "worst_A_absolute_error_bound",
                "worst_B_relative_error_bound",
                "worst_B_absolute_error_bound",
            )
        )
        row_counts_valid = True
        for family in ("A", "B"):
            count_names = (
                f"count_{family}_CERT_NEG",
                f"count_{family}_ZERO_COMPATIBLE",
                f"count_{family}_CERT_POS",
                f"count_{family}_scale_zero",
            )
            for count_name in count_names:
                value = record[count_name]
                valid = (
                    np.isfinite(value)
                    and value == round(value)
                    and 0 <= value <= WIDTH
                )
                row_counts_valid &= bool(valid)
                if valid:
                    record[count_name] = int(round(value))
            row_counts_valid &= (
                record[f"count_{family}_CERT_NEG"]
                + record[f"count_{family}_ZERO_COMPATIBLE"]
                + record[f"count_{family}_CERT_POS"]
                == WIDTH
            )
            any_cert_negative |= record[f"count_{family}_CERT_NEG"] > 0
            raw_margin = record[f"worst_{family}_raw_margin"]
            absolute_error = record[f"worst_{family}_absolute_error_bound"]
            if raw_margin < -absolute_error:
                record[f"worst_{family}_cert_class"] = "CERT_NEG"
            elif raw_margin > absolute_error:
                record[f"worst_{family}_cert_class"] = "CERT_POS"
            else:
                record[f"worst_{family}_cert_class"] = "ZERO_COMPATIBLE"
        row_ready = (
            row_finite
            and row_indices_valid
            and row_scales_positive
            and row_error_bounds_valid
            and row_counts_valid
        )
        finite_gate &= row_finite
        index_gate &= row_indices_valid
        positive_scale_gate &= row_scales_positive
        error_bound_gate &= row_error_bounds_valid
        class_count_gate &= row_counts_valid
        rows.append({"layer": layer, **record, "row_classification_ready": row_ready})

    measured_flops = int(summary["flops_used"])
    measured_increment = measured_flops - base_flops
    exact_price_gate = measured_increment == EXPECTED_INCREMENTAL_FLOPS
    resource_gate = (
        measured_flops <= FLOP_BUDGET
        and float(summary["wall_time_s"]) <= WALL_LIMIT_S
        and float(summary["residual_wall_time_s"]) <= RESIDUAL_LIMIT_S
    )
    technical_gate = (
        finite_gate and index_gate and positive_scale_gate
        and error_bound_gate and class_count_gate
        and exact_price_gate and resource_gate
    )
    if not technical_gate:
        status = "TECHNICAL_HOLD__SEE_GATES"
    elif any_cert_negative:
        status = "TECHNICAL_PASS__SCIENTIFIC_HOLD__STORED_TUPLE_CERT_NEG_PRESENT"
    else:
        status = "TECHNICAL_PASS__V1_STORED_TUPLE_MOMENT_HOLD_CLEARED__NO_CERT_NEG"
    receipt = {
        "schema": "frozen-v4-target-free-normalized-moment-replay-v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "source": {
            "diagnostic_estimator_sha256": _sha256(ESTIMATOR),
            "diagnostic_source_manifest_sha256": _sha256(MANIFEST),
            "diagnostic_freeze_receipt_sha256": _sha256(FREEZE_RECEIPT),
            "v4_estimator_sha256": V4_ESTIMATOR_SHA256,
            "v4_independent_audit_sha256": V4_AUDIT_SHA256,
            "manifest_status": manifest["status"],
            "freeze_status": freeze_receipt["status"],
        },
        "v4_meter_receipt": {
            "path": str(args.v4_meter_receipt.resolve()),
            "sha256": _sha256(args.v4_meter_receipt.resolve()),
            "status": meter_receipt.get("status"),
            "base_flops": base_flops,
        },
        "superseded_v1_hold": {
            "path": str(v1_hold_path),
            "sha256": V1_HOLD_RECEIPT_SHA256,
            "status": v1_hold["status"],
            "reason_for_v2": (
                "V1 found sparse raw A/B negatives of order 1e-30 to 1e-28; "
                "V2 measures each worst signed margin relative to its own "
                "difference-of-positive-terms cancellation scale."
            ),
        },
        "scope": {
            "width": WIDTH,
            "depth": DEPTH,
            "seed": args.seed,
            "dataset_accesses": 0,
            "target_or_truth_accesses": 0,
            "score_or_fit_calls": 0,
            "sampling_estimator_calls": 0,
            "scalar_moment_clips_floors_or_repairs": 0,
            "weight_stream": "SeedSequence(seed).spawn(3)[0], official He-normal recipe",
        },
        "weights": {
            "aggregate_sha256": aggregate_weights,
            "layers": weight_records,
        },
        "channels": list(CHANNELS),
        "rows": rows,
        "gates": {
            "all_28_values_finite_each_layer": finite_gate,
            "worst_indices_integral_and_in_range": index_gate,
            "reported_cancellation_scales_strictly_positive": positive_scale_gate,
            "outward_error_bounds_nonnegative_and_finite": error_bound_gate,
            "certification_class_counts_integral_and_partition_width": class_count_gate,
            "any_stored_tuple_CERT_NEG": any_cert_negative,
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
            "The outward interval classifies only determinant evaluation on the "
            "stored fp64 raw r1..r4 tuple; raw-input errors are defined as zero and "
            "upstream error in generating those moments is outside the interval. "
            "CERT_NEG means the represented tuple is infeasible at this observation "
            "point; ZERO_COMPATIBLE means determinant-evaluation cancellation is "
            "consistent with zero. This proves no joint law or broader closure "
            "accuracy, performs no repair, establishes no MSE, and authorizes no "
            "leaderboard submission."
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
