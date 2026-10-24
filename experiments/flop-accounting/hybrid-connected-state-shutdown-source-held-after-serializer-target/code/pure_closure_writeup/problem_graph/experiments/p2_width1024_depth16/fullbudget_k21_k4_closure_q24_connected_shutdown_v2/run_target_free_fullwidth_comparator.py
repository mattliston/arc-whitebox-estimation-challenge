#!/usr/bin/env python3
"""One-use, release-gated, weights-only width-1024 parent/CS2 comparator."""

from __future__ import annotations

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
SOURCE = HERE / "estimator.py"
SOURCE_SHA256 = "4efb2c40d21d16c7b51ee87741323c3b99aded664fd75a2d5e00164f7b584779"
SOURCE_HOLD = HERE / "SOURCE_HOLD_V2.json"
SOURCE_HOLD_SHA256 = "c35332c6df5bfb2c7ad7105f200117078d55901956164d7717abdb8f85c2417c"
PARENT = ROOT / "fullbudget_k21_k4_closure_q24_manual_rho_v1" / "estimator.py"
PARENT_SHA256 = "7d0c9e304c59dc21e59888313d19cf71bb2df0a895996ef6c7456bf82ea33e00"
PARENT_METER = (
    ROOT
    / "meter_protocol"
    / "runs"
    / "fb_ec4_q24_manual_rho_v1_7d0c9e30_n1_s8_seed20260824"
    / "receipt.json"
)
PARENT_METER_SHA256 = "4ad5d4bb8e70a3f57ef7559f5dceffe0d92fc7b77626088cff87431bdbf18160"
PREREGISTER = HERE / "PREREGISTER_FULLWIDTH_COMPARATOR_AND_METER_V1.json"
PREREGISTER_SHA256 = "e06a68a92a139c670b0fe2fcb38615e96fa8d96a0019b3eef58bc2e995d36dd0"
RELEASE = HERE / "ROOT_FULLWIDTH_COMPARATOR_RELEASE_SCAFFOLD_V1.json"
OUTPUT = (
    HERE
    / "runs"
    / "fullwidth_official_he_seeds20260824_20260826_v1"
    / "receipt.json"
)
CLAIM = OUTPUT.parent / "CLAIM.json"

SEEDS = (20260824, 20260825, 20260826)
WIDTH = 1024
DEPTH = 16
FLOP_BUDGET = 2**41
WALL_LIMIT_S = 120.0
OFFICIAL_SEED20260824_WEIGHTS_SHA256 = (
    "4073521961bbac02491a2d1ba620c568fea73ed21bf052c73e42c807286a56ec"
)
AUTHORIZED_RELEASE_STATUS = (
    "ROOT_AUTHORIZED_ONE_TARGET_FREE_FULLWIDTH_COMPARATOR__NOT_YET_CONSUMED"
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


def _load(tag: str, path: Path):
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _operation_calls(budget: Any) -> int:
    operations = budget.summary_dict(by_namespace=False).get("operations", {})
    return sum(int(value.get("calls", 0)) for value in operations.values())


def _verify_static_bindings() -> None:
    expected = (
        (SOURCE, SOURCE_SHA256, "candidate source"),
        (SOURCE_HOLD, SOURCE_HOLD_SHA256, "source HOLD"),
        (PARENT, PARENT_SHA256, "parent source"),
        (PARENT_METER, PARENT_METER_SHA256, "parent literal-meter receipt"),
        (PREREGISTER, PREREGISTER_SHA256, "preregister"),
    )
    for path, digest, label in expected:
        if not path.is_file() or _sha256(path) != digest:
            raise RuntimeError(f"{label} binding mismatch")
    preregister = json.loads(PREREGISTER.read_text(encoding="utf-8"))
    if preregister["fullwidth_comparator"]["seeds_exact_in_order"] != list(SEEDS):
        raise RuntimeError("preregistered seed order mismatch")
    if preregister["access_boundary"]["truth_computations_in_comparator"] != 0:
        raise RuntimeError("preregister no-truth boundary mismatch")


def _verify_release() -> tuple[str, dict[str, Any]]:
    if not RELEASE.is_file():
        raise RuntimeError("fullwidth comparator release does not exist")
    release_sha256 = _sha256(RELEASE)
    release = json.loads(RELEASE.read_text(encoding="utf-8"))
    if release.get("status") != AUTHORIZED_RELEASE_STATUS:
        raise RuntimeError("fullwidth comparator release remains on HOLD")
    authorization = release.get("authorization", {})
    if authorization.get("authorized") is not True or authorization.get("one_use") is not True:
        raise RuntimeError("fullwidth comparator is not explicitly one-use authorized")
    if authorization.get("authorized_at_utc") is None or authorization.get("authorizer") is None:
        raise RuntimeError("fullwidth comparator authorization identity is incomplete")
    bindings = release.get("bindings", {})
    exact = {
        "candidate_estimator_sha256": SOURCE_SHA256,
        "source_hold_sha256": SOURCE_HOLD_SHA256,
        "preregister_sha256": PREREGISTER_SHA256,
        "parent_estimator_sha256": PARENT_SHA256,
        "parent_literal_meter_receipt_sha256": PARENT_METER_SHA256,
    }
    for key, expected in exact.items():
        if bindings.get(key) != expected:
            raise RuntimeError(f"release binding mismatch: {key}")
    audit = bindings.get("independent_strict_audit", {})
    if not all(isinstance(audit.get(key), str) and audit[key] for key in (
        "path", "sha256", "status_exact"
    )):
        raise RuntimeError("independent strict-audit binding remains null")
    audit_path = (HERE / audit["path"]).resolve()
    if audit_path.parent != HERE.resolve() or not audit_path.is_file():
        raise RuntimeError("strict-audit path must name a file in the candidate HOLD")
    if _sha256(audit_path) != audit["sha256"]:
        raise RuntimeError("strict-audit hash mismatch")
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit_payload.get("status") != audit["status_exact"]:
        raise RuntimeError("strict-audit status mismatch")
    serialized_audit = json.dumps(audit_payload, sort_keys=True)
    if SOURCE_SHA256 not in serialized_audit or SOURCE_HOLD_SHA256 not in serialized_audit:
        raise RuntimeError("strict audit does not bind candidate and source HOLD")
    if release.get("fixed_output") != str(OUTPUT.relative_to(HERE)):
        raise RuntimeError("release output namespace mismatch")
    return release_sha256, audit


def _fixture(seed: int) -> tuple[MLP, dict[str, Any]]:
    streams = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(streams[0])
    scale = float(np.sqrt(2.0 / WIDTH))
    weights = []
    layer_digests = []
    named_digests = []
    for layer in range(DEPTH):
        weight = (rng.standard_normal((WIDTH, WIDTH)) * scale).astype(np.float32)
        digest = _array_digest(weight)
        weights.append(weight)
        layer_digests.append(digest)
        named_digests.append((f"mlp0/layer{layer}", digest))
    aggregate = _combine_named_digests(named_digests)
    if seed == 20260824 and aggregate != OFFICIAL_SEED20260824_WEIGHTS_SHA256:
        raise RuntimeError("official seed-20260824 weight-law reconstruction mismatch")
    estimator_seed = int(streams[2].generate_state(1)[0])
    mlp = MLP(
        width=WIDTH,
        depth=DEPTH,
        weights=weights,
        seed=estimator_seed,
        name=f"target-free-official-he-seed{seed}",
    )
    mlp.validate()
    return mlp, {
        "seed": seed,
        "width": WIDTH,
        "depth": DEPTH,
        "seed_tree": "numpy.random.SeedSequence(seed).spawn(3)",
        "weights_stream": 0,
        "estimator_seed_stream": 2,
        "estimator_seed": estimator_seed,
        "weight_generation": "float32(N(0,1) float64 * sqrt(2/width))",
        "aggregate_weights_sha256": aggregate,
        "layer_weight_sha256": layer_digests,
        "dataset_rows_accessed": 0,
        "target_values_accessed": 0,
        "truth_computed": False,
    }


def _write_durable_claim(release_sha256: str, audit: dict[str, Any]) -> str:
    """Consume the one-use release before allocating any full-width fixture."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=False)
    claim = {
        "schema": "q24-cs2-target-free-fullwidth-one-use-claim-v1",
        "status": "CLAIMED_ONE_USE_RELEASE__NO_RETRY_IF_PROCESS_CRASHES",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "candidate_estimator_sha256": SOURCE_SHA256,
            "source_hold_sha256": SOURCE_HOLD_SHA256,
            "preregister_sha256": PREREGISTER_SHA256,
            "runner_sha256": _sha256(Path(__file__)),
            "fullwidth_release_sha256": release_sha256,
            "independent_strict_audit": audit,
        },
        "fixed_execution": {
            "seeds_exact_in_order": list(SEEDS),
            "width": WIDTH,
            "depth": DEPTH,
            "parent_child_arm_count": 6,
            "fullwidth_fixture_generation_started": False,
        },
        "scope": {
            "dataset_arguments": 0,
            "dataset_rows_accessed": 0,
            "target_values_accessed": 0,
            "truth_computations": 0,
            "fits": 0,
            "packages": 0,
            "submissions": 0,
        },
    }
    payload = (json.dumps(claim, indent=2, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(
        CLAIM,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o644,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        raise
    directory_descriptor = os.open(
        OUTPUT.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return hashlib.sha256(payload).hexdigest()


def _install_shutdown_observer(module: Any) -> tuple[list[dict[str, int]], dict[str, Any]]:
    records: list[dict[str, int]] = []
    holder: dict[str, Any] = {"invalid": None, "cp_post": None, "cp4_provider": None}
    original_certificate = module._certified_invalid_marginals
    original_cp_post = module._cp_post_read
    original_cp4_reset = module._cp4_reset
    original_reset = module._reset_stars

    def observed_certificate(*args: Any, **kwargs: Any):
        invalid = original_certificate(*args, **kwargs)
        holder["invalid"] = invalid
        holder["cp_post"] = None
        holder["cp4_provider"] = None
        return invalid

    def observed_cp_post(*args: Any, **kwargs: Any):
        value = original_cp_post(*args, **kwargs)
        holder["cp_post"] = value
        return value

    def observed_cp4_reset(*args: Any, **kwargs: Any):
        value = original_cp4_reset(*args, **kwargs)
        holder["cp4_provider"] = value[2]
        return value

    def observed_reset(*args: Any, **kwargs: Any):
        value = original_reset(*args, **kwargs)
        if holder["invalid"] is None:
            return value
        if holder["cp_post"] is None or holder["cp4_provider"] is None:
            raise RuntimeError("shutdown observer call-order invariant failed")
        cubic, quartic, pairing_alpha = value
        invalid = np.asarray(holder["invalid"], dtype=bool)
        invalid_plane = np.logical_or(invalid[:, None], invalid[None, :])
        cp_post = np.asarray(holder["cp_post"])
        provider_d, provider_a, provider_b = (
            np.asarray(item) for item in holder["cp4_provider"]
        )
        cubic_array = np.asarray(cubic)
        quartic_array = np.asarray(quartic)
        invalid_units = int(np.count_nonzero(invalid))
        pairing_violations = int(
            invalid_units > 0 and np.count_nonzero(np.asarray(pairing_alpha))
        )
        star_violations = int(np.count_nonzero(cubic_array[invalid_plane]))
        star_violations += int(np.count_nonzero(quartic_array[invalid_plane]))
        cp_violations = int(np.count_nonzero(cp_post[invalid_plane]))
        cp_violations += int(np.count_nonzero(provider_d[invalid]))
        cp_violations += int(np.count_nonzero(provider_a[invalid_plane]))
        cp_violations += int(np.count_nonzero(provider_b[invalid_plane]))
        combined_violations = int(
            np.count_nonzero((cubic_array + cp_post)[invalid_plane])
        )
        combined_violations += int(
            np.count_nonzero((np.diag(quartic_array) + provider_d)[invalid])
        )
        combined_violations += int(
            np.count_nonzero((quartic_array + provider_a)[invalid_plane])
        )
        combined_violations += int(np.count_nonzero(provider_b[invalid_plane]))
        records.append({
            "invalid_unit_count": invalid_units,
            "pairing_alpha_violation_count": pairing_violations,
            "connected_star_incident_nonzero_count": star_violations,
            "cp_carrier_incident_nonzero_count": cp_violations,
            "combined_carrier_incident_nonzero_count": combined_violations,
        })
        return value

    module._certified_invalid_marginals = observed_certificate
    module._cp_post_read = observed_cp_post
    module._cp4_reset = observed_cp4_reset
    module._reset_stars = observed_reset
    return records, holder


def _shutdown_summary(records: list[dict[str, int]]) -> dict[str, Any]:
    violation_keys = (
        "pairing_alpha_violation_count",
        "connected_star_incident_nonzero_count",
        "cp_carrier_incident_nonzero_count",
        "combined_carrier_incident_nonzero_count",
    )
    return {
        "certificate_reset_count": len(records),
        "affected_reset_count": sum(row["invalid_unit_count"] > 0 for row in records),
        "invalid_unit_reset_activation_count": sum(
            row["invalid_unit_count"] for row in records
        ),
        "violation_totals": {
            key: sum(row[key] for row in records) for key in violation_keys
        },
        "by_reset": records,
    }


def _run_arm(path: Path, tag: str, mlp: MLP, observe_shutdown: bool) -> tuple[np.ndarray | None, dict[str, Any]]:
    module = _load(tag, path)
    records: list[dict[str, int]] = []
    if observe_shutdown:
        records, _ = _install_shutdown_observer(module)
    budget = None
    started = time.perf_counter()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with flops.BudgetContext(
                flop_budget=FLOP_BUDGET,
                quiet=True,
                wall_time_limit_s=WALL_LIMIT_S,
            ) as budget:
                prediction = module.Estimator().predict(mlp, FLOP_BUDGET)
        materialized = np.asarray(prediction).copy()
        del prediction
        telemetry: dict[str, Any] = {
            "completed": True,
            "exception_type": None,
            "exception_message": None,
            "flops_used": int(budget.flops_used),
            "operation_calls": _operation_calls(budget),
            "direct_context_wall_seconds": float(time.perf_counter() - started),
            "shape": list(materialized.shape),
            "dtype": str(materialized.dtype),
            "all_finite": bool(np.isfinite(materialized).all()),
            "prediction_sha256": _array_digest(materialized),
        }
    except Exception as error:
        materialized = None
        telemetry = {
            "completed": False,
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "flops_used_before_failure": int(budget.flops_used) if budget is not None else None,
            "operation_calls_before_failure": _operation_calls(budget) if budget is not None else None,
            "direct_context_wall_seconds": float(time.perf_counter() - started),
            "shape": None,
            "dtype": None,
            "all_finite": False,
            "prediction_sha256": None,
        }
    if observe_shutdown:
        telemetry["shutdown_observer"] = _shutdown_summary(records)
    return materialized, telemetry


def _contract_pass(telemetry: dict[str, Any]) -> bool:
    return bool(
        telemetry["completed"]
        and telemetry["shape"] == [DEPTH, WIDTH]
        and telemetry["dtype"] == "float32"
        and telemetry["all_finite"]
    )


def _compare(parent: np.ndarray, child: np.ndarray) -> dict[str, Any]:
    parent64 = parent.astype(np.float64)
    child64 = child.astype(np.float64)
    delta = child64 - parent64
    final_delta = delta[-1]
    parent_rms = float(np.sqrt(np.mean(parent64 * parent64)))
    parent_maxabs = float(np.max(np.abs(parent64)))
    disagreement_rms = float(np.sqrt(np.mean(delta * delta)))
    disagreement_maxabs = float(np.max(np.abs(delta)))
    raw_byte_parity = bool(
        parent.dtype == child.dtype
        and parent.shape == child.shape
        and parent.tobytes(order="C") == child.tobytes(order="C")
    )
    return {
        "all_layer_disagreement_mse": float(np.mean(delta * delta)),
        "all_layer_disagreement_rms": disagreement_rms,
        "all_layer_disagreement_maxabs": disagreement_maxabs,
        "final_layer_disagreement_mse": float(np.mean(final_delta * final_delta)),
        "final_layer_disagreement_rms": float(np.sqrt(np.mean(final_delta * final_delta))),
        "final_layer_disagreement_maxabs": float(np.max(np.abs(final_delta))),
        "parent_all_layer_rms": parent_rms,
        "parent_all_layer_maxabs": parent_maxabs,
        "rms_parent_scale_gate": (
            disagreement_rms <= parent_rms if parent_rms > 0.0 else disagreement_rms == 0.0
        ),
        "maxabs_parent_scale_gate": (
            disagreement_maxabs <= parent_maxabs
            if parent_maxabs > 0.0
            else disagreement_maxabs == 0.0
        ),
        "raw_byte_parity": raw_byte_parity,
    }


def main() -> int:
    if not sys.dont_write_bytecode:
        raise SystemExit("run with Python -B / PYTHONDONTWRITEBYTECODE=1")
    _verify_static_bindings()
    release_sha256, audit_binding = _verify_release()
    if OUTPUT.exists() or OUTPUT.parent.exists():
        raise SystemExit(f"refusing to overwrite or reuse one-use output namespace: {OUTPUT.parent}")
    claim_sha256 = _write_durable_claim(release_sha256, audit_binding)

    rows = []
    for seed in SEEDS:
        mlp, fixture = _fixture(seed)
        parent_array, parent = _run_arm(
            PARENT, f"q24_cs2_fullwidth_parent_{seed}", mlp, False
        )
        gc.collect()
        child_array, child = _run_arm(
            SOURCE, f"q24_cs2_fullwidth_child_{seed}", mlp, True
        )
        comparison = None
        if parent_array is not None and child_array is not None:
            comparison = _compare(parent_array, child_array)
        rows.append({
            "fixture": fixture,
            "parent": parent,
            "child": child,
            "comparison": comparison,
        })
        del parent_array, child_array, mlp
        gc.collect()

    parent_contracts = all(_contract_pass(row["parent"]) for row in rows)
    child_contracts = all(_contract_pass(row["child"]) for row in rows)
    activation_count = sum(
        row["child"]["shutdown_observer"]["invalid_unit_reset_activation_count"]
        for row in rows
    )
    violation_keys = (
        "pairing_alpha_violation_count",
        "connected_star_incident_nonzero_count",
        "cp_carrier_incident_nonzero_count",
        "combined_carrier_incident_nonzero_count",
    )
    carrier_violations = {
        key: sum(
            row["child"]["shutdown_observer"]["violation_totals"][key]
            for row in rows
        )
        for key in violation_keys
    }
    carrier_gate = all(value == 0 for value in carrier_violations.values())
    comparisons_available = all(row["comparison"] is not None for row in rows)
    no_activation_parity = all(
        row["child"]["shutdown_observer"]["invalid_unit_reset_activation_count"] != 0
        or row["comparison"] is not None and row["comparison"]["raw_byte_parity"]
        for row in rows
    )
    parent_scale_gate = all(
        row["comparison"] is not None
        and row["comparison"]["rms_parent_scale_gate"]
        and row["comparison"]["maxabs_parent_scale_gate"]
        for row in rows
    )
    passed = bool(
        parent_contracts
        and child_contracts
        and activation_count > 0
        and carrier_gate
        and comparisons_available
        and no_activation_parity
        and parent_scale_gate
    )
    receipt = {
        "schema": "q24-cs2-target-free-fullwidth-official-he-comparator-v1",
        "status": (
            "PASS_TARGET_FREE_FULLWIDTH_STABILITY__LITERAL_METER_AND_ACCURACY_HOLD"
            if passed
            else "REJECT_TARGET_FREE_FULLWIDTH_STABILITY__SOURCE_CLOSED"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bindings": {
            "candidate_estimator_sha256": SOURCE_SHA256,
            "source_hold_sha256": SOURCE_HOLD_SHA256,
            "parent_estimator_sha256": PARENT_SHA256,
            "parent_literal_meter_receipt_sha256": PARENT_METER_SHA256,
            "preregister_sha256": PREREGISTER_SHA256,
            "runner_sha256": _sha256(Path(__file__)),
            "fullwidth_release_sha256": release_sha256,
            "one_use_claim_sha256": claim_sha256,
            "independent_strict_audit": audit_binding,
        },
        "gates": {
            "parent_contracts_all_three": parent_contracts,
            "child_contracts_all_three": child_contracts,
            "invalid_unit_reset_activation_count": activation_count,
            "at_least_one_activation": activation_count > 0,
            "carrier_violation_totals": carrier_violations,
            "carrier_invariants": carrier_gate,
            "comparisons_available_all_three": comparisons_available,
            "zero_activation_rows_have_raw_byte_parity": no_activation_parity,
            "all_rows_within_parent_output_scale": parent_scale_gate,
            "fullwidth_comparator_pass": passed,
        },
        "rows": rows,
        "scope": {
            "seeds_executed_exact_in_order": list(SEEDS),
            "seed_sweep": False,
            "dataset_arguments": 0,
            "dataset_rows_accessed": 0,
            "public_target_values_accessed": 0,
            "private_target_values_accessed": 0,
            "truth_computations": 0,
            "fits": 0,
            "samples_used_by_estimator": 0,
            "packages": 0,
            "submissions": 0,
            "graph_edits": 0,
            "claim_boundary": (
                "Weights-only stability and carrier evidence only; no raw-MSE, "
                "accuracy, board, or literal-meter claim."
            ),
        },
    }
    temporary = OUTPUT.with_name(f".{OUTPUT.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, OUTPUT)
    print(json.dumps(receipt["gates"], indent=2, sort_keys=True), flush=True)
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
