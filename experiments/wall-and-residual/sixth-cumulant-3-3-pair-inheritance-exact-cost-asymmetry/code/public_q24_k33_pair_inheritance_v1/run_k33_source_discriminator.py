#!/usr/bin/env python3
"""Sealed target-blind RAW-K33 source/action discriminator.

The default action is a standard-library-only static audit.  Numerical work
is impossible unless a separately created ``K33_RELEASE.json`` pins this
runner and every frozen authority byte for byte.  An authorized call uses two
fresh subprocesses: the frozen public D4 prediction first, then the read-only
K33 source/action instrumentation.  It opens one sealed weights-only
``observer_fit`` row, never reads a target or score, and never feeds K33 back
into the estimator.
"""

from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import secrets
import subprocess
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
DERIVATION = HERE / "DERIVATION.md"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
STATIC_VERIFICATION = HERE / "STATIC_VERIFICATION_V5.json"
PROTOCOL_AUTHORITY = HERE / "K33_PROTOCOL_AUTHORITY_V1.json"
COORDINATE_AUTHORITY = HERE / "K33_COORDINATE_PRESERVATION_AUTHORITY_V1.json"
DTYPE_PRECEDENCE = HERE / "K33_PROTOCOL_ACCOUNTING_PRECEDENCE_V2.json"
DTYPE_AUDIT = HERE / "INDEPENDENT_DTYPE_ACCOUNTING_HOLD_V1.json"
SOURCE = HERE / "k33_instrumented_source.py"
RUNNER = HERE / "run_k33_source_discriminator.py"
RELEASE = HERE / "K33_RELEASE.json"
RELEASE_SCAFFOLD = HERE / "K33_RELEASE_SCAFFOLD.json"
BASE_SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py"
PRICING = P2 / "p2r0c1d28_k33_local_reset_static_pricing_v1/K33_STATIC_PRICING_RECEIPT.json"
PRICING_ERRATUM = P2 / "p2r0c1d28_k33_local_reset_static_pricing_v1/K33_STATIC_PRICING_ACCOUNTING_ERRATUM_V1.json"
STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)

BASE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
PRICING_SHA256 = "379d85b23513cf2d0a4ca0937b6bdf628b2b855ef29c533d5d4a7c682b30e954"
PRICING_ERRATUM_SHA256 = "351e12bd2e491d8e91cc9da6901e2763332461ab74cee17526fa6a20df05d3af"
PROTOCOL_SHA256 = "9c84584c7cc54ec6efd42eec9b41e2eff1c49c272c93aed5d9703b1eba7fc823"
COORDINATE_AUTHORITY_SHA256 = "7c95892746654c68db34b84fe18174a35e108016256ad92ecc75d6b194465295"
DTYPE_PRECEDENCE_SHA256 = "a950b1af3a8a211e9682def2a53596258022709f99e6bd6fbd6e4fbb33bf4778"
DTYPE_AUDIT_SHA256 = "9ad342149f4a9628eadbcc2dd144ef84f7389d66705c0f0149969273e4fd05c6"
STATIC_VERIFICATION_SHA256 = "a1daa778b6081ffed9ce8d296f3a81e66f4d475cac201b11852648dd582c4857"

ALLOWED_RANKS = (201, 643)
CAPTURE_SOURCE_LAYERS = (4, 9, 14)
PUBLIC_COMPONENTS = (
    "host",
    "native_raw20",
    "native_sigma",
    "native_j_correction",
    "final_public_prediction",
)
PHASE2_BUDGET = 2**41
EXPECTED_SOURCE_INCREMENT = 76_382_558_208
SOURCE_INCREMENT_CAP = 76_500_000_000
SOURCE_SCRATCH_CAP = 153_665_536
LARGEST_NEW_ARRAY_CAP = 142_606_336
REMOTE_SINGLE_ARRAY_CAP = 4 * 1024**3
PROCESS_RSS_CAP = 8 * 1024**3
RESIDUAL_SECONDS_CAP = 0.4
WORKER_WALL_SECONDS = 120.0
OFFICIAL_INTERPRETER = (
    P2 / "data_gate/.venv/bin/python"
).absolute()
OFFICIAL_RUNTIME_ENVIRONMENT = {
    "interpreter": str(OFFICIAL_INTERPRETER),
    "flopscope_version": "0.12.0",
    "whestbench_version": "0.16.0",
    "numpy_version": "2.4.6",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_content_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    """Create one immutable control receipt; an existing path is terminal."""

    with path.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_npy(path: Path, array: Any) -> dict[str, Any]:
    import numpy as np

    value = np.ascontiguousarray(array, dtype=np.dtype("<f8"))
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.save(handle, value, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "content_sha256": array_content_sha256(value),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
    }


def peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def load_module(tag: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(tag, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def source_ast_facts() -> dict[str, Any]:
    tree = ast.parse(SOURCE.read_text())
    string_arguments: list[str] = []
    calls: list[str] = []
    assignments: list[str] = []

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = dotted(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        if isinstance(node, ast.Call):
            return f"{dotted(node.func)}()"
        return ""

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            calls.append(dotted(node.func))
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                string_arguments.append(node.args[0].value)
        if isinstance(node, ast.Assign):
            assignments.extend(dotted(target) for target in node.targets)
    return {
        "calls": calls,
        "string_arguments": string_arguments,
        "assignments": assignments,
    }


def static_receipt() -> dict[str, Any]:
    prereg = json.loads(PREREGISTRATION.read_text())
    static = json.loads(STATIC_VERIFICATION.read_text())
    scaffold = json.loads(RELEASE_SCAFFOLD.read_text())
    source_text = SOURCE.read_text()
    runner_text = RUNNER.read_text()
    runner_tree = ast.parse(runner_text)
    runner_function_text = {
        node.name: ast.get_source_segment(runner_text, node) or ""
        for node in runner_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    execute_text = runner_function_text["execute"]
    worker_text = runner_function_text["worker_phase"]
    facts = source_ast_facts()
    weighted_reductions = facts["string_arguments"].count("dij,dij->ij")
    source_einsums = facts["calls"].count("fnp.einsum")
    lock_create_offset = execute_text.index("exclusive_json(lock_path, lock_payload)")
    base_launch_offset = execute_text.index('base = run_subprocess_phase(\n                "base"')
    base_preflight_offset = execute_text.index("base_preflight = {")
    source_launch_offset = execute_text.index('source = run_subprocess_phase(\n                "source"')
    worker_validate_offset = worker_text.index(
        "launch_authority = validate_worker_ticket(ticket_path, phase, rank)"
    )
    sealed_row_offset = worker_text.index("row, weights, store_hashes = load_sealed_row(rank)")
    checks = {
        "derivation_exists": DERIVATION.is_file(),
        "preregistration_json_valid": True,
        "static_v5_pass": static.get("status") == "PASS_STATIC_TARGET_FREE",
        "static_v5_sha256": sha256_file(STATIC_VERIFICATION) == STATIC_VERIFICATION_SHA256,
        "base_source_sha256": sha256_file(BASE_SOURCE) == BASE_SHA256,
        "pricing_sha256": sha256_file(PRICING) == PRICING_SHA256,
        "pricing_erratum_sha256": sha256_file(PRICING_ERRATUM) == PRICING_ERRATUM_SHA256,
        "protocol_authority_sha256": sha256_file(PROTOCOL_AUTHORITY) == PROTOCOL_SHA256,
        "coordinate_authority_sha256": sha256_file(COORDINATE_AUTHORITY) == COORDINATE_AUTHORITY_SHA256,
        "dtype_accounting_precedence_sha256": sha256_file(DTYPE_PRECEDENCE) == DTYPE_PRECEDENCE_SHA256,
        "independent_dtype_accounting_sha256": sha256_file(DTYPE_AUDIT) == DTYPE_AUDIT_SHA256,
        "official_interpreter_exists": OFFICIAL_INTERPRETER.is_file(),
        "release_scaffold_is_nonexecuting_hold": scaffold.get("status") == "HOLD_SCAFFOLD_NOT_EXECUTABLE" and scaffold.get("independent_final_go") is False,
        "release_scaffold_binds_runner": scaffold.get("runner_sha256") == sha256_file(RUNNER),
        "release_scaffold_binds_runtime": scaffold.get("runtime_environment") == OFFICIAL_RUNTIME_ENVIRONMENT,
        "source_runtime_checks_base_sha": "frozen D4 source hash mismatch before import" in source_text,
        "source_calls_full_public_estimator": "base.Estimator().predict(mlp, base._PHASE2_FLOP_BUDGET)" in source_text,
        "source_does_not_call_native_tape_shortcut": "predict_with_native_tape(" not in source_text,
        "source_wraps_internal_tape_once": "base._predict = predict_with_tape_capture" in source_text,
        "source_wraps_native_j_once": "base.transport_correction = transport_with_capture" in source_text,
        "source_has_three_einsum_spellings": source_einsums == 3,
        "source_has_exact_weighted_reduction_spelling": weighted_reductions == 1,
        "source_uses_writable_fill_diagonal_zero": "fnp.fill_diagonal(k33, 0.0)" in source_text,
        "source_exposes_five_public_components": "state[\"public_correction\"]" in source_text and "final_output" in source_text,
        "source_passes_l3_l4_l3sq_diagnostics": "coefficients[0, 3]" in source_text and "coefficients[0, 4]" in source_text and "coefficients[0, 6]" in source_text,
        "no_retry_lock_precedes_base_launch": lock_create_offset < base_launch_offset,
        "lock_creation_and_durability_are_inside_terminal_failure_boundary": (
            "try:\n        exclusive_json(lock_path, lock_payload)\n"
            "        fsync_directory(HERE)\n"
            "        lock_sha256 = sha256_file(lock_path)"
            in execute_text
        ),
        "base_preflight_precedes_source_launch": (
            base_launch_offset < base_preflight_offset < source_launch_offset
        ),
        "worker_validates_lock_capability_before_sealed_row_open": (
            worker_validate_offset < sealed_row_offset
        ),
        "lock_commits_distinct_phase_capability_hashes": (
            'for phase, capability in phase_capabilities.items()' in execute_text
            and 'for phase in ("base", "source")' in execute_text
            and '"phase_capability_sha256"' in execute_text
        ),
        "worker_creates_exclusive_phase_consumption_receipt": (
            "exclusive_json(\n        consumption_path," in runner_text
            and '"PHASE_CAPABILITY_IRREVOCABLY_CONSUMED"' in runner_text
        ),
        "result_and_coordinates_have_one_atomic_bundle_rename": (
            execute_text.count("os.replace(staged_bundle, bundle_path)") == 1
            and "fsync_directory(staged_bundle)" in execute_text
            and 'staged_result_path = staged_bundle / result_name' in execute_text
            and 'commit_manifest_path = staged_bundle / "COMMIT_MANIFEST.json"'
            in execute_text
        ),
        "failure_receipt_is_suppressed_after_visible_bundle_commit": (
            "if not bundle_path.exists() and not failure_path.exists():" in execute_text
        ),
        "prereg_forbids_rank35": prereg["release_protocol"]["rank35_allowed"] is False,
        "prereg_orders_201_then_643": prereg["release_protocol"]["ordered_role_ranks"] == [201, 643],
        "prereg_forbids_targets": prereg["scope"]["targets_allowed"] is False,
        "prereg_binds_coordinate_outputs": prereg["source_protocol"]["coordinate_preservation"]["full_planes_persisted"] is False,
        "release_absent": not RELEASE.exists(),
        "rank201_bundle_absent": not (HERE / "K33_RANK0201_BUNDLE").exists(),
        "rank643_bundle_absent": not (HERE / "K33_RANK0643_BUNDLE").exists(),
        "rank201_failure_absent": not (HERE / "K33_RANK0201_FAILURE.json").exists(),
        "rank643_failure_absent": not (HERE / "K33_RANK0643_FAILURE.json").exists(),
        "rank201_lock_absent": not (HERE / "K33_RANK0201_RUN.lock").exists(),
        "rank643_lock_absent": not (HERE / "K33_RANK0643_RUN.lock").exists(),
        "rank201_phase_consumption_absent": all(not (HERE / f"K33_RANK0201_{phase}_WORKER_CONSUMED.json").exists() for phase in ("BASE", "SOURCE")),
        "rank643_phase_consumption_absent": all(not (HERE / f"K33_RANK0643_{phase}_WORKER_CONSUMED.json").exists() for phase in ("BASE", "SOURCE")),
    }
    return {
        "schema": "q24-raw-k33-source-runner-static-v2-dtype-accounting",
        "status": "STATIC_READY_HOLD_NO_RELEASE" if all(checks.values()) else "STATIC_HOLD",
        "checks": checks,
        "bindings": {
            "derivation_sha256": sha256_file(DERIVATION),
            "preregistration_sha256": sha256_file(PREREGISTRATION),
            "static_verification_v5_sha256": sha256_file(STATIC_VERIFICATION),
            "protocol_authority_sha256": sha256_file(PROTOCOL_AUTHORITY),
            "coordinate_authority_sha256": sha256_file(COORDINATE_AUTHORITY),
            "dtype_accounting_precedence_sha256": sha256_file(DTYPE_PRECEDENCE),
            "independent_dtype_accounting_sha256": sha256_file(DTYPE_AUDIT),
            "base_source_sha256": sha256_file(BASE_SOURCE),
            "pricing_sha256": sha256_file(PRICING),
            "pricing_erratum_sha256": sha256_file(PRICING_ERRATUM),
            "instrumented_source_sha256": sha256_file(SOURCE),
            "runner_sha256": sha256_file(RUNNER),
            "release_scaffold_sha256": sha256_file(RELEASE_SCAFFOLD),
        },
        "literal_remote_contract": {
            "source_runtime_einsums": "one initial C33 plus fourteen kernel and fourteen weighted-reduction calls",
            "source_incremental_flops": EXPECTED_SOURCE_INCREMENT,
            "source_incremental_flop_cap": SOURCE_INCREMENT_CAP,
            "source_incremental_scratch_bytes": SOURCE_SCRATCH_CAP,
            "largest_new_array_bytes": LARGEST_NEW_ARRAY_CAP,
            "public_components": list(PUBLIC_COMPONENTS),
            "fixed_full_plane_capture_layers": list(CAPTURE_SOURCE_LAYERS),
            "coordinate_slot_mapping": "slot s=0..14: source h(s+1), outgoing mlp.weights[s+1], consumer/birth h(s+2)",
            "required_runtime_environment": OFFICIAL_RUNTIME_ENVIRONMENT,
        },
        "release_path": str(RELEASE),
        "release_present": RELEASE.exists(),
        "release_scaffold_path": str(RELEASE_SCAFFOLD),
        "numerical_work_performed": False,
        "sealed_rows_opened": 0,
        "targets_read": 0,
    }


def expected_release_hashes() -> dict[str, str]:
    return {
        "derivation_sha256": sha256_file(DERIVATION),
        "preregistration_sha256": sha256_file(PREREGISTRATION),
        "static_verification_v5_sha256": sha256_file(STATIC_VERIFICATION),
        "protocol_authority_sha256": sha256_file(PROTOCOL_AUTHORITY),
        "coordinate_authority_sha256": sha256_file(COORDINATE_AUTHORITY),
        "dtype_accounting_precedence_sha256": sha256_file(DTYPE_PRECEDENCE),
        "independent_dtype_accounting_sha256": sha256_file(DTYPE_AUDIT),
        "instrumented_source_sha256": sha256_file(SOURCE),
        "runner_sha256": sha256_file(RUNNER),
        "base_source_sha256": sha256_file(BASE_SOURCE),
        "pricing_sha256": sha256_file(PRICING),
        "pricing_erratum_sha256": sha256_file(PRICING_ERRATUM),
    }


def require_release(rank: int) -> dict[str, Any]:
    if rank not in ALLOWED_RANKS:
        raise RuntimeError(f"rank must be one of {ALLOWED_RANKS}; rank 35 is forbidden")
    if not RELEASE.is_file():
        raise RuntimeError("K33_RELEASE.json is absent; numerical execution is not authorized")
    release = json.loads(RELEASE.read_text())
    if release.get("status") != "RELEASE_K33_SOURCE_ACTION_ONLY":
        raise RuntimeError("release status is not RELEASE_K33_SOURCE_ACTION_ONLY")
    if release.get("allowed_role") != "observer_fit" or release.get("allowed_role_ranks") != [rank]:
        raise RuntimeError("release must authorize exactly one observer_fit rank")
    for key, value in expected_release_hashes().items():
        if release.get(key) != value:
            raise RuntimeError(f"release hash mismatch: {key}")
    if release.get("independent_final_go") is not True:
        raise RuntimeError("release lacks independent FINAL GO")
    if release.get("runtime_environment") != OFFICIAL_RUNTIME_ENVIRONMENT:
        raise RuntimeError("release does not bind the exact official runtime environment")
    if Path(sys.executable).absolute() != OFFICIAL_INTERPRETER:
        raise RuntimeError("numerical execution requires the bound official interpreter")
    observed_versions = {
        "interpreter": str(Path(sys.executable).absolute()),
        "flopscope_version": importlib.metadata.version("flopscope"),
        "whestbench_version": importlib.metadata.version("whestbench"),
        "numpy_version": importlib.metadata.version("numpy"),
    }
    if observed_versions != OFFICIAL_RUNTIME_ENVIRONMENT:
        raise RuntimeError(f"official runtime version mismatch: {observed_versions}")
    review = release.get("review")
    if not isinstance(review, dict) or not review.get("reviewer_identity") or not review.get("summary"):
        raise RuntimeError("release lacks reviewer identity/summary")
    if release.get("no_retry") is not True or release.get("concurrent_heavy_run_allowed") is not False:
        raise RuntimeError("release weakens no-retry or single-heavy-owner protocol")
    forbidden = (
        "targets_allowed",
        "scores_allowed",
        "truth_join_allowed",
        "training_allowed",
        "observer_selection_allowed",
        "autonomous_feedback_allowed",
        "full_estimator_allowed",
        "submission_allowed",
    )
    if any(release.get(key, False) for key in forbidden):
        raise RuntimeError("release attempts to broaden target-blind source-only authority")

    prerequisites = release.get("prerequisite_result_sha256", {})
    if rank == 201:
        if prerequisites != {}:
            raise RuntimeError("rank201 release must have no K33 numerical prerequisite")
    else:
        name = "K33_RANK0201_RESULT.json"
        if set(prerequisites) != {name}:
            raise RuntimeError("rank643 release must pin exactly the immutable rank201 result")
        bundle = HERE / "K33_RANK0201_BUNDLE"
        path = bundle / name
        if not path.is_file() or sha256_file(path) != prerequisites[name]:
            raise RuntimeError("rank201 prerequisite is missing or hash-mismatched")
        result = json.loads(path.read_text())
        if (
            result.get("status") != "COMPLETE_TARGET_BLIND_K33_SOURCE_ACTION"
            or result.get("decision") != "PASS_SOURCE_ACTION_RANK201__HOLD_RANK643_PENDING_AUDIT_RELEASE"
            or result.get("all_gates_pass") is not True
            or result.get("role") != "observer_fit"
            or int(result.get("role_rank", -1)) != 201
            or result.get("firewall", {}).get("targets_read") != 0
        ):
            raise RuntimeError("rank201 prerequisite is not an audited passing source-only result")
        manifest_path = bundle / "COMMIT_MANIFEST.json"
        if (
            not manifest_path.is_file()
            or sha256_file(manifest_path)
            != result.get("authority", {}).get("commit_manifest_sha256")
        ):
            raise RuntimeError("rank201 atomic-bundle commit manifest is missing or hash-mismatched")
        for artifact in result.get("coordinate_artifacts", {}).values():
            artifact_path = bundle / Path(artifact["path"]).name
            if not artifact_path.is_file() or sha256_file(artifact_path) != artifact["file_sha256"]:
                raise RuntimeError("rank201 coordinate prerequisite is missing or hash-mismatched")
    return release


def load_sealed_row(rank: int) -> tuple[dict[str, Any], Any, dict[str, str]]:
    import numpy as np

    manifest_path = STORE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise RuntimeError("sealed-store role-rank mismatch")
    row_path = STORE / entry["path"]
    if sha256_file(row_path) != entry["sha256"]:
        raise RuntimeError("sealed row-manifest hash mismatch")
    row = json.loads(row_path.read_text())
    if row.get("role") != "observer_fit" or int(row.get("role_rank", -1)) != rank:
        raise RuntimeError("sealed row is not the requested observer_fit rank")
    if set(row.get("decoded_columns", ())) != {"mlp_id", "mlp_name", "mlp_seed", "weights"}:
        raise RuntimeError("sealed row decoded-column firewall changed")
    weights_path = row_path.parent / "weights.npy"
    expected_weight_hash = row["files"]["weights.npy"]["sha256"]
    if sha256_file(weights_path) != expected_weight_hash:
        raise RuntimeError("sealed weight-file hash mismatch")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.shape != (16, 1024, 1024) or weights.dtype != np.dtype("<f4"):
        raise RuntimeError("sealed weights have unexpected shape or dtype")
    hashes = {
        "store_manifest_sha256": sha256_file(manifest_path),
        "row_manifest_sha256": sha256_file(row_path),
        "weights_file_sha256": expected_weight_hash,
    }
    return row, weights, hashes


def make_mlp(weights: Any, row: dict[str, Any], rank: int) -> Any:
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed

    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[index] for index in range(16)],
        seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
        name=f"k33-target-blind-observer-fit-{rank:04d}",
    )
    mlp.validate()
    return mlp


def public_with_components(module: Any, mlp: Any) -> tuple[Any, Any, Any, Any, Any]:
    original_predict = module._predict
    original_transport = module.transport_correction
    state: dict[str, Any] = {"tape": None, "correction": None}

    def capture_predict(*args: Any, **kwargs: Any) -> Any:
        answer = original_predict(*args, **kwargs)
        if state["tape"] is not None or not isinstance(answer, tuple) or len(answer) != 3:
            raise RuntimeError("frozen public D4 internal tape call count/shape changed")
        state["tape"] = answer
        return answer

    def capture_transport(*args: Any, **kwargs: Any) -> Any:
        answer = original_transport(*args, **kwargs)
        if state["correction"] is not None:
            raise RuntimeError("frozen public D4 native-J executed more than once")
        state["correction"] = answer
        return answer

    module._predict = capture_predict
    module.transport_correction = capture_transport
    try:
        final = module.Estimator().predict(mlp, module._PHASE2_FLOP_BUDGET)
    finally:
        module._predict = original_predict
        module.transport_correction = original_transport
    if state["tape"] is None or state["correction"] is None:
        raise RuntimeError("frozen public D4 did not expose the required tape/J path")
    host, native_raw, native_sigma = state["tape"]
    return host, native_raw, native_sigma, state["correction"], final


def local_array(remote: Any) -> Any:
    import numpy as np

    return np.array(remote, dtype=np.dtype("<f8"), copy=True)


def vector_metrics(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value, dtype=np.float64)
    count = int(array.size)
    return {
        "l2_norm": float(np.linalg.norm(array)),
        "rms": float(np.sqrt(np.mean(array * array, dtype=np.float64))),
        "max_abs": float(np.max(np.abs(array))),
        "positive_fraction": float(np.count_nonzero(array > 0.0) / count),
        "negative_fraction": float(np.count_nonzero(array < 0.0) / count),
        "zero_fraction": float(np.count_nonzero(array == 0.0) / count),
        "finite": bool(np.isfinite(array).all()),
        "content_sha256": array_content_sha256(array),
    }


class StreamingK33Sink:
    """Eagerly fetch fixed planes before scratch reuse; persist no full plane."""

    def __init__(self) -> None:
        self.c33: dict[int, Any] = {}
        self.pre_k33: dict[int, Any] = {}
        self.post_k33: dict[int, Any] = {}
        self.raw_k6: dict[int, Any] = {}
        self.births: dict[int, dict[str, Any]] = {}

    def capture_c33(self, layer: int, value: Any) -> None:
        if layer in CAPTURE_SOURCE_LAYERS:
            if layer in self.c33:
                raise RuntimeError(f"duplicate C33 capture at h{layer}")
            self.c33[layer] = local_array(value)

    def capture_pre_symmetry(self, layer: int, value: Any) -> None:
        if layer in CAPTURE_SOURCE_LAYERS:
            if layer in self.pre_k33:
                raise RuntimeError(f"duplicate pre-symmetry K33 capture at h{layer}")
            self.pre_k33[layer] = local_array(value)

    def capture_action(self, layer: int, k33: Any, raw_k6: Any) -> None:
        if layer in self.raw_k6:
            raise RuntimeError(f"duplicate raw-k6 capture at h{layer}")
        self.raw_k6[layer] = local_array(raw_k6)
        if layer in CAPTURE_SOURCE_LAYERS:
            self.post_k33[layer] = local_array(k33)

    def capture_birth(
        self,
        target_layer: int,
        raw_k6: Any,
        c1_3: Any,
        c1_4: Any,
        c1_6: Any,
        sigma: Any,
        k3: Any,
        k4: Any,
        birth: Any,
        tangent: Any,
    ) -> None:
        if target_layer in self.births:
            raise RuntimeError(f"duplicate L6 birth capture at h{target_layer}")
        self.births[target_layer] = {
            "raw_k6": local_array(raw_k6),
            "c1_3": local_array(c1_3),
            "c1_4": local_array(c1_4),
            "c1_6": local_array(c1_6),
            "sigma": local_array(sigma),
            "k3": local_array(k3),
            "k4": local_array(k4),
            "birth": local_array(birth),
            "tangent": local_array(tangent),
        }

    def finalize(self) -> tuple[list[dict[str, Any]], Any, Any]:
        import numpy as np

        if set(self.raw_k6) != set(range(1, 16)):
            raise RuntimeError("raw-k6 layer set is not exactly h1:h15")
        if set(self.births) != set(range(2, 17)):
            raise RuntimeError("L6 birth layer set is not exactly h2:h16")
        if set(self.c33) != set(CAPTURE_SOURCE_LAYERS):
            raise RuntimeError("C33 fixed capture layer set changed")
        if set(self.pre_k33) != set(CAPTURE_SOURCE_LAYERS):
            raise RuntimeError("pre-K33 fixed capture layer set changed")
        if set(self.post_k33) != set(CAPTURE_SOURCE_LAYERS):
            raise RuntimeError("post-K33 fixed capture layer set changed")

        rows: list[dict[str, Any]] = []
        raw_bank = np.empty((15, 1024), dtype=np.dtype("<f8"))
        birth_bank = np.empty((15, 1024), dtype=np.dtype("<f8"))
        for slot in range(15):
            source_layer = slot + 1
            consumer_layer = slot + 2
            raw = self.raw_k6[source_layer]
            birth_row = self.births[consumer_layer]
            if not np.array_equal(raw.view(np.uint8), birth_row["raw_k6"].view(np.uint8)):
                raise RuntimeError(f"raw-k6 action/birth association changed at slot {slot}")
            sigma = np.maximum(birth_row["sigma"], 1.0e-300)
            inv_sigma3 = sigma ** -3
            l3 = birth_row["c1_3"] * (birth_row["k3"] * inv_sigma3 / 6.0)
            l4 = birth_row["c1_4"] * (birth_row["k4"] * (inv_sigma3 / sigma) / 24.0)
            l3sq = birth_row["c1_6"] * (
                (birth_row["k3"] * birth_row["k3"]) * (inv_sigma3 * inv_sigma3) / 72.0
            )
            exact_birth = birth_row["birth"]
            columns = np.stack((l3, l4, l3sq), axis=1)
            coefficients, _residuals, _rank, _singular = np.linalg.lstsq(
                columns, exact_birth, rcond=None
            )
            projected = columns @ coefficients
            residual = exact_birth - projected
            birth_energy = float(np.vdot(exact_birth, exact_birth).real)
            residual_energy = float(np.vdot(residual, residual).real)
            l3sq_rms = float(np.sqrt(np.mean(l3sq * l3sq, dtype=np.float64)))
            l6_rms = float(np.sqrt(np.mean(exact_birth * exact_birth, dtype=np.float64)))

            full_plane: dict[str, Any] | None = None
            if source_layer in CAPTURE_SOURCE_LAYERS:
                c33 = self.c33[source_layer]
                pre = self.pre_k33[source_layer]
                post = self.post_k33[source_layer]
                c33_norm = float(np.linalg.norm(c33))
                pre_norm = float(np.linalg.norm(pre))
                post_norm = float(np.linalg.norm(post))
                asymmetry = float(np.linalg.norm(pre - pre.T) / max(pre_norm, 1.0e-300))
                eigenvalues = np.linalg.eigvalsh(post)
                singular_values = np.abs(eigenvalues)
                spectral_sum = float(np.sum(singular_values, dtype=np.float64))
                if spectral_sum > 0.0:
                    probabilities = singular_values[singular_values > 0.0] / spectral_sum
                    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
                    spectral_entropy = float(-np.sum(probabilities * np.log(probabilities)))
                else:
                    effective_rank = float("nan")
                    spectral_entropy = float("nan")
                offdiag_count = post.size - post.shape[0]
                full_plane = {
                    "c33_frobenius_norm": c33_norm,
                    "c33_rms": float(np.sqrt(np.mean(c33 * c33, dtype=np.float64))),
                    "pre_k33_frobenius_norm": pre_norm,
                    "post_k33_frobenius_norm": post_norm,
                    "post_k33_rms": float(np.sqrt(np.mean(post * post, dtype=np.float64))),
                    "connected_survival_to_c33_ratio": pre_norm / max(c33_norm, 1.0e-300),
                    "connected_subtraction_to_c33_ratio": float(np.linalg.norm(c33 - pre) / max(c33_norm, 1.0e-300)),
                    "pre_symmetrization_relative_asymmetry": asymmetry,
                    "post_symmetry_max_abs": float(np.max(np.abs(post - post.T))),
                    "post_diagonal_max_abs": float(np.max(np.abs(np.diag(post)))),
                    "post_positive_offdiagonal_fraction": float(np.count_nonzero(post > 0.0) / offdiag_count),
                    "post_negative_offdiagonal_fraction": float(np.count_nonzero(post < 0.0) / offdiag_count),
                    "signed_spectral_entropy": spectral_entropy,
                    "signed_spectral_effective_rank": effective_rank,
                    "all_values_finite": bool(
                        np.isfinite(c33).all()
                        and np.isfinite(pre).all()
                        and np.isfinite(post).all()
                        and np.isfinite(eigenvalues).all()
                    ),
                    "c33_content_sha256": array_content_sha256(c33),
                    "pre_k33_content_sha256": array_content_sha256(pre),
                    "post_k33_content_sha256": array_content_sha256(post),
                }

            raw_bank[slot] = raw
            birth_bank[slot] = exact_birth
            rows.append(
                {
                    "slot": slot,
                    "source_layer_one_based": source_layer,
                    "outgoing_weight_index_zero_based": slot + 1,
                    "outgoing_weight_literal": f"mlp.weights[{slot + 1}]",
                    "consumer_birth_layer_one_based": consumer_layer,
                    "raw_k6": vector_metrics(raw),
                    "l6_mean_birth": vector_metrics(exact_birth),
                    "existing_l3_mean_emission": vector_metrics(l3),
                    "existing_l4_mean_emission": vector_metrics(l4),
                    "existing_l3_squared_mean_emission": vector_metrics(l3sq),
                    "l6_to_l3_squared_rms_ratio_diagnostic_only": (
                        l6_rms / l3sq_rms if l3sq_rms > 0.0 else None
                    ),
                    "l6_residual_energy_fraction_after_l3_l4_l3sq_projection_diagnostic_only": (
                        residual_energy / birth_energy if birth_energy > 0.0 else None
                    ),
                    "mean_only_tangent_diagnostic_only": vector_metrics(birth_row["tangent"]),
                    "fixed_full_plane_capture": full_plane,
                }
            )

        self.c33.clear()
        self.pre_k33.clear()
        self.post_k33.clear()
        self.raw_k6.clear()
        self.births.clear()
        return rows, raw_bank, birth_bank


def component_summaries(values: tuple[Any, ...]) -> dict[str, dict[str, Any]]:
    import numpy as np

    if len(values) != len(PUBLIC_COMPONENTS):
        raise RuntimeError("public component count changed")
    summaries: dict[str, dict[str, Any]] = {}
    for name, value in zip(PUBLIC_COMPONENTS, values):
        local = np.array(value, copy=True)
        summaries[name] = {
            "dtype": local.dtype.str,
            "shape": list(local.shape),
            "finite": bool(np.isfinite(local).all()),
            "content_sha256": array_content_sha256(local),
        }
    return summaries


def validate_worker_ticket(ticket_path: Path, phase: str, rank: int) -> dict[str, Any]:
    nonce = os.environ.get("K33_INTERNAL_WORKER_NONCE")
    if not nonce:
        raise RuntimeError("internal worker nonce is absent")
    ticket = json.loads(ticket_path.read_text())
    lock_path = HERE / f"K33_RANK{rank:04d}_RUN.lock"
    if not lock_path.is_file():
        raise RuntimeError("internal worker has no immutable parent no-retry lock")
    lock_sha256 = sha256_file(lock_path)
    lock = json.loads(lock_path.read_text())
    capability = ticket.get("phase_capability")
    capability_sha256 = (
        hashlib.sha256(capability.encode("ascii")).hexdigest()
        if isinstance(capability, str)
        else None
    )
    if (
        ticket.get("schema") != "k33-internal-worker-ticket-v1"
        or ticket.get("phase") != phase
        or int(ticket.get("rank", -1)) != rank
        or ticket.get("nonce") != nonce
        or ticket.get("runner_sha256") != sha256_file(RUNNER)
        or ticket.get("release_sha256") != sha256_file(RELEASE)
        or ticket.get("lock_sha256") != lock_sha256
        or ticket.get("run_id") != lock.get("run_id")
        or lock.get("schema") != "q24-raw-k33-no-retry-lock-v2"
        or lock.get("status") != "CONSUMED_SINGLETON_AUTHORITY"
        or int(lock.get("rank", -1)) != rank
        or lock.get("release_sha256") != sha256_file(RELEASE)
        or lock.get("runner_sha256") != sha256_file(RUNNER)
        or capability_sha256
        != lock.get("phase_capability_sha256", {}).get(phase)
    ):
        raise RuntimeError("internal worker ticket is invalid")
    consumption_name = lock.get("phase_consumption_receipts", {}).get(phase)
    if consumption_name != f"K33_RANK{rank:04d}_{phase.upper()}_WORKER_CONSUMED.json":
        raise RuntimeError("parent lock has an invalid phase-consumption path")
    consumption_path = HERE / consumption_name
    consumed = ticket_path.with_suffix(ticket_path.suffix + ".consumed")
    if consumed.exists() or consumption_path.exists():
        raise RuntimeError("internal worker ticket was already consumed")
    ticket_sha256 = sha256_file(ticket_path)
    exclusive_json(
        consumption_path,
        {
            "schema": "q24-raw-k33-worker-consumption-v1",
            "status": "PHASE_CAPABILITY_IRREVOCABLY_CONSUMED",
            "rank": rank,
            "phase": phase,
            "run_id": lock["run_id"],
            "lock_sha256": lock_sha256,
            "ticket_sha256": ticket_sha256,
            "release_sha256": sha256_file(RELEASE),
            "runner_sha256": sha256_file(RUNNER),
        },
    )
    fsync_directory(HERE)
    os.replace(ticket_path, consumed)
    return {
        "run_id": lock["run_id"],
        "lock_sha256": lock_sha256,
        "ticket_sha256": ticket_sha256,
        "consumption_receipt": consumption_path.name,
        "consumption_receipt_sha256": sha256_file(consumption_path),
    }


def worker_phase(phase: str, rank: int, ticket_path: Path, output_path: Path, bank_dir: Path) -> None:
    import flopscope as flops
    import numpy as np
    from threadpoolctl import threadpool_limits

    release = require_release(rank)
    launch_authority = validate_worker_ticket(ticket_path, phase, rank)
    if output_path.exists():
        raise RuntimeError("worker output already exists")
    if sha256_file(BASE_SOURCE) != BASE_SHA256:
        raise RuntimeError("frozen D4 SHA changed before worker import")
    flops.configure(symmetry_warnings=False)
    row, weights, store_hashes = load_sealed_row(rank)
    mlp = make_mlp(weights, row, rank)
    started = time.perf_counter()
    remote_values: tuple[Any, ...] | None = None
    sink: StreamingK33Sink | None = None
    module_tag = f"k33_worker_{phase}_{rank}_{os.getpid()}"
    try:
        if phase == "base":
            module = load_module(module_tag, BASE_SOURCE)
        elif phase == "source":
            source_module = load_module(module_tag, SOURCE)
            sink = StreamingK33Sink()
        else:
            raise RuntimeError("worker phase must be base or source")

        with threadpool_limits(limits=2):
            with flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=WORKER_WALL_SECONDS,
                quiet=True,
                namespace=f"k33_source_action_{phase}_rank{rank:04d}",
            ) as meter:
                if phase == "base":
                    remote_values = public_with_components(module, mlp)
                    counts = None
                else:
                    remote_values, counts = source_module.predict_with_k33_tape(mlp, sink)
        meter_summary = meter.summary_dict(by_namespace=False)
        components = component_summaries(remote_values)
        layer_rows: list[dict[str, Any]] | None = None
        coordinate_artifacts: dict[str, Any] | None = None
        if phase == "source":
            if sink is None:
                raise RuntimeError("source sink was not initialized")
            layer_rows, raw_bank, birth_bank = sink.finalize()
            raw_path = bank_dir / "raw_k6.npy"
            birth_path = bank_dir / "l6_mean_birth.npy"
            coordinate_artifacts = {
                "raw_k6": atomic_npy(raw_path, raw_bank),
                "l6_mean_birth": atomic_npy(birth_path, birth_bank),
            }
            del raw_bank, birth_bank
        elapsed = time.perf_counter() - started
        payload = {
            "schema": "q24-raw-k33-source-worker-v1",
            "phase": phase,
            "role": "observer_fit",
            "role_rank": rank,
            "mlp_id": int(row["mlp_id"]),
            "public_components": components,
            "literal_meter": meter_summary,
            "outer_wall_seconds": elapsed,
            "process_peak_rss_bytes": peak_rss_bytes(),
            "source_call_counts": counts,
            "layers": layer_rows,
            "coordinate_artifacts": coordinate_artifacts,
            "store_hashes": store_hashes,
            "release_sha256": sha256_file(RELEASE),
            "validated_release_payload": release,
            "launch_authority": launch_authority,
            "firewall": {
                "targets_read": 0,
                "scores_read": 0,
                "truth_joins": 0,
                "training_steps": 0,
                "observer_selection_rows_opened": 0,
            },
        }
        atomic_json(output_path, payload)
    finally:
        del remote_values, mlp
        sys.modules.pop(module_tag, None)
        gc.collect()


def run_subprocess_phase(
    phase: str,
    rank: int,
    directory: Path,
    release_sha: str,
    run_id: str,
    phase_capability: str,
    lock_sha256: str,
) -> dict[str, Any]:
    nonce = secrets.token_hex(32)
    ticket_path = directory / f"{phase}.ticket.json"
    output_path = directory / f"{phase}.result.json"
    bank_dir = directory / "coordinate_banks"
    bank_dir.mkdir(exist_ok=True)
    ticket = {
        "schema": "k33-internal-worker-ticket-v1",
        "phase": phase,
        "rank": rank,
        "nonce": nonce,
        "run_id": run_id,
        "phase_capability": phase_capability,
        "lock_sha256": lock_sha256,
        "release_sha256": release_sha,
        "runner_sha256": sha256_file(RUNNER),
    }
    atomic_json(ticket_path, ticket)
    environment = os.environ.copy()
    environment["K33_INTERNAL_WORKER_NONCE"] = nonce
    command = [
        sys.executable,
        str(RUNNER),
        "--internal-worker-phase",
        phase,
        "--rank",
        str(rank),
        "--worker-ticket",
        str(ticket_path),
        "--worker-output",
        str(output_path),
        "--worker-bank-dir",
        str(bank_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=str(HERE),
        env=environment,
        capture_output=True,
        text=True,
        timeout=360.0,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{phase} worker failed rc={completed.returncode}: "
            f"{completed.stderr[-4000:]}"
        )
    if not output_path.is_file():
        raise RuntimeError(f"{phase} worker returned without an immutable phase receipt")
    return json.loads(output_path.read_text())


def component_parity(base: dict[str, Any], source: dict[str, Any]) -> dict[str, bool]:
    return {
        name: (
            base[name]["dtype"] == source[name]["dtype"]
            and base[name]["shape"] == source[name]["shape"]
            and base[name]["content_sha256"] == source[name]["content_sha256"]
        )
        for name in PUBLIC_COMPONENTS
    }


def execute(rank: int) -> dict[str, Any]:
    import numpy as np

    release = require_release(rank)
    release_sha = sha256_file(RELEASE)
    result_name = f"K33_RANK{rank:04d}_RESULT.json"
    bundle_path = HERE / f"K33_RANK{rank:04d}_BUNDLE"
    failure_path = HERE / f"K33_RANK{rank:04d}_FAILURE.json"
    lock_path = HERE / f"K33_RANK{rank:04d}_RUN.lock"
    consumption_paths = {
        phase: HERE / f"K33_RANK{rank:04d}_{phase.upper()}_WORKER_CONSUMED.json"
        for phase in ("base", "source")
    }
    for path in (bundle_path, failure_path, lock_path, *consumption_paths.values()):
        if path.exists():
            raise RuntimeError(f"refusing retry/overwrite because {path.name} exists")

    run_id = secrets.token_hex(32)
    phase_capabilities = {
        phase: secrets.token_hex(32) for phase in ("base", "source")
    }
    lock_payload = {
        "schema": "q24-raw-k33-no-retry-lock-v2",
        "status": "CONSUMED_SINGLETON_AUTHORITY",
        "rank": rank,
        "run_id": run_id,
        "release_sha256": release_sha,
        "runner_sha256": sha256_file(RUNNER),
        "phase_capability_sha256": {
            phase: hashlib.sha256(capability.encode("ascii")).hexdigest()
            for phase, capability in phase_capabilities.items()
        },
        "phase_consumption_receipts": {
            phase: path.name for phase, path in consumption_paths.items()
        },
        "planned_final_bundle": bundle_path.name,
        "planned_result_basename": result_name,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    lock_sha256: str | None = None
    try:
        exclusive_json(lock_path, lock_payload)
        fsync_directory(HERE)
        lock_sha256 = sha256_file(lock_path)
        with tempfile.TemporaryDirectory(prefix=f".k33_rank{rank:04d}_", dir=HERE) as temporary:
            directory = Path(temporary)
            base = run_subprocess_phase(
                "base",
                rank,
                directory,
                release_sha,
                run_id,
                phase_capabilities["base"],
                lock_sha256,
            )
            base_preflight = {
                "flops_below_phase2_budget": int(base["literal_meter"]["flops_used"]) < PHASE2_BUDGET,
                "residual_seconds_below_0_4": float(base["literal_meter"]["residual_wall_time_s"]) < RESIDUAL_SECONDS_CAP,
                "process_peak_rss_below_8gib": int(base["process_peak_rss_bytes"]) < PROCESS_RSS_CAP,
                "all_public_components_finite": all(
                    item["finite"] for item in base["public_components"].values()
                ),
            }
            if not all(base_preflight.values()):
                raise RuntimeError(f"base preflight gate failed before source launch: {base_preflight}")
            source = run_subprocess_phase(
                "source",
                rank,
                directory,
                release_sha,
                run_id,
                phase_capabilities["source"],
                lock_sha256,
            )

            if base["validated_release_payload"] != release or source["validated_release_payload"] != release:
                raise RuntimeError("worker did not embed the exact validated release payload")
            if base["mlp_id"] != source["mlp_id"] or base["store_hashes"] != source["store_hashes"]:
                raise RuntimeError("fresh base/source workers did not use the same sealed row")
            for phase, payload in (("base", base), ("source", source)):
                launch = payload.get("launch_authority", {})
                receipt_path = consumption_paths[phase]
                if (
                    launch.get("run_id") != run_id
                    or launch.get("lock_sha256") != lock_sha256
                    or launch.get("consumption_receipt") != receipt_path.name
                    or not receipt_path.is_file()
                    or launch.get("consumption_receipt_sha256")
                    != sha256_file(receipt_path)
                ):
                    raise RuntimeError(f"{phase} worker launch is not bound to the parent lock")

            parity = component_parity(base["public_components"], source["public_components"])
            base_flops = int(base["literal_meter"]["flops_used"])
            source_flops = int(source["literal_meter"]["flops_used"])
            added_flops = source_flops - base_flops
            layers = source["layers"]
            if not isinstance(layers, list) or len(layers) != 15:
                raise RuntimeError("source worker did not return exactly 15 slot rows")

            temporary_coordinates = source["coordinate_artifacts"]
            raw_temporary = Path(temporary_coordinates["raw_k6"]["path"])
            birth_temporary = Path(temporary_coordinates["l6_mean_birth"]["path"])
            raw_bank = np.load(raw_temporary, allow_pickle=False)
            birth_bank = np.load(birth_temporary, allow_pickle=False)
            if raw_bank.shape != (15, 1024) or raw_bank.dtype != np.dtype("<f8"):
                raise RuntimeError("raw-k6 coordinate bank shape/dtype changed")
            if birth_bank.shape != (15, 1024) or birth_bank.dtype != np.dtype("<f8"):
                raise RuntimeError("L6-birth coordinate bank shape/dtype changed")
            if array_content_sha256(raw_bank) != temporary_coordinates["raw_k6"]["content_sha256"]:
                raise RuntimeError("raw-k6 coordinate content hash changed")
            if array_content_sha256(birth_bank) != temporary_coordinates["l6_mean_birth"]["content_sha256"]:
                raise RuntimeError("L6-birth coordinate content hash changed")
            del raw_bank, birth_bank
            staged_bundle = directory / f"K33_RANK{rank:04d}_BUNDLE_STAGE"
            staged_bundle.mkdir()
            raw_name = f"K33_RANK{rank:04d}_RAW_K6.npy"
            birth_name = f"K33_RANK{rank:04d}_L6_MEAN_BIRTH.npy"
            raw_staged = staged_bundle / raw_name
            birth_staged = staged_bundle / birth_name
            os.replace(raw_temporary, raw_staged)
            os.replace(birth_temporary, birth_staged)
            coordinate_artifacts = {
                "raw_k6": {
                    **temporary_coordinates["raw_k6"],
                    "path": raw_name,
                    "file_sha256": sha256_file(raw_staged),
                    "slot_mapping": "s=0..14: source h(s+1), outgoing mlp.weights[s+1], consumer/birth h(s+2)",
                },
                "l6_mean_birth": {
                    **temporary_coordinates["l6_mean_birth"],
                    "path": birth_name,
                    "file_sha256": sha256_file(birth_staged),
                    "slot_mapping": "s=0..14: source h(s+1), outgoing mlp.weights[s+1], consumer/birth h(s+2)",
                },
            }

            fixed_plane_rows = [
                row for row in layers if row["fixed_full_plane_capture"] is not None
            ]
            raw_action_norms = [float(row["raw_k6"]["l2_norm"]) for row in layers]
            all_layer_finite = all(
                row["raw_k6"]["finite"]
                and row["l6_mean_birth"]["finite"]
                and row["mean_only_tangent_diagnostic_only"]["finite"]
                for row in layers
            )
            all_component_finite = all(
                item["finite"]
                for phase in (base, source)
                for item in phase["public_components"].values()
            )
            source_residual = float(source["literal_meter"]["residual_wall_time_s"])
            base_residual = float(base["literal_meter"]["residual_wall_time_s"])
            source_peak = int(source["process_peak_rss_bytes"])
            base_peak = int(base["process_peak_rss_bytes"])
            gates = {
                "full_public_host_native_raw_sigma_j_final_byte_identical": all(parity.values()),
                "source_call_counts_exact": source["source_call_counts"] == {
                    "q24_calls": 15,
                    "connected_calls": 15,
                    "source_calls": 15,
                    "push_calls": 15,
                    "action_calls": 15,
                    "reset_calls": 14,
                    "univariate_calls": 15,
                },
                "literal_incremental_flops_exactly_76382558208": added_flops == EXPECTED_SOURCE_INCREMENT,
                "literal_incremental_flops_at_most_76500000000": 0 <= added_flops <= SOURCE_INCREMENT_CAP,
                "source_total_flops_strictly_below_phase2_budget": source_flops < PHASE2_BUDGET,
                "base_and_source_residual_seconds_strictly_below_0_4": base_residual < RESIDUAL_SECONDS_CAP and source_residual < RESIDUAL_SECONDS_CAP,
                "fresh_process_peak_rss_strictly_below_8gib": max(base_peak, source_peak) < PROCESS_RSS_CAP,
                "static_incremental_scratch_at_most_153665536": SOURCE_SCRATCH_CAP <= 153_665_536,
                "static_largest_new_array_at_most_142606336": LARGEST_NEW_ARRAY_CAP <= 142_606_336,
                "static_largest_new_array_below_4gib": LARGEST_NEW_ARRAY_CAP < REMOTE_SINGLE_ARRAY_CAP,
                "all_public_source_action_and_tangent_values_finite": all_component_finite and all_layer_finite and all(row["fixed_full_plane_capture"]["all_values_finite"] for row in fixed_plane_rows),
                "fixed_full_plane_capture_layers_exactly_4_9_14": [row["source_layer_one_based"] for row in fixed_plane_rows] == list(CAPTURE_SOURCE_LAYERS),
                "pre_symmetrization_relative_asymmetry_at_most_1e_10": all(row["fixed_full_plane_capture"]["pre_symmetrization_relative_asymmetry"] <= 1.0e-10 for row in fixed_plane_rows),
                "post_symmetry_exact": all(row["fixed_full_plane_capture"]["post_symmetry_max_abs"] == 0.0 for row in fixed_plane_rows),
                "post_diagonal_exactly_zero": all(row["fixed_full_plane_capture"]["post_diagonal_max_abs"] == 0.0 for row in fixed_plane_rows),
                "fixed_k33_frobenius_norm_strictly_positive": all(row["fixed_full_plane_capture"]["post_k33_frobenius_norm"] > 0.0 for row in fixed_plane_rows),
                "raw_k6_l2_norm_strictly_positive_all_15_actions": all(value > 0.0 and math.isfinite(value) for value in raw_action_norms),
                "coordinate_banks_shape_dtype_hash_exact": all(artifact["shape"] == [15, 1024] and artifact["dtype"] == "<f8" and sha256_file(staged_bundle / artifact["path"]) == artifact["file_sha256"] for artifact in coordinate_artifacts.values()),
                "slot_mapping_exact_all_15": all(row["slot"] == slot and row["source_layer_one_based"] == slot + 1 and row["outgoing_weight_index_zero_based"] == slot + 1 and row["outgoing_weight_literal"] == f"mlp.weights[{slot + 1}]" and row["consumer_birth_layer_one_based"] == slot + 2 for slot, row in enumerate(layers)),
                "role_and_rank_match_singleton_release": base["role"] == source["role"] == "observer_fit" and base["role_rank"] == source["role_rank"] == rank and release["allowed_role_ranks"] == [rank],
                "targets_scores_truth_joins_training_selection_exactly_zero": all(value == 0 for phase in (base, source) for value in phase["firewall"].values()),
                "no_hard_source_materiality_threshold_applied": True,
            }
            passed = all(gates.values())
            decision = (
                "PASS_SOURCE_ACTION_RANK201__HOLD_RANK643_PENDING_AUDIT_RELEASE"
                if passed and rank == 201
                else "GO_K33_AUTONOMOUS_RESET_DESIGN_ONLY"
                if passed and rank == 643
                else "CLOSE_THIS_RAW_D4_Q7_K33_33_SOURCE_ACTION_SPELLING"
            )
            commit_manifest = {
                "schema": "q24-raw-k33-atomic-bundle-commit-manifest-v1",
                "status": "FSYNCED_PRECOMMIT__VISIBLE_ONLY_BY_ATOMIC_DIRECTORY_RENAME",
                "role": "observer_fit",
                "role_rank": rank,
                "run_id": run_id,
                "release_sha256": release_sha,
                "runner_sha256": sha256_file(RUNNER),
                "lock_sha256": lock_sha256,
                "planned_final_bundle": bundle_path.name,
                "planned_files": {
                    "result": result_name,
                    "raw_k6": raw_name,
                    "l6_mean_birth": birth_name,
                    "commit_manifest": "COMMIT_MANIFEST.json",
                },
                "coordinate_artifacts": coordinate_artifacts,
                "planned_terminal": {
                    "status": "COMPLETE_TARGET_BLIND_K33_SOURCE_ACTION",
                    "decision": decision,
                    "all_gates_pass": passed,
                },
                "phase_consumption_receipts": {
                    phase: {
                        "path": path.name,
                        "sha256": sha256_file(path),
                    }
                    for phase, path in consumption_paths.items()
                },
                "full_planes_persisted": False,
            }
            commit_manifest_path = staged_bundle / "COMMIT_MANIFEST.json"
            atomic_json(commit_manifest_path, commit_manifest)
            fsync_directory(staged_bundle)
            commit_manifest_sha256 = sha256_file(commit_manifest_path)
            result = {
                "schema": "q24-raw-k33-target-blind-source-action-result-v1",
                "status": "COMPLETE_TARGET_BLIND_K33_SOURCE_ACTION",
                "decision": decision,
                "all_gates_pass": passed,
                "role": "observer_fit",
                "role_rank": rank,
                "mlp_id": int(base["mlp_id"]),
                "slot_mapping": "slot s=0..14 = source h(s+1), outgoing mlp.weights[s+1], consumer/birth h(s+2)",
                "parity": {
                    "components": parity,
                    "base": base["public_components"],
                    "source": source["public_components"],
                },
                "literal_meter": {
                    "phase2_budget": PHASE2_BUDGET,
                    "base_preflight_before_source_launch": base_preflight,
                    "base": base["literal_meter"],
                    "source": source["literal_meter"],
                    "source_minus_base_flops": added_flops,
                    "expected_source_minus_base_flops": EXPECTED_SOURCE_INCREMENT,
                },
                "rss": {
                    "base_fresh_process_peak_bytes": base_peak,
                    "source_fresh_process_peak_bytes": source_peak,
                    "process_cap_bytes": PROCESS_RSS_CAP,
                    "static_incremental_scratch_bytes": SOURCE_SCRATCH_CAP,
                    "static_largest_new_remote_array_bytes": LARGEST_NEW_ARRAY_CAP,
                    "fixed_plane_fetches_are_transient_client_memory_only": True,
                    "full_c33_or_k33_planes_persisted": False,
                },
                "layers": layers,
                "coordinate_artifacts": coordinate_artifacts,
                "gates": gates,
                "firewall": {
                    "targets_read": 0,
                    "scores_read": 0,
                    "truth_joins": 0,
                    "training_steps": 0,
                    "observer_selection_rows_opened": 0,
                    "autonomous_feedback_runs": 0,
                    "full_estimator_candidate_runs": 0,
                    "packages": 0,
                    "submissions": 0,
                    "rank35_k33_calls": 0,
                    "pass_authorizes_targets": False,
                },
                "association_boundary": {
                    "incoming_k6_in_this_source_screen": 0,
                    "augmented_mean_or_weight_recurrence_validated": False,
                    "future_autonomous_reset_gate_still_required": True,
                    "future_native_j_must_be_recomputed_on_augmented_tape": True,
                },
                "authority": {
                    **expected_release_hashes(),
                    **base["store_hashes"],
                    "release_sha256": release_sha,
                    "validated_release_payload": release,
                    "run_id": run_id,
                    "no_retry_lock_sha256": lock_sha256,
                    "phase_consumption_receipts": commit_manifest["phase_consumption_receipts"],
                    "final_bundle": bundle_path.name,
                    "commit_manifest": "COMMIT_MANIFEST.json",
                    "commit_manifest_sha256": commit_manifest_sha256,
                },
            }
            staged_result_path = staged_bundle / result_name
            atomic_json(staged_result_path, result)
            fsync_directory(staged_bundle)
            os.replace(staged_bundle, bundle_path)
            fsync_directory(HERE)
            result_path = bundle_path / result_name
            return {
                "output": str(result_path),
                "bundle": str(bundle_path),
                "decision": decision,
                "all_gates_pass": passed,
            }
    except Exception as exc:
        durable_lock_sha256 = (
            lock_sha256
            if lock_sha256 is not None
            else sha256_file(lock_path)
            if lock_path.is_file()
            else None
        )
        phase_consumption = {
            phase: {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for phase, path in consumption_paths.items()
            if path.is_file()
        }
        failure = {
            "schema": "q24-raw-k33-target-blind-source-action-terminal-failure-v1",
            "status": "FAILED_TARGET_BLIND_K33_SOURCE_ACTION__NO_RETRY",
            "decision": "CLOSE_THIS_RAW_D4_Q7_K33_33_SOURCE_ACTION_SPELLING",
            "all_gates_pass": False,
            "role": "observer_fit",
            "role_rank": rank,
            "terminal_exception": {"type": type(exc).__name__, "message": str(exc)},
            "firewall": {
                "targets_read": 0,
                "scores_read": 0,
                "truth_joins": 0,
                "training_steps": 0,
                "observer_selection_rows_opened": 0,
                "rank35_k33_calls": 0,
                "retry_allowed": False,
            },
            "authority": {
                **expected_release_hashes(),
                "release_sha256": release_sha,
                "validated_release_payload": release,
                "run_id": run_id,
                "no_retry_lock_sha256": durable_lock_sha256,
                "phase_consumption_receipts": phase_consumption,
                "planned_final_bundle": bundle_path.name,
                "final_bundle_visible": bundle_path.exists(),
            },
        }
        if not bundle_path.exists() and not failure_path.exists():
            atomic_json(failure_path, failure)
            fsync_directory(HERE)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rank", type=int, default=201)
    parser.add_argument("--internal-worker-phase", choices=("base", "source"))
    parser.add_argument("--worker-ticket", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--worker-bank-dir", type=Path)
    args = parser.parse_args()

    if args.internal_worker_phase is not None:
        if args.worker_ticket is None or args.worker_output is None or args.worker_bank_dir is None:
            raise SystemExit("internal worker requires ticket, output, and bank directory")
        worker_phase(
            args.internal_worker_phase,
            args.rank,
            args.worker_ticket,
            args.worker_output,
            args.worker_bank_dir,
        )
        return
    if not args.execute:
        print(json.dumps(static_receipt(), indent=2, sort_keys=True))
        return
    print(json.dumps(execute(args.rank), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
