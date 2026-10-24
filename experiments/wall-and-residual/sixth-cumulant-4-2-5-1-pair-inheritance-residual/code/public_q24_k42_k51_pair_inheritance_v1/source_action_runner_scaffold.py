#!/usr/bin/env python3
"""Sealed target-blind K42+K51 source/action transaction runner.

With no independent ``K42_K51_SOURCE_ACTION_RELEASE.json`` the default action
is a standard-library-only static HOLD receipt.  A future hash-pinned release
may authorize exactly one observer-fit row.  The runner then consumes a
durable no-retry claim before any row loader, launches three internally owned
official-environment workers in the order A1/B/A2, and commits one combined
raw-k6 coordinate plus its result and manifest by one directory rename.

This file contains no target, training, autonomous-feedback, estimator-
promotion, or submission path.  No callback or injected worker is accepted.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
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
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
SOURCE = HERE / "k42_k51_instrumented_source.py"
DERIVATION = HERE / "DERIVATION.md"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
VERIFIER = HERE / "verify_static.py"
STATIC_RESULT = HERE / "STATIC_VERIFICATION_V3.json"
FREEZE = HERE / "SOURCE_FREEZE_V3.json"
ACCOUNTING_ERRATUM = HERE / "K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json"
METER_AMENDMENT = HERE / "K42_K51_RUNTIME_METER_GATE_AMENDMENT_V1.json"
INTEGRATED_ACCOUNTING = HERE / "K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json"
INDEPENDENT_HOLD = HERE / "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json"
BASE_SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py"
BASE_RESULT = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/D4_CANDIDATE_RESULT.json"
D29_PRICING = (
    P2
    / "p2r0c1d29_k42_k51_pair_inheritance_static_pricing_v1"
    / "K42_K51_STATIC_DESIGN_PRICING_RECEIPT.json"
)
STORE = (
    P2
    / "learned_residual_observer_design/v4_feature_store_v3"
    / "sealed_target_blind_feature_store_v3"
)
RUNNER = HERE / "source_action_runner_scaffold.py"
RELEASE = HERE / "K42_K51_SOURCE_ACTION_RELEASE.json"

ROLE = "observer_fit"
ALLOWED_ORDERED_RANKS = (201, 643)
PUBLIC_COMPONENTS = (
    "host",
    "native_raw20",
    "native_sigma",
    "native_j_correction",
    "final_public_prediction",
)
COORDINATE_NAME = "combined_raw_k6.npy"
COORDINATE_SHAPE = (15, 1024)
OUTPUT_SHAPE = (16, 1024)
FIXED_SUMMARY_LAYERS = (4, 9, 14)

PHASE2_BUDGET = 2**41
STATIC_DENSE_DIAGNOSTIC_FLOPS = 152_024_121_344
PREDICTED_INTEGRATED_INCREMENTAL_FLOPS = 151_825_424_384
INCREMENTAL_FLOP_CAP = 152_200_000_000
RSS_CAP_BYTES = 2**33
RESIDUAL_CAP_SECONDS = 0.4
WORKER_WALL_SECONDS = 120.0
REMOTE_SCRATCH_CAP_BYTES = 309_092_352
LARGEST_NEW_ARRAY_CAP_BYTES = 285_212_672
REMOTE_SINGLE_ARRAY_CAP_BYTES = 2**32

EXPECTED_SOURCE_CALL_COUNTS = {
    "q24_calls": 15,
    "connected_calls": 15,
    "source_calls": 15,
    "source_einsum_calls": 29,
    "push_calls": 15,
    "action_calls": 15,
    "batched_matmul_calls": 15,
    "logical_matmuls": 30,
    "fill_diagonal_calls": 30,
    "reset_calls": 14,
}
EXPECTED_OPERATION_CALL_DELTAS = {
    "empty": 16,
    "add": 165,
    "concatenate": 30,
    "copy": 15,
    "getitem": 43,
    "multiply": 1020,
    "reshape": 15,
    "stack": 45,
    "subtract": 210,
    "einsum": 29,
    "diagonal": 15,
    "fill_diagonal": 30,
    "transpose": 15,
    "copyto": 30,
    "matmul": 15,
    "sum": 15,
}
SEALED_ROW_ZERO_ACCESS_SENTINELS = {
    "scores": 0,
    "target_values_decoded": 0,
    "target_columns_projected": 0,
    "fits": 0,
    "optimizers_constructed": 0,
}

BASE_SOURCE_SHA256 = "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e"
BASE_RESULT_SHA256 = "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1"
D29_SHA256 = "bb8a55e3224da56139dc79fa2b65acc4cf8320f0991c7d77da5ec0892ad3b6cc"
INDEPENDENT_HOLD_SHA256 = "1d5786631f2923c606d4be53f428f9e6fe88f9418dbf15ecc6abb111a592f7ff"
ACCOUNTING_ERRATUM_SHA256 = "63fbf1da297b292b6340c688da3e20b16a850c296e165e05f111b0847534b8c3"
METER_AMENDMENT_SHA256 = "6ae06a43e0093849437a3eae784a19604e1bd2b451cf13cd9b484586e2f80de2"

OFFICIAL_INTERPRETER = (P2 / "data_gate/.venv/bin/python").absolute()
OFFICIAL_RUNTIME_ENVIRONMENT = {
    "interpreter": str(OFFICIAL_INTERPRETER),
    "flopscope_version": "0.12.0",
    "whestbench_version": "0.16.0",
    "numpy_version": "2.4.6",
}

REQUIRED_FROZEN_FILES = {
    "DERIVATION.md",
    "PREREGISTRATION.json",
    "k42_k51_instrumented_source.py",
    "source_action_runner_scaffold.py",
    "verify_static.py",
    "STATIC_VERIFICATION_V3.json",
    "K42_K51_STATIC_ACCOUNTING_ERRATUM_V1.json",
    "K42_K51_RUNTIME_METER_GATE_AMENDMENT_V1.json",
    "K42_K51_INTEGRATED_ACCOUNTING_RECEIPT_V1.json",
    "INDEPENDENT_STATIC_REVIEW_HOLD_V1.json",
}

REQUIRED_RESULT_GATES = {
    "authority_and_release_hashes_match",
    "role_and_rank_match_singleton_release",
    "target_training_selection_firewall",
    "full_public_A1_B_A2_byte_parity",
    "public_output_shape_finite_repeat",
    "centered_coefficient_quadrature_and_recurrence",
    "partition_203_and_formula_parity",
    "batched_source_vs_scalar_parity",
    "factor15_action_parity",
    "factor6_action_parity",
    "raw_vs_standardized_parity",
    "k24_equals_k42_transpose",
    "k15_equals_k51_transpose",
    "correlated_gaussian_null",
    "exact_two_zero_diagonals_each_source",
    "source_and_action_dtype_exact_float64_before_copy",
    "all_source_sector_action_and_combined_values_finite",
    "k42_plane_norm_positive_all_sources",
    "k51_plane_norm_positive_all_sources",
    "raw_action42_norm_positive_all_actions",
    "raw_action51_norm_positive_all_actions",
    "combined_raw_norm_positive_all_actions",
    "one_shared_endpoint_and_coordinate_mapping",
    "combined_raw_bank_dtype_shape_file_and_content_hash",
    "exact_source_call_inventory",
    "exact_official_operation_call_deltas",
    "base_A1_A2_flop_bills_equal",
    "equal_nonnegative_incremental_flop_deltas_at_most_152200000000",
    "all_total_flops_strictly_below_2pow41",
    "all_process_rss_at_most_8gib",
    "all_official_residual_seconds_strictly_below_0_4",
    "static_remote_scratch_within_cap",
    "static_largest_new_array_within_caps",
    "no_hard_magnitude_rank_or_selection_threshold",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_sealed_row_zero_access_sentinels(row: dict[str, Any]) -> dict[str, int]:
    """Require the sealed row's explicit target/training sentinels before weights."""

    observed: dict[str, int] = {}
    for key, expected in SEALED_ROW_ZERO_ACCESS_SENTINELS.items():
        value = row.get(key)
        if type(value) is not int or value != expected:
            raise RuntimeError(f"sealed row zero-access sentinel changed: {key}")
        observed[key] = value
    return observed


def array_content_sha256(array: Any) -> str:
    import numpy as np

    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.view(np.uint8))
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def exclusive_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, indent=2, sort_keys=True, allow_nan=False)
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
        "c_order": bool(value.flags.c_contiguous),
        "nbytes": int(value.nbytes),
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


def bundle_path(rank: int) -> Path:
    return HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_BUNDLE"


def result_name(rank: int) -> str:
    return f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_RESULT.json"


def lock_path(rank: int) -> Path:
    return HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_RUN.lock"


def failure_path(rank: int) -> Path:
    return HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_FAILURE.json"


def consumption_path(rank: int, phase: str) -> Path:
    return HERE / f"K42_K51_SOURCE_ACTION_RANK{rank:04d}_{phase}_WORKER_CONSUMED.json"


def require_distinct_phase_capabilities(capabilities: dict[str, str]) -> None:
    if (
        set(capabilities) != {"A1", "B", "A2"}
        or any(not isinstance(value, str) or not value for value in capabilities.values())
        or len(set(capabilities.values())) != 3
    ):
        raise RuntimeError("A1/B/A2 phase capabilities are not distinct")


def validate_official_runtime() -> dict[str, str]:
    observed = {
        "interpreter": str(Path(sys.executable).absolute()),
        "flopscope_version": importlib.metadata.version("flopscope"),
        "whestbench_version": importlib.metadata.version("whestbench"),
        "numpy_version": importlib.metadata.version("numpy"),
    }
    if observed != OFFICIAL_RUNTIME_ENVIRONMENT:
        raise RuntimeError(f"official runtime mismatch: {observed}")
    return observed


def _verify_foundational_authorities() -> None:
    expected = {
        BASE_SOURCE: BASE_SOURCE_SHA256,
        BASE_RESULT: BASE_RESULT_SHA256,
        D29_PRICING: D29_SHA256,
        INDEPENDENT_HOLD: INDEPENDENT_HOLD_SHA256,
        ACCOUNTING_ERRATUM: ACCOUNTING_ERRATUM_SHA256,
        METER_AMENDMENT: METER_AMENDMENT_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"foundational authority mismatch: {path.name}")


def _verify_release_freeze(release: dict[str, Any]) -> tuple[dict[str, Any], str]:
    if release.get("freeze_path") != FREEZE.name or not _is_sha256(release.get("freeze_sha256")):
        raise RuntimeError("release does not pin the exact V3 freeze path/hash")
    if not FREEZE.is_file() or sha256_file(FREEZE) != release["freeze_sha256"]:
        raise RuntimeError("release-pinned V3 freeze is missing or hash-mismatched")
    freeze = json.loads(FREEZE.read_text())
    if (
        freeze.get("schema") != "q24-k42-k51-pair-inheritance-static-source-freeze-v3"
        or freeze.get("status")
        != "STATIC_V3_REVIEW_READY_HOLD_NO_RELEASE_NO_ROW_OR_ESTIMATOR_CALL"
    ):
        raise RuntimeError("V3 freeze schema/status mismatch")
    frozen_files = freeze.get("frozen_files")
    if not isinstance(frozen_files, dict) or set(frozen_files) != REQUIRED_FROZEN_FILES:
        raise RuntimeError("V3 freeze does not pin the complete operative file set")
    for relative, digest in frozen_files.items():
        path = HERE / relative
        if not _is_sha256(digest) or not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"V3 frozen file mismatch: {relative}")
    if release.get("frozen_files") != frozen_files:
        raise RuntimeError("release must repeat the complete V3 frozen-file map")
    foundational = freeze.get("foundational_authorities")
    if foundational != {
        "d4_source_sha256": BASE_SOURCE_SHA256,
        "d4_result_sha256": BASE_RESULT_SHA256,
        "d29_pricing_sha256": D29_SHA256,
        "independent_hold_sha256": INDEPENDENT_HOLD_SHA256,
        "accounting_erratum_sha256": ACCOUNTING_ERRATUM_SHA256,
        "meter_gate_amendment_sha256": METER_AMENDMENT_SHA256,
    }:
        raise RuntimeError("V3 freeze foundational authority map mismatch")
    return freeze, sha256_file(FREEZE)


def _validated_embedded_rank201_release_payload(
    result: dict[str, Any], freeze: dict[str, Any], freeze_hash: str
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Validate the rank201 release bytes that survive replacement by rank643."""

    authority = result.get("authority")
    if not isinstance(authority, dict):
        raise RuntimeError("rank201 result lacks embedded authority")
    old_payload = authority.get("validated_release_payload")
    old_payload_sha = authority.get("release_payload_sha256")
    if (
        not isinstance(old_payload, dict)
        or not _is_sha256(old_payload_sha)
        or canonical_json_sha256(old_payload) != old_payload_sha
    ):
        raise RuntimeError("rank201 embedded release payload/hash binding is invalid")
    old_required = {
        "schema": "q24-k42-k51-source-action-singleton-release-v3",
        "status": "RELEASE_TARGET_BLIND_K42_K51_SOURCE_ACTION_ONCE",
        "allowed_role": ROLE,
        "allowed_role_ranks": [201],
        "singleton_row_count": 1,
        "no_retry": True,
        "concurrent_heavy_run_allowed": False,
        "targets_allowed": False,
        "scores_allowed": False,
        "truth_join_allowed": False,
        "training_allowed": False,
        "selection_allowed": False,
        "autonomous_feedback_allowed": False,
        "full_estimator_allowed": False,
        "submission_allowed": False,
        "independent_final_go": True,
        "runtime_environment": OFFICIAL_RUNTIME_ENVIRONMENT,
        "incremental_flop_cap": INCREMENTAL_FLOP_CAP,
        "freeze_path": FREEZE.name,
        "freeze_sha256": freeze_hash,
        "frozen_files": freeze["frozen_files"],
        "prerequisite_rank201_sha256": {},
    }
    if any(old_payload.get(key) != expected for key, expected in old_required.items()):
        raise RuntimeError("rank201 embedded release payload singleton/firewall semantics changed")
    if (
        authority.get("frozen_files") != freeze["frozen_files"]
        or authority.get("runner_sha256") != freeze["frozen_files"][RUNNER.name]
        or authority.get("base_source_sha256") != BASE_SOURCE_SHA256
        or authority.get("row_manifest_sha256") != old_payload.get("row_manifest_sha256")
        or authority.get("weights_sha256") != old_payload.get("weights_sha256")
        or authority.get("sealed_row_zero_access_sentinels")
        != SEALED_ROW_ZERO_ACCESS_SENTINELS
        or authority.get("sealed_row_zero_access_sentinels_sha256")
        != canonical_json_sha256(SEALED_ROW_ZERO_ACCESS_SENTINELS)
        or not _is_sha256(authority.get("release_file_sha256"))
        or not _is_sha256(authority.get("no_retry_lock_sha256"))
        or not isinstance(authority.get("run_id"), str)
        or not authority["run_id"]
    ):
        raise RuntimeError("rank201 result authority does not join frozen row/weight/run bindings")
    return authority, old_payload, old_payload_sha


def require_manifest_binding_subset(
    manifest: dict[str, Any], expected: dict[str, Any]
) -> None:
    bindings = manifest.get("bindings")
    if not isinstance(bindings, dict) or any(
        bindings.get(key) != value for key, value in expected.items()
    ):
        raise RuntimeError("commit manifest authority cross-binding mismatch")


def _validate_rank201_prerequisite(
    release: dict[str, Any], freeze: dict[str, Any], freeze_hash: str
) -> dict[str, Any]:
    prior_bundle = bundle_path(201)
    prior_result = prior_bundle / result_name(201)
    prior_manifest = prior_bundle / "COMMIT_MANIFEST.json"
    pins = release.get("prerequisite_rank201_sha256")
    expected_names = {result_name(201), "COMMIT_MANIFEST.json"}
    if not isinstance(pins, dict) or set(pins) != expected_names:
        raise RuntimeError("rank643 release must pin rank201 result and commit manifest")
    for path in (prior_result, prior_manifest):
        if not path.is_file() or sha256_file(path) != pins[path.name]:
            raise RuntimeError("rank201 prerequisite artifact is missing or hash-mismatched")
    result = json.loads(prior_result.read_text())
    required = {
        "schema": "q24-k42-k51-target-blind-source-action-result-v3",
        "status": "COMPLETE_TARGET_BLIND_K42_K51_SOURCE_ACTION",
        "decision": "PASS_K42_K51_SOURCE_ACTION_RANK201__HOLD_RANK643_PENDING_INDEPENDENT_AUDIT_AND_NEW_RELEASE",
        "all_gates_pass": True,
        "role": ROLE,
        "role_rank": 201,
        "freeze_sha256": freeze_hash,
        "source_sha256": freeze["frozen_files"][SOURCE.name],
        "preregistration_sha256": freeze["frozen_files"][PREREGISTRATION.name],
        "retry_allowed": False,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise RuntimeError(f"rank201 semantic prerequisite mismatch: {key}")
    gates = result.get("gates")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_RESULT_GATES or not all(gates.values()):
        raise RuntimeError("rank201 result lacks the complete conjunctive V3 PASS gate set")
    firewall = result.get("firewall", {})
    if firewall != {
        "targets_read": 0,
        "scores_read": 0,
        "truth_joins": 0,
        "training_steps": 0,
        "selection_steps": 0,
        "autonomous_feedback_runs": 0,
        "full_estimator_candidate_runs": 0,
        "submissions": 0,
        "rank35_calls": 0,
        "pass_authorizes_targets": False,
    }:
        raise RuntimeError("rank201 prerequisite firewall is not zero")
    authority, _old_payload, old_payload_sha = _validated_embedded_rank201_release_payload(
        result, freeze, freeze_hash
    )
    manifest = json.loads(prior_manifest.read_text())
    files = manifest.get("files", {})
    coordinate_path = prior_bundle / COORDINATE_NAME
    coordinate = _coordinate_receipt(coordinate_path)
    expected_binding_subset = {
        "role": ROLE,
        "role_rank": 201,
        "run_id": authority["run_id"],
        "release_file_sha256": authority["release_file_sha256"],
        "release_payload_sha256": old_payload_sha,
        "freeze_sha256": freeze_hash,
        "frozen_files": freeze["frozen_files"],
        "runner_sha256": authority["runner_sha256"],
        "source_sha256": result["source_sha256"],
        "row_manifest_sha256": authority["row_manifest_sha256"],
        "weights_sha256": authority["weights_sha256"],
        "sealed_row_zero_access_sentinels_sha256": authority[
            "sealed_row_zero_access_sentinels_sha256"
        ],
        "no_retry_lock_sha256": authority["no_retry_lock_sha256"],
        "phase_consumption_receipts": authority.get("phase_consumption_receipts"),
        "terminal_status": result["status"],
        "terminal_decision": result["decision"],
        "all_gates_pass": True,
    }
    require_manifest_binding_subset(manifest, expected_binding_subset)
    if (
        manifest.get("status") != "FSYNCED_PRECOMMIT__VISIBLE_ONLY_BY_ATOMIC_DIRECTORY_RENAME"
        or manifest.get("planned_final_bundle") != prior_bundle.name
        or manifest.get("planned_files")
        != [COORDINATE_NAME, result_name(201), "COMMIT_MANIFEST.json"]
        or set(files) != {COORDINATE_NAME, result_name(201)}
        or files[result_name(201)].get("file_sha256") != sha256_file(prior_result)
        or files[COORDINATE_NAME] != coordinate
        or result.get("coordinate_artifact") != coordinate
        or set(path.name for path in prior_bundle.iterdir())
        != {COORDINATE_NAME, result_name(201), "COMMIT_MANIFEST.json"}
    ):
        raise RuntimeError("rank201 atomic bundle semantics are invalid")
    parent_lock = lock_path(201)
    phase_receipts = authority.get("phase_consumption_receipts")
    if (
        not parent_lock.is_file()
        or sha256_file(parent_lock) != authority["no_retry_lock_sha256"]
        or not isinstance(phase_receipts, dict)
        or set(phase_receipts) != {"A1", "B", "A2"}
    ):
        raise RuntimeError("rank201 durable claim/phase receipt chain is incomplete")
    lock = json.loads(parent_lock.read_text())
    if (
        lock.get("run_id") != authority["run_id"]
        or lock.get("role") != ROLE
        or int(lock.get("role_rank", -1)) != 201
        or lock.get("release_sha256") != authority["release_file_sha256"]
        or lock.get("release_payload_sha256") != old_payload_sha
        or lock.get("row_manifest_sha256") != authority["row_manifest_sha256"]
        or lock.get("weights_sha256") != authority["weights_sha256"]
    ):
        raise RuntimeError("rank201 durable parent claim does not join embedded release")
    for phase, receipt in phase_receipts.items():
        path = HERE / receipt.get("path", "")
        if (
            not path.is_file()
            or sha256_file(path) != receipt.get("sha256")
            or path.name != consumption_path(201, phase).name
        ):
            raise RuntimeError("rank201 phase consumption receipt chain is invalid")
        payload = json.loads(path.read_text())
        if (
            payload.get("phase") != phase
            or payload.get("run_id") != authority["run_id"]
            or payload.get("lock_sha256") != authority["no_retry_lock_sha256"]
            or payload.get("release_sha256") != authority["release_file_sha256"]
            or payload.get("runner_sha256") != authority["runner_sha256"]
            or payload.get("phase_capability_sha256")
            != lock.get("phase_capability_sha256", {}).get(phase)
        ):
            raise RuntimeError("rank201 phase receipt does not join parent capability")
    return result


def require_release(rank: int) -> dict[str, Any]:
    if rank not in ALLOWED_ORDERED_RANKS:
        raise RuntimeError("rank must be 201 or 643; rank35 and every other rank are forbidden")
    if not RELEASE.is_file():
        raise RuntimeError("K42+K51 release is absent; numerical execution is not authorized")
    _verify_foundational_authorities()
    release = json.loads(RELEASE.read_text())
    required = {
        "schema": "q24-k42-k51-source-action-singleton-release-v3",
        "status": "RELEASE_TARGET_BLIND_K42_K51_SOURCE_ACTION_ONCE",
        "allowed_role": ROLE,
        "allowed_role_ranks": [rank],
        "singleton_row_count": 1,
        "no_retry": True,
        "concurrent_heavy_run_allowed": False,
        "targets_allowed": False,
        "scores_allowed": False,
        "truth_join_allowed": False,
        "training_allowed": False,
        "selection_allowed": False,
        "autonomous_feedback_allowed": False,
        "full_estimator_allowed": False,
        "submission_allowed": False,
        "independent_final_go": True,
        "runtime_environment": OFFICIAL_RUNTIME_ENVIRONMENT,
        "incremental_flop_cap": INCREMENTAL_FLOP_CAP,
    }
    for key, expected in required.items():
        if release.get(key) != expected:
            raise RuntimeError(f"release field mismatch: {key}")
    for key in ("row_manifest_sha256", "weights_sha256"):
        if not _is_sha256(release.get(key)):
            raise RuntimeError(f"release lacks singleton row binding: {key}")
    review = release.get("review")
    if not isinstance(review, dict) or not review.get("reviewer_identity") or not review.get("summary"):
        raise RuntimeError("release lacks independent reviewer identity and summary")
    freeze, freeze_hash = _verify_release_freeze(release)
    if rank == 201:
        if release.get("prerequisite_rank201_sha256") != {}:
            raise RuntimeError("rank201 release may not carry a numerical prerequisite")
        prior = None
    else:
        prior = _validate_rank201_prerequisite(release, freeze, freeze_hash)
    return {
        "release": release,
        "release_file_sha256": sha256_file(RELEASE),
        "release_payload_sha256": canonical_json_sha256(release),
        "freeze": freeze,
        "freeze_sha256": freeze_hash,
        "prior_rank201_result": prior,
    }


def load_sealed_row(
    rank: int, release: dict[str, Any]
) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import numpy as np

    manifest_path = STORE / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entry = manifest["roles"][ROLE]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise RuntimeError("sealed-store role-rank mismatch")
    row_path = STORE / entry["path"]
    if sha256_file(row_path) != entry["sha256"]:
        raise RuntimeError("sealed row-manifest hash mismatch")
    if entry["sha256"] != release["row_manifest_sha256"]:
        raise RuntimeError("sealed row does not match the singleton release")
    row = json.loads(row_path.read_text())
    if row.get("role") != ROLE or int(row.get("role_rank", -1)) != rank:
        raise RuntimeError("sealed row role/rank mismatch")
    if set(row.get("decoded_columns", ())) != {"mlp_id", "mlp_name", "mlp_seed", "weights"}:
        raise RuntimeError("sealed row decoded-column firewall changed")
    zero_access_sentinels = validate_sealed_row_zero_access_sentinels(row)
    zero_access_sha256 = canonical_json_sha256(zero_access_sentinels)
    weights_path = row_path.parent / "weights.npy"
    weights_hash = row["files"]["weights.npy"]["sha256"]
    if weights_hash != release["weights_sha256"] or sha256_file(weights_path) != weights_hash:
        raise RuntimeError("sealed weights do not match the singleton release")
    weights = np.load(weights_path, mmap_mode="r", allow_pickle=False)
    if weights.shape != (16, 1024, 1024) or weights.dtype != np.dtype("<f4"):
        raise RuntimeError("sealed weights shape/dtype changed")
    return row, weights, {
        "store_manifest_sha256": sha256_file(manifest_path),
        "row_manifest_sha256": entry["sha256"],
        "weights_sha256": weights_hash,
        "sealed_row_zero_access_sentinels": zero_access_sentinels,
        "sealed_row_zero_access_sentinels_sha256": zero_access_sha256,
    }


def make_mlp(weights: Any, row: dict[str, Any], rank: int) -> Any:
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed

    mlp = MLP(
        width=1024,
        depth=16,
        weights=[weights[index] for index in range(16)],
        seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
        name=f"k42-k51-target-blind-observer-fit-{rank:04d}",
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
            raise RuntimeError("frozen D4 public internal tape call count/shape changed")
        state["tape"] = answer
        return answer

    def capture_transport(*args: Any, **kwargs: Any) -> Any:
        answer = original_transport(*args, **kwargs)
        if state["correction"] is not None:
            raise RuntimeError("frozen D4 native-J correction ran more than once")
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
        raise RuntimeError("Estimator.predict->_predict_corrected did not expose tape/J")
    host, native_raw, native_sigma = state["tape"]
    return host, native_raw, native_sigma, state["correction"], final


def checked_local_array(remote: Any, label: str) -> tuple[Any, str]:
    import numpy as np

    observed = np.asarray(remote)
    observed_dtype = observed.dtype.str
    if observed.dtype != np.dtype("<f8"):
        raise RuntimeError(f"{label} dtype changed before copy: {observed_dtype}")
    # np.asarray is the authoritative fetch.  Preserve its exact dtype and
    # make only the contiguity copy, if one is actually required.
    return np.ascontiguousarray(observed), observed_dtype


def vector_metrics(value: Any) -> dict[str, Any]:
    import numpy as np

    array = np.asarray(value)
    if array.dtype != np.dtype("<f8"):
        raise RuntimeError(f"vector metric dtype changed: {array.dtype.str}")
    count = int(array.size)
    return {
        "l2_norm": float(np.linalg.norm(array)),
        "rms": float(np.sqrt(np.mean(array * array, dtype=np.float64))),
        "max_abs": float(np.max(np.abs(array))),
        "positive_fraction": float(np.count_nonzero(array > 0.0) / count),
        "negative_fraction": float(np.count_nonzero(array < 0.0) / count),
        "zero_fraction": float(np.count_nonzero(array == 0.0) / count),
        "finite": bool(np.isfinite(array).all()),
        "observed_dtype_before_copy": array.dtype.str,
        "content_sha256": array_content_sha256(array),
    }


def plane_metrics(value: Any) -> dict[str, Any]:
    import numpy as np

    array, observed_dtype = checked_local_array(value, "source plane")
    diagonal = np.diag(array)
    metrics = {
        "frobenius_norm": float(np.linalg.norm(array)),
        "rms": float(np.sqrt(np.mean(array * array, dtype=np.float64))),
        "max_abs": float(np.max(np.abs(array))),
        "diagonal_max_abs": float(np.max(np.abs(diagonal))),
        "finite": bool(np.isfinite(array).all()),
        "observed_dtype_before_copy": observed_dtype,
        "positive_fraction": float(np.count_nonzero(array > 0.0) / array.size),
        "negative_fraction": float(np.count_nonzero(array < 0.0) / array.size),
    }
    del array, diagonal
    return metrics


class StreamingPairSink:
    """Exec-cold, internally owned sink; retain only scalars and one sum bank."""

    def __init__(self) -> None:
        self.sources: dict[int, dict[str, Any]] = {}
        self.actions: dict[int, dict[str, Any]] = {}
        self.combined: dict[int, Any] = {}

    def capture_source(self, layer: int, k42: Any, k51: Any) -> None:
        if layer in self.sources:
            raise RuntimeError(f"duplicate source capture at h{layer}")
        self.sources[layer] = {
            "k42": plane_metrics(k42),
            "k51": plane_metrics(k51),
            "fixed_summary_layer": layer in FIXED_SUMMARY_LAYERS,
        }

    def capture_action(self, layer: int, raw_by_sector: Any, raw_total: Any) -> None:
        import numpy as np

        if layer in self.actions or layer in self.combined:
            raise RuntimeError(f"duplicate action capture at h{layer}")
        sectors, sector_dtype = checked_local_array(raw_by_sector, "sector actions")
        combined, combined_dtype = checked_local_array(raw_total, "combined raw-k6")
        if sectors.shape != (2, 1024) or combined.shape != (1024,):
            raise RuntimeError("source action shape changed")
        exact_sum = sectors[0] + sectors[1]
        if not np.array_equal(exact_sum.view(np.uint8), combined.view(np.uint8)):
            raise RuntimeError("combined raw-k6 is not the exact two-sector sum")
        self.actions[layer] = {
            "raw_action42": vector_metrics(sectors[0]),
            "raw_action51": vector_metrics(sectors[1]),
            "combined_raw_k6": vector_metrics(combined),
            "sector_cosine": float(
                np.vdot(sectors[0], sectors[1]).real
                / max(np.linalg.norm(sectors[0]) * np.linalg.norm(sectors[1]), 1.0e-300)
            ),
            "exact_sector_sum": True,
            "sector_observed_dtype_before_copy": sector_dtype,
            "combined_observed_dtype_before_copy": combined_dtype,
        }
        self.combined[layer] = combined
        del sectors, exact_sum

    def finalize(self) -> tuple[list[dict[str, Any]], Any]:
        import numpy as np

        expected = set(range(1, 16))
        if set(self.sources) != expected or set(self.actions) != expected or set(self.combined) != expected:
            raise RuntimeError("source/action layer set is not exactly h1:h15")
        bank = np.empty(COORDINATE_SHAPE, dtype=np.dtype("<f8"))
        rows: list[dict[str, Any]] = []
        for slot, layer in enumerate(range(1, 16)):
            bank[slot] = self.combined[layer]
            rows.append({
                "slot": slot,
                "source_layer_one_based": layer,
                "outgoing_weight_index_zero_based": layer,
                "outgoing_weight_literal": f"mlp.weights[{layer}]",
                "consumer_layer_one_based": layer + 1,
                "source": self.sources[layer],
                "action": self.actions[layer],
            })
        self.sources.clear()
        self.actions.clear()
        self.combined.clear()
        return rows, np.ascontiguousarray(bank)


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


def runtime_science_proof() -> dict[str, Any]:
    tag = f"k42_k51_runtime_proof_{os.getpid()}"
    proof = load_module(tag, VERIFIER)
    try:
        coefficient = proof.coefficient_and_centering_proof()
        partition = proof.partition_proof()
        contraction = proof.source_contraction_proof()
        gaussian = proof.connected_gaussian_null_proof()
        action = proof.action_and_orientation_proof()
        endpoint = proof.endpoint_proof()
    finally:
        sys.modules.pop(tag, None)
    gates = {
        "centered_coefficient_quadrature_and_recurrence": (
            coefficient["coefficient_quadrature_max_relative_error"] <= 5.0e-11
            and coefficient["centered_quadrature_max_relative_error"] <= 5.0e-11
            and coefficient["raw5_recurrence_relative_error"] <= 5.0e-13
            and coefficient["c4_tail_recurrence_relative_error"] <= 5.0e-13
            and coefficient["c5_tail_recurrence_relative_error"] <= 5.0e-13
        ),
        "partition_203_and_formula_parity": (
            partition["all_set_partitions"] == 203
            and max(partition["k42_formula_relative_error"], partition["k51_formula_relative_error"]) <= 1.0e-12
        ),
        "batched_source_vs_scalar_parity": max(contraction.values()) <= 1.0e-12,
        "factor15_action_parity": action["factor15_brute_relative_error"] <= 1.0e-12,
        "factor6_action_parity": action["factor6_brute_relative_error"] <= 1.0e-12,
        "raw_vs_standardized_parity": max(
            action["k42_raw_standardized_relative_error"],
            action["k51_raw_standardized_relative_error"],
        ) <= 1.0e-12,
        "k24_equals_k42_transpose": action["k24_transpose_action_relative_error"] <= 1.0e-12,
        "k15_equals_k51_transpose": action["k15_transpose_action_relative_error"] <= 1.0e-12,
        "correlated_gaussian_null": max(gaussian.values()) <= 1.0e-12,
        "one_shared_endpoint_and_coordinate_mapping": max(endpoint.values()) <= 1.0e-12,
    }
    if not all(gates.values()):
        raise RuntimeError(f"runner-owned runtime science proof failed: {gates}")
    return {
        "gates": gates,
        "coefficient_and_centering": coefficient,
        "partition": partition,
        "batched_source": contraction,
        "correlated_gaussian_null": gaussian,
        "actions_orientation_scale": action,
        "shared_endpoint": endpoint,
    }


def validate_worker_ticket(ticket_path: Path, phase: str, rank: int) -> dict[str, Any]:
    nonce = os.environ.get("K42_K51_INTERNAL_WORKER_NONCE")
    if not nonce:
        raise RuntimeError("internal worker nonce is absent")
    ticket = json.loads(ticket_path.read_text())
    parent_lock_path = lock_path(rank)
    if not parent_lock_path.is_file():
        raise RuntimeError("internal worker has no durable parent no-retry claim")
    parent_lock_sha = sha256_file(parent_lock_path)
    parent_lock = json.loads(parent_lock_path.read_text())
    capability = ticket.get("phase_capability")
    capability_sha = (
        hashlib.sha256(capability.encode("ascii")).hexdigest()
        if isinstance(capability, str)
        else None
    )
    expected_source = BASE_SOURCE_SHA256 if phase in ("A1", "A2") else sha256_file(SOURCE)
    if (
        ticket.get("schema") != "k42-k51-internal-worker-ticket-v3"
        or ticket.get("phase") != phase
        or int(ticket.get("rank", -1)) != rank
        or ticket.get("nonce") != nonce
        or ticket.get("run_id") != parent_lock.get("run_id")
        or ticket.get("runner_sha256") != sha256_file(RUNNER)
        or ticket.get("source_sha256") != expected_source
        or ticket.get("release_sha256") != sha256_file(RELEASE)
        or ticket.get("lock_sha256") != parent_lock_sha
        or parent_lock.get("schema") != "q24-k42-k51-no-retry-lock-v3"
        or parent_lock.get("status") != "CONSUMED_SINGLETON_AUTHORITY_BEFORE_ROW_OPEN"
        or int(parent_lock.get("role_rank", -1)) != rank
        or parent_lock.get("role") != ROLE
        or parent_lock.get("runner_sha256") != sha256_file(RUNNER)
        or parent_lock.get("release_sha256") != sha256_file(RELEASE)
        or capability_sha != parent_lock.get("phase_capability_sha256", {}).get(phase)
    ):
        raise RuntimeError("internal worker ticket/claim/capability binding is invalid")
    receipt_path = consumption_path(rank, phase)
    if parent_lock.get("phase_consumption_receipts", {}).get(phase) != receipt_path.name:
        raise RuntimeError("parent claim has an invalid worker-consumption path")
    consumed_ticket = ticket_path.with_suffix(ticket_path.suffix + ".consumed")
    if consumed_ticket.exists() or receipt_path.exists():
        raise RuntimeError("worker phase capability was already consumed")
    ticket_sha = sha256_file(ticket_path)
    exclusive_json(
        receipt_path,
        {
            "schema": "q24-k42-k51-worker-consumption-v3",
            "status": "PHASE_CAPABILITY_IRREVOCABLY_CONSUMED_BEFORE_ROW_OPEN",
            "role": ROLE,
            "role_rank": rank,
            "phase": phase,
            "run_id": parent_lock["run_id"],
            "lock_sha256": parent_lock_sha,
            "ticket_sha256": ticket_sha,
            "phase_capability_sha256": capability_sha,
            "release_sha256": sha256_file(RELEASE),
            "runner_sha256": sha256_file(RUNNER),
            "source_sha256": expected_source,
        },
    )
    fsync_directory(HERE)
    os.replace(ticket_path, consumed_ticket)
    fsync_directory(ticket_path.parent)
    return {
        "run_id": parent_lock["run_id"],
        "lock_sha256": parent_lock_sha,
        "ticket_sha256": ticket_sha,
        "consumption_receipt": receipt_path.name,
        "consumption_receipt_sha256": sha256_file(receipt_path),
        "phase_capability_sha256": capability_sha,
        "process_nonce": nonce,
    }


def worker_phase(
    phase: str,
    rank: int,
    ticket_path: Path,
    output_path: Path,
    bank_dir: Path,
) -> None:
    """Private subprocess entry; capability is consumed before current-row open."""

    authority = require_release(rank)
    runtime = validate_official_runtime()
    launch_authority = validate_worker_ticket(ticket_path, phase, rank)
    # The irreversible capability receipt above must precede this exact loader.
    row, weights, store_hashes = load_sealed_row(rank, authority["release"])

    import flopscope as flops
    import numpy as np
    from threadpoolctl import threadpool_limits

    if output_path.exists():
        raise RuntimeError("worker output already exists")
    if sha256_file(BASE_SOURCE) != BASE_SOURCE_SHA256:
        raise RuntimeError("frozen D4 source changed before worker import")
    flops.configure(symmetry_warnings=False)
    mlp = make_mlp(weights, row, rank)
    started = time.perf_counter()
    remote_values: tuple[Any, ...] | None = None
    sink: StreamingPairSink | None = None
    source_counts: dict[str, int] | None = None
    layer_rows: list[dict[str, Any]] | None = None
    coordinate_artifact: dict[str, Any] | None = None
    science_proof: dict[str, Any] | None = None
    module_tag = f"k42_k51_worker_{phase}_{rank}_{os.getpid()}"
    try:
        if phase in ("A1", "A2"):
            module = load_module(module_tag, BASE_SOURCE)
            branch_source_sha = BASE_SOURCE_SHA256
        elif phase == "B":
            module = load_module(module_tag, SOURCE)
            branch_source_sha = sha256_file(SOURCE)
            sink = StreamingPairSink()
            # Tiny target-free proofs run outside the literal public-call meter.
            science_proof = runtime_science_proof()
        else:
            raise RuntimeError("worker phase must be A1, B, or A2")

        with threadpool_limits(limits=2):
            with flops.BudgetContext(
                flop_budget=PHASE2_BUDGET,
                wall_time_limit_s=WORKER_WALL_SECONDS,
                quiet=True,
                namespace=f"k42_k51_source_action_{phase}_rank{rank:04d}",
            ) as meter:
                if phase in ("A1", "A2"):
                    remote_values = public_with_components(module, mlp)
                else:
                    remote_values, source_counts = module.predict_with_pair_tape(mlp, sink)
        meter_summary = meter.summary_dict(by_namespace=False)
        public_components = component_summaries(remote_values)
        if phase == "B":
            if sink is None:
                raise RuntimeError("instrumented sink was not initialized")
            layer_rows, combined_bank = sink.finalize()
            bank_dir.mkdir(exist_ok=True)
            coordinate_artifact = atomic_npy(bank_dir / COORDINATE_NAME, combined_bank)
            del combined_bank
        payload = {
            "schema": "q24-k42-k51-internal-worker-result-v3",
            "phase": phase,
            "role": ROLE,
            "role_rank": rank,
            "mlp_id": int(row["mlp_id"]),
            "source_sha256": branch_source_sha,
            "runner_sha256": sha256_file(RUNNER),
            "public_components": public_components,
            "literal_meter": meter_summary,
            "outer_wall_seconds": float(time.perf_counter() - started),
            "process_peak_rss_bytes": peak_rss_bytes(),
            "source_call_counts": source_counts,
            "runtime_science_proof": science_proof,
            "layers": layer_rows,
            "coordinate_artifact": coordinate_artifact,
            "store_hashes": store_hashes,
            "row_binding": {
                "role": ROLE,
                "role_rank": rank,
                "row_manifest_sha256": store_hashes["row_manifest_sha256"],
                "weights_sha256": store_hashes["weights_sha256"],
                "singleton_row_count": 1,
                "sealed_row_zero_access_sentinels": store_hashes[
                    "sealed_row_zero_access_sentinels"
                ],
                "sealed_row_zero_access_sentinels_sha256": store_hashes[
                    "sealed_row_zero_access_sentinels_sha256"
                ],
            },
            "release_file_sha256": authority["release_file_sha256"],
            "release_payload_sha256": authority["release_payload_sha256"],
            "validated_release_payload": authority["release"],
            "launch_authority": launch_authority,
            "worker_identity": {
                "process_id": os.getpid(),
                "process_nonce": launch_authority["process_nonce"],
                "phase": phase,
                "official_runtime": runtime,
            },
            "firewall": {
                "targets_read": 0,
                "scores_read": 0,
                "truth_joins": 0,
                "training_steps": 0,
                "selection_steps": 0,
                "autonomous_feedback_runs": 0,
                "full_estimator_candidate_runs": 0,
                "submissions": 0,
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
    parent_lock_sha: str,
) -> dict[str, Any]:
    nonce = secrets.token_hex(32)
    phase_dir = directory / phase
    phase_dir.mkdir()
    ticket_path = phase_dir / "worker.ticket.json"
    output_path = phase_dir / "worker.result.json"
    bank_dir = phase_dir / "coordinate"
    expected_source = BASE_SOURCE_SHA256 if phase in ("A1", "A2") else sha256_file(SOURCE)
    ticket = {
        "schema": "k42-k51-internal-worker-ticket-v3",
        "phase": phase,
        "rank": rank,
        "nonce": nonce,
        "run_id": run_id,
        "phase_capability": phase_capability,
        "lock_sha256": parent_lock_sha,
        "release_sha256": release_sha,
        "runner_sha256": sha256_file(RUNNER),
        "source_sha256": expected_source,
    }
    atomic_json(ticket_path, ticket)
    environment = os.environ.copy()
    environment["K42_K51_INTERNAL_WORKER_NONCE"] = nonce
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        str(OFFICIAL_INTERPRETER),
        "-B",
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
    process = subprocess.Popen(
        command,
        cwd=str(HERE),
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=360.0)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        raise RuntimeError(f"{phase} worker timed out: {stderr[-4000:]}")
    if process.returncode != 0:
        raise RuntimeError(f"{phase} worker failed rc={process.returncode}: {stderr[-4000:]}")
    if stdout.strip():
        raise RuntimeError(f"{phase} internal worker emitted unauthorized stdout")
    if not output_path.is_file():
        raise RuntimeError(f"{phase} worker returned without an immutable phase receipt")
    payload = json.loads(output_path.read_text())
    identity = payload.get("worker_identity", {})
    launch = payload.get("launch_authority", {})
    if (
        identity.get("process_id") != process.pid
        or identity.get("process_nonce") != nonce
        or identity.get("phase") != phase
        or identity.get("official_runtime") != OFFICIAL_RUNTIME_ENVIRONMENT
        or launch.get("process_nonce") != nonce
    ):
        raise RuntimeError(f"{phase} worker identity is not parent-observed/bound")
    return payload


def operation_call_deltas(
    instrumented_meter: dict[str, Any], base_meter: dict[str, Any]
) -> dict[str, int]:
    instrumented = instrumented_meter.get("operations", {})
    base = base_meter.get("operations", {})
    names = set(instrumented) | set(base)
    deltas: Counter[str] = Counter()
    for name in names:
        delta = int(instrumented.get(name, {}).get("calls", 0)) - int(
            base.get(name, {}).get("calls", 0)
        )
        if delta:
            # FlopScope 0.12 reports the public ``diagonal`` call as ``diag``.
            canonical = "diagonal" if name == "diag" else name
            deltas[canonical] += delta
    return dict(sorted(deltas.items()))


def component_parity(
    a1: dict[str, Any], b: dict[str, Any], a2: dict[str, Any]
) -> dict[str, bool]:
    return {
        name: (
            a1[name]["dtype"] == b[name]["dtype"] == a2[name]["dtype"]
            and a1[name]["shape"] == b[name]["shape"] == a2[name]["shape"]
            and a1[name]["content_sha256"]
            == b[name]["content_sha256"]
            == a2[name]["content_sha256"]
        )
        for name in PUBLIC_COMPONENTS
    }


def _coordinate_receipt(path: Path) -> dict[str, Any]:
    import numpy as np

    value = np.load(path, allow_pickle=False)
    if (
        value.dtype != np.dtype("<f8")
        or value.shape != COORDINATE_SHAPE
        or not value.flags.c_contiguous
        or not bool(np.isfinite(value).all())
    ):
        raise RuntimeError("combined raw-k6 bank shape/dtype/order/finite gate failed")
    return {
        "path": COORDINATE_NAME,
        "file_sha256": sha256_file(path),
        "content_sha256": array_content_sha256(value),
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "c_order": bool(value.flags.c_contiguous),
        "nbytes": int(value.nbytes),
        "source_order": "h1..h15",
        "consumer_order": "h2..h16",
        "contains_sector_banks": False,
        "contains_full_planes": False,
        "contains_targets": False,
    }


def commit_atomic_bundle(
    staging_root: Path,
    final_bundle: Path,
    terminal_result_name: str,
    coordinate_source: Path,
    expected_coordinate: dict[str, Any],
    result: dict[str, Any],
    manifest_bindings: dict[str, Any],
    *,
    _selftest_fail_after_fsync_before_visibility: bool = False,
) -> dict[str, Any]:
    """Commit exactly coordinate/result/manifest, with rename as visibility point."""

    if final_bundle.exists():
        raise RuntimeError("final bundle already exists; overwrite/retry forbidden")
    staged_bundle = staging_root / "ATOMIC_BUNDLE_STAGE"
    staged_bundle.mkdir()
    committed = False
    try:
        staged_coordinate = staged_bundle / COORDINATE_NAME
        os.replace(coordinate_source, staged_coordinate)
        coordinate = _coordinate_receipt(staged_coordinate)
        comparable_keys = ("file_sha256", "content_sha256", "dtype", "shape", "c_order", "nbytes")
        if any(coordinate[key] != expected_coordinate.get(key) for key in comparable_keys):
            raise RuntimeError("staged coordinate does not match worker receipt")
        if result.get("coordinate_artifact") != coordinate:
            raise RuntimeError("terminal result does not bind the exact staged coordinate")

        staged_result = staged_bundle / terminal_result_name
        atomic_json(staged_result, result)
        result_sha = sha256_file(staged_result)
        manifest = {
            "schema": "q24-k42-k51-atomic-bundle-commit-manifest-v3",
            "status": "FSYNCED_PRECOMMIT__VISIBLE_ONLY_BY_ATOMIC_DIRECTORY_RENAME",
            "planned_final_bundle": final_bundle.name,
            "planned_files": [COORDINATE_NAME, terminal_result_name, "COMMIT_MANIFEST.json"],
            "files": {
                COORDINATE_NAME: coordinate,
                terminal_result_name: {
                    "path": terminal_result_name,
                    "file_sha256": result_sha,
                    "schema": result["schema"],
                    "status": result["status"],
                    "decision": result["decision"],
                    "all_gates_pass": result["all_gates_pass"],
                },
            },
            "bindings": manifest_bindings,
            "full_planes_persisted": False,
            "sector_action_banks_persisted": False,
            "derived_endpoint_banks_persisted": False,
        }
        manifest_path = staged_bundle / "COMMIT_MANIFEST.json"
        atomic_json(manifest_path, manifest)
        if set(path.name for path in staged_bundle.iterdir()) != {
            COORDINATE_NAME,
            terminal_result_name,
            "COMMIT_MANIFEST.json",
        }:
            raise RuntimeError("staged atomic bundle contains an unauthorized file")
        if sha256_file(staged_result) != manifest["files"][terminal_result_name]["file_sha256"]:
            raise RuntimeError("result changed after commit-manifest binding")
        if sha256_file(staged_coordinate) != manifest["files"][COORDINATE_NAME]["file_sha256"]:
            raise RuntimeError("coordinate changed after commit-manifest binding")
        fsync_directory(staged_bundle)
        if _selftest_fail_after_fsync_before_visibility:
            raise RuntimeError("mocked failure after complete stage fsync before visibility rename")
        os.replace(staged_bundle, final_bundle)
        committed = True
        fsync_directory(final_bundle.parent)
        return {
            "bundle": str(final_bundle),
            "result": str(final_bundle / terminal_result_name),
            "commit_manifest": str(final_bundle / "COMMIT_MANIFEST.json"),
            "commit_manifest_sha256": sha256_file(final_bundle / "COMMIT_MANIFEST.json"),
            "coordinate_artifact": coordinate,
        }
    except Exception:
        if not committed and staged_bundle.exists():
            shutil.rmtree(staged_bundle)
        raise


def _runtime_artifact_paths(rank: int) -> tuple[Path, ...]:
    return (
        bundle_path(rank),
        failure_path(rank),
        lock_path(rank),
        *(consumption_path(rank, phase) for phase in ("A1", "B", "A2")),
    )


def persist_failure_if_lock_owned(
    *,
    lock_acquired_by_this_process: bool,
    final_bundle: Path,
    failure: Path,
    payload: dict[str, Any],
) -> bool:
    """A claim-race loser is read-only and cannot publish a failure receipt."""

    if (
        not lock_acquired_by_this_process
        or final_bundle.exists()
        or failure.exists()
    ):
        return False
    atomic_json(failure, payload)
    fsync_directory(failure.parent)
    return True


def execute(rank: int) -> dict[str, Any]:
    authority = require_release(rank)
    validate_official_runtime()
    final_bundle = bundle_path(rank)
    terminal_result_name = result_name(rank)
    failure = failure_path(rank)
    parent_lock_path = lock_path(rank)
    phase_receipts = {
        phase: consumption_path(rank, phase) for phase in ("A1", "B", "A2")
    }
    for path in _runtime_artifact_paths(rank):
        if path.exists():
            raise RuntimeError(f"refusing retry/overwrite because {path.name} exists")

    release_sha = authority["release_file_sha256"]
    run_id = secrets.token_hex(32)
    phase_capabilities = {phase: secrets.token_hex(32) for phase in ("A1", "B", "A2")}
    require_distinct_phase_capabilities(phase_capabilities)
    lock_payload = {
        "schema": "q24-k42-k51-no-retry-lock-v3",
        "status": "CONSUMED_SINGLETON_AUTHORITY_BEFORE_ROW_OPEN",
        "role": ROLE,
        "role_rank": rank,
        "run_id": run_id,
        "release_sha256": release_sha,
        "release_payload_sha256": authority["release_payload_sha256"],
        "freeze_sha256": authority["freeze_sha256"],
        "runner_sha256": sha256_file(RUNNER),
        "source_sha256": sha256_file(SOURCE),
        "row_manifest_sha256": authority["release"]["row_manifest_sha256"],
        "weights_sha256": authority["release"]["weights_sha256"],
        "phase_capability_sha256": {
            phase: hashlib.sha256(capability.encode("ascii")).hexdigest()
            for phase, capability in phase_capabilities.items()
        },
        "phase_consumption_receipts": {
            phase: path.name for phase, path in phase_receipts.items()
        },
        "planned_worker_order": ["A1", "B", "A2"],
        "planned_final_bundle": final_bundle.name,
        "planned_result": terminal_result_name,
        "retry_allowed": False,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    parent_lock_sha: str | None = None
    lock_acquired_by_this_process = False
    try:
        # This exclusive durable claim is the transaction's first mutable act.
        exclusive_json(parent_lock_path, lock_payload)
        lock_acquired_by_this_process = True
        fsync_directory(HERE)
        parent_lock_sha = sha256_file(parent_lock_path)
        with tempfile.TemporaryDirectory(
            prefix=f".k42_k51_rank{rank:04d}_{run_id[:12]}_", dir=HERE
        ) as temporary:
            directory = Path(temporary)
            a1 = run_subprocess_phase(
                "A1", rank, directory, release_sha, run_id,
                phase_capabilities["A1"], parent_lock_sha,
            )
            a1_meter = a1["literal_meter"]
            a1_preflight = {
                "total_flops_below_budget": int(a1_meter["flops_used"]) < PHASE2_BUDGET,
                "residual_below_cap": float(a1_meter["residual_wall_time_s"]) < RESIDUAL_CAP_SECONDS,
                "rss_below_cap": int(a1["process_peak_rss_bytes"]) <= RSS_CAP_BYTES,
                "all_public_components_finite": all(
                    item["finite"] for item in a1["public_components"].values()
                ),
            }
            if not all(a1_preflight.values()):
                raise RuntimeError(f"A1 preflight failed before B launch: {a1_preflight}")

            b = run_subprocess_phase(
                "B", rank, directory, release_sha, run_id,
                phase_capabilities["B"], parent_lock_sha,
            )
            a2 = run_subprocess_phase(
                "A2", rank, directory, release_sha, run_id,
                phase_capabilities["A2"], parent_lock_sha,
            )

            workers = {"A1": a1, "B": b, "A2": a2}
            for phase, payload in workers.items():
                if (
                    payload.get("schema") != "q24-k42-k51-internal-worker-result-v3"
                    or payload.get("phase") != phase
                    or payload.get("role") != ROLE
                    or int(payload.get("role_rank", -1)) != rank
                    or payload.get("runner_sha256") != sha256_file(RUNNER)
                    or payload.get("release_file_sha256") != release_sha
                    or payload.get("release_payload_sha256") != authority["release_payload_sha256"]
                    or payload.get("validated_release_payload") != authority["release"]
                ):
                    raise RuntimeError(f"{phase} worker authority payload mismatch")
                launch = payload.get("launch_authority", {})
                receipt = phase_receipts[phase]
                if (
                    launch.get("run_id") != run_id
                    or launch.get("lock_sha256") != parent_lock_sha
                    or launch.get("consumption_receipt") != receipt.name
                    or not receipt.is_file()
                    or launch.get("consumption_receipt_sha256") != sha256_file(receipt)
                ):
                    raise RuntimeError(f"{phase} worker is not bound to its one-use capability")

            if len({payload["worker_identity"]["process_id"] for payload in workers.values()}) != 3:
                raise RuntimeError("A1/B/A2 process IDs are not distinct")
            if len({payload["worker_identity"]["process_nonce"] for payload in workers.values()}) != 3:
                raise RuntimeError("A1/B/A2 process nonces are not distinct")
            if len({payload["mlp_id"] for payload in workers.values()}) != 1:
                raise RuntimeError("A1/B/A2 did not use the same singleton MLP")
            if len({canonical_json_sha256(payload["store_hashes"]) for payload in workers.values()}) != 1:
                raise RuntimeError("A1/B/A2 sealed-store hashes differ")
            expected_row_binding = {
                "role": ROLE,
                "role_rank": rank,
                "row_manifest_sha256": authority["release"]["row_manifest_sha256"],
                "weights_sha256": authority["release"]["weights_sha256"],
                "singleton_row_count": 1,
                "sealed_row_zero_access_sentinels": SEALED_ROW_ZERO_ACCESS_SENTINELS,
                "sealed_row_zero_access_sentinels_sha256": canonical_json_sha256(
                    SEALED_ROW_ZERO_ACCESS_SENTINELS
                ),
            }
            sealed_row_firewall = all(
                payload.get("row_binding") == expected_row_binding
                and payload.get("store_hashes", {}).get(
                    "sealed_row_zero_access_sentinels"
                ) == SEALED_ROW_ZERO_ACCESS_SENTINELS
                and payload.get("store_hashes", {}).get(
                    "sealed_row_zero_access_sentinels_sha256"
                ) == canonical_json_sha256(SEALED_ROW_ZERO_ACCESS_SENTINELS)
                for payload in workers.values()
            )
            if not sealed_row_firewall:
                raise RuntimeError("A1/B/A2 row/weight singleton binding differs")
            if (
                a1["source_sha256"] != BASE_SOURCE_SHA256
                or a2["source_sha256"] != BASE_SOURCE_SHA256
                or b["source_sha256"] != authority["freeze"]["frozen_files"][SOURCE.name]
            ):
                raise RuntimeError("A1/B/A2 source hash binding differs")

            parity = component_parity(
                a1["public_components"], b["public_components"], a2["public_components"]
            )
            a1_flops = int(a1["literal_meter"]["flops_used"])
            b_flops = int(b["literal_meter"]["flops_used"])
            a2_flops = int(a2["literal_meter"]["flops_used"])
            delta_a1 = b_flops - a1_flops
            delta_a2 = b_flops - a2_flops
            operation_delta_a1 = operation_call_deltas(b["literal_meter"], a1["literal_meter"])
            operation_delta_a2 = operation_call_deltas(b["literal_meter"], a2["literal_meter"])

            layers = b.get("layers")
            if not isinstance(layers, list) or len(layers) != 15:
                raise RuntimeError("B worker did not return exactly 15 transient layer summaries")
            for slot, row in enumerate(layers):
                if (
                    row.get("slot") != slot
                    or row.get("source_layer_one_based") != slot + 1
                    or row.get("outgoing_weight_index_zero_based") != slot + 1
                    or row.get("outgoing_weight_literal") != f"mlp.weights[{slot + 1}]"
                    or row.get("consumer_layer_one_based") != slot + 2
                ):
                    raise RuntimeError("combined coordinate slot/source/weight/consumer mapping changed")

            worker_coordinate = b.get("coordinate_artifact")
            if not isinstance(worker_coordinate, dict):
                raise RuntimeError("B worker lacks its combined coordinate receipt")
            coordinate_source = Path(worker_coordinate["path"])
            coordinate = _coordinate_receipt(coordinate_source)
            for key in ("file_sha256", "content_sha256", "dtype", "shape", "c_order", "nbytes"):
                if coordinate[key] != worker_coordinate.get(key):
                    raise RuntimeError("B worker coordinate receipt mismatch")

            source_finite = all(
                row["source"][sector]["finite"]
                for row in layers for sector in ("k42", "k51")
            )
            action_finite = all(
                row["action"][sector]["finite"]
                for row in layers
                for sector in ("raw_action42", "raw_action51", "combined_raw_k6")
            )
            exact_diagonals = all(
                row["source"][sector]["diagonal_max_abs"] == 0.0
                for row in layers for sector in ("k42", "k51")
            )
            science = b.get("runtime_science_proof")
            if not isinstance(science, dict) or not all(science.get("gates", {}).values()):
                raise RuntimeError("B worker lacks complete runner-owned runtime science proof")
            science_gates = science["gates"]
            all_public_finite = all(
                component["finite"]
                for payload in workers.values()
                for component in payload["public_components"].values()
            )
            firewall_zero = all(
                value == 0
                for payload in workers.values()
                for value in payload["firewall"].values()
            )
            residuals = {
                phase: float(payload["literal_meter"]["residual_wall_time_s"])
                for phase, payload in workers.items()
            }
            rss = {
                phase: int(payload["process_peak_rss_bytes"])
                for phase, payload in workers.items()
            }
            totals = {
                "A1": a1_flops,
                "B": b_flops,
                "A2": a2_flops,
            }
            final_shapes = {
                tuple(payload["public_components"]["final_public_prediction"]["shape"])
                for payload in workers.values()
            }

            gates = {
                "authority_and_release_hashes_match": True,
                "role_and_rank_match_singleton_release": True,
                "target_training_selection_firewall": (
                    firewall_zero and sealed_row_firewall
                ),
                "full_public_A1_B_A2_byte_parity": all(parity.values()),
                "public_output_shape_finite_repeat": (
                    final_shapes == {OUTPUT_SHAPE} and all_public_finite
                    and a1["public_components"]["final_public_prediction"]["content_sha256"]
                    == a2["public_components"]["final_public_prediction"]["content_sha256"]
                ),
                "centered_coefficient_quadrature_and_recurrence": science_gates["centered_coefficient_quadrature_and_recurrence"],
                "partition_203_and_formula_parity": science_gates["partition_203_and_formula_parity"],
                "batched_source_vs_scalar_parity": science_gates["batched_source_vs_scalar_parity"],
                "factor15_action_parity": science_gates["factor15_action_parity"],
                "factor6_action_parity": science_gates["factor6_action_parity"],
                "raw_vs_standardized_parity": science_gates["raw_vs_standardized_parity"],
                "k24_equals_k42_transpose": science_gates["k24_equals_k42_transpose"],
                "k15_equals_k51_transpose": science_gates["k15_equals_k51_transpose"],
                "correlated_gaussian_null": science_gates["correlated_gaussian_null"],
                "exact_two_zero_diagonals_each_source": (
                    exact_diagonals and b["source_call_counts"].get("fill_diagonal_calls") == 30
                ),
                "source_and_action_dtype_exact_float64_before_copy": all(
                    row["source"][sector]["observed_dtype_before_copy"] == "<f8"
                    for row in layers for sector in ("k42", "k51")
                ) and all(
                    row["action"][key] == "<f8"
                    for row in layers
                    for key in (
                        "sector_observed_dtype_before_copy",
                        "combined_observed_dtype_before_copy",
                    )
                ),
                "all_source_sector_action_and_combined_values_finite": source_finite and action_finite,
                "k42_plane_norm_positive_all_sources": all(
                    row["source"]["k42"]["frobenius_norm"] > 0.0 for row in layers
                ),
                "k51_plane_norm_positive_all_sources": all(
                    row["source"]["k51"]["frobenius_norm"] > 0.0 for row in layers
                ),
                "raw_action42_norm_positive_all_actions": all(
                    row["action"]["raw_action42"]["l2_norm"] > 0.0 for row in layers
                ),
                "raw_action51_norm_positive_all_actions": all(
                    row["action"]["raw_action51"]["l2_norm"] > 0.0 for row in layers
                ),
                "combined_raw_norm_positive_all_actions": all(
                    row["action"]["combined_raw_k6"]["l2_norm"] > 0.0 for row in layers
                ),
                "one_shared_endpoint_and_coordinate_mapping": (
                    science_gates["one_shared_endpoint_and_coordinate_mapping"]
                    and all(row["action"]["exact_sector_sum"] for row in layers)
                ),
                "combined_raw_bank_dtype_shape_file_and_content_hash": (
                    coordinate["dtype"] == "<f8"
                    and coordinate["shape"] == list(COORDINATE_SHAPE)
                    and coordinate["c_order"] is True
                    and coordinate["nbytes"] == 122_880
                ),
                "exact_source_call_inventory": b["source_call_counts"] == EXPECTED_SOURCE_CALL_COUNTS,
                "exact_official_operation_call_deltas": (
                    operation_delta_a1 == EXPECTED_OPERATION_CALL_DELTAS
                    and operation_delta_a2 == EXPECTED_OPERATION_CALL_DELTAS
                ),
                "base_A1_A2_flop_bills_equal": a1_flops == a2_flops,
                "equal_nonnegative_incremental_flop_deltas_at_most_152200000000": (
                    delta_a1 == delta_a2
                    and 0 <= delta_a1 <= INCREMENTAL_FLOP_CAP
                    and 0 <= delta_a2 <= INCREMENTAL_FLOP_CAP
                ),
                "all_total_flops_strictly_below_2pow41": all(
                    value < PHASE2_BUDGET for value in totals.values()
                ),
                "all_process_rss_at_most_8gib": all(
                    value <= RSS_CAP_BYTES for value in rss.values()
                ),
                "all_official_residual_seconds_strictly_below_0_4": all(
                    value < RESIDUAL_CAP_SECONDS for value in residuals.values()
                ),
                "static_remote_scratch_within_cap": REMOTE_SCRATCH_CAP_BYTES <= 309_092_352,
                "static_largest_new_array_within_caps": (
                    LARGEST_NEW_ARRAY_CAP_BYTES <= 285_212_672
                    and LARGEST_NEW_ARRAY_CAP_BYTES < REMOTE_SINGLE_ARRAY_CAP_BYTES
                ),
                "no_hard_magnitude_rank_or_selection_threshold": True,
            }
            if set(gates) != REQUIRED_RESULT_GATES:
                raise RuntimeError("internal V3 result-gate inventory changed")
            passed = all(gates.values())
            decision = (
                "PASS_K42_K51_SOURCE_ACTION_RANK201__HOLD_RANK643_PENDING_INDEPENDENT_AUDIT_AND_NEW_RELEASE"
                if passed and rank == 201
                else "GO_K42_K51_AUTONOMOUS_RESET_DESIGN_ONLY"
                if passed and rank == 643
                else "CLOSE_THIS_COMBINED_RAW_D4_Q7_K42_K51_SOURCE_ACTION_SPELLING"
            )
            phase_consumption = {
                phase: {"path": path.name, "sha256": sha256_file(path)}
                for phase, path in phase_receipts.items()
            }
            result = {
                "schema": "q24-k42-k51-target-blind-source-action-result-v3",
                "status": "COMPLETE_TARGET_BLIND_K42_K51_SOURCE_ACTION",
                "decision": decision,
                "all_gates_pass": passed,
                "role": ROLE,
                "role_rank": rank,
                "mlp_id": int(a1["mlp_id"]),
                "freeze_sha256": authority["freeze_sha256"],
                "source_sha256": authority["freeze"]["frozen_files"][SOURCE.name],
                "preregistration_sha256": authority["freeze"]["frozen_files"][PREREGISTRATION.name],
                "retry_allowed": False,
                "coordinate_artifact": coordinate,
                "slot_mapping": "s=0..14: source h(s+1), outgoing mlp.weights[s+1], consumer h(s+2)",
                "parity": {
                    "components": parity,
                    "A1": a1["public_components"],
                    "B": b["public_components"],
                    "A2": a2["public_components"],
                },
                "literal_meter": {
                    "phase2_budget": PHASE2_BUDGET,
                    "static_dense_diagnostic_flops": STATIC_DENSE_DIAGNOSTIC_FLOPS,
                    "predicted_integrated_incremental_flops": PREDICTED_INTEGRATED_INCREMENTAL_FLOPS,
                    "hard_incremental_cap_flops": INCREMENTAL_FLOP_CAP,
                    "A1_preflight_before_B_launch": a1_preflight,
                    "A1": a1["literal_meter"],
                    "B": b["literal_meter"],
                    "A2": a2["literal_meter"],
                    "B_minus_A1_flops": delta_a1,
                    "B_minus_A2_flops": delta_a2,
                    "operation_call_delta_B_minus_A1": operation_delta_a1,
                    "operation_call_delta_B_minus_A2": operation_delta_a2,
                },
                "rss": {
                    "fresh_process_peak_bytes": rss,
                    "process_cap_bytes": RSS_CAP_BYTES,
                    "static_remote_scratch_ceiling_bytes": REMOTE_SCRATCH_CAP_BYTES,
                    "static_largest_new_array_bytes": LARGEST_NEW_ARRAY_CAP_BYTES,
                    "client_coordinate_bank_bytes": 122_880,
                    "full_planes_persisted": False,
                    "sector_action_banks_persisted": False,
                },
                "residual_seconds": residuals,
                "runtime_science_proof": science,
                "layers": layers,
                "source_call_counts": b["source_call_counts"],
                "gates": gates,
                "firewall": {
                    "targets_read": 0,
                    "scores_read": 0,
                    "truth_joins": 0,
                    "training_steps": 0,
                    "selection_steps": 0,
                    "autonomous_feedback_runs": 0,
                    "full_estimator_candidate_runs": 0,
                    "submissions": 0,
                    "rank35_calls": 0,
                    "pass_authorizes_targets": False,
                },
                "authority": {
                    "release_file_sha256": release_sha,
                    "release_payload_sha256": authority["release_payload_sha256"],
                    "validated_release_payload": authority["release"],
                    "frozen_files": authority["freeze"]["frozen_files"],
                    "runner_sha256": sha256_file(RUNNER),
                    "base_source_sha256": BASE_SOURCE_SHA256,
                    "store_hashes": a1["store_hashes"],
                    "row_manifest_sha256": authority["release"]["row_manifest_sha256"],
                    "weights_sha256": authority["release"]["weights_sha256"],
                    "sealed_row_zero_access_sentinels": SEALED_ROW_ZERO_ACCESS_SENTINELS,
                    "sealed_row_zero_access_sentinels_sha256": canonical_json_sha256(
                        SEALED_ROW_ZERO_ACCESS_SENTINELS
                    ),
                    "run_id": run_id,
                    "no_retry_lock_sha256": parent_lock_sha,
                    "worker_process_ids": {
                        phase: payload["worker_identity"]["process_id"]
                        for phase, payload in workers.items()
                    },
                    "worker_process_nonces": {
                        phase: payload["worker_identity"]["process_nonce"]
                        for phase, payload in workers.items()
                    },
                    "phase_consumption_receipts": phase_consumption,
                    "final_bundle": final_bundle.name,
                    "commit_manifest": "COMMIT_MANIFEST.json",
                },
            }
            manifest_bindings = {
                "role": ROLE,
                "role_rank": rank,
                "run_id": run_id,
                "release_file_sha256": release_sha,
                "release_payload_sha256": authority["release_payload_sha256"],
                "freeze_sha256": authority["freeze_sha256"],
                "frozen_files": authority["freeze"]["frozen_files"],
                "runner_sha256": sha256_file(RUNNER),
                "source_sha256": sha256_file(SOURCE),
                "row_manifest_sha256": authority["release"]["row_manifest_sha256"],
                "weights_sha256": authority["release"]["weights_sha256"],
                "sealed_row_zero_access_sentinels_sha256": canonical_json_sha256(
                    SEALED_ROW_ZERO_ACCESS_SENTINELS
                ),
                "no_retry_lock_sha256": parent_lock_sha,
                "phase_consumption_receipts": phase_consumption,
                "terminal_status": result["status"],
                "terminal_decision": decision,
                "all_gates_pass": passed,
            }
            committed = commit_atomic_bundle(
                directory,
                final_bundle,
                terminal_result_name,
                coordinate_source,
                worker_coordinate,
                result,
                manifest_bindings,
            )
            return {
                **committed,
                "decision": decision,
                "all_gates_pass": passed,
            }
    except Exception as exc:
        durable_lock_sha = (
            parent_lock_sha
            if parent_lock_sha is not None
            else sha256_file(parent_lock_path)
            if lock_acquired_by_this_process and parent_lock_path.is_file()
            else None
        )
        consumed = {
            phase: {"path": path.name, "sha256": sha256_file(path)}
            for phase, path in phase_receipts.items()
            if path.is_file()
        }
        terminal_failure = {
            "schema": "q24-k42-k51-target-blind-source-action-terminal-failure-v3",
            "status": "FAILED_TARGET_BLIND_K42_K51_SOURCE_ACTION__NO_RETRY",
            "decision": "CLOSE_THIS_COMBINED_RAW_D4_Q7_K42_K51_SOURCE_ACTION_SPELLING",
            "all_gates_pass": False,
            "role": ROLE,
            "role_rank": rank,
            "retry_allowed": False,
            "terminal_exception": {"type": type(exc).__name__, "message": str(exc)},
            "firewall": {
                "targets_read": 0,
                "scores_read": 0,
                "truth_joins": 0,
                "training_steps": 0,
                "selection_steps": 0,
                "autonomous_feedback_runs": 0,
                "full_estimator_candidate_runs": 0,
                "submissions": 0,
            },
            "authority": {
                "release_file_sha256": release_sha,
                "release_payload_sha256": authority["release_payload_sha256"],
                "freeze_sha256": authority["freeze_sha256"],
                "frozen_files": authority["freeze"]["frozen_files"],
                "runner_sha256": sha256_file(RUNNER),
                "source_sha256": sha256_file(SOURCE),
                "run_id": run_id,
                "no_retry_lock_sha256": durable_lock_sha,
                "phase_consumption_receipts": consumed,
                "planned_final_bundle": final_bundle.name,
                "final_bundle_visible": final_bundle.exists(),
            },
        }
        persist_failure_if_lock_owned(
            lock_acquired_by_this_process=lock_acquired_by_this_process,
            final_bundle=final_bundle,
            failure=failure,
            payload=terminal_failure,
        )
        raise


def transaction_primitive_selftest() -> dict[str, bool]:
    """Tiny synthetic replay/failure/atomicity test; no release or row access."""

    import numpy as np

    checks: dict[str, bool] = {}
    with tempfile.TemporaryDirectory(prefix="k42_k51_txn_selftest_") as temporary:
        root = Path(temporary)
        duplicate_capability_rejected = False
        try:
            require_distinct_phase_capabilities({"A1": "same", "B": "same", "A2": "other"})
        except RuntimeError:
            duplicate_capability_rejected = True
        checks["duplicate_phase_capability_rejected_before_claim"] = duplicate_capability_rejected
        fake_freeze_hash = "f" * 64
        fake_frozen = {RUNNER.name: "1" * 64}
        fake_freeze = {"frozen_files": fake_frozen}
        fake_payload = {
            "schema": "q24-k42-k51-source-action-singleton-release-v3",
            "status": "RELEASE_TARGET_BLIND_K42_K51_SOURCE_ACTION_ONCE",
            "allowed_role": ROLE,
            "allowed_role_ranks": [201],
            "singleton_row_count": 1,
            "no_retry": True,
            "concurrent_heavy_run_allowed": False,
            "targets_allowed": False,
            "scores_allowed": False,
            "truth_join_allowed": False,
            "training_allowed": False,
            "selection_allowed": False,
            "autonomous_feedback_allowed": False,
            "full_estimator_allowed": False,
            "submission_allowed": False,
            "independent_final_go": True,
            "runtime_environment": OFFICIAL_RUNTIME_ENVIRONMENT,
            "incremental_flop_cap": INCREMENTAL_FLOP_CAP,
            "freeze_path": FREEZE.name,
            "freeze_sha256": fake_freeze_hash,
            "frozen_files": fake_frozen,
            "prerequisite_rank201_sha256": {},
            "row_manifest_sha256": "2" * 64,
            "weights_sha256": "3" * 64,
        }
        fake_result = {
            "authority": {
                "validated_release_payload": fake_payload,
                "release_payload_sha256": canonical_json_sha256(fake_payload),
                "frozen_files": fake_frozen,
                "runner_sha256": "1" * 64,
                "base_source_sha256": BASE_SOURCE_SHA256,
                "row_manifest_sha256": "2" * 64,
                "weights_sha256": "3" * 64,
                "sealed_row_zero_access_sentinels": SEALED_ROW_ZERO_ACCESS_SENTINELS,
                "sealed_row_zero_access_sentinels_sha256": canonical_json_sha256(
                    SEALED_ROW_ZERO_ACCESS_SENTINELS
                ),
                "release_file_sha256": "4" * 64,
                "no_retry_lock_sha256": "5" * 64,
                "run_id": "synthetic-rank201",
            }
        }
        _validated_embedded_rank201_release_payload(
            fake_result, fake_freeze, fake_freeze_hash
        )
        mutant = json.loads(json.dumps(fake_result))
        mutant_payload = mutant["authority"]["validated_release_payload"]
        mutant_payload["row_manifest_sha256"] = "6" * 64
        mutant["authority"]["release_payload_sha256"] = canonical_json_sha256(mutant_payload)
        mutant_rejected = False
        try:
            _validated_embedded_rank201_release_payload(
                mutant, fake_freeze, fake_freeze_hash
            )
        except RuntimeError:
            mutant_rejected = True
        checks["mutant_rank201_embedded_release_cross_binding_rejected"] = mutant_rejected
        valid_zero_sentinels = validate_sealed_row_zero_access_sentinels(
            dict(SEALED_ROW_ZERO_ACCESS_SENTINELS)
        )
        mutant_zero_sentinel_rejected = False
        mutant_zero_sentinels = dict(SEALED_ROW_ZERO_ACCESS_SENTINELS)
        mutant_zero_sentinels["target_values_decoded"] = 1
        try:
            validate_sealed_row_zero_access_sentinels(mutant_zero_sentinels)
        except RuntimeError:
            mutant_zero_sentinel_rejected = True
        checks["mutant_sealed_row_zero_access_sentinel_rejected_before_weights"] = (
            valid_zero_sentinels == SEALED_ROW_ZERO_ACCESS_SENTINELS
            and mutant_zero_sentinel_rejected
        )
        dtype_lock = root / "wrong_dtype.lock"
        dtype_failure = root / "wrong_dtype.failure.json"
        dtype_bundle = root / "wrong_dtype.bundle"
        exclusive_json(dtype_lock, {"status": "CONSUMED", "retry_allowed": False})
        wrong_dtype_rejected = False
        try:
            checked_local_array(np.zeros((2, 4), dtype=np.float32), "mock wrong-dtype action")
        except RuntimeError as dtype_error:
            wrong_dtype_rejected = True
            persist_failure_if_lock_owned(
                lock_acquired_by_this_process=True,
                final_bundle=dtype_bundle,
                failure=dtype_failure,
                payload={
                    "status": "FAILED_TARGET_BLIND_K42_K51_SOURCE_ACTION__NO_RETRY",
                    "retry_allowed": False,
                    "terminal_exception": str(dtype_error),
                },
            )
        dtype_replay_rejected = False
        try:
            exclusive_json(dtype_lock, {"status": "REPLAY"})
        except FileExistsError:
            dtype_replay_rejected = True
        dtype_failure_payload = json.loads(dtype_failure.read_text()) if dtype_failure.is_file() else {}
        checks["wrong_dtype_claim_owned_terminal_failure_no_bundle_or_orphan"] = (
            wrong_dtype_rejected
            and dtype_replay_rejected
            and dtype_failure_payload.get("status")
            == "FAILED_TARGET_BLIND_K42_K51_SOURCE_ACTION__NO_RETRY"
            and dtype_failure_payload.get("retry_allowed") is False
            and not dtype_bundle.exists()
            and not (root / "ATOMIC_BUNDLE_STAGE").exists()
            and not (root / COORDINATE_NAME).exists()
        )
        claim = root / "claim.json"
        exclusive_json(claim, {"status": "CONSUMED"})
        replay_rejected = False
        try:
            exclusive_json(claim, {"status": "REPLAY"})
        except FileExistsError:
            replay_rejected = True
        checks["exclusive_claim_replay_rejected"] = replay_rejected
        race_failure = root / "failure.json"
        winner_marker = root / "winner.marker"
        winner_marker.write_text("winner\n")
        loser_wrote = persist_failure_if_lock_owned(
            lock_acquired_by_this_process=False,
            final_bundle=root / "winner.bundle",
            failure=race_failure,
            payload={"status": "LOSER_MUST_NOT_WRITE"},
        )
        checks["lock_race_loser_creates_no_failure_receipt"] = (
            not loser_wrote
            and not race_failure.exists()
            and winner_marker.read_text() == "winner\n"
        )

        work = root / "work"
        work.mkdir()
        source = work / "coordinate.npy"
        synthetic = np.arange(15 * 1024, dtype=np.float64).reshape(COORDINATE_SHAPE)
        worker_receipt = atomic_npy(source, synthetic)
        coordinate = _coordinate_receipt(source)
        result = {
            "schema": "tiny-transaction-selftest",
            "status": "COMPLETE",
            "decision": "PASS_TINY",
            "all_gates_pass": True,
            "coordinate_artifact": coordinate,
        }
        final = root / "FINAL_BUNDLE"
        committed = commit_atomic_bundle(
            work, final, "RESULT.json", source, worker_receipt, result,
            {"test_only": True},
        )
        checks["one_atomic_bundle_visible"] = final.is_dir()
        checks["bundle_exact_three_files"] = {
            path.name for path in final.iterdir()
        } == {COORDINATE_NAME, "RESULT.json", "COMMIT_MANIFEST.json"}
        manifest = json.loads((final / "COMMIT_MANIFEST.json").read_text())
        require_manifest_binding_subset(manifest, {"test_only": True})
        checks["manifest_binds_result_file_sha"] = (
            manifest["files"]["RESULT.json"]["file_sha256"]
            == sha256_file(final / "RESULT.json")
        )
        checks["manifest_binds_coordinate_file_and_content_sha"] = (
            manifest["files"][COORDINATE_NAME]["file_sha256"]
            == sha256_file(final / COORDINATE_NAME)
            and manifest["files"][COORDINATE_NAME]["content_sha256"]
            == array_content_sha256(np.load(final / COORDINATE_NAME, allow_pickle=False))
            and committed["commit_manifest_sha256"]
            == sha256_file(final / "COMMIT_MANIFEST.json")
        )
        mutant_manifest = json.loads(json.dumps(manifest))
        mutant_manifest["bindings"]["test_only"] = False
        mutant_manifest_rejected = False
        try:
            require_manifest_binding_subset(mutant_manifest, {"test_only": True})
        except RuntimeError:
            mutant_manifest_rejected = True
        checks["mutant_commit_manifest_cross_binding_rejected"] = mutant_manifest_rejected

        failed_work = root / "failed_work"
        failed_work.mkdir()
        failed_source = failed_work / "coordinate.npy"
        failed_receipt = atomic_npy(failed_source, synthetic)
        failed_receipt["content_sha256"] = "0" * 64
        failed_final = root / "FAILED_FINAL"
        failure_rejected = False
        try:
            commit_atomic_bundle(
                failed_work,
                failed_final,
                "RESULT.json",
                failed_source,
                failed_receipt,
                result,
                {"test_only": True},
            )
        except RuntimeError:
            failure_rejected = True
        checks["mocked_precommit_failure_rejected"] = failure_rejected
        checks["mocked_failure_has_no_visible_bundle"] = not failed_final.exists()
        checks["mocked_failure_has_no_orphan_stage_or_coordinate"] = (
            not (failed_work / "ATOMIC_BUNDLE_STAGE").exists()
            and not failed_source.exists()
        )

        rename_work = root / "rename_failure_work"
        rename_work.mkdir()
        rename_source = rename_work / "coordinate.npy"
        rename_receipt = atomic_npy(rename_source, synthetic)
        rename_coordinate = _coordinate_receipt(rename_source)
        rename_result = {
            "schema": "tiny-transaction-selftest",
            "status": "COMPLETE",
            "decision": "PASS_TINY",
            "all_gates_pass": True,
            "coordinate_artifact": rename_coordinate,
        }
        rename_final = root / "RENAME_FAILURE_FINAL"
        rename_failure_rejected = False
        try:
            commit_atomic_bundle(
                rename_work,
                rename_final,
                "RESULT.json",
                rename_source,
                rename_receipt,
                rename_result,
                {"test_only": True},
                _selftest_fail_after_fsync_before_visibility=True,
            )
        except RuntimeError:
            rename_failure_rejected = True
        checks["mocked_post_fsync_previsibility_rename_failure_rejected"] = rename_failure_rejected
        checks["post_fsync_failure_has_no_visible_or_orphan_bundle"] = (
            not rename_final.exists()
            and not (rename_work / "ATOMIC_BUNDLE_STAGE").exists()
            and not rename_source.exists()
        )
    if not all(checks.values()):
        raise RuntimeError(f"transaction primitive selftest failed: {checks}")
    return checks


def static_status() -> dict[str, Any]:
    runtime_artifacts = [
        path.name
        for rank in ALLOWED_ORDERED_RANKS
        for path in _runtime_artifact_paths(rank)
        if path.exists()
    ]
    frozen_ok = False
    frozen_files: dict[str, Any] | None = None
    if FREEZE.is_file():
        try:
            freeze = json.loads(FREEZE.read_text())
            frozen_files = freeze.get("frozen_files")
            frozen_ok = (
                set(frozen_files or {}) == REQUIRED_FROZEN_FILES
                and all(
                    (HERE / name).is_file() and sha256_file(HERE / name) == digest
                    for name, digest in (frozen_files or {}).items()
                )
            )
        except Exception:
            frozen_ok = False
    return {
        "schema": "q24-k42-k51-source-action-runner-static-status-v3",
        "status": "HOLD_NO_RELEASE_NO_ROW_OR_ESTIMATOR_CALL",
        "release_path": str(RELEASE),
        "release_present": RELEASE.exists(),
        "freeze_path": str(FREEZE),
        "freeze_present": FREEZE.exists(),
        "complete_frozen_map_valid": frozen_ok,
        "required_frozen_files": sorted(REQUIRED_FROZEN_FILES),
        "runtime_artifacts_present": runtime_artifacts,
        "official_interpreter_path": str(OFFICIAL_INTERPRETER),
        "official_interpreter_exists": OFFICIAL_INTERPRETER.is_file(),
        "worker_order": ["A1", "B", "A2"],
        "workers_are_internal_subprocesses": True,
        "callback_worker_api_present": False,
        "static_dense_formula_diagnostic_flops": STATIC_DENSE_DIAGNOSTIC_FLOPS,
        "predicted_integrated_incremental_flops": PREDICTED_INTEGRATED_INCREMENTAL_FLOPS,
        "hard_incremental_flop_cap": INCREMENTAL_FLOP_CAP,
        "expected_operation_call_deltas": EXPECTED_OPERATION_CALL_DELTAS,
        "syntactic_fill_diagonal_sites": SOURCE.read_text().count("fnp.fill_diagonal("),
        "runtime_fill_diagonal_calls": EXPECTED_SOURCE_CALL_COUNTS["fill_diagonal_calls"],
        "atomic_bundle_files": [COORDINATE_NAME, "RESULT.json", "COMMIT_MANIFEST.json"],
        "numerical_work_performed": False,
        "sealed_rows_opened": 0,
        "weights_opened": 0,
        "targets_read": 0,
        "estimator_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--selftest-transaction", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--rank", type=int, default=201)
    parser.add_argument("--internal-worker-phase", choices=("A1", "B", "A2"))
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
    if args.selftest_transaction:
        print(json.dumps(transaction_primitive_selftest(), indent=2, sort_keys=True))
        return
    if args.execute:
        print(json.dumps(execute(args.rank), indent=2, sort_keys=True))
        return
    print(json.dumps(static_status(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
