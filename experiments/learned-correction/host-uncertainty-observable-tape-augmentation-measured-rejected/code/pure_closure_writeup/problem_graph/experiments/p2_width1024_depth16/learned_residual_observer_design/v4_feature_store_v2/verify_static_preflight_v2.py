#!/usr/bin/env python3
"""Strict target-free static and synthetic verifier for additive V2."""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any

from contract_v2 import (
    CAPS, CLAIM_PATH, ENVIRONMENT_PYTHON, FROZEN_BINDINGS, HERE, MAX_HOST_WORKERS,
    MAX_WATCHDOG_GAP_SECONDS, OUTPUT_PATH, PER_PROCESS_LIMIT_BYTES,
    RELEASE_SCHEMA_PATH, ROLE_BINDINGS, SCRATCH_PATH, SOURCE_MANIFEST_PATH,
    STAGING_PATH, V1_BINDINGS, WATCHDOG_INTERVAL_SECONDS, assert_runtime_absent,
    load_exact_roster, sha256_file, verify_v1_bindings,
)
import generate_feature_store_once_v2 as generator
import test_injected_resource_failures_v2 as injected_tests
import verify_artifacts_v2 as artifact_verifier


SOURCE_FILES = (
    "PROTOCOL_V2.md",
    "RELEASE_SCHEMA_V2.json",
    "STATIC_CONTRACT_V2.json",
    "INJECTED_RESOURCE_TEST_RECEIPT_V2.json",
    "contract_v2.py",
    "resource_safety_v2.py",
    "target_blind_cursor_v2.py",
    "host_worker_exec_gate_v2.py",
    "host_feature_worker_v2.py",
    "generate_feature_store_once_v2.py",
    "sealed_feature_store_provider_v2.py",
    "verify_artifacts_v2.py",
    "synthetic_group_tree_v2.py",
    "test_injected_resource_failures_v2.py",
    "verify_static_preflight_v2.py",
)
PYTHON_FILES = tuple(name for name in SOURCE_FILES if name.endswith(".py"))


def _top_level_imports(tree: ast.Module) -> set[str]:
    result: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            result.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.add(node.module or "")
    return result


def _ordered(text: str, *tokens: str) -> None:
    position = -1
    for token in tokens:
        found = text.find(token, position + 1)
        if found < 0:
            raise AssertionError(f"ordered safety token absent: {token}")
        position = found


def _source_laws() -> dict[str, Any]:
    sources = {name: (HERE / name).read_text(encoding="utf-8") for name in PYTHON_FILES}
    trees = {name: ast.parse(text, filename=name) for name, text in sources.items()}
    generator_imports = _top_level_imports(trees["generate_feature_store_once_v2.py"])
    if generator_imports & {"numpy", "pyarrow", "flopscope", "torch"}:
        raise AssertionError("V2 generator imports a data/host/training package before claim")
    cursor_imports = _top_level_imports(trees["target_blind_cursor_v2.py"])
    if cursor_imports & {"numpy", "pyarrow"}:
        raise AssertionError("V2 cursor imports a row-decoding package at boot")

    launcher = sources["generate_feature_store_once_v2.py"]
    execute = launcher[launcher.index("def execute_generation"):]
    _ordered(execute, "claim = _claim(release, release_sha)", "ResourceWatchdog(", "_start_cursor(")
    _ordered(
        launcher[launcher.index("def _start_cursor"):launcher.index("def _launch_host")],
        'watchdog.guard("pre_iterator_construction")',
        '_send_cursor(process, {"op": "START"})',
    )
    _ordered(
        execute,
        'watchdog.guard("pre_decode_next_64mib_full_row")',
        '_send_cursor(cursor, {"op": "NEXT"})',
    )
    _ordered(
        execute,
        'watchdog.guard("pre_partial_row_directory_creation")',
        "partial.mkdir(mode=0o700)",
        'watchdog.guard("pre_64mib_weight_write_allocation")',
        '"op": "WRITE"',
    )
    launch_host = launcher[launcher.index("def _launch_host"):launcher.index("def _finish_host")]
    _ordered(
        launch_host,
        "start_new_session=True",
        "record = registry.register(process",
        'watchdog.guard("pre_monitored_host_heavy_import_and_compute_exec")',
        "os.replace(gate_partial, gate_path)",
    )
    _ordered(
        launcher[launcher.index("def _finish_host"):launcher.index("def _wait_host_slot")],
        "retire_completed_group(registry, record)",
        "validate_row_directory",
        "os.replace(item[\"partial\"], destination)",
        "fsync_directory(destination.parent)",
    )

    cursor = sources["target_blind_cursor_v2.py"]
    _ordered(cursor, 'operation == "START"', "iterator = iter(reader.iter_target_blind_full())")
    _ordered(cursor, 'operation == "NEXT"', "current = next(iterator)")
    if cursor.index("current = next(iterator)") < cursor.index('operation == "NEXT"'):
        raise AssertionError("V2 cursor can decode outside NEXT")

    safety = sources["resource_safety_v2.py"]
    for token in (
        "os.killpg(pgid, sig)", "signal.SIGTERM", "signal.SIGKILL",
        "record.process.wait", "_quiescent_three_polls", "empty >= 3",
        "MAX_WATCHDOG_GAP_SECONDS", "PER_PROCESS_LIMIT_BYTES",
    ):
        if token not in safety:
            raise AssertionError(f"V2 group-supervisor token absent: {token}")
    if "if self._violation.is_set()" not in safety:
        raise AssertionError("V2 watchdog can log a false post-termination sample gap")

    worker = sources["host_feature_worker_v2.py"]
    if worker.index("resource.setrlimit") > worker.index("import numpy as np"):
        raise AssertionError("V2 worker does not attempt its address-space guard before NumPy")
    if "strict_group_rss_watchdog_required" not in worker:
        raise AssertionError("V2 worker lacks authoritative Darwin group-watchdog receipt")
    if "frozen._host_tape(weights, cap)" not in worker or "for cap in CAPS" not in worker:
        raise AssertionError("V2 worker changed frozen CP1536/CP1792 semantics")

    tests = sources["test_injected_resource_failures_v2.py"]
    for token in (
        '"system_25_percent"', '"group_8gib"', '"exception"',
        "decode_callable_invocations != 0", "sigkill_groups",
        "process_group_pids(record.pgid)",
    ):
        if token not in tests:
            raise AssertionError(f"V2 injected proof token absent: {token}")

    all_source = "\n".join(sources.values())
    if re.search(r"['\"][^'\"]*\.parquet['\"]", all_source, flags=re.IGNORECASE):
        raise AssertionError("V2 source names a Parquet payload")
    for forbidden in ("sealed_" + "role_targets", "authorized_" + "full_targets"):
        if forbidden in all_source:
            raise AssertionError(f"V2 source names forbidden target interface: {forbidden}")
    return {
        "generator_top_level_imports": sorted(generator_imports),
        "cursor_top_level_imports": sorted(cursor_imports),
        "claim_before_cursor_or_reader": True,
        "guard_before_iterator_decode_partial_and_weight_write": True,
        "continuous_watchdog_max_gap_seconds": MAX_WATCHDOG_GAP_SECONDS,
        "whole_group_term_grace_kill_reap_three_empty_polls": True,
        "host_heavy_exec_after_group_registration_and_guard": True,
        "group_quiescence_before_atomic_row_seal": True,
        "parquet_payload_paths_in_sources": 0,
    }


def _exact_environment_boot_probes() -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    reports = {}
    for name in ("target_blind_cursor_v2.py", "host_feature_worker_v2.py"):
        process = subprocess.run(
            [str(ENVIRONMENT_PYTHON), str(HERE / name), "--help"],
            cwd=HERE, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, timeout=60, check=False,
        )
        if process.returncode != 0:
            raise AssertionError(
                f"exact-environment zero-row boot probe failed for {name}: {process.stderr}"
            )
        reports[name] = {"returncode": 0, "row_decodes": 0}
    return reports


def verify() -> dict[str, Any]:
    v1 = verify_v1_bindings()
    v1_audit = json.loads(
        V1_BINDINGS["independent_source_audit_hold"]["path"].read_text(encoding="utf-8")
    )
    if v1_audit.get("status") != (
        "HOLD_PHASE2_OBSERVER_V4_FEATURE_STORE_SOURCE_AUDIT__RESOURCE_SUPERVISOR_INCOMPLETE"
    ):
        raise AssertionError("V1 independent HOLD status changed")
    assert_runtime_absent()
    roster = load_exact_roster()
    if tuple(CAPS) != (1536, 1792) or MAX_HOST_WORKERS != 2:
        raise AssertionError("V2 cap/worker contract changed")
    if WATCHDOG_INTERVAL_SECONDS != 0.05 or MAX_WATCHDOG_GAP_SECONDS != 0.20:
        raise AssertionError("V2 watchdog cadence changed")
    if PER_PROCESS_LIMIT_BYTES != 8 * (1 << 30):
        raise AssertionError("V2 group memory boundary changed")
    if {role: len(rows) for role, rows in roster.items()} != {
        "observer_fit": 688, "observer_selection": 88,
    }:
        raise AssertionError("V2 exact fit/selection roster changed")

    schema = json.loads(RELEASE_SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema.get("$id") != "phase2-observer-v4-feature-store-v2-release-schema-v1":
        raise AssertionError("V2 release schema ID changed")
    if len(schema.get("oneOf", [])) != 2:
        raise AssertionError("V2 release schema must separate generation/finalization")
    static_contract = json.loads((HERE / "STATIC_CONTRACT_V2.json").read_text(encoding="utf-8"))
    if static_contract.get("status") != (
        "SOURCE_ONLY__HOLD_REAL_GENERATION_PENDING_INDEPENDENT_V2_SOURCE_AUDIT_AND_EXACT_RELEASE"
    ):
        raise AssertionError("V2 static contract overclaims readiness")

    manifest = json.loads(SOURCE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("schema") != "phase2-observer-v4-feature-store-source-manifest-v2":
        raise AssertionError("V2 source manifest schema changed")
    if manifest.get("status") != "FROZEN_FOR_INDEPENDENT_V2_SOURCE_REAUDIT__HOLD_REAL_GENERATION":
        raise AssertionError("V2 source manifest overclaims readiness")
    files = manifest.get("files")
    if set(files or {}) != set(SOURCE_FILES):
        raise AssertionError("V2 source manifest membership changed")
    for name in SOURCE_FILES:
        if files[name] != sha256_file(HERE / name):
            raise AssertionError(f"V2 source hash changed: {name}")
    if manifest.get("immutable_v1_bindings") != v1:
        raise AssertionError("V2 source manifest V1 bindings changed")
    upstream = {name: record["sha256"] for name, record in FROZEN_BINDINGS.items()}
    if manifest.get("frozen_upstream_bindings") != upstream:
        raise AssertionError("V2 source manifest upstream bindings changed")

    injected_receipt = json.loads(
        (HERE / "INJECTED_RESOURCE_TEST_RECEIPT_V2.json").read_text(encoding="utf-8")
    )
    if injected_receipt.get("status") != (
        "PASS_V2_INJECTED_RESOURCE_ABORTS_ZERO_DECODE_AND_GROUP_QUIESCENCE"
    ):
        raise AssertionError("V2 injected receipt is not PASS")
    if injected_receipt.get("v1_hold_sha256") != V1_BINDINGS["independent_source_audit_hold"]["sha256"]:
        raise AssertionError("V2 injected receipt does not bind the V1 HOLD")

    laws = _source_laws()
    boot = _exact_environment_boot_probes()
    injected = injected_tests.verify()
    if any(
        case["decoded_rows"] != 0
        or case["decode_callable_invocations"] != 0
        or not case["descendant_dead"]
        or not case["process_group_quiescent"]
        or not case["term_sent_to_group"]
        or not case["kill_sent_to_group_after_grace"]
        or case["maximum_sample_gap_seconds"] > MAX_WATCHDOG_GAP_SECONDS
        for case in injected["cases"]
    ):
        raise AssertionError("V2 dynamic injected proof failed")
    generation = generator.preflight()
    artifacts = artifact_verifier.preflight()
    if any(generation["runtime_paths"].values()) or artifacts["feature_rows_opened_or_hashed"] != 0:
        raise AssertionError("V2 target-free preflight touched runtime artifacts")
    assert_runtime_absent()
    return {
        "schema": "phase2-observer-v4-feature-store-v2-static-preflight-receipt-v1",
        "status": "PASS_TARGET_FREE_V2_RESOURCE_SUPERVISOR_PREFLIGHT__HOLD_REAL_GENERATION_FOR_INDEPENDENT_AUDIT_AND_RELEASE",
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST_PATH),
        "source_files": files,
        "immutable_v1_bindings": v1,
        "frozen_upstream_bindings": upstream,
        "v1_hold_status": v1_audit["status"],
        "source_laws": laws,
        "exact_environment_zero_row_boot_probes": boot,
        "injected_cases": [
            {
                "label": case["label"],
                "decoded_rows": case["decoded_rows"],
                "decode_callable_invocations": case["decode_callable_invocations"],
                "descendant_dead": case["descendant_dead"],
                "group_quiescent": case["process_group_quiescent"],
                "term_then_kill": case["term_sent_to_group"] and case["kill_sent_to_group_after_grace"],
                "maximum_active_sample_gap_seconds": case["maximum_sample_gap_seconds"],
            }
            for case in injected["cases"]
        ],
        "generation_preflight_status": generation["status"],
        "artifact_preflight_status": artifacts["status"],
        "roles": {role: ROLE_BINDINGS[role]["rows"] for role in ROLE_BINDINGS},
        "caps": list(CAPS),
        "generation_release_instances": 0,
        "claims": 0,
        "real_full_rows_decoded": 0,
        "feature_rows": 0,
        "role_target_artifacts_opened_or_hashed": 0,
        "target_values_decoded": 0,
        "optimizers_constructed": 0,
        "fits": 0,
        "scores": 0,
        "generation_authorized": False,
        "independent_v2_source_reaudit_required": True,
    }


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2, sort_keys=True))
