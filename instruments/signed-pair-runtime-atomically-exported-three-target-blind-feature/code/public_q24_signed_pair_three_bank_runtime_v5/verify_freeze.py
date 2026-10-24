#!/usr/bin/env python3
"""Read-only freeze replay for lean signed-pair three-bank runtime V5."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_json_bytes(value):
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def read_canonical_json(path):
    raw = Path(path).read_bytes()
    value = json.loads(raw.decode("ascii"))
    if raw != canonical_json_bytes(value):
        raise AssertionError(Path(path).name + " is not canonical JSON")
    return value


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    manifest_path = HERE / "PACKAGE_MANIFEST.json"
    manifest = read_canonical_json(manifest_path)
    require(manifest["schema"] == "public-q24-three-bank-runtime-package-manifest-v5", "manifest schema")
    require(manifest["status"] == "FROZEN_STATIC_HOLD_NO_EXECUTION", "manifest status")
    components = manifest["components"]
    require(isinstance(components, dict) and components, "manifest components")
    for name, expected in components.items():
        require(Path(name).name == name and not Path(name).is_absolute(), "unsafe component name")
        require(sha256(HERE / name) == expected, "component drift: " + name)

    receipt = read_canonical_json(HERE / "FREEZE_RECEIPT.json")
    require(receipt["schema"] == "public-q24-three-bank-runtime-freeze-receipt-v5", "receipt schema")
    require(receipt["status"] == "FROZEN_STATIC_HOLD_NO_EXECUTION", "receipt status")
    require(
        receipt["file_sha256"]
        == {name: digest for name, digest in components.items() if name != "FREEZE_RECEIPT.json"},
        "freeze receipt/component graph mismatch",
    )
    require(manifest["freeze_receipt_sha256"] == sha256(HERE / "FREEZE_RECEIPT.json"), "receipt binding")
    require(manifest["callable_bundle_manifest_sha256"] == sha256(HERE / "CALLABLE_BUNDLE_MANIFEST.json"), "bundle binding")
    require(manifest["execution_stack_manifest_sha256"] == sha256(HERE / "EXECUTION_STACK_MANIFEST.json"), "stack binding")
    require(manifest["runner_sha256"] == sha256(HERE / "runtime_runner.py"), "runner binding")
    require(manifest["static_verification_sha256"] == sha256(HERE / "STATIC_VERIFICATION.json"), "static binding")
    require(manifest["flopscope_reset_verification_sha256"] == sha256(HERE / "FLOPSCOPE_RESET_VERIFICATION_V5.json"), "reset proof binding")
    require(manifest["prior_runtime_v4_terminal_failure_lineage_sha256"] == sha256(HERE / "V4_TERMINAL_FAILURE_LINEAGE_V5.json"), "V4 terminal binding")

    environment = dict(os.environ)
    with tempfile.TemporaryDirectory(prefix="q24-v5-freeze-pycache-", dir="/private/tmp") as cache:
        environment["PYTHONPYCACHEPREFIX"] = cache
        replay = subprocess.run(
            [sys.executable, str(HERE / "verify_static.py")],
            cwd=str(HERE),
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        reset_replay = subprocess.run(
            [sys.executable, str(HERE / "verify_reset.py")],
            cwd=str(HERE),
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    require(replay.returncode == 0, "static verifier replay failed")
    require(replay.stderr == b"", "static verifier wrote stderr")
    require(replay.stdout == (HERE / "STATIC_VERIFICATION.json").read_bytes(), "static replay bytes drift")
    require(reset_replay.returncode == 0, "reset verifier replay failed")
    require(reset_replay.stderr == b"", "reset verifier wrote stderr")
    require(
        reset_replay.stdout == (HERE / "FLOPSCOPE_RESET_VERIFICATION_V5.json").read_bytes(),
        "reset replay bytes drift",
    )
    require(not any(HERE.rglob("__pycache__")), "package pycache present")

    result = {
        "schema": "public-q24-three-bank-runtime-freeze-verification-v5",
        "status": "PASS_FROZEN_STATIC_HOLD_NO_EXECUTION",
        "package_manifest_sha256": sha256(manifest_path),
        "freeze_receipt_sha256": sha256(HERE / "FREEZE_RECEIPT.json"),
        "static_verification_sha256": sha256(HERE / "STATIC_VERIFICATION.json"),
        "static_replay_exit_code": replay.returncode,
        "static_replay_stdout_byte_identical": True,
        "reset_verification_sha256": sha256(HERE / "FLOPSCOPE_RESET_VERIFICATION_V5.json"),
        "reset_replay_exit_code": reset_replay.returncode,
        "reset_replay_stdout_byte_identical": True,
        "component_count": len(components),
        "package_pycache_absent": True,
        "access_counters": {
            "numerical_runtime_modules_imported": 0,
            "estimator_or_source_calls": 0,
            "ordered_scope_files_opened": 0,
            "row_manifests_opened": 0,
            "weight_arrays_opened": 0,
            "targets_opened": 0,
            "banks_created": 0,
            "training_runs": 0,
            "scores_computed": 0,
            "packages_built": 0,
            "submissions_made": 0,
        },
        "verdict": "GO_REQUEST_INDEPENDENT_STATIC_REVIEW__EXECUTION_REMAINS_HOLD",
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
