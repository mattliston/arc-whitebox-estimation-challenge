#!/usr/bin/env python3
"""Zero-call serialization/import guard for the P2R0C1D6 preparation.

This deliberately does not import the estimator.  It prevents a repeat of the
D5 common-driver lowercase-JSON spelling failure, proves every current JSON
artifact is canonical and round-trippable, and confirms that all executable
preparation modules are protected by ``if __name__ == '__main__'`` guards.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RESULT = HERE / "SERIALIZATION_GUARD_RESULT.json"

EXPECTED = {
    "build_candidate.py":
        "302e2ca5f3c3a437ed3fb348dae4fc5318efc7e5ba59387793d918270298a8dc",
    "estimator.py":
        "e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58",
    "SOURCE_DIFF.patch":
        "73dd4e286bf09160abc41102c080607aabbf981c8bd3822ce37250e9a663df58",
    "ACCOUNTING_AND_LIVENESS_MANIFEST.json":
        "bced5170098ccef0a702414a212dea606ef3c16e1c823c2ed2973888d5f0558c",
    "BUILD_RECEIPT.json":
        "7a7c4f07975cca45a413656b44017571ccec7b65d4784ade0bfa54b365d28c2a",
    "verify_static_liveness.py":
        "b6cef81fab76d6ca63a47554a79c5ba4970c26fda614ebc738a5f156a6d3fcd1",
    "STATIC_VERIFICATION.json":
        "4756f8d22b032d3a96058caafab92ea18d3e76bb64a763642636b82944af2fe7",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loaded_json_spellings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in {"false", "true", "null"}
    }


def guarded_module(path: Path) -> bool:
    tree = ast.parse(path.read_text())
    if not tree.body or not isinstance(tree.body[-1], ast.If):
        return False
    test = tree.body[-1].test
    return (
        isinstance(test, ast.Compare)
        and isinstance(test.left, ast.Name)
        and test.left.id == "__name__"
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq)
        and len(test.comparators) == 1
        and isinstance(test.comparators[0], ast.Constant)
        and test.comparators[0].value == "__main__"
    )


def main() -> None:
    for relative, expected in EXPECTED.items():
        path = HERE / relative
        assert path.is_file(), path
        assert sha256(path) == expected, (relative, sha256(path), expected)

    scripts = (
        HERE / "build_candidate.py",
        HERE / "verify_static_liveness.py",
        HERE / "estimator.py",
    )
    for script in scripts:
        assert loaded_json_spellings(script) == set(), script
        compile(script.read_text(), str(script), "exec")
    assert guarded_module(HERE / "build_candidate.py")
    assert guarded_module(HERE / "verify_static_liveness.py")

    json_files = (
        HERE / "ACCOUNTING_AND_LIVENESS_MANIFEST.json",
        HERE / "BUILD_RECEIPT.json",
        HERE / "STATIC_VERIFICATION.json",
    )
    decoded = {}
    for path in json_files:
        payload = json.loads(path.read_text())
        canonical = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        assert canonical == path.read_text(), path
        assert json.loads(canonical) == payload
        decoded[path.name] = payload

    manifest = decoded["ACCOUNTING_AND_LIVENESS_MANIFEST.json"]
    build = decoded["BUILD_RECEIPT.json"]
    static = decoded["STATIC_VERIFICATION.json"]
    assert manifest["execution_authority"]["fixture_call"] is False
    assert manifest["execution_authority"]["common_driver_call"] is False
    assert manifest["execution_authority"]["exact_shape_smoke"] is False
    assert manifest["execution_authority"]["tiny_synthetic_kernel_smoke"] is False
    assert manifest["execution_authority"]["targets_scores_names_board"] is False
    assert manifest["execution_authority"]["production_submission_network"] is False
    assert build["estimator_imported"] is False
    assert build["estimator_calls"] == 0
    assert static["estimator_imported"] is False
    assert static["estimator_calls"] == 0

    # Representative future receipt skeleton: use native Python JSON values,
    # serialize, and round-trip before any runner is allowed to be prepared.
    skeleton = {
        "schema": "p2r0c1d6-future-gate-skeleton-v1",
        "node": "P2R0C1D6",
        "status": "SERIALIZATION_GUARD_ONLY_NOT_A_LAUNCH",
        "candidate_source_sha256": EXPECTED["estimator.py"],
        "reference_estimator_calls": 0,
        "candidate_estimator_calls": 0,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "training": False,
        "submission": False,
        "production": False,
        "network": False,
        "result": None,
    }
    encoded = json.dumps(skeleton, indent=2, sort_keys=True) + "\n"
    roundtrip = json.loads(encoded)
    assert roundtrip == skeleton
    assert '"board_opened": false' in encoded
    assert '"result": null' in encoded

    forbidden_runtime_names = (
        "EXACT_SHAPE_SMOKE_RESULT.json",
        "KERNEL_SMOKE_RESULT.json",
        "COMMON_DRIVER_RESULT.json",
        "RECOVERY_DESCENDANT_RESULT.json",
        "LAUNCH_SEAL.json",
    )
    assert all(not (HERE / name).exists() for name in forbidden_runtime_names)
    assert tuple(HERE.glob("*.npy")) == ()

    result = {
        "schema": "p2r0c1d6-serialization-guard-v1",
        "status": "PASS_SERIALIZATION_GUARD_ZERO_CALL",
        "node": "P2R0C1D6",
        "candidate_source_sha256": EXPECTED["estimator.py"],
        "canonical_json_artifacts": [path.name for path in json_files],
        "lowercase_python_json_names_loaded": [],
        "builder_main_guard": True,
        "static_verifier_main_guard": True,
        "future_receipt_skeleton_roundtrip": True,
        "estimator_imported": False,
        "estimator_calls": 0,
        "exact_shape_calls": 0,
        "fixture_calls": 0,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "training": False,
        "submission": False,
        "production": False,
        "network": False,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
