#!/usr/bin/env python3
"""Read-only verifier for the draft/future D6 exact-shape smoke seal.

In the checked-in HOLD state this proves the runner cannot pass its seal guard.
If a future authorized seal appears, the same verifier requires a frozen
manifest plus PASS implementation and tiny-runtime prerequisite receipts.
It never imports or calls the estimator.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "EXACT_SHAPE_SMOKE_DRAFT_MANIFEST.json"
RUNNER = HERE / "run_exact_shape_kernel_resource_smoke.py"
SCHEMA = HERE / "EXACT_SHAPE_SMOKE_TERMINAL_RECEIPT_SCHEMA.json"
SEAL = HERE / "EXACT_SHAPE_SMOKE_SEAL.json"
RESULT = HERE / "EXACT_SHAPE_SMOKE_RESULT.json"

EXPECTED = {
    HERE / "estimator.py":
        "e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58",
    HERE / "ACCOUNTING_AND_LIVENESS_MANIFEST.json":
        "bced5170098ccef0a702414a212dea606ef3c16e1c823c2ed2973888d5f0558c",
    HERE / "STATIC_VERIFICATION.json":
        "4756f8d22b032d3a96058caafab92ea18d3e76bb64a763642636b82944af2fe7",
    HERE / "SERIALIZATION_GUARD_RESULT.json":
        "9a3ffa19dab76b59443f86b2dbace7a1e2c172b66f3791f27d75d960fc01b45b",
    RUNNER:
        "50f073ee7aafd54cbe9d1625a803ea820459c372fbf74ab213592a847023751f",
    SCHEMA:
        "b967980d31c3c94cdb0aa4d60e3339f4f4c6b9a60e63e923cdc2d5116800f16d",
    MANIFEST:
        "8b1e09d9255057886a353c98ca8364c9da39e6a6d4c48c1a25cc012e9907c111",
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


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d6(m: int, k: int, n: int) -> int:
    return (
        7**4 * mm(m // 16, k // 16, n // 16)
        + sum(
            7**level * 11 * (m * k + k * n) // 4 ** (level + 1)
            for level in range(4)
        )
        + sum(
            7**level * 7 * (m * n) // 4 ** (level + 1)
            for level in range(4)
        )
    )


def verify_runner_ast() -> None:
    tree = ast.parse(RUNNER.read_text())
    assert isinstance(tree.body[-1], ast.If)
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    assert {
        "verify_seal", "load_candidate", "run_smoke", "finish_once", "main"
    } <= functions.keys()
    main_calls = [
        ast.unparse(node.func)
        for node in ast.walk(functions["main"])
        if isinstance(node, ast.Call)
    ]
    assert main_calls[:2] == ["verify_seal", "run_smoke"]
    verify_text = ast.unparse(functions["verify_seal"])
    assert verify_text.index("if not SEAL.is_file()") < verify_text.index(
        "json.loads(SEAL.read_text())"
    )
    assert "load_candidate" not in verify_text
    run_text = ast.unparse(functions["run_smoke"])
    assert run_text.index("load_candidate(SOURCE)") < run_text.index(
        "fnp.empty(ARENA_ELEMENTS"
    )
    assert "Estimator(" not in run_text
    assert ".predict(" not in run_text
    runner_text = RUNNER.read_text()
    for forbidden in (
        "whestbench",
        "MLP(",
        "np.load(",
        "target",
        "requests",
        "urllib",
        "socket",
    ):
        if forbidden == "target":
            # Access declarations and the docstring intentionally contain it.
            continue
        assert forbidden not in runner_text, forbidden
    assert loaded_json_spellings(RUNNER) == set()


def main() -> None:
    for path, expected in EXPECTED.items():
        assert path.is_file(), path
        assert sha256(path) == expected, (path, sha256(path), expected)
    verify_runner_ast()
    manifest = json.loads(MANIFEST.read_text())
    schema = json.loads(SCHEMA.read_text())
    assert manifest["schema"] == "p2r0c1d6-exact-shape-smoke-manifest-v1"
    assert schema["$id"] == "p2r0c1d6-exact-shape-smoke-terminal-receipt-schema-v1"
    assert manifest["bindings"]["runner_sha256"] == EXPECTED[RUNNER]
    assert manifest["bindings"]["terminal_receipt_schema_sha256"] == EXPECTED[SCHEMA]
    assert manifest["execution_environment"] == {
        "flopscope_version_exact": "0.12.0",
        "numpy_version_exact": "2.4.6",
        "threads": 2,
        "venv": "../data_gate/.venv",
        "version_mismatch_disposition": "FAIL_TERMINAL_NO_RETRY",
    }
    roster = (
        [(1024, 1024, 1024 * layer) for layer in range(1, 16) for _ in range(2)]
        + [(1024, 1024 * layer, 1024) for layer in range(1, 15) for _ in range(2)]
    )
    assert len(roster) == manifest["roster"]["total_channel_contractions"] == 58
    assert sum(mm(*shape) for shape in roster) == 966_086_623_232
    assert sum(d6(*shape) for shape in roster) == 591_553_830_912
    gates = manifest["gates"]
    assert gates["candidate_roster_flops_exact"] == 591_553_830_912
    assert gates["direct_roster_flops_exact"] == 966_086_623_232
    assert gates["cold_view_binding_flops_exact"] == 1_919_598_592
    assert gates["cold_view_binding_reshape_calls_exact"] == 52
    assert gates["candidate_roster_inventory_exact"] == {
        "add": 1624,
        "copyto": 3248,
        "matmul": 58,
        "subtract": 1856,
    }
    assert sum(gates["candidate_roster_inventory_exact"].values()) == 6786
    assert gates["candidate_minus_direct_residual_seconds_max"] == 0.010
    assert gates["process_rss_increment_bytes_at_most"] == 3_913_531_392
    assert gates["absolute_process_rss_bytes_strictly_below"] == 7 * 1024**3
    assert gates["pooled_rms_delta_vs_direct_float32"] == 1.714909904361275e-6
    assert gates["maximum_absolute_delta_vs_direct_float32"] == 8.574549521806375e-6
    assert gates["candidate_full_calls_35_279_35_exact"] == [
        1_319_635_780_983,
        1_356_359_358_984,
        1_317_713_029_632,
    ]
    assert gates["incumbent_minus_candidate_projected"] == [
        372_849_123_328,
        374_768_721_920,
        374_768_721_920,
    ]
    access = manifest["access_declarations"]
    assert access == {
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "estimator_fixtures_opened": [],
        "estimator_predict_calls": 0,
        "training": False,
        "production": False,
        "submission": False,
        "network": False,
    }
    assert not RESULT.exists()
    assert tuple(HERE.glob("*.npy")) == ()

    if not SEAL.exists():
        assert manifest["status"] == (
            "DRAFT_HOLD_TINY_EXACT_RUNTIME_RESIDUAL_GATE_FAILED"
        )
        assert manifest["seal"] == {
            "authorized": False,
            "exists": False,
            "path": "EXACT_SHAPE_SMOKE_SEAL.json",
        }
        prerequisites = manifest["prerequisites"]
        assert prerequisites["independent_implementation_audit_status"] == (
            "FAIL_RESIDUAL_GATE_TERMINAL_CLOSE"
        )
        assert prerequisites["independent_implementation_audit_sha256"] == (
            "5ae738215f76b008384346d58fdb6cef0db39c61a056a99f6b7f8f05d907807a"
        )
        assert prerequisites[
            "independent_implementation_audit_verification_status"
        ] == "PASS_VERIFIED_TERMINAL_CLOSE_D6_RESIDUAL_GATE_FAILED"
        assert prerequisites["tiny_exact_runtime_smoke_status"] == (
            "FAIL_RESIDUAL_GATE_ADVERSE_EVIDENCE"
        )
        assert prerequisites["tiny_exact_runtime_smoke_sha256"] == (
            "c08ce3f9a6ed054f2a3c5a8942ca0e4b93970ba5987d4699c6171765538ca7f8"
        )
        assert prerequisites["tiny_residual_delta_seconds"] == (
            0.04961721133440733
        )
        assert prerequisites["tiny_residual_delta_seconds"] > 0.010
        print("HOLD_EXACT_SHAPE_SMOKE_DRAFT_NOT_SEALED_TINY_RESIDUAL_FAIL")
        return

    seal = json.loads(SEAL.read_text())
    assert seal["status"] == "FROZEN_ONE_SHOT_EXACT_SHAPE_SMOKE_AUTHORIZED"
    assert manifest["status"] == "FROZEN_BEFORE_ONE_SHOT_EXACT_SHAPE_SMOKE"
    assert manifest["seal"]["authorized"] is True
    assert manifest["seal"]["exists"] is True
    for relative, expected in seal["sha256"].items():
        path = (HERE / relative).resolve()
        assert path.is_file(), path
        assert sha256(path) == expected, (relative, sha256(path), expected)
    prerequisites = manifest["prerequisites"]
    assert prerequisites["independent_implementation_audit_status"] == "PASS"
    assert prerequisites["tiny_exact_runtime_smoke_status"] == "PASS"
    assert float(prerequisites["tiny_residual_delta_seconds"]) <= 0.010
    print("PASS_FROZEN_ONE_SHOT_EXACT_SHAPE_SMOKE_SEAL")


if __name__ == "__main__":
    main()
