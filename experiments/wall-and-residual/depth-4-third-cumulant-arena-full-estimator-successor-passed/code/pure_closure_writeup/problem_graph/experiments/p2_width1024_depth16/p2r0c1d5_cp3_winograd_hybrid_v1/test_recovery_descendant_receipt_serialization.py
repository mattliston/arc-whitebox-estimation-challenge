#!/usr/bin/env python3
"""Target-free, zero-estimator-call receipt serialization smoke."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_recovery_descendant_candidate_gate.py"
BUILDER = HERE / "build_recovery_descendant_runner.py"
RESULT = HERE / "RECOVERY_DESCENDANT_RESULT.json"
CANDIDATE_OUTPUTS = tuple(HERE.glob("recovery_candidate_rank*.npy"))


def loaded_json_spellings(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in {"false", "true", "null"}
    }


def main() -> None:
    assert not RESULT.exists()
    assert CANDIDATE_OUTPUTS == ()
    assert loaded_json_spellings(RUNNER) == set()
    assert loaded_json_spellings(BUILDER) == set()

    tree = ast.parse(RUNNER.read_text())
    top_level_calls = [node for node in tree.body if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)]
    assert top_level_calls == []
    final_guard = tree.body[-1]
    assert isinstance(final_guard, ast.If)

    spec = importlib.util.spec_from_file_location("p2r0c1d5r1_serialization_smoke", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.load_module = lambda _path: (_ for _ in ()).throw(AssertionError("Estimator load forbidden"))
    module.load_fixture = lambda _rank, _manifest: (_ for _ in ()).throw(AssertionError("fixture load forbidden"))

    payload = module.build_result(
        "SERIALIZATION_TEST_ONLY",
        [],
        {"conjunctive_pass": True},
        None,
        {
            "candidate_source_sha256": module.SOURCE_SHA256,
            "test_only": "no estimator call",
        },
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    decoded = json.loads(encoded)
    assert decoded["schema"] == "p2r0c1d5r1-target-free-recovery-descendant-gate-v1"
    assert decoded["node"] == "P2R0C1D5R1"
    assert decoded["original_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
    assert decoded["reference_estimator_calls_in_descendant"] == 0
    assert decoded["board_opened"] is False
    assert decoded["submission"] is False
    assert decoded["training"] is False
    assert decoded["network"] is False
    assert decoded["target_columns_opened"] == []
    assert decoded["scores_opened"] == []
    assert decoded["names_opened"] == []
    assert not RESULT.exists()
    assert tuple(HERE.glob("recovery_candidate_rank*.npy")) == ()
    print("PASS_RECOVERY_DESCENDANT_RECEIPT_SERIALIZATION_ZERO_CALL")


if __name__ == "__main__":
    main()
