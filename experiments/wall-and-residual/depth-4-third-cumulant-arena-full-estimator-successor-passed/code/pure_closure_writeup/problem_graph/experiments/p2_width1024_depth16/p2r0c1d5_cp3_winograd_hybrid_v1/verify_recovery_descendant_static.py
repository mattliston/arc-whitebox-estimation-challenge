#!/usr/bin/env python3
"""Static verifier for the zero-reference-call P2R0C1D5R1 descendant."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ORIGINAL = HERE / "run_common_driver_gate.py"
RUNNER = HERE / "run_recovery_descendant_candidate_gate.py"
BUILDER = HERE / "build_recovery_descendant_runner.py"
MANIFEST = HERE / "RECOVERY_DESCENDANT_MANIFEST.json"
EXPECTED_SHA256 = {
    "run_common_driver_gate.py": "e0acba52fa730bc6d448ce873e953f0ddf50c6a908f5c8e69c5b02ff14a5ec0f",
    "run_recovery_descendant_candidate_gate.py": "42ca250d6686e5e48b2730063d22dfd3407f870a16dab59be0929f6a5d81928f",
    "build_recovery_descendant_runner.py": "5e51e00e52a4e9d45c93ffba5369cb3dcf86360ee2374f489b0ce134917b51df",
    "RECOVERY_DESCENDANT_MANIFEST.json": "cafa3d88a46d4fc45dca6cfe52acaf148bb9123e5e88b31d58fb37d0135e24c1",
    "RECOVERED_REFERENCE_EVIDENCE.json": "cbf63a2c30cc2e1f8eca4f4506ad7510de10b72bd721d1bdf54640458f1fcfb6",
    "estimator.py": "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5",
}
SCIENCE_CONSTANTS = (
    "SOURCE_SHA256",
    "CALL_ORDER",
    "EXPECTED_FLOPS",
    "EXPECTED_REFERENCE_FLOPS",
    "EXACT_SAVING",
    "PHASE2_BUDGET",
    "THREADS",
    "MAX_RSS_BYTES",
    "MAX_WALL_SECONDS",
    "MAX_RESIDUAL_SECONDS",
    "MAX_DEPLOYMENT_RESIDUAL_SECONDS",
    "MAX_POOLED_FINAL_RMS_DELTA",
    "MAX_FINAL_ABS_DELTA",
)
UNCHANGED_FUNCTIONS = (
    "sha256",
    "array_sha256",
    "resident_bytes",
    "write_json_once",
    "write_array_once",
    "load_fixture",
    "layer_drift",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def functions(tree: ast.Module) -> dict[str, ast.FunctionDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}


def assignment_nodes(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node.value
    return result


class NormalizePlumbing(ast.NodeTransformer):
    def visit_Constant(self, node: ast.Constant):
        if node.value == "p2r0c1d5r1_candidate_gate":
            return ast.copy_location(ast.Constant("p2r0c1d5_candidate_gate"), node)
        if node.value == "p2r0c1d5r1_candidate_":
            return ast.copy_location(ast.Constant("p2r0c1d5_candidate_"), node)
        return node


def normalized_dump(node: ast.AST) -> str:
    node = NormalizePlumbing().visit(ast.fix_missing_locations(node))
    return ast.dump(node, include_attributes=False)


def main() -> None:
    for relative, expected in EXPECTED_SHA256.items():
        assert sha256((HERE / relative).resolve()) == expected, relative
    original_tree = ast.parse(ORIGINAL.read_text())
    runner_tree = ast.parse(RUNNER.read_text())
    builder_tree = ast.parse(BUILDER.read_text())
    original_functions = functions(original_tree)
    runner_functions = functions(runner_tree)
    original_assignments = assignment_nodes(original_tree)
    runner_assignments = assignment_nodes(runner_tree)

    for name in SCIENCE_CONSTANTS:
        assert ast.dump(runner_assignments[name], include_attributes=False) == ast.dump(
            original_assignments[name], include_attributes=False
        ), name
    for name in UNCHANGED_FUNCTIONS:
        assert normalized_dump(runner_functions[name]) == normalized_dump(original_functions[name]), name
    assert normalized_dump(runner_functions["load_module"]) == normalized_dump(original_functions["load_module"])
    assert normalized_dump(runner_functions["main"]) == normalized_dump(original_functions["main"])

    forbidden = {
        node.id
        for tree in (runner_tree, builder_tree)
        for node in ast.walk(tree)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in {"false", "true", "null"}
    }
    assert forbidden == set()
    runner_source = RUNNER.read_text()
    assert 'MANIFEST = HERE / "RECOVERY_DESCENDANT_MANIFEST.json"' in runner_source
    assert 'SEAL = HERE / "RECOVERY_DESCENDANT_LAUNCH_SEAL.json"' in runner_source
    assert 'REFERENCE_RESULT = HERE / "RECOVERED_REFERENCE_EVIDENCE.json"' in runner_source
    assert 'RESULT = HERE / "RECOVERY_DESCENDANT_RESULT.json"' in runner_source
    assert runner_source.count("module.Estimator()") == 1
    assert runner_source.count("estimator.predict(") == 1
    assert '"node": "P2R0C1D5R1"' in runner_source
    assert '"original_protocol_status": "FAIL_TERMINAL_NO_RETRY"' in runner_source
    assert '"reference_estimator_calls_in_descendant": 0' in runner_source
    assert '"board_opened": False' in runner_source
    assert '"submission": False' in runner_source
    assert '"training": False' in runner_source
    assert '"network": False' in runner_source
    assert 'reference_result.get("call_order", ())' in runner_source
    assert 'trial["ordinal"] != ordinal or trial["role_rank"] != CALL_ORDER[ordinal]' in runner_source
    assert 'trial["output_file"] != path.name' in runner_source
    assert 'reference_result.get("access_declarations") != {' in runner_source

    manifest = json.loads(MANIFEST.read_text())
    assert manifest["candidate_node"] == "P2R0C1D5R1"
    assert manifest["parent_protocol_status"] == "FAIL_TERMINAL_NO_RETRY"
    assert manifest["call_order"] == [35, 279, 35]
    assert manifest["process_protocol"]["reference_estimator_calls_in_descendant"] == 0
    assert manifest["process_protocol"]["original_candidate_calls"] == 0
    assert manifest["process_protocol"]["descendant_candidate_calls_at_freeze"] == 0
    assert manifest["candidate_source_sha256"] == EXPECTED_SHA256["estimator.py"]
    assert manifest["candidate_runner_sha256"] == EXPECTED_SHA256[RUNNER.name]
    assert manifest["recovered_reference_evidence_sha256"] == EXPECTED_SHA256["RECOVERED_REFERENCE_EVIDENCE.json"]
    assert manifest["expected_incumbent_flops"] == [1_692_484_904_311, 1_731_128_080_904, 1_692_481_751_552]
    assert manifest["expected_candidate_flops"] == [1_320_876_655_991, 1_359_519_832_584, 1_320_873_503_232]
    assert manifest["amended_exact_net_saving"] == 371_608_248_320
    assert manifest["scope_guards"] == {
        "weights_only": True,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "network": False,
        "training": False,
    }
    assert not (HERE / "REFERENCE_RESULT.json").exists()
    assert not (HERE / "COMMON_DRIVER_RESULT.json").exists()
    assert not (HERE / "RECOVERY_DESCENDANT_RESULT.json").exists()
    assert not (HERE / "RECOVERY_DESCENDANT_LAUNCH_SEAL.json").exists()
    assert not any(HERE.glob("candidate_rank*.npy"))
    assert not any(HERE.glob("recovery_candidate_rank*.npy"))
    print("PASS_RECOVERY_DESCENDANT_STATIC_PRESEAL")


if __name__ == "__main__":
    main()
