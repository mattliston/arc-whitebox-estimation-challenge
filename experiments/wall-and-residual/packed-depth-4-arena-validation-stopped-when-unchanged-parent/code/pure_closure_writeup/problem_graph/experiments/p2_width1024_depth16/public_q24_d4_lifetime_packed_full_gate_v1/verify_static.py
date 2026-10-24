#!/usr/bin/env python3
"""No-estimator static verifier for the frozen packed-D4 full gate."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
STATIC = P2 / "public_q24_d4_lifetime_packed_reinvestment_v1"
D4_DIR = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1"
D15_DIR = P2 / "p2r0c1d15_cp3_arena_depth5_lifetime_packed_v1"

SOURCE = HERE / "estimator.py"
BUILDER = HERE / "build_candidate.py"
RUNNER = HERE / "run_full_gate.py"
SCAFFOLD = HERE / "RELEASE_SCAFFOLD.json"
FREEZE = HERE / "GATE_FREEZE.json"
RELEASE = HERE / "FULL_GATE_RELEASE.json"
RESULT = HERE / "FULL_GATE_RESULT.json"
LOCK = HERE / "FULL_GATE_RUN.lock"
PACKED_KERNEL = STATIC / "packed_d4_kernel.py"
PACKED_RECEIPT = STATIC / "PACKED_D4_STATIC_RECEIPT.json"
D4_SOURCE = D4_DIR / "estimator.py"
D4_RESULT = D4_DIR / "D4_CANDIDATE_RESULT.json"
D15_RESULT = D15_DIR / "D15_CANDIDATE_RESULT.json"

EXPECTED = {
    SOURCE: "982775b3296709156e1d05c24d703a0982db8bab6dce83dc16744fb09361b066",
    BUILDER: "41e3a047b9433cd721089cdea872ba92127c29bfaf4e2e7997e999cb1e27895b",
    PACKED_KERNEL: "bc23fe4f8a46bd194390b53523d63d5ab136fc1feb0240d589fcaf3fad681a37",
    PACKED_RECEIPT: "832220c4c9467d8d57569649198e2465e97bc908ebb96103e549e683964c4b51",
    D4_SOURCE: "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e",
    D4_RESULT: "6d8e7cf98d45b1c667d7d16c373cf9935f96d1f07e44674ca8f9584d51c111d1",
    D15_RESULT: "7571c8948ef992643642fb39731295b341fea9e7929aa8f9e53cbbe5c4623964",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def literal_assignment(tree: ast.Module, name: str):
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise RuntimeError((name, len(matches)))
    return ast.literal_eval(matches[0].value)


def main() -> None:
    checks: dict[str, bool] = {}
    for path, expected in EXPECTED.items():
        checks[f"sha256::{path.name}::{path.parent.name}"] = sha256(path) == expected

    builder = load("packed_d4_full_gate_builder_static", BUILDER)
    checks["mechanical_rebuild_byte_identical"] = builder.build() == SOURCE.read_bytes()

    source_text = SOURCE.read_text()
    source_tree = ast.parse(source_text)
    push_specs = literal_assignment(source_tree, "_CP3_PUSH_ARENA_VIEW_SPECS")
    read_specs = literal_assignment(source_tree, "_CP3_READ_ARENA_VIEW_SPECS")
    arena_elements = literal_assignment(source_tree, "_CP3_WINOGRAD_ARENA_ELEMENTS")
    checks["arena_elements_exact"] = arena_elements == 511_393_792
    checks["twenty_four_push_views"] = len(push_specs) == 24
    checks["twenty_four_read_views"] = len(read_specs) == 24
    checks["all_views_within_arena"] = all(
        0 <= int(start) <= int(stop) <= arena_elements
        for specs in (push_specs, read_specs)
        for start, stop, _shape in specs
    )
    checks["two_literal_independent_read_allocations"] = (
        source_text.count("fnp.empty((width, width), dtype=fnp.float32)") == 2
    )
    checks["no_product_tail_read_output_alias"] = (
        "read_output_start" not in source_text
        and "first, second = read_outputs" in source_text
    )
    checks["packed_artifact_not_runtime_imported"] = "packed_d4_kernel" not in source_text

    runner = load("packed_d4_full_gate_runner_static", RUNNER)
    checks["call_order_35_279_35"] = runner.CALL_ORDER == (35, 279, 35)
    checks["all_five_public_components"] = runner.PUBLIC_COMPONENTS == (
        "host",
        "native_raw20",
        "native_sigma",
        "native_j_correction",
        "final_public_prediction",
    )
    checks["candidate_bills_exact"] = tuple(
        runner.expected_inventory("candidate", ordinal)["flops"]
        for ordinal in range(3)
    ) == (1_319_838_156_151, 1_356_595_288_584, 1_317_948_959_232)
    checks["candidate_total_calls_exact"] = tuple(
        runner.expected_inventory("candidate", ordinal)["total_calls"]
        for ordinal in range(3)
    ) == (15_414, 15_341, 15_331)
    checks["cold_call_delta_exact"] = (
        runner.expected_inventory("candidate", 0)["calls"]["reshape"]
        == runner.expected_inventory("parent", 0)["calls"]["reshape"] - 4
        and runner.expected_inventory("candidate", 0)["calls"]["empty"]
        == runner.expected_inventory("parent", 0)["calls"]["empty"] + 2
    )
    checks["hot_inventory_unchanged"] = all(
        runner.expected_inventory("candidate", ordinal)
        == runner.expected_inventory("parent", ordinal)
        for ordinal in (1, 2)
    )
    runner_text = RUNNER.read_text()
    checks["parent_then_candidate_separate_workers"] = (
        'parent = run_worker("parent", staging, lock_sha256)' in runner_text
        and 'candidate = run_worker("candidate", staging, lock_sha256)' in runner_text
        and runner_text.index('parent = run_worker("parent", staging, lock_sha256)')
        < runner_text.index('candidate = run_worker("candidate", staging, lock_sha256)')
    )
    checks["release_required_before_lock_and_fixture"] = (
        runner_text.index("release = require_release()")
        < runner_text.index("exclusive_json(LOCK, lock_payload)")
        and runner_text.index("exclusive_json(LOCK, lock_payload)")
        < runner_text.index('parent = run_worker("parent", staging, lock_sha256)')
    )
    checks["all_component_byte_comparison_present"] = (
        "for name in PUBLIC_COMPONENTS:" in runner_text
        and "component_parity[name] = byte_equal_files(left, right)" in runner_text
    )
    checks["all_component_repeat_comparison_present"] = (
        'repeat[phase] = {' in runner_text
        and 'worker["trials"][0]["components"][name]["content_sha256"]' in runner_text
        and 'worker["trials"][2]["components"][name]["content_sha256"]' in runner_text
    )
    checks["rss_and_residual_gates_exact"] = (
        runner.MAX_RSS_BYTES == 8 * 2**30
        and runner.MAX_RESIDUAL_SECONDS == 0.4
    )
    checks["d15_full_result_fail_terminal"] = (
        json.loads(D15_RESULT.read_text()).get("status") == "FAIL_TERMINAL"
    )
    checks["d15_role_static_only"] = (
        runner.D15_PRECEDENT_ROLE
        == "STATIC_LIFETIME_ONLY__FULL_D15_RESULT_IS_FAIL_TERMINAL"
    )

    scaffold = json.loads(SCAFFOLD.read_text()) if SCAFFOLD.is_file() else {}
    freeze = json.loads(FREEZE.read_text()) if FREEZE.is_file() else {}
    checks["release_scaffold_hold"] = (
        scaffold.get("status") == "HOLD_AWAITING_FRESH_INDEPENDENT_GO_HOLD"
        and scaffold.get("independent_final_go") is False
    )
    checks["gate_freeze_hold"] = (
        freeze.get("status") == "FROZEN_AWAITING_FRESH_INDEPENDENT_GO_HOLD"
    )
    checks["release_absent"] = not RELEASE.exists()
    checks["result_absent"] = not RESULT.exists()
    checks["lock_absent"] = not LOCK.exists()

    payload = {
        "schema": "packed-d4-full-gate-static-verification-v1",
        "status": "PASS_STATIC_NO_EXECUTION" if all(checks.values()) else "FAIL_STATIC",
        "checks": checks,
        "bindings": {
            "source_sha256": sha256(SOURCE),
            "builder_sha256": sha256(BUILDER),
            "runner_sha256": sha256(RUNNER),
            "release_scaffold_sha256": sha256(SCAFFOLD) if SCAFFOLD.is_file() else None,
            "gate_freeze_sha256": sha256(FREEZE) if FREEZE.is_file() else None,
            "packed_kernel_sha256": sha256(PACKED_KERNEL),
            "packed_receipt_sha256": sha256(PACKED_RECEIPT),
            "d4_source_sha256": sha256(D4_SOURCE),
            "d4_result_sha256": sha256(D4_RESULT),
            "d15_fail_terminal_result_sha256": sha256(D15_RESULT),
        },
        "scope": {
            "estimator_imports": 0,
            "estimator_calls": 0,
            "fixtures_opened": [],
            "targets_read": 0,
            "scores_read": 0,
            "network": False,
            "submission": False,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "PASS_STATIC_NO_EXECUTION":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
