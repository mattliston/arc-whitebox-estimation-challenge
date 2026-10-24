#!/usr/bin/env python3
"""Independent static/runtime-receipt verifier for terminal P2R0C1D6 audit.

The candidate is parsed, never imported or called.  This verifier binds the
sealed design, production source, manifests, isolated tiny semantic authority,
and same-source residual diagnostic.  It opens no estimator fixture, target,
score, name, board, exact-shape operand, common driver, or network resource.
"""

from __future__ import annotations

import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import statistics


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D5 = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1"
TINY = P2 / "p2r0c1d6_tiny_prototype_v1"
DATA_GATE = P2 / "data_gate"
FLOPSCOPE = (
    DATA_GATE / ".venv/lib/python3.11/site-packages/flopscope"
)
CANDIDATE = HERE / "estimator.py"
MANIFEST = HERE / "ACCOUNTING_AND_LIVENESS_MANIFEST.json"
AUDIT = HERE / "INDEPENDENT_REDTEAM_D6_IMPLEMENTATION_AUDIT.md"
DIAGNOSTIC = HERE / "INDEPENDENT_TINY_RESIDUAL_DIAGNOSTIC.json"
RESULT = HERE / "INDEPENDENT_D6_IMPLEMENTATION_AUDIT_VERIFICATION.json"


EXPECTED = {
    D5 / "D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md":
        "808a41b8fa1429e1efaf978cd3f17f18ecf7ae7f21c20a9359546922b3696da7",
    D5 / "INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md":
        "df76169237166cf685ca9e0474643d1d668dcc2ae762f5e9e57af0e6b8afd6f2",
    HERE / "build_candidate.py":
        "302e2ca5f3c3a437ed3fb348dae4fc5318efc7e5ba59387793d918270298a8dc",
    CANDIDATE:
        "e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58",
    HERE / "SOURCE_DIFF.patch":
        "73dd4e286bf09160abc41102c080607aabbf981c8bd3822ce37250e9a663df58",
    MANIFEST:
        "bced5170098ccef0a702414a212dea606ef3c16e1c823c2ed2973888d5f0558c",
    HERE / "BUILD_RECEIPT.json":
        "7a7c4f07975cca45a413656b44017571ccec7b65d4784ade0bfa54b365d28c2a",
    HERE / "verify_static_liveness.py":
        "b6cef81fab76d6ca63a47554a79c5ba4970c26fda614ebc738a5f156a6d3fcd1",
    HERE / "STATIC_VERIFICATION.json":
        "4756f8d22b032d3a96058caafab92ea18d3e76bb64a763642636b82944af2fe7",
    HERE / "verify_serialization_guard.py":
        "09b6b953f4386c6ba213f221bd595ff107ad39d2ec62ce6128b72557f2502a1b",
    HERE / "SERIALIZATION_GUARD_RESULT.json":
        "9a3ffa19dab76b59443f86b2dbace7a1e2c172b66f3791f27d75d960fc01b45b",
    HERE / "run_independent_tiny_residual_diagnostic.py":
        "4e1f58c73d5d7d7fcc1622fdfb0c62f4f1d495f4f5d67a0a8ab8f00e49d45102",
    DIAGNOSTIC:
        "c08ce3f9a6ed054f2a3c5a8942ca0e4b93970ba5987d4699c6171765538ca7f8",
    AUDIT:
        "5ae738215f76b008384346d58fdb6cef0db39c61a056a99f6b7f8f05d907807a",
    TINY / "TINY_PROTOTYPE_FINAL_SEAL_002.json":
        "2f2f8d26a4892b727ce7be43664bcf001e48fa61f35ff184b98f1d6f891b85ed",
    TINY / "TINY_PROTOTYPE_RECEIPT.json":
        "fcddb6013402ae896aeda204705be543b6811c747bd79092c5feb99b417ae20c",
    TINY / "tiny_d6_kernel.py":
        "a09710a67b0f2e571b82b62909d5f699f2993507d7bd8c739d30c8d0d397a9f1",
    TINY / "verify_tiny_prototype_final_v2.py":
        "292fdb157c45eef934815e62aec858c841758a86ca0e1c84599af84766fb25db",
    FLOPSCOPE / "_array_ops.py":
        "e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c",
    FLOPSCOPE / "_pointwise.py":
        "7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7",
    FLOPSCOPE / "_validation.py":
        "6a047b60aaa5954b9301a15a6e3e5f646d10a95ace60c6b47c54cd9f2b52531f",
    FLOPSCOPE / "_einsum.py":
        "0f25b8f5709c5a96404f20c38afc9b705568ab2654ab6c5fc81f3de292b73284",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


class CallOrder(ast.NodeVisitor):
    def __init__(self) -> None:
        self.calls: list[ast.Call] = []

    def visit_Call(self, node: ast.Call) -> None:
        self.calls.append(node)
        self.generic_visit(node)


def fnp_call_order(node: ast.AST) -> list[tuple[str, str]]:
    visitor = CallOrder()
    visitor.visit(node)
    result = []
    for call in visitor.calls:
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "fnp"
        ):
            result.append((call.func.attr, ast.unparse(call)))
    return result


def assert_out_calls(node: ast.AST, names: set[str]) -> None:
    for call in ast.walk(node):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Attribute):
            continue
        if not isinstance(call.func.value, ast.Name) or call.func.value.id != "fnp":
            continue
        if call.func.attr in names:
            assert any(keyword.arg == "out" for keyword in call.keywords), ast.unparse(call)


def verify_layout(layout: dict, expected_elements: int) -> None:
    regions = layout["physical_regions"]
    assert len(regions) == 15
    cursor = 0
    by_name = {}
    for region in regions:
        assert region["start"] == cursor
        assert region["elements"] == region["stop"] - region["start"] > 0
        by_name[region["name"]] = region
        cursor = region["stop"]
    assert cursor == expected_elements == layout["physical_elements"]
    assert layout["physical_bytes_float32"] == 4 * expected_elements
    assert layout["physical_regions_contiguous_and_nonoverlapping"] is True
    views = layout["view_specs"]
    assert len(views) == 26
    for view in views:
        owner = by_name[view["physical_region"]]
        assert owner["start"] <= view["start"] < view["stop"] <= owner["stop"]
        assert view["elements"] == view["stop"] - view["start"]
        assert view["elements"] == product(view["shape"])
    for index, left in enumerate(views):
        for right in views[index + 1:]:
            overlap = max(left["start"], right["start"]) < min(
                left["stop"], right["stop"]
            )
            if overlap:
                assert left["physical_region"] == right["physical_region"]
    assert layout["view_binding_reshape_bill"] == sum(
        view["elements"] for view in views
    )


def arena_elements(m: int, k: int, n: int) -> int:
    q = Fraction(7, 4)
    a, b, c = m * k, k * n, m * n
    value = sum(q**level * (a + b) for level in range(1, 5))
    value += q**3 * max(a, b) + q**4 * c + 3 * q**3 * c + 2 * c
    assert value.denominator == 1
    return value.numerator


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d5(m: int, k: int, n: int) -> int:
    return 7**4 * mm(m // 16, k // 16, n // 16) + sum(
        7**level * 11 * (m * k + k * n + m * n) // 4 ** (level + 1)
        for level in range(4)
    )


def d6(m: int, k: int, n: int) -> int:
    return 7**4 * mm(m // 16, k // 16, n // 16) + sum(
        7**level * (
            11 * (m * k + k * n) + 7 * m * n
        ) // 4 ** (level + 1)
        for level in range(4)
    )


def main() -> None:
    if RESULT.exists():
        raise FileExistsError(RESULT)
    for path, expected in EXPECTED.items():
        assert path.is_file(), path
        assert sha256(path) == expected, (path, sha256(path), expected)

    manifest = json.loads(MANIFEST.read_text())
    arena = manifest["arena"]
    push, read = arena["push"], arena["read"]
    assert arena_elements(1024, 1024, 15360) == 844_165_120
    assert arena_elements(1024, 14336, 1024) == 681_480_192
    assert arena["single_flat_arena_elements"] == 844_165_120
    assert arena["single_flat_arena_bytes"] == 3_376_660_480
    verify_layout(push, 844_165_120)
    verify_layout(read, 681_480_192)
    assert push["view_binding_reshape_bill"] == 986_193_920
    assert read["view_binding_reshape_bill"] == 933_404_672
    assert push["view_binding_reshape_bill"] + read["view_binding_reshape_bill"] == 1_919_598_592
    assert arena["anchor_plus_push_arena_bytes"] == 6_798_163_968
    assert arena["static_margin_below_8gib_bytes"] == 1_791_770_624
    assert arena["future_smoke_increment_cap_bytes"] == 3_913_531_392
    assert arena["future_smoke_absolute_peak_strictly_below_bytes"] == 7 * 1024**3

    source = CANDIDATE.read_text()
    tree = ast.parse(source)
    transform = function(tree, "_cp3_transform_level")
    reconstruct = function(tree, "_cp3_reconstruct_level")
    kernel = function(tree, "_cp3_winograd_arena")
    bind = function(tree, "_cp3_bind_arena_views")
    transform_order = [name for name, _ in fnp_call_order(transform)]
    reconstruct_order = [name for name, _ in fnp_call_order(reconstruct)]
    assert transform_order == (
        ["add", "subtract", "subtract", "subtract"]
        + ["copyto"] * 7
        + ["subtract"] * 4
        + ["copyto"] * 7
    )
    assert reconstruct_order == [
        "add", "add", "add", "add", "add", "subtract", "add"
    ]
    assert_out_calls(transform, {"add", "subtract"})
    assert_out_calls(reconstruct, {"add", "subtract"})
    assert_out_calls(kernel, {"matmul"})
    bind_calls = fnp_call_order(bind)
    assert [name for name, _ in bind_calls] == ["reshape"]
    assert "fnp.reshape(" in ast.unparse(bind)
    for call in ast.walk(bind):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
            assert not (
                call.func.attr == "reshape"
                and not (
                    isinstance(call.func.value, ast.Name)
                    and call.func.value.id == "fnp"
                )
            ), ast.unparse(call)
    kernel_text = ast.unparse(kernel)
    assert kernel_text.count("_cp3_transform_level(") == 1
    assert kernel_text.count("_cp3_reconstruct_level(") == 1
    assert not any(
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_cp3_winograd_arena"
        for call in ast.walk(kernel)
    )
    for forbidden in ("fnp.stack", "fnp.block", "fnp.concatenate", "fnp.einsum", "fnp.tensordot"):
        for node in (bind, transform, reconstruct, kernel):
            assert forbidden not in ast.unparse(node), (node.name, forbidden)

    transform_counts = Counter(transform_order)
    reconstruct_counts = Counter(reconstruct_order)
    inventory = {
        "copyto": 4 * transform_counts["copyto"],
        "subtract": 4 * (
            transform_counts["subtract"] + reconstruct_counts["subtract"]
        ),
        "add": 4 * (transform_counts["add"] + reconstruct_counts["add"]),
        "matmul": 1,
    }
    assert inventory == {"copyto": 56, "subtract": 32, "add": 28, "matmul": 1}
    assert sum(inventory.values()) == 117

    # Prepending each new child axis must match recursive D5's deepest-first
    # stack layout and C-order leaf flattening.
    paths = [()]
    for _ in range(4):
        paths = [(child,) + parent for child in range(7) for parent in paths]
    encoded = [
        path[0] * 7**3 + path[1] * 7**2 + path[2] * 7 + path[3]
        for path in paths
    ]
    assert encoded == list(range(2401))

    accounting = manifest["accounting"]
    roster = (
        [(1024, 1024, 1024 * layer) for layer in range(1, 16) for _ in range(2)]
        + [(1024, 1024 * layer, 1024) for layer in range(1, 15) for _ in range(2)]
    )
    assert len(roster) == 58
    assert sum(mm(*shape) for shape in roster) == 966_086_623_232
    assert sum(d5(*shape) for shape in roster) == 594_693_332_992
    assert sum(d6(*shape) for shape in roster) == 591_553_830_912
    assert accounting["reconstruction_assembly_copy_saving_per_prediction"] == 3_139_502_080
    assert accounting["paired_common_left_stack_saving_per_prediction"] == 20_971_520
    assert accounting["projected_d6_calls_35_279_35"] == [
        1_319_635_780_983, 1_356_359_358_984, 1_317_713_029_632,
    ]
    assert accounting["projected_incumbent_minus_d6"] == [
        372_849_123_328, 374_768_721_920, 374_768_721_920,
    ]
    assert min(accounting["projected_incumbent_minus_d6"]) >= 371_608_248_320
    assert accounting["net_saving_vs_d5"]["cold_call0"] == 1_240_875_008

    tiny_seal = json.loads((TINY / "TINY_PROTOTYPE_FINAL_SEAL_002.json").read_text())
    tiny_receipt = json.loads((TINY / "TINY_PROTOTYPE_RECEIPT.json").read_text())
    assert tiny_seal["status"] == "SEALED_PASS_D6_TINY_PROTOTYPE"
    assert tiny_seal["disposition"]["exact_shape_smoke_authorized"] is False
    assert tiny_receipt["status"] == "PASS_SEALED_D6_TINY_PROTOTYPE"
    assert all(tiny_receipt["checks"].values())
    assert all(tiny_receipt["gapped_stride_and_guard_checks"].values())
    assert all(tiny_receipt["branch_order_proof"]["checks"].values())
    assert tiny_receipt["branch_order_proof"]["count"] == 2401
    assert tiny_receipt["branch_order_proof"]["path_sequence_sha256"] == (
        "a32d9b1e570b05f0de01ede892bd26aec3a520723826c836be6fdde1112c6281"
    )
    for case in tiny_receipt["cases"]:
        assert all(case["checks"].values())
        assert case["meter"] == case["expected_meter"]

    diagnostic = json.loads(DIAGNOSTIC.read_text())
    assert diagnostic["status"] == "FAIL_RESIDUAL_GATE_ADVERSE_EVIDENCE"
    assert diagnostic["candidate_source_sha256"] == EXPECTED[CANDIDATE]
    assert diagnostic["diagnostic_script_sha256"] == EXPECTED[
        HERE / "run_independent_tiny_residual_diagnostic.py"
    ]
    assert diagnostic["flopscope_version"] == "0.12.0+np2.4.6"
    assert diagnostic["numpy_version"] == "2.4.6"
    assert diagnostic["roster_calls_per_arm"] == 58
    assert diagnostic["alternated_repetitions"] == 8
    assert all(diagnostic["checks"].values())
    assert diagnostic["residual_delta_gate_seconds"] == 0.010
    assert diagnostic["residual_gate_pass"] is False
    deltas = diagnostic["candidate_minus_direct_residual_seconds"]
    assert len(deltas) == 8 and all(value > 0.010 for value in deltas)
    assert statistics.median(deltas) == diagnostic[
        "median_candidate_minus_direct_residual_seconds"
    ] == 0.04961721133440733
    assert diagnostic["expected_candidate_roster_calls"] == {
        "add": 1624, "copyto": 3248, "matmul": 58, "subtract": 1856,
    }
    assert diagnostic["expected_direct_roster_calls"] == {"matmul": 58}
    for field in (
        "estimator_imported", "targets_scores_names_board_opened", "submission", "network"
    ):
        assert diagnostic[field] is False
    for field in ("estimator_calls", "fixture_calls", "exact_shape_calls", "common_driver_calls"):
        assert diagnostic[field] == 0

    audit = AUDIT.read_text()
    required_audit_phrases = (
        "FAIL the prerequisite residual-wall gate",
        "D6 implementation form",
        "Do not authorize the exact-shape smoke",
        "No live spatial collision was found",
        "memory formula is internally feasible",
        "median D6-minus-direct residual",
        "Rejected: exact-shape execution",
    )
    for phrase in required_audit_phrases:
        assert phrase in audit, phrase

    result = {
        "schema": "p2r0c1d6-independent-implementation-audit-verification-v1",
        "status": "PASS_VERIFIED_TERMINAL_CLOSE_D6_RESIDUAL_GATE_FAILED",
        "terminal_verdict": (
            "PASS_STATIC_AND_TINY_SEMANTICS; FAIL_RESIDUAL_GATE; "
            "CLOSE_D6_ARENA_FORM; DO_NOT_AUTHORIZE_EXACT_SHAPE_OR_COMMON_DRIVER"
        ),
        "candidate_source_sha256": EXPECTED[CANDIDATE],
        "audit_sha256": EXPECTED[AUDIT],
        "diagnostic_sha256": EXPECTED[DIAGNOSTIC],
        "tiny_final_seal_sha256": EXPECTED[TINY / "TINY_PROTOTYPE_FINAL_SEAL_002.json"],
        "arena_formula_consistent": True,
        "memory_formula_impossibility_found": False,
        "physical_regions_per_layout": 15,
        "persistent_views_per_layout": 26,
        "cold_binding_reshape_calls": 52,
        "cold_binding_reshape_flops": 1_919_598_592,
        "per_channel_hot_inventory": inventory,
        "per_channel_hot_operation_count": 117,
        "roster_calls": 58,
        "d6_roster_flops": 591_553_830_912,
        "minimum_projected_incumbent_saving": min(
            accounting["projected_incumbent_minus_d6"]
        ),
        "tiny_semantic_authority_pass": True,
        "tiny_residual_gate_seconds": 0.010,
        "tiny_median_candidate_minus_direct_residual_seconds": statistics.median(deltas),
        "tiny_residual_gate_pass": False,
        "exact_shape_authorized": False,
        "common_driver_authorized": False,
        "estimator_imported": False,
        "estimator_calls": 0,
        "fixture_calls": 0,
        "exact_shape_calls": 0,
        "targets_scores_names_board_opened": False,
        "submission": False,
        "network": False,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
