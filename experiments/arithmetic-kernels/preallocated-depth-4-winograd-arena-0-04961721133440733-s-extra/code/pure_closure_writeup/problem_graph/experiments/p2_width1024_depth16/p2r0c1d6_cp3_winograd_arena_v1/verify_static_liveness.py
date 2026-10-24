#!/usr/bin/env python3
"""Target-free static/liveness verifier for P2R0C1D6.

The candidate estimator is parsed but never imported or called.  This verifier
independently checks the mechanical source derivation, physical arena maps,
alias/liveness schedule, AST operation law, 58-contraction roster, and cold /
warm literal-accounting formulas.  It opens no fixture, target, score, name,
board, production, submission, or network resource.
"""

from __future__ import annotations

import ast
import difflib
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
D5 = P2 / "p2r0c1d5_cp3_winograd_hybrid_v1"
INCUMBENT = P2 / "public_noeigh_card_v1" / "estimator.py"
CANDIDATE = HERE / "estimator.py"
BUILDER = HERE / "build_candidate.py"
DIFF = HERE / "SOURCE_DIFF.patch"
MANIFEST = HERE / "ACCOUNTING_AND_LIVENESS_MANIFEST.json"
BUILD_RECEIPT = HERE / "BUILD_RECEIPT.json"
RESULT = HERE / "STATIC_VERIFICATION.json"

EXPECTED = {
    INCUMBENT:
        "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7",
    D5 / "D5R1_RESOURCE_CAUSAL_AUDIT_AND_D6_SUCCESSOR.md":
        "808a41b8fa1429e1efaf978cd3f17f18ecf7ae7f21c20a9359546922b3696da7",
    D5 / "INDEPENDENT_REDTEAM_D6_SUCCESSOR_AUDIT.md":
        "df76169237166cf685ca9e0474643d1d668dcc2ae762f5e9e57af0e6b8afd6f2",
    D5 / "verify_independent_d6_successor.py":
        "58f28d61bcb7047f84d254d5785e7c10ffd6707e6a8e3d61f634d53546933855",
    BUILDER:
        "302e2ca5f3c3a437ed3fb348dae4fc5318efc7e5ba59387793d918270298a8dc",
    CANDIDATE:
        "e9ebe9abebe4d632e20d81396aea2bc64a511277d88bc6fd13fb6a80721ffd58",
    DIFF:
        "73dd4e286bf09160abc41102c080607aabbf981c8bd3822ce37250e9a663df58",
    MANIFEST:
        "bced5170098ccef0a702414a212dea606ef3c16e1c823c2ed2973888d5f0558c",
    BUILD_RECEIPT:
        "7a7c4f07975cca45a413656b44017571ccec7b65d4784ade0bfa54b365d28c2a",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def product(values) -> int:
    result = 1
    for value in values:
        result *= int(value)
    return result


def calls_in(function: ast.FunctionDef) -> Counter:
    calls = Counter()
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            try:
                calls[ast.unparse(node.func)] += 1
            except Exception:
                pass
    return calls


def assert_all_out(function: ast.FunctionDef, names: set[str]) -> None:
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        try:
            name = ast.unparse(node.func)
        except Exception:
            continue
        if name in names:
            assert any(keyword.arg == "out" for keyword in node.keywords), (
                function.name,
                name,
                ast.unparse(node),
            )


def verify_physical_layout(layout: dict, expected_elements: int) -> None:
    regions = layout["physical_regions"]
    assert regions
    cursor = 0
    by_name = {}
    for region in regions:
        assert region["start"] == cursor
        assert region["stop"] > region["start"]
        assert region["elements"] == region["stop"] - region["start"]
        assert region["name"] not in by_name
        by_name[region["name"]] = region
        cursor = region["stop"]
    assert cursor == expected_elements == layout["physical_elements"]
    assert layout["physical_bytes_float32"] == 4 * expected_elements
    assert layout["physical_regions_contiguous_and_nonoverlapping"] is True

    views = layout["view_specs"]
    assert len(views) == 26
    assert len({view["name"] for view in views}) == 26
    for view in views:
        owner = by_name[view["physical_region"]]
        assert owner["start"] <= view["start"] < view["stop"] <= owner["stop"]
        assert view["elements"] == view["stop"] - view["start"]
        assert view["elements"] == product(view["shape"])

    # Physical overlap is forbidden.  Alias overlap is allowed only inside a
    # single owner interval and is scheduled sequentially by source AST below.
    for left_index, left in enumerate(regions):
        for right in regions[left_index + 1:]:
            assert left["stop"] <= right["start"]
    for left_index, left in enumerate(views):
        for right in views[left_index + 1:]:
            overlap = max(left["start"], right["start"]) < min(
                left["stop"], right["stop"]
            )
            if overlap:
                assert left["physical_region"] == right["physical_region"]

    assert layout["view_binding_reshape_bill"] == sum(
        view["elements"] for view in views
    )


def independent_arena_elements(m: int, k: int, n: int) -> int:
    q = Fraction(7, 4)
    a, b, c = m * k, k * n, m * n
    result = sum(q**level * (a + b) for level in range(1, 5))
    result += q**3 * max(a, b)
    result += q**4 * c
    result += 3 * q**3 * c
    result += 2 * c
    assert result.denominator == 1
    return result.numerator


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def d5_kernel(m: int, k: int, n: int) -> int:
    return (
        7**4 * mm(m // 16, k // 16, n // 16)
        + sum(
            7**level * 11 * (m * k + k * n + m * n) // 4 ** (level + 1)
            for level in range(4)
        )
    )


def d6_kernel(m: int, k: int, n: int) -> int:
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


def main() -> None:
    for path, expected in EXPECTED.items():
        assert path.is_file(), path
        assert sha256(path) == expected, (path, sha256(path), expected)

    manifest = json.loads(MANIFEST.read_text())
    receipt = json.loads(BUILD_RECEIPT.read_text())
    assert manifest["schema"] == "p2r0c1d6-accounting-liveness-v1"
    assert manifest["status"] == "FROZEN_PRE_STATIC_REVIEW_NO_EXECUTION"
    assert receipt["status"] == "BUILT_PRE_STATIC_REVIEW_NO_EXECUTION"
    assert receipt["candidate_source_sha256"] == EXPECTED[CANDIDATE]
    assert receipt["source_diff_sha256"] == EXPECTED[DIFF]
    assert receipt["accounting_liveness_manifest_sha256"] == EXPECTED[MANIFEST]
    assert receipt["estimator_calls"] == 0
    assert receipt["estimator_imported"] is False
    assert receipt["exact_shape_calls"] == 0
    assert receipt["fixture_calls"] == 0
    assert receipt["target_columns_opened"] == []
    assert receipt["scores_opened"] == []
    assert receipt["names_opened"] == []
    assert receipt["board_opened"] is False
    assert receipt["submission"] is False
    assert receipt["network"] is False

    arena = manifest["arena"]
    assert arena["single_flat_arena_elements"] == 844_165_120
    assert arena["single_flat_arena_bytes"] == 3_376_660_480
    assert arena["allocation_count"] == 1
    assert arena["layouts_share_same_flat_arena"] is True
    push = arena["push"]
    read = arena["read"]
    assert independent_arena_elements(1024, 1024, 15360) == 844_165_120
    assert independent_arena_elements(1024, 14336, 1024) == 681_480_192
    verify_physical_layout(push, 844_165_120)
    verify_physical_layout(read, 681_480_192)
    assert push["matrix_shape"] == [1024, 1024, 15360]
    assert read["matrix_shape"] == [1024, 14336, 1024]
    assert push["view_binding_reshape_bill"] == 986_193_920
    assert read["view_binding_reshape_bill"] == 933_404_672
    assert len(push["view_specs"]) + len(read["view_specs"]) == 52
    assert arena["anchor_plus_push_arena_bytes"] == 6_798_163_968
    assert arena["static_margin_below_8gib_bytes"] == 1_791_770_624
    assert arena["future_smoke_increment_cap_bytes"] == 3_913_531_392
    assert arena["future_smoke_absolute_peak_strictly_below_bytes"] == 7 * 1024**3

    # Import only the guarded builder, never the estimator.  Rebuild in memory
    # and independently reconstruct the exact unified diff; no file is written.
    builder_tree = ast.parse(BUILDER.read_text())
    assert isinstance(builder_tree.body[-1], ast.If)
    assert isinstance(builder_tree.body[-1].test, ast.Compare)
    spec = importlib.util.spec_from_file_location("p2r0c1d6_builder_verify", BUILDER)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = builder
    spec.loader.exec_module(builder)
    rebuilt_push = builder.make_layout("push", 1024, 1024, 15360)
    rebuilt_read = builder.make_layout("read", 1024, 14336, 1024)
    assert rebuilt_push == push
    assert rebuilt_read == read
    rebuilt_candidate = builder.build_candidate(rebuilt_push, rebuilt_read)
    assert hashlib.sha256(rebuilt_candidate).hexdigest() == EXPECTED[CANDIDATE]
    assert rebuilt_candidate == CANDIDATE.read_bytes()
    rebuilt_diff = "".join(difflib.unified_diff(
        INCUMBENT.read_text().splitlines(keepends=True),
        rebuilt_candidate.decode("utf-8").splitlines(keepends=True),
        fromfile="public_noeigh_card_v1/estimator.py",
        tofile="p2r0c1d6_cp3_winograd_arena_v1/estimator.py",
        n=5,
    ))
    assert rebuilt_diff == DIFF.read_text()
    assert builder.accounting(rebuilt_push, rebuilt_read) == manifest["accounting"]

    candidate_text = CANDIDATE.read_text()
    compile(candidate_text, str(CANDIDATE), "exec")
    tree = ast.parse(candidate_text)
    source_tree = ast.parse(INCUMBENT.read_text())
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    source_functions = {
        node.name: node
        for node in source_tree.body
        if isinstance(node, ast.FunctionDef)
    }
    required = {
        "_cp3_bind_arena_views",
        "_cp3_transform_level",
        "_cp3_reconstruct_level",
        "_cp3_winograd_arena",
        "_cp_read",
        "_cp_k3",
        "_cp_push",
        "_cp_grow",
    }
    assert required <= functions.keys()

    changed = {"_cp_read", "_cp_k3", "_cp_push", "_initialize_h1", "_predict"}
    for name in sorted(set(source_functions) - changed):
        assert ast.dump(
            source_functions[name], include_attributes=False
        ) == ast.dump(functions[name], include_attributes=False), name

    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id.startswith("_CP3_"):
                constants[target.id] = ast.literal_eval(node.value)
    assert constants["_CP3_CAP"] == 16_384
    assert constants["_CP3_WINOGRAD_LEVELS"] == 4
    assert constants["_CP3_WINOGRAD_ARENA_ELEMENTS"] == 844_165_120
    push_source_specs = tuple(
        (view["start"], view["stop"], tuple(view["shape"]))
        for view in push["view_specs"]
    )
    read_source_specs = tuple(
        (view["start"], view["stop"], tuple(view["shape"]))
        for view in read["view_specs"]
    )
    assert constants["_CP3_PUSH_ARENA_VIEW_SPECS"] == push_source_specs
    assert constants["_CP3_READ_ARENA_VIEW_SPECS"] == read_source_specs

    bind_calls = calls_in(functions["_cp3_bind_arena_views"])
    transform_calls = calls_in(functions["_cp3_transform_level"])
    reconstruct_calls = calls_in(functions["_cp3_reconstruct_level"])
    kernel_calls = calls_in(functions["_cp3_winograd_arena"])
    read_calls = calls_in(functions["_cp_read"])
    push_calls = calls_in(functions["_cp_push"])
    assert bind_calls["fnp.reshape"] == 1
    assert transform_calls["fnp.add"] == 1
    assert transform_calls["fnp.subtract"] == 7
    assert transform_calls["fnp.copyto"] == 14
    assert reconstruct_calls["fnp.add"] == 6
    assert reconstruct_calls["fnp.subtract"] == 1
    assert kernel_calls["fnp.matmul"] == 1
    assert kernel_calls["_cp3_transform_level"] == 1
    assert kernel_calls["_cp3_reconstruct_level"] == 1
    assert kernel_calls["_cp3_winograd_arena"] == 0
    assert read_calls["_cp3_winograd_arena"] == 2
    assert read_calls["fnp.multiply"] == 5
    assert read_calls["fnp.add"] == 1
    assert push_calls["_cp3_winograd_arena"] == 2
    per_channel_kernel_inventory = {
        "copyto": 4 * transform_calls["fnp.copyto"],
        "subtract": (
            4 * transform_calls["fnp.subtract"]
            + 4 * reconstruct_calls["fnp.subtract"]
        ),
        "add": (
            4 * transform_calls["fnp.add"]
            + 4 * reconstruct_calls["fnp.add"]
        ),
        "matmul": kernel_calls["fnp.matmul"],
    }
    assert per_channel_kernel_inventory == {
        "copyto": 56,
        "subtract": 32,
        "add": 28,
        "matmul": 1,
    }
    assert sum(per_channel_kernel_inventory.values()) == 117

    assert_all_out(functions["_cp3_transform_level"], {"fnp.add", "fnp.subtract"})
    assert_all_out(functions["_cp3_reconstruct_level"], {"fnp.add", "fnp.subtract"})
    assert_all_out(functions["_cp3_winograd_arena"], {"fnp.matmul"})
    assert_all_out(functions["_cp_read"], {"fnp.add", "fnp.multiply"})

    cp3_functions = (
        "_cp3_bind_arena_views",
        "_cp3_transform_level",
        "_cp3_reconstruct_level",
        "_cp3_winograd_arena",
        "_cp_read",
        "_cp_k3",
        "_cp_push",
    )
    for name in cp3_functions:
        body = ast.unparse(functions[name])
        for forbidden in (
            "fnp.stack",
            "fnp.block",
            "fnp.concatenate",
            "fnp.einsum",
            "fnp.tensordot",
            "numpy",
        ):
            assert forbidden not in body, (name, forbidden)
        loaded_names = {
            node.id
            for node in ast.walk(functions[name])
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
        }
        assert not loaded_names & {"np", "numpy"}, name
    for name in ("_cp3_transform_level", "_cp3_reconstruct_level", "_cp3_winograd_arena", "_cp_push"):
        body_calls = calls_in(functions[name])
        assert not any(
            call in body_calls
            for call in (
                "fnp.empty", "fnp.zeros", "fnp.ones", "fnp.full",
                "fnp.empty_like", "fnp.zeros_like", "fnp.copy",
            )
        ), name

    # Source storage rewiring: one flat arena, four existing CP state planes,
    # exactly two persistent read product planes, and both view maps bound once.
    predict_body = ast.unparse(functions["_predict"])
    assert predict_body.count(
        "fnp.empty(_CP3_WINOGRAD_ARENA_ELEMENTS, dtype=fnp.float32)"
    ) == 1
    assert predict_body.count(
        "fnp.empty((2, 2, width, _CP3_CAP), dtype=fnp.float32)"
    ) == 1
    assert predict_body.count(
        "fnp.empty((2, width, _CP3_CAP), dtype=fnp.float32)"
    ) == 1
    assert predict_body.count("_cp3_bind_arena_views") == 2
    assert "cp3_context[0][1 - cp_bank_index]" in predict_body
    assert "cp3_context[0][cp_bank_index]" in predict_body
    assert "cp3_context[2], cp3_context[5]" in predict_body

    accounting = manifest["accounting"]
    roster = (
        [(1024, 1024, 1024 * layer) for layer in range(1, 16) for _ in range(2)]
        + [(1024, 1024 * layer, 1024) for layer in range(1, 15) for _ in range(2)]
    )
    assert len(roster) == 58
    assert accounting["roster"]["push_channel_contractions"] == 30
    assert accounting["roster"]["read_channel_contractions"] == 28
    assert accounting["roster"]["total_channel_contractions"] == 58
    assert accounting["roster"]["leaf_batch_matmul_dispatches_per_prediction"] == 58
    assert accounting["roster"]["leaf_products_per_dispatch"] == 2401
    assert accounting["roster"]["shapes"] == [list(shape) for shape in roster]
    assert sum(mm(*shape) for shape in roster) == 966_086_623_232
    assert sum(d5_kernel(*shape) for shape in roster) == 594_693_332_992
    assert sum(d6_kernel(*shape) for shape in roster) == 591_553_830_912
    assert accounting["reconstruction_assembly_copy_saving_per_prediction"] == 3_139_502_080
    assert accounting["paired_common_left_stack_saving_per_prediction"] == 20_971_520
    cold_bind = push["view_binding_reshape_bill"] + read["view_binding_reshape_bill"]
    assert cold_bind == 1_919_598_592
    assert accounting["cold_view_binding_reshape_bill"] == cold_bind
    assert accounting["net_saving_vs_d5"] == {
        "cold_call0": 1_240_875_008,
        "warm_calls1_and2": 3_160_473_600,
    }
    assert accounting["projected_d6_calls_35_279_35"] == [
        1_319_635_780_983,
        1_356_359_358_984,
        1_317_713_029_632,
    ]
    assert accounting["projected_incumbent_minus_d6"] == [
        372_849_123_328,
        374_768_721_920,
        374_768_721_920,
    ]
    assert all(
        value >= 371_608_248_320
        for value in accounting["projected_incumbent_minus_d6"]
    )
    assert all(
        candidate <= d5
        for candidate, d5 in zip(
            accounting["projected_d6_calls_35_279_35"],
            accounting["d5_exact_calls_35_279_35"],
        )
    )

    liveness = manifest["liveness"]
    assert liveness["large_dynamic_allocations"] == []
    assert liveness["top_output_written_directly"] is True
    assert liveness["dummy_billed_work"] is False
    authority = manifest["execution_authority"]
    assert authority["source_build"] is True
    assert authority["static_verification"] is True
    for forbidden_authority in (
        "tiny_synthetic_kernel_smoke",
        "exact_shape_smoke",
        "fixture_call",
        "common_driver_call",
        "targets_scores_names_board",
        "production_submission_network",
    ):
        assert authority[forbidden_authority] is False

    result = {
        "schema": "p2r0c1d6-static-liveness-verification-v1",
        "status": "PASS_STATIC_LIVENESS_PREPARATION_ONLY",
        "node": "P2R0C1D6",
        "candidate_source_sha256": EXPECTED[CANDIDATE],
        "builder_sha256": EXPECTED[BUILDER],
        "manifest_sha256": EXPECTED[MANIFEST],
        "source_diff_sha256": EXPECTED[DIFF],
        "single_flat_arena_elements": arena["single_flat_arena_elements"],
        "push_layout_elements": push["physical_elements"],
        "read_layout_elements": read["physical_elements"],
        "cold_view_binding_reshape_bill": cold_bind,
        "cold_view_binding_reshape_calls": 52,
        "physical_regions_nonoverlapping": True,
        "alias_views_within_owner": True,
        "recursive_cp3_kernel_calls": 0,
        "cp3_stack_block_concatenate_calls": 0,
        "leaf_batch_matmul_dispatches_per_prediction": 58,
        "per_channel_kernel_operation_inventory": per_channel_kernel_inventory,
        "per_channel_kernel_operation_count": 117,
        "cold_net_saving_vs_d5": 1_240_875_008,
        "projected_d6_calls_35_279_35": accounting["projected_d6_calls_35_279_35"],
        "projection_only_meter_required": True,
        "estimator_imported": False,
        "estimator_calls": 0,
        "exact_shape_calls": 0,
        "fixture_calls": 0,
        "target_columns_opened": [],
        "scores_opened": [],
        "names_opened": [],
        "board_opened": False,
        "submission": False,
        "network": False,
    }
    RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
