#!/usr/bin/env python3
"""Static and provenance gates for V4 normalized-moment diagnostic v2."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from build_v2_from_v1 import (
    V4_AUDIT,
    V4_AUDIT_SHA256,
    V4_ESTIMATOR,
    V4_ESTIMATOR_SHA256,
    V1_ESTIMATOR,
    V1_ESTIMATOR_SHA256,
    V1_FREEZE,
    V1_FREEZE_SHA256,
    V1_HOLD_RECEIPT,
    V1_HOLD_RECEIPT_SHA256,
    V1_MANIFEST,
    V1_MANIFEST_SHA256,
    build_source,
)


HERE = Path(__file__).resolve().parent
ESTIMATOR = HERE / "estimator.py"
MANIFEST = HERE / "SOURCE_MANIFEST_V2.json"
EXPECTED_CHANNELS = (
    "min_relative_A", "worst_A_relative_error_bound",
    "worst_A_raw_margin", "worst_A_absolute_error_bound",
    "worst_A_index", "worst_A_scale",
    "worst_A_r1", "worst_A_r2", "worst_A_r3", "worst_A_r4",
    "count_A_CERT_NEG", "count_A_ZERO_COMPATIBLE",
    "count_A_CERT_POS", "count_A_scale_zero",
    "min_relative_B", "worst_B_relative_error_bound",
    "worst_B_raw_margin", "worst_B_absolute_error_bound",
    "worst_B_index", "worst_B_scale",
    "worst_B_r1", "worst_B_r2", "worst_B_r3", "worst_B_r4",
    "count_B_CERT_NEG", "count_B_ZERO_COMPATIBLE",
    "count_B_CERT_POS", "count_B_scale_zero",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _call_name(node: ast.Call) -> str:
    value = node.func
    parts: list[str] = []
    while isinstance(value, ast.Attribute):
        parts.append(value.attr)
        value = value.value
    if isinstance(value, ast.Name):
        parts.append(value.id)
    return ".".join(reversed(parts))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    assert len(matches) == 1, name
    return matches[0]


def verify() -> dict[str, object]:
    assert _sha256(V4_ESTIMATOR) == V4_ESTIMATOR_SHA256
    assert _sha256(V4_AUDIT) == V4_AUDIT_SHA256
    assert _sha256(V1_ESTIMATOR) == V1_ESTIMATOR_SHA256
    assert _sha256(V1_MANIFEST) == V1_MANIFEST_SHA256
    assert _sha256(V1_FREEZE) == V1_FREEZE_SHA256
    assert _sha256(V1_HOLD_RECEIPT) == V1_HOLD_RECEIPT_SHA256
    parent = V1_ESTIMATOR.read_text(encoding="utf-8")
    source = ESTIMATOR.read_text(encoding="utf-8")
    assert source == build_source(parent)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "FROZEN_V2_SOURCE_GATES_PASS__REPLAY_READY"
    assert manifest["parents"]["frozen_v4"]["estimator"]["sha256"] == V4_ESTIMATOR_SHA256
    assert manifest["parents"]["frozen_v4"]["independent_audit"]["sha256"] == V4_AUDIT_SHA256
    assert manifest["parents"]["v1_diagnostic_source"]["estimator"]["sha256"] == V1_ESTIMATOR_SHA256
    assert manifest["parents"]["v1_diagnostic_source"]["source_manifest"]["sha256"] == V1_MANIFEST_SHA256
    assert manifest["parents"]["v1_diagnostic_source"]["freeze_receipt"]["sha256"] == V1_FREEZE_SHA256
    assert manifest["parents"]["v1_hold_result93124980"]["sha256"] == V1_HOLD_RECEIPT_SHA256
    assert manifest["negative_controls"] == {
        "no_dataset_access": True,
        "no_target_or_truth_access": True,
        "no_score_or_fit": True,
        "no_sampling_estimator": True,
        "no_scalar_moment_clip_floor_or_repair": True,
        "passing_v4_meter_receipt_bound": True,
        "no_physical_replay_executed_at_source_freeze": True,
        "frozen_v4_estimator_unmodified": True,
        "problem_graph_unmodified_here": True,
    }
    for binding in manifest["files"]:
        path = HERE / binding["path"]
        assert path.is_file(), path
        assert _sha256(path) == binding["sha256"], path

    tree = ast.parse(source)
    imports: set[str | None] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    assert imports == {
        "flopscope", "flopscope.numpy", "whestbench", "whestbench.domain"
    }
    for forbidden in (
        "numpy", "scipy", "subprocess", "threading", "asyncio", "ctypes",
        "cffi", "multiprocessing", "datasets", "huggingface_hub",
    ):
        assert forbidden not in imports

    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert ast.literal_eval(assignments["_DIAGNOSTIC_CHANNELS"]) == EXPECTED_CHANNELS

    helper = _function(tree, "_normalized_moment_worst_row")
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None
    helper_calls = {
        _call_name(node) for node in ast.walk(helper) if isinstance(node, ast.Call)
    }
    assert helper_calls == {
        "_add_sub_error", "_multiply_error", "fnp.abs", "fnp.argmin",
        "fnp.count_nonzero", "fnp.divide", "fnp.stack", "fnp.where",
        "fnp.zeros_like", "int",
    }
    for repair in (
        "fnp.maximum", "fnp.minimum", "fnp.clip", "fnp.nan_to_num",
    ):
        assert repair not in helper_source
    for exact_formula in (
        "variance = raw2 - raw1_squared",
        "a_left = raw1 * raw3",
        "margin_a = a_left - raw2_squared",
        "raw1_raw2 = raw1 * raw2",
        "centered_21 = raw3 - raw1_raw2",
        "raw4_minus_raw2_squared = raw4 - raw2_squared",
        "b_left = raw4_minus_raw2_squared * variance",
        "centered_21_squared = centered_21 * centered_21",
        "margin_b = b_left - centered_21_squared",
        "scale_a = fnp.abs(a_left) + fnp.abs(raw2_squared)",
        "scale_b = fnp.abs(b_left) + fnp.abs(centered_21_squared)",
        "cert_negative_a = margin_a < -error_margin_a",
        "cert_positive_a = margin_a > error_margin_a",
        "cert_negative_b = margin_b < -error_margin_b",
        "cert_positive_b = margin_b > error_margin_b",
        "fnp.divide(margin_a, scale_a, out=relative_a, where=positive_scale_a)",
        "reduction_a = fnp.where(positive_scale_a, relative_a, fnp.inf)",
    ):
        assert exact_formula in helper_source

    assert ast.literal_eval(assignments["_TWO_U"]) == 2.220446049250313e-16
    up = _function(tree, "_up")
    add_sub_error = _function(tree, "_add_sub_error")
    multiply_error = _function(tree, "_multiply_error")
    up_source = ast.get_source_segment(source, up)
    add_source = ast.get_source_segment(source, add_sub_error)
    multiply_source = ast.get_source_segment(source, multiply_error)
    assert up_source is not None and "fnp.nextafter(nonnegative, fnp.inf)" in up_source
    assert add_source is not None and all(
        formula in add_source for formula in (
            "inherited = _up(error_x + error_y)",
            "local = _up(_TWO_U * fnp.abs(result))",
            "return _up(inherited + local)",
        )
    )
    assert multiply_source is not None and all(
        formula in multiply_source for formula in (
            "left = _up(fnp.abs(x) * error_y)",
            "right = _up(fnp.abs(y) * error_x)",
            "inherited = _up(left + right)",
            "cross = _up(error_x * error_y)",
            "inherited = _up(inherited + cross)",
            "local = _up(_TWO_U * fnp.abs(result))",
            "return _up(inherited + local)",
        )
    )

    assert source.count("_normalized_moment_worst_row(") == 4
    assert "diagnostic = _normalized_moment_worst_row(mean, raw2, raw3, raw4)" in source
    assert "diagnostic = _normalized_moment_worst_row(raw1, raw2, raw3, raw4)" in source
    assert "_normalized_moment_worst_row(*final_raw)" in source
    assert "diagnostic_rows = [diagnostic]" in source
    assert "fnp.stack(diagnostic_rows, axis=0)" in source
    assert "return output, diagnostics" in source

    reset = _function(tree, "_full_relu_reset")
    diagnostic_lines = [
        node.lineno for node in ast.walk(reset)
        if isinstance(node, ast.Call) and _call_name(node) == "_normalized_moment_worst_row"
    ]
    repair_lines = [
        node.lineno for node in ast.walk(reset)
        if isinstance(node, ast.Call) and _call_name(node) == "_repair_covariance"
    ]
    assert len(diagnostic_lines) == len(repair_lines) == 1
    assert diagnostic_lines[0] < repair_lines[0]

    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    assert [node.name for node in classes] == ["Estimator", "MomentDiagnostic"]
    assert "return _predict_with_diagnostics(mlp)[0]" in source
    assert "return _predict_with_diagnostics(mlp)[1]" in source
    assert "or int(budget) < _PHASE2_FLOP_BUDGET" in source
    assert "open(" not in source
    assert "eval(" not in source
    assert "exec(" not in source
    return {
        "status": "PASS_V4_NORMALIZED_MOMENT_DIAGNOSTIC_V2_STATIC_AND_PROVENANCE",
        "v4_estimator_sha256": V4_ESTIMATOR_SHA256,
        "v4_independent_audit_sha256": V4_AUDIT_SHA256,
        "v1_estimator_sha256": V1_ESTIMATOR_SHA256,
        "v1_hold_receipt_sha256": V1_HOLD_RECEIPT_SHA256,
        "diagnostic_estimator_sha256": _sha256(ESTIMATOR),
        "channels": list(EXPECTED_CHANNELS),
        "observation_sites": 3,
        "actual_layer_rows_at_depth16": 16,
        "repair_feedback": False,
    }


if __name__ == "__main__":
    print(verify())
