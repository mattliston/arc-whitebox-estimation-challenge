#!/usr/bin/env python3
"""Fail-closed, non-importing source-audit harness for corrected CS3B.

The default mode verifies only the immutable scaffold and corrected design
chain.  ``--static-handoff`` additionally parses and diffs frozen candidate
source bytes but never imports or executes them.  ``--probe-handoff`` replays
an already-sealed target-free author result; it still never imports or
executes estimator source.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
WIDTH_ROOT = HERE.parent
SCAFFOLD = HERE / "INDEPENDENT_SOURCE_AUDIT_SCAFFOLD_CS3B_V1.json"
SCAFFOLD_CORRECTION = (
    HERE / "INDEPENDENT_SOURCE_AUDIT_SCAFFOLD_CORRECTION_CS3B_V1.json"
)
NAMESPACE_CLARIFICATION = (
    HERE / "INDEPENDENT_SOURCE_AUDIT_NAMESPACE_CLARIFICATION_CS3B_V1.json"
)

SCAFFOLD_SHA256 = "5b8a212ff282154d3507d6dfa39bbb102f929e3de640cdb354b34d6e45a9e467"
SCAFFOLD_CORRECTION_SHA256 = (
    "ede02815e007c0d776271a66714912eafb001df6689770b29414ee5829988341"
)
SOURCE_HOLD_SHA256 = "ac29611a20280de259c7ab2f970d865b09ca1c417d1ac09d64aeab5a89d23bb5"
NAMESPACE_CLARIFICATION_SHA256 = (
    "ce8b642ac27e213d85a417b439b75a5d51f6c8fcc385fcbca6030eb9cb0bfcd2"
)

EXPECTED_CERTIFICATE_LEDGER = {
    "nextafter": 57,
    "abs": 23,
    "multiply": 35,
    "add": 28,
    "subtract": 5,
    "negative": 2,
    "less": 2,
    "bitwise_or": 1,
    "zeros_like": 1,
}
EXPECTED_CERTIFICATE_TOTAL = 154
EXPECTED_LR_POLY = {
    (1, 2): 16,
    (2, 1): 4,
    (0, 2): 8,
    (1, 1): 20,
    (2, 0): 8,
    (0, 1): 2,
}
EXPECTED_D0_POLY = {(0, 3): 4, (0, 2): 8}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def resolve_local(relative: str) -> Path:
    path = (HERE / relative).resolve()
    require(path.is_relative_to(WIDTH_ROOT), f"binding escapes width root: {relative}")
    return path


def verify_binding(binding: dict[str, Any], label: str) -> Path:
    require(set(binding) >= {"path", "sha256"}, f"malformed binding: {label}")
    require(isinstance(binding["path"], str), f"null path in populated binding: {label}")
    require(isinstance(binding["sha256"], str), f"null hash in populated binding: {label}")
    path = resolve_local(binding["path"])
    require(path.is_file(), f"missing binding: {label}: {path}")
    require(sha256(path) == binding["sha256"], f"hash mismatch: {label}")
    return path


def lr_star(rank: int, width: int) -> int:
    return (
        16 * rank * width * width
        + 4 * rank * rank * width
        + 8 * width * width
        + 20 * rank * width
        + 8 * rank * rank
        + 2 * width
    )


def d0(width: int) -> int:
    return 4 * width * width * width + 8 * width * width


def dh(width: int) -> int:
    return d0(width) + 2 * width


def selected_mode(rank: int, width: int) -> str:
    return "LOWRANK64" if lr_star(rank, width) <= d0(width) else "DENSE32"


def debris(roots: Iterable[Path]) -> list[str]:
    found: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
                found.add(str(path))
    return sorted(found)


def function_map(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def class_map(tree: ast.Module) -> dict[str, ast.ClassDef]:
    return {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}


def ast_key(node: ast.AST) -> str:
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def without_docstring(node: ast.AST) -> ast.AST:
    """Return a parsed copy with only the leading docstring removed."""
    copy = ast.parse(ast.unparse(node)).body[0]
    body = getattr(copy, "body", None)
    if (
        isinstance(body, list)
        and body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        del body[0]
    return copy


def call_name(call: ast.Call) -> str:
    node: ast.AST = call.func
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def local_operation_ledger(node: ast.AST) -> Counter[str]:
    result: Counter[str] = Counter()
    for item in ast.walk(node):
        if isinstance(item, ast.BinOp):
            if isinstance(item.op, ast.Mult):
                result["multiply"] += 1
            elif isinstance(item.op, ast.Add):
                result["add"] += 1
            elif isinstance(item.op, ast.Sub):
                result["subtract"] += 1
        elif isinstance(item, ast.UnaryOp) and isinstance(item.op, ast.USub):
            result["negative"] += 1
        elif isinstance(item, ast.Compare):
            result["less"] += sum(isinstance(op, ast.Lt) for op in item.ops)
        elif isinstance(item, ast.Call):
            name = call_name(item)
            if name == "fnp.nextafter":
                result["nextafter"] += 1
            elif name == "fnp.abs":
                result["abs"] += 1
            elif name == "fnp.bitwise_or":
                result["bitwise_or"] += 1
            elif name == "fnp.zeros_like":
                result["zeros_like"] += 1
    return result


def expanded_reference_ledger(
    name: str,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> Counter[str]:
    node = functions[name]
    result = local_operation_ledger(node)
    for item in ast.walk(node):
        if isinstance(item, ast.Call):
            child = call_name(item)
            if child in {"_up", "_add_sub_error", "_multiply_error"}:
                result.update(expanded_reference_ledger(child, functions))
    return result


def assignment_map(function: ast.AST) -> dict[str, list[ast.AST]]:
    result: dict[str, list[ast.AST]] = {}
    for node in ast.walk(function):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result.setdefault(target.id, []).append(node.value)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.value is not None:
                result.setdefault(node.target.id, []).append(node.value)
    return result


Polynomial = dict[tuple[int, int], int]


def poly_add(left: Polynomial, right: Polynomial, sign: int = 1) -> Polynomial:
    result = dict(left)
    for key, value in right.items():
        result[key] = result.get(key, 0) + sign * value
        if result[key] == 0:
            del result[key]
    return result


def poly_mul(left: Polynomial, right: Polynomial) -> Polynomial:
    result: Polynomial = {}
    for (lr, lw), lv in left.items():
        for (rr, rw), rv in right.items():
            key = (lr + rr, lw + rw)
            result[key] = result.get(key, 0) + lv * rv
    return {key: value for key, value in result.items() if value}


def polynomial(node: ast.AST, rank_name: str, width_name: str) -> Polynomial:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return {(0, 0): int(node.value)}
    if isinstance(node, ast.Name):
        if node.id == rank_name:
            return {(1, 0): 1}
        if node.id == width_name:
            return {(0, 1): 1}
        raise AssertionError(f"unbound name in resource polynomial: {node.id}")
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return {key: -value for key, value in polynomial(node.operand, rank_name, width_name).items()}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return poly_add(
            polynomial(node.left, rank_name, width_name),
            polynomial(node.right, rank_name, width_name),
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return poly_add(
            polynomial(node.left, rank_name, width_name),
            polynomial(node.right, rank_name, width_name),
            -1,
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return poly_mul(
            polynomial(node.left, rank_name, width_name),
            polynomial(node.right, rank_name, width_name),
        )
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Pow)
        and isinstance(node.right, ast.Constant)
        and isinstance(node.right.value, int)
        and 0 <= node.right.value <= 4
    ):
        result: Polynomial = {(0, 0): 1}
        base = polynomial(node.left, rank_name, width_name)
        for _ in range(node.right.value):
            result = poly_mul(result, base)
        return result
    raise AssertionError(f"unsupported resource polynomial AST: {ast.dump(node)}")


def normalized_source(node: ast.AST) -> str:
    return "".join(ast.unparse(node).split())


def matrix_shape(value: Any, label: str) -> tuple[int, int]:
    require(isinstance(value, list) and value, f"{label} must be a nonempty matrix")
    require(all(isinstance(row, list) for row in value), f"{label} rows malformed")
    width = len(value[0])
    require(width > 0 and all(len(row) == width for row in value), f"{label} ragged")
    return len(value), width


def decimals(value: list[list[Any]]) -> list[list[Decimal]]:
    return [[Decimal(str(cell)) for cell in row] for row in value]


def transpose(value: list[list[Decimal]]) -> list[list[Decimal]]:
    return [list(column) for column in zip(*value)]


def matmul(
    left: list[list[Decimal]], right: list[list[Decimal]]
) -> list[list[Decimal]]:
    require(len(left[0]) == len(right), "matrix contraction mismatch")
    right_t = transpose(right)
    return [[sum((x * y for x, y in zip(row, column)), Decimal(0)) for column in right_t] for row in left]


def matrix_add(
    left: list[list[Decimal]], right: list[list[Decimal]], sign: int = 1
) -> list[list[Decimal]]:
    return [
        [x + Decimal(sign) * y for x, y in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def direct_masked_push(
    covariance: list[list[Decimal]], weight: list[list[Decimal]], indices: list[int]
) -> list[list[Decimal]]:
    invalid = set(indices)
    masked = [
        [Decimal(0) if i in invalid or j in invalid else value for j, value in enumerate(row)]
        for i, row in enumerate(covariance)
    ]
    return matmul(matmul(transpose(weight), masked), weight)


def lowrank_push(
    covariance: list[list[Decimal]], weight: list[list[Decimal]], indices: list[int]
) -> list[list[Decimal]]:
    left = matmul(transpose(weight), covariance)
    q = matmul(left, weight)
    u = [weight[index][:] for index in indices]
    crows = [covariance[index][:] for index in indices]
    g = [[crows[i][index] for index in indices] for i in range(len(indices))]
    a = [[row[index] for index in indices] for row in left]
    b = matmul(crows, weight)
    u_t = transpose(u)
    lc = matmul(u_t, b)
    rc = matmul(a, u)
    h = matmul(g, u)
    joint = matmul(u_t, h)
    return matrix_add(matrix_add(matrix_add(q, lc, -1), rc, -1), joint)


def max_matrix_error(reference: list[list[Decimal]], observed: Any) -> Decimal:
    require(matrix_shape(observed, "observed matrix") == (len(reference), len(reference[0])), "observed shape mismatch")
    converted = decimals(observed)
    return max(abs(x - y) for row_x, row_y in zip(reference, converted) for x, y in zip(row_x, row_y))


def verify_scaffold() -> dict[str, Any]:
    require(sha256(SCAFFOLD) == SCAFFOLD_SHA256, "scaffold bytes changed")
    require(
        sha256(SCAFFOLD_CORRECTION) == SCAFFOLD_CORRECTION_SHA256,
        "scaffold correction bytes changed",
    )
    require(
        sha256(NAMESPACE_CLARIFICATION) == NAMESPACE_CLARIFICATION_SHA256,
        "namespace clarification bytes changed",
    )
    scaffold = load_json(SCAFFOLD)
    correction = load_json(SCAFFOLD_CORRECTION)
    namespace_clarification = load_json(NAMESPACE_CLARIFICATION)
    require(scaffold["schema"] == "independent-cs3b-source-audit-scaffold-v1", "scaffold schema")
    require(scaffold["status"].startswith("PASS_STATIC_SCAFFOLD__FAIL_CLOSED"), "scaffold status")
    require(correction["bindings"]["immutable_scaffold"]["sha256"] == SCAFFOLD_SHA256, "correction scaffold binding")
    require(correction["bindings"]["frozen_source_hold"]["sha256"] == SOURCE_HOLD_SHA256, "correction source-HOLD binding")
    require(
        namespace_clarification["bindings"]["immutable_scaffold_sha256"]
        == SCAFFOLD_SHA256
        and namespace_clarification["bindings"]["additive_seam_correction_sha256"]
        == SCAFFOLD_CORRECTION_SHA256,
        "namespace clarification chain mismatch",
    )

    bound_paths: dict[str, Path] = {}
    for name, binding in scaffold["bindings"].items():
        bound_paths[name] = verify_binding(binding, name)
    for name, binding in scaffold["explicitly_void_non_authoritative_artifacts"].items():
        path = HERE / name
        require(path.is_file() and sha256(path) == binding["sha256"], f"void artifact mismatch: {name}")

    future = scaffold["future_bindings_immutable_null_at_seal"]
    for name, binding in future.items():
        require(all(value is None for value in binding.values()), f"future binding not null: {name}")

    audit = load_json(bound_paths["independent_corrected_design_audit"])
    receipt = load_json(bound_paths["author_corrected_design_receipt"])
    release = load_json(bound_paths["one_source_build_release"])
    require(audit["status"] == scaffold["bindings"]["independent_corrected_design_audit"]["status_exact"], "audit status")
    require(receipt["status"] == scaffold["bindings"]["author_corrected_design_receipt"]["status_exact"], "design receipt status")
    require(release["status"] == scaffold["bindings"]["one_source_build_release"]["status_exact"], "build release status")
    require(release["authority"]["candidate_execution"] is False, "build release unexpectedly authorizes execution")
    require(release["authoritative_overrides"]["lowrank_cost"].endswith("+20*r*w+8*r^2+2*w"), "release uses stale cost")

    parent_text = bound_paths["parent_estimator"].read_text(encoding="utf-8")
    parent_tree = ast.parse(parent_text, filename=str(bound_paths["parent_estimator"]))
    parent_functions = function_map(parent_tree)
    reference_ledger = expanded_reference_ledger("_certified_invalid_marginals", parent_functions)
    require(dict(reference_ledger) == EXPECTED_CERTIFICATE_LEDGER, f"parent certificate ledger mismatch: {reference_ledger}")
    require(sum(reference_ledger.values()) == EXPECTED_CERTIFICATE_TOTAL, "certificate total mismatch")

    witnesses = scaffold["corrected_resource_contract"]["required_selector_witnesses"]
    for witness in witnesses:
        require(selected_mode(witness["r"], witness["w"]) == witness["mode"], f"selector witness failed: {witness}")
    require(lr_star(241, 1024) == 4294999816, "r241 arithmetic")
    require(lr_star(242, 1024) == 4313779744, "r242 arithmetic")
    require(d0(1024) == 4303355904 and dh(1024) == 4303357952, "dense arithmetic")
    total = 2131652533303 + 14 * dh(1024) - 112
    require(total == 2191899544519, "worst projection arithmetic")

    with localcontext() as context:
        context.prec = 160
        c = decimals([["1.0", "0.25", "-0.125"], ["0.5", "2.0", "0.75"], ["0.0625", "-0.5", "3.0"]])
        w = decimals([["1.0", "0.5", "0.25"], ["-0.25", "1.0", "0.75"], ["0.125", "-0.5", "1.0"]])
        require(direct_masked_push(c, w, [1]) == lowrank_push(c, w, [1]), "internal nonsymmetric identity self-test")

    require(not debris([HERE]), "pycache/bytecode debris in audit namespace")
    return {
        "status": "PASS_STATIC_SCAFFOLD__SOURCE_AUDIT_REMAINS_FAIL_CLOSED",
        "scaffold_sha256": SCAFFOLD_SHA256,
        "scaffold_correction_sha256": SCAFFOLD_CORRECTION_SHA256,
        "namespace_clarification_sha256": NAMESPACE_CLARIFICATION_SHA256,
        "bound_artifacts": len(bound_paths),
        "parent_certificate_ledger_total": sum(reference_ledger.values()),
        "corrected_resource_arithmetic": True,
        "nonsymmetric_identity_self_test": True,
        "candidate_path_or_hash_read": False,
        "candidate_imports_or_executions": 0,
    }


def top_level_assignments(tree: ast.Module) -> dict[str, ast.AST]:
    result: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            result[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            result[node.target.id] = node
    return result


def import_keys(tree: ast.Module) -> list[str]:
    return [ast_key(node) for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]


def verify_candidate_ast(
    parent_path: Path,
    candidate_path: Path,
    roles: dict[str, str],
    expected_changed: list[str],
    observer_functions: list[str],
) -> dict[str, Any]:
    parent_text = parent_path.read_text(encoding="utf-8")
    candidate_text = candidate_path.read_text(encoding="utf-8")
    parent_tree = ast.parse(parent_text, filename=str(parent_path))
    candidate_tree = ast.parse(candidate_text, filename=str(candidate_path))
    parent_functions = function_map(parent_tree)
    candidate_functions = function_map(candidate_tree)
    parent_classes = class_map(parent_tree)
    candidate_classes = class_map(candidate_tree)

    correction = load_json(SCAFFOLD_CORRECTION)
    allowed_changed = set(correction["correction"]["allowed_changed_functions_effective"])
    require(set(expected_changed) <= allowed_changed, "handoff expects unauthorized changed function")
    require(set(parent_functions) <= set(candidate_functions), "parent function removed")
    actual_changed = sorted(
        name for name in parent_functions if ast_key(parent_functions[name]) != ast_key(candidate_functions[name])
    )
    require(actual_changed == sorted(expected_changed), f"changed-function set mismatch: {actual_changed}")
    require(set(actual_changed) <= allowed_changed, f"unapproved changed functions: {set(actual_changed)-allowed_changed}")
    new_functions = set(candidate_functions) - set(parent_functions)
    require(new_functions == set(observer_functions), f"unbound new functions: {new_functions}")
    require(all(name.startswith("_cs3b_observer_") for name in new_functions), "observer prefix violation")

    require(set(parent_classes) == set(candidate_classes), "class inventory changed")
    for name in parent_classes:
        parent_class = without_docstring(parent_classes[name])
        candidate_class = without_docstring(candidate_classes[name])
        require(ast_key(parent_class) == ast_key(candidate_class), f"class changed beyond namespace docstring: {name}")
    require(import_keys(parent_tree) == import_keys(candidate_tree), "import AST changed")

    parent_assignments = top_level_assignments(parent_tree)
    candidate_assignments = top_level_assignments(candidate_tree)
    require(set(parent_assignments) <= set(candidate_assignments), "parent top-level assignment removed")
    for name, node in parent_assignments.items():
        require(ast_key(node) == ast_key(candidate_assignments[name]), f"parent top-level assignment changed: {name}")
    new_assignments = set(candidate_assignments) - set(parent_assignments)
    allowed_modes = {"_PAIRING_NOMINAL", "_PAIRING_LOWRANK64", "_PAIRING_DENSE32"}
    require(new_assignments == allowed_modes, f"unexpected new top-level assignments: {new_assignments}")
    mode_values = []
    for name in sorted(allowed_modes):
        node = candidate_assignments[name]
        require(isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, int), f"mode is not literal int: {name}")
        mode_values.append(node.value.value)
    require(sorted(mode_values) == [0, 1, 2], "mode integers are not distinct 0/1/2")

    forbidden_names = {"np", "numpy", "subprocess", "requests", "urllib", "socket"}
    forbidden_calls = {"open", "eval", "exec", "compile", "__import__"}
    forbidden_attrs = {"item", "tolist", "view", "data", "ctypes", "tobytes", "tofile"}
    for function_name in actual_changed:
        function = candidate_functions[function_name]
        for node in ast.walk(function):
            if isinstance(node, ast.Name):
                require(node.id not in forbidden_names, f"forbidden name in {function_name}: {node.id}")
            elif isinstance(node, ast.Attribute):
                require(node.attr not in forbidden_attrs, f"forbidden attribute in {function_name}: {node.attr}")
            elif isinstance(node, ast.Call):
                require(call_name(node) not in forbidden_calls, f"forbidden call in {function_name}: {call_name(node)}")

    certificate = candidate_functions["_certified_invalid_marginals"]
    helper_calls = [
        call_name(node)
        for node in ast.walk(certificate)
        if isinstance(node, ast.Call) and call_name(node) in {"_up", "_add_sub_error", "_multiply_error"}
    ]
    require(not helper_calls, f"certificate helper calls remain: {helper_calls}")
    candidate_ledger = local_operation_ledger(certificate)
    require(dict(candidate_ledger) == EXPECTED_CERTIFICATE_LEDGER, f"candidate certificate ledger mismatch: {candidate_ledger}")
    require(sum(candidate_ledger.values()) == EXPECTED_CERTIFICATE_TOTAL, "candidate certificate total mismatch")

    predict = candidate_functions["_predict"]
    assignments = assignment_map(predict)
    for role, target in roles.items():
        require(isinstance(target, str) and target, f"empty AST role: {role}")
        if role.endswith("_target") and role != "selector_compare_target":
            require(target in assignments, f"AST role target absent: {role}={target}")

    rank_name = roles["rank_name"]
    width_name = roles["width_name"]
    lowrank_exprs = assignments[roles["lowrank_cost_target"]]
    dense_exprs = assignments[roles["dense_cost_target"]]
    require(len(lowrank_exprs) == 1 and len(dense_exprs) == 1, "resource target assigned more than once")
    require(polynomial(lowrank_exprs[0], rank_name, width_name) == EXPECTED_LR_POLY, "candidate Lr_star AST mismatch")
    require(polynomial(dense_exprs[0], rank_name, width_name) == EXPECTED_D0_POLY, "candidate D0 AST mismatch")
    selector_role = roles["selector_compare_target"]
    if selector_role == "__direct_if_compare__":
        selector_exprs = [
            node.test
            for node in ast.walk(predict)
            if isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
        ]
    else:
        selector_exprs = assignments[selector_role]
    matching_selectors = [
        selector
        for selector in selector_exprs
        if len(selector.ops) == 1
        and isinstance(selector.ops[0], ast.LtE)
        and normalized_source(selector.left) == roles["lowrank_cost_target"]
        and normalized_source(selector.comparators[0]) == roles["dense_cost_target"]
    ]
    require(len(matching_selectors) == 1, "exact lowrank<=dense selector not found once")

    pre_left = assignments[roles["pre_left_target"]]
    pre_covariance = assignments[roles["pre_covariance_target"]]
    require(len(pre_left) == 1 and normalized_source(pre_left[0]) == "weight64_t@covariance", "pre_left grouping mismatch")
    require(len(pre_covariance) == 1 and normalized_source(pre_covariance[0]) == f"{roles['pre_left_target']}@weight64", "pre_covariance grouping mismatch")

    expected_expressions = {
        "U_target": f"{roles['weight_name']}[{roles['index_name']},:]",
        "Crows_target": f"{roles['covariance_name']}[{roles['index_name']},:]",
        "G_target": f"{roles['Crows_target']}[:,{roles['index_name']}]",
        "A_target": f"{roles['pre_left_target']}[:,{roles['index_name']}]",
        "B_target": f"{roles['Crows_target']}@{roles['weight_name']}",
        "U_transpose_target": f"fnp.transpose({roles['U_target']})",
        "Lc_target": f"{roles['U_transpose_target']}@{roles['B_target']}",
        "Rc_target": f"{roles['A_target']}@{roles['U_target']}",
        "H_target": f"{roles['G_target']}@{roles['U_target']}",
        "J_target": f"{roles['U_transpose_target']}@{roles['H_target']}",
        "Qmask_target": f"(({roles['pre_covariance_target']}-{roles['Lc_target']})-{roles['Rc_target']})+{roles['J_target']}",
    }
    for role, expected in expected_expressions.items():
        values = assignments[roles[role]]
        expected_normalized = normalized_source(ast.parse(expected, mode="eval").body)
        matching = [
            value
            for value in values
            if normalized_source(value) == expected_normalized
        ]
        require(len(matching) == 1, f"exact algebra expression not found once: {role}")

    require("fnp.all(valid_plane)" not in candidate_text, "forbidden w-by-w all reduction")
    require("pairing_enabled" not in candidate_text, "old global pairing gate remains")
    require("+ 8 * invalid_count * width" not in candidate_text, "stale gather-law spelling detected")
    return {
        "syntax_and_ast_parse": True,
        "changed_functions": actual_changed,
        "new_mode_constants": sorted(new_assignments),
        "certificate_ledger": dict(candidate_ledger),
        "certificate_total": sum(candidate_ledger.values()),
        "corrected_resource_polynomials": True,
        "selector_ast": True,
        "parent_pre_left_grouping": True,
        "nonsymmetric_five_matmul_operand_order": True,
        "native_escape_scan": True,
        "candidate_imported_or_executed": False,
    }


def verify_static_handoff(path: Path) -> dict[str, Any]:
    scaffold_result = verify_scaffold()
    handoff = load_json(path)
    require(handoff["schema"] == "independent-cs3b-static-source-audit-handoff-v1", "static handoff schema")
    require(handoff["scaffold_sha256"] == SCAFFOLD_SHA256, "static handoff scaffold hash")
    require(handoff["scaffold_correction_sha256"] == SCAFFOLD_CORRECTION_SHA256, "static handoff correction hash")
    require(
        handoff["namespace_clarification_sha256"]
        == NAMESPACE_CLARIFICATION_SHA256,
        "static handoff namespace clarification hash",
    )
    candidate = verify_binding(handoff["bindings"]["candidate_estimator"], "candidate estimator")
    source_hold = verify_binding(handoff["bindings"]["candidate_source_hold"], "candidate source HOLD")
    require(handoff["bindings"]["author_source_probe_result"] == {"path": None, "sha256": None, "status_exact": None}, "static handoff author result must remain null")
    hold = load_json(source_hold)
    require(hold["status"] == "IMMUTABLE_SOURCE_HOLD__ZERO_EXECUTIONS__INDEPENDENT_SOURCE_AUDIT_REQUIRED", "source HOLD status")
    require(hold["bindings"]["estimator"]["sha256"] == sha256(candidate), "source HOLD candidate binding")
    require(hold["execution_counters"]["candidate_imports"] == 0 and hold["execution_counters"]["candidate_estimator_executions"] == 0, "source HOLD claims execution")

    parent = resolve_local(load_json(SCAFFOLD)["bindings"]["parent_estimator"]["path"])
    ast_result = verify_candidate_ast(
        parent,
        candidate,
        handoff["ast_role_bindings"],
        handoff["expected_changed_functions"],
        handoff.get("observer_functions", []),
    )
    roots = {HERE, candidate.parent, source_hold.parent}
    require(not debris(roots), "pycache/bytecode debris in static source audit scope")
    return {
        "status": "PASS_INDEPENDENT_STATIC_SOURCE_AUDIT__SOURCE_PROBE_RESULT_NULL__NO_EXECUTION_AUTHORITY",
        "scaffold": scaffold_result,
        "handoff_sha256": sha256(path),
        "candidate_sha256": sha256(candidate),
        "source_hold_sha256": sha256(source_hold),
        "ast": ast_result,
        "author_source_probe_result": None,
        "candidate_imports_or_executions": 0,
        "source_probe_runs": 0,
    }


def verify_certificate_result(section: dict[str, Any]) -> None:
    required_classes = {"boundary", "one_ulp_below", "one_ulp_above", "signed_zero", "subnormal", "large_finite", "fixed_fuzz"}
    require(required_classes <= set(section["case_classes_present"]), "certificate case class missing")
    require(section["all_named_intermediate_bytes_equal"] is True, "certificate intermediate mismatch")
    require(section["all_invalid_mask_bytes_equal"] is True, "certificate mask mismatch")
    require(section["operation_ledger"] == {**EXPECTED_CERTIFICATE_LEDGER, "total": 154}, "certificate result ledger")
    require(section["tracked_call_delta"] == 0 and section["tracked_flop_delta"] == 0, "certificate tracked delta")
    require(section["helper_frames_removed_per_reset"] == 68, "certificate frame count")


def verify_nonsymmetric_result(section: dict[str, Any]) -> None:
    require(section["bound_frozen_before_results"] is True, "lowrank bound not preregistered")
    fixtures = section["fixtures"]
    require(isinstance(fixtures, list) and fixtures, "no nonsymmetric fixtures")
    asymmetric = False
    for index, fixture in enumerate(fixtures):
        c_raw = fixture["C"]
        w_raw = fixture["W"]
        c_shape = matrix_shape(c_raw, f"C[{index}]")
        w_shape = matrix_shape(w_raw, f"W[{index}]")
        require(c_shape[0] == c_shape[1] == w_shape[0] == w_shape[1], "fixture shape mismatch")
        indices = fixture["idx"]
        require(isinstance(indices, list) and indices and len(set(indices)) == len(indices), "bad invalid indices")
        require(all(isinstance(value, int) and 0 <= value < c_shape[0] for value in indices), "index out of range")
        asymmetric = asymmetric or any(c_raw[i][j] != c_raw[j][i] for i in range(c_shape[0]) for j in range(c_shape[1]))
        with localcontext() as context:
            context.prec = 200
            c = decimals(c_raw)
            w = decimals(w_raw)
            direct = direct_masked_push(c, w, indices)
            lowrank = lowrank_push(c, w, indices)
            require(direct == lowrank, f"exact lowrank identity failed: fixture {index}")
            actual_error = max_matrix_error(direct, fixture["candidate_Qmask"])
            bound = Decimal(str(fixture["frozen_absolute_bound"]))
            reported = Decimal(str(fixture["observed_max_absolute_error"]))
            require(actual_error <= bound, f"candidate lowrank bound failed: fixture {index}")
            require(abs(actual_error - reported) <= Decimal(str(fixture["reporting_tolerance"])), f"reported lowrank error mismatch: fixture {index}")
    require(asymmetric, "nonsymmetric roster contains no asymmetric C")


def verify_resource_result(section: dict[str, Any]) -> None:
    records = section["activation_records"]
    require(isinstance(records, list) and records, "no resource activation records")
    seen: set[tuple[int, int, str]] = set()
    for record in records:
        width = record["w"]
        rank = record["r"]
        mode = selected_mode(rank, width)
        require(record["mode"] == mode, f"mode mismatch: {record}")
        expected_flops = lr_star(rank, width) if mode == "LOWRANK64" else dh(width)
        expected_calls = 18 if mode == "LOWRANK64" else 10
        require(record["delta_flops"] == expected_flops, f"activation FLOPs mismatch: {record}")
        require(record["delta_calls"] == expected_calls, f"activation calls mismatch: {record}")
        seen.add((width, rank, mode))
    required = {(5, 1, "DENSE32"), (6, 1, "LOWRANK64"), (128, 29, "LOWRANK64"), (128, 30, "DENSE32")}
    require(required <= seen, f"resource witnesses missing: {required-seen}")
    for mixed in section["mixed_records"]:
        low = mixed["lowrank"]
        dense = mixed["dense_count"]
        expected_flops = sum(lr_star(item["r"], item["w"]) for item in low)
        require(len({item["w"] for item in low} | {mixed["w"]}) == 1, "mixed record widths differ")
        expected_flops += dense * dh(mixed["w"]) - 112
        expected_calls = 18 * len(low) + 10 * dense - 14
        require(mixed["delta_flops"] == expected_flops and mixed["delta_calls"] == expected_calls, "mixed resource law mismatch")


def flatten_numbers(value: Any) -> list[float]:
    if isinstance(value, list):
        result: list[float] = []
        for item in value:
            result.extend(flatten_numbers(item))
        return result
    require(isinstance(value, (int, float)), "plane contains non-number")
    return [float(value)]


def nested_sum(values: list[Any]) -> Any:
    first = values[0]
    if isinstance(first, list):
        require(all(isinstance(value, list) and len(value) == len(first) for value in values), "component shape mismatch")
        return [nested_sum([value[index] for value in values]) for index in range(len(first))]
    return sum(float(value) for value in values)


def max_nested_error(left: Any, right: Any) -> float:
    left_flat = flatten_numbers(left)
    right_flat = flatten_numbers(right)
    require(len(left_flat) == len(right_flat), "nested comparison shape mismatch")
    return max((abs(x - y) for x, y in zip(left_flat, right_flat)), default=0.0)


def incident_values(planes: dict[str, Any], invalid: list[bool]) -> list[float]:
    result: list[float] = []
    k4 = planes["k4"]
    for i, flag in enumerate(invalid):
        if flag:
            result.append(float(k4[i]))
    for name in ("k21", "k31", "k22"):
        matrix = planes[name]
        for i in range(len(invalid)):
            for j in range(len(invalid)):
                if invalid[i] or invalid[j]:
                    result.append(float(matrix[i][j]))
    return result


def verify_carriers(section: dict[str, Any]) -> None:
    records = section["records"]
    require(isinstance(records, list) and records, "no carrier records")
    stages = {record["stage"] for record in records}
    require({"current", "one_push"} <= stages, "carrier stage missing")
    required_components = {"pairing", "star", "cp3", "cp4"}
    for record in records:
        components = record["components"]
        require(set(components) == required_components, "carrier component roster")
        tolerance = float(record["reconstruction_tolerance"])
        for plane in ("k4", "k21", "k31", "k22"):
            reconstructed = nested_sum([components[name][plane] for name in sorted(required_components)])
            require(max_nested_error(reconstructed, record["aggregate"][plane]) <= tolerance, f"aggregate reconstruction failed: {record['stage']} {plane}")
        invalid = record["invalid_mask"]
        require(all(type(value) is bool for value in invalid), "invalid mask not Boolean")
        if record["stage"] == "current" or record["map_kind"] in {"identity", "signed_permutation"}:
            for name, planes in components.items():
                require(all(value == 0.0 for value in incident_values(planes, invalid)), f"incident carrier nonzero: {record['stage']} {name}")
        else:
            require(record["dense_spread_bound_pass"] is True, "dense spread bound failed")
        selected = record["selected_pairing_covariance"]
        width, width2 = matrix_shape(selected, "selected pairing covariance")
        require(width == width2 == len(invalid), "selected covariance shape")
        alpha = float(record["pairing_alpha"])
        diagonal = [float(selected[i][i]) for i in range(width)]
        expected_k4 = [3.0 * alpha * value * value for value in diagonal]
        expected_k31 = [[3.0 * alpha * diagonal[i] * float(selected[i][j]) for j in range(width)] for i in range(width)]
        expected_k22 = [[alpha * (diagonal[i] * diagonal[j] + 2.0 * float(selected[i][j]) * float(selected[i][j])) for j in range(width)] for i in range(width)]
        pairing = components["pairing"]
        require(max_nested_error(expected_k4, pairing["k4"]) <= tolerance, "pairing K4 reconstruction")
        require(max_nested_error(expected_k31, pairing["k31"]) <= tolerance, "pairing K31 reconstruction")
        require(max_nested_error(expected_k22, pairing["k22"]) <= tolerance, "pairing K22 reconstruction")


def verify_null_face(section: dict[str, Any]) -> None:
    records = section["records"]
    require(isinstance(records, list) and records, "no null-face record")
    expected_division = {"operation": "divide", "dtype": "float64", "shape": [], "calls": 1, "flops": 2}
    for record in records:
        require(record["denominator_hex"] == "0x0.0p+0", "null denominator not positive zero")
        require(record["alpha_hex"] == "0x0.0p+0", "null alpha not positive zero")
        require(record["ordinary_division_ledger"] == expected_division, "ordinary division ledger")
        require(record["null_division_ledger"] == expected_division, "null division ledger")
        require(record["constructor_calls"] == 0, "null constructor call present")
        require(record["k4_k31_reconstruction_pass"] is True, "null K4/K31 reconstruction")
        require(math.isfinite(float(record["k22_residual"])), "null K22 residual nonfinite")
    require(section["diagonal_nonzero_wick_counterexample_present"] is True, "null-face counterexample absent")


def verify_no_activation(section: dict[str, Any]) -> None:
    records = section["records"]
    require(isinstance(records, list) and records, "no nominal parity records")
    for record in records:
        require(record["mode_roster"] == ["NOMINAL"] * 14, "nominal mode roster")
        require(record["parent_output_bytes_sha256"] == record["candidate_output_bytes_sha256"], "nominal output bytes differ")
        require(record["parent_named_state_bytes_sha256"] == record["candidate_named_state_bytes_sha256"], "nominal state bytes differ")
        require(record["localized_flops"] == 0 and record["localized_calls"] == 0, "nominal localized work")
        require(record["scalar_where_flop_delta"] == -112 and record["scalar_where_call_delta"] == -14, "nominal scalar where delta")
        require(record["certificate_flop_delta"] == 0 and record["certificate_call_delta"] == 0, "nominal certificate delta")
        require(record["two_run_byte_deterministic"] is True and record["finite"] is True, "nominal determinism/finiteness")


def verify_probe_handoff(path: Path) -> dict[str, Any]:
    handoff = load_json(path)
    require(handoff["schema"] == "independent-cs3b-source-probe-audit-handoff-v1", "probe handoff schema")
    static_handoff = verify_binding(handoff["bindings"]["static_source_audit_handoff"], "static source handoff")
    static_result = verify_static_handoff(static_handoff)
    prereg = verify_binding(handoff["bindings"]["author_source_probe_preregister"], "probe preregister")
    runner = verify_binding(handoff["bindings"]["author_source_probe_runner"], "probe runner")
    result_path = verify_binding(handoff["bindings"]["author_source_probe_result"], "probe result")
    result = load_json(result_path)
    require(result["status"] == handoff["bindings"]["author_source_probe_result"]["status_exact"], "probe result status")
    require(result["bindings"]["candidate_estimator_sha256"] == static_result["candidate_sha256"], "probe candidate mismatch")
    require(result["bindings"]["preregister_sha256"] == sha256(prereg), "probe preregister mismatch")
    require(result["bindings"]["runner_sha256"] == sha256(runner), "probe runner mismatch")
    access = result["access"]
    require(access["targets_rows_teachers_scores_private_state"] == 0, "probe accessed target-like data")
    require(access["width1024_runs"] == 0 and access["literal_meter_runs"] == 0, "probe exceeded source scope")
    verify_certificate_result(result["certificate_equivalence"])
    verify_nonsymmetric_result(result["nonsymmetric_lowrank"])
    verify_resource_result(result["resource_law"])
    verify_carriers(result["carrier_reconstruction"])
    verify_null_face(result["null_face"])
    verify_no_activation(result["no_activation_parity"])
    require(result["determinism_and_finiteness"]["two_run_byte_determinism"] is True, "probe nondeterministic")
    require(result["determinism_and_finiteness"]["parent_finite_regressions"] == 0, "parent-finite regression")
    require(not debris({HERE, prereg.parent, runner.parent, result_path.parent}), "pycache in probe audit scope")
    return {
        "status": "PASS_INDEPENDENT_TARGET_FREE_SOURCE_PROBE_REPLAY__NO_FULLWIDTH_OR_METER_AUTHORITY",
        "static_source_audit": static_result,
        "probe_handoff_sha256": sha256(path),
        "probe_preregister_sha256": sha256(prereg),
        "probe_runner_sha256": sha256(runner),
        "probe_result_sha256": sha256(result_path),
        "candidate_imports_or_executions_by_this_harness": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--static-handoff", type=Path)
    group.add_argument("--probe-handoff", type=Path)
    args = parser.parse_args()
    if args.static_handoff is not None:
        result = verify_static_handoff(args.static_handoff.resolve())
    elif args.probe_handoff is not None:
        result = verify_probe_handoff(args.probe_handoff.resolve())
    else:
        result = verify_scaffold()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
