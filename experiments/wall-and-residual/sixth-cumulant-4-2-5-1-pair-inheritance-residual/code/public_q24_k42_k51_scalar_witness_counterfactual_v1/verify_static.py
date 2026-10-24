#!/usr/bin/env python3
"""Static verifier for the forbidden-retry K42+K51 witness counterfactual.

The verifier parses source, proves accounting and small scalar equivalences,
and imports only the non-executing runner.  It never imports the D4 estimator,
opens a sealed row or weight, selects a rank, creates a release, or runs a
numerical source path.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import re
import struct
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
P2 = HERE.parent
GRAPH_ROOT = HERE.parents[2]
SOURCE = HERE / "k42_k51_scalar_witness_source.py"
RUNNER = HERE / "counterfactual_runner_scaffold.py"
DERIVATION = HERE / "DERIVATION.md"
PREREGISTRATION = HERE / "PREREGISTRATION.json"
STATIC_RESULT = HERE / "STATIC_VERIFICATION_V1.json"
FREEZE = HERE / "SOURCE_FREEZE_V1.json"
PARENT = P2 / "public_q24_k42_k51_pair_inheritance_v1"
PARENT_SOURCE = PARENT / "k42_k51_instrumented_source.py"
PARENT_RUNNER = PARENT / "source_action_runner_scaffold.py"
PARENT_FREEZE = PARENT / "SOURCE_FREEZE_V3.json"
PARENT_CLOSURE = PARENT / "K42_K51_RANK0201_TERMINAL_CLOSURE_RECEIPT.json"
BASE_SOURCE = P2 / "p2r0c1d11_cp3_arena_depth4_full_estimator_v1/estimator.py"
D29_PRICING = (
    P2
    / "p2r0c1d29_k42_k51_pair_inheritance_static_pricing_v1"
    / "K42_K51_STATIC_DESIGN_PRICING_RECEIPT.json"
)
GRAPH = GRAPH_ROOT / "graph.json"
RECONCILER = GRAPH_ROOT / "reconcile_20260827_k4251_terminal.py"
MARKDOWN = GRAPH_ROOT / "problem-graph.md"
HTML = GRAPH_ROOT / "problem-graph.html"

EXPECTED_PARENT_HASHES = {
    "parent_source": "547b45c219fd868d359ebb8deec9f564a89d433cd6fbdc9cc637940e164cd601",
    "parent_runner": "fd9a5f8f395f96af3569d7f2d422033bcd714c18d8b22760bb5e1fce0c9fbc8e",
    "parent_freeze": "2248c8dc67cceabdef1f1228a4d77acfd7b5d9aecd250dfb162a9e19b57a5b2a",
    "parent_closure": "3334080f68af40fa02509639e7c9a6ebc87c8888c30b21aea399d9362bae793d",
    "base_source": "6aedce54f2006efcb9d87eeb41aae6fd785d7344534634371c48a7979158342e",
    "d29_pricing": "bb8a55e3224da56139dc79fa2b65acc4cf8320f0991c7d77da5ec0892ad3b6cc",
    "graph": "84ff484740ec9390d61272e9807784396db9d944abcc84b2d2490e4e23f47e76",
    "reconciler": "e8ac39f470ae130673f2b8dca3efa55063e10778fe6d865342ea83c75d878437",
    "markdown": "ce55ed4e07680ab35c53bbec0430f4f2e61bb628ef96bc073c0e4ef67ef2264d",
    "html": "3e2a2bfa4fb07b90e88f3517f5bf6db262b27d13b4c99b0b78b41fdd5441ed23",
}
PARENT_OPERATION_CALLS = {
    "empty": 16,
    "add": 165,
    "concatenate": 30,
    "copy": 15,
    "getitem": 43,
    "multiply": 1020,
    "reshape": 15,
    "stack": 45,
    "subtract": 210,
    "einsum": 29,
    "diagonal": 15,
    "fill_diagonal": 30,
    "transpose": 15,
    "copyto": 30,
    "matmul": 15,
    "sum": 15,
}
ADDED_WITNESS_CALLS = {
    "min": 30,
    "max": 30,
    "diagonal": 15,
    "count_nonzero": 15,
    "add": 15,
    "array_equal": 15,
}
EXPECTED_COUNTERFACTUAL_CALLS = {
    "empty": 16,
    "add": 180,
    "array_equal": 15,
    "concatenate": 30,
    "copy": 15,
    "count_nonzero": 15,
    "getitem": 43,
    "max": 30,
    "min": 30,
    "multiply": 1020,
    "reshape": 15,
    "stack": 45,
    "subtract": 210,
    "einsum": 29,
    "diagonal": 30,
    "fill_diagonal": 30,
    "transpose": 15,
    "copyto": 30,
    "matmul": 15,
    "sum": 15,
}
EXPECTED_SYNTACTIC_FNP_CALLS = Counter(
    {
        "multiply": 18,
        "subtract": 9,
        "add": 5,
        "stack": 3,
        "einsum": 3,
        "concatenate": 2,
        "min": 2,
        "max": 2,
        "diagonal": 2,
        "fill_diagonal": 2,
        "empty": 2,
        "copyto": 2,
        "copy": 1,
        "reshape": 1,
        "matmul": 1,
        "sum": 1,
        "count_nonzero": 1,
        "array_equal": 1,
        "transpose": 1,
    }
)
REMOVED_DIAGNOSTIC_TOKENS = {
    "frobenius_norm",
    "l2_norm",
    "rms",
    "max_abs",
    "positive_fraction",
    "negative_fraction",
    "zero_fraction",
    "sector_cosine",
    "plane_metrics",
    "vector_metrics",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def find_function(tree: ast.AST, name: str) -> ast.FunctionDef:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise AssertionError(f"expected one function {name}, found {len(matches)}")
    return matches[0]


def fnp_calls(node: ast.AST) -> Counter[str]:
    calls: Counter[str] = Counter()
    for item in ast.walk(node):
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and isinstance(item.func.value, ast.Name)
            and item.func.value.id == "fnp"
        ):
            calls[item.func.attr] += 1
    return calls


def is_sink_call(statement: ast.stmt, method: str) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    function = statement.value.func
    return (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "sink"
        and function.attr == method
    )


def split_around_sink(function: ast.FunctionDef, method: str) -> tuple[list[ast.stmt], ast.Call, list[ast.stmt]]:
    locations = [index for index, statement in enumerate(function.body) if is_sink_call(statement, method)]
    if len(locations) != 1:
        raise AssertionError(f"expected one sink.{method} statement")
    index = locations[0]
    call = function.body[index].value
    assert isinstance(call, ast.Call)
    return function.body[:index], call, function.body[index + 1 :]


def ast_equal(left: Any, right: Any) -> bool:
    if isinstance(left, list) or isinstance(right, list):
        if not isinstance(left, list) or not isinstance(right, list):
            return False
        return len(left) == len(right) and all(
            ast_equal(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return ast.dump(left, include_attributes=False) == ast.dump(right, include_attributes=False)


def literal_assignment(tree: ast.Module, name: str) -> Any:
    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == name
        ):
            return ast.literal_eval(statement.value)
    raise AssertionError(f"literal assignment missing: {name}")


def function_assignment(function: ast.FunctionDef, name: str) -> ast.Assign:
    matches = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one {name} assignment in {function.name}")
    return matches[0]


def leading_runtime_error_guard(function: ast.FunctionDef) -> dict[str, Any]:
    """Prove a direct unconditional RuntimeError is first after the docstring."""

    body = list(function.body)
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    first = body[0] if body else None
    direct = (
        isinstance(first, ast.Raise)
        and first.cause is None
        and isinstance(first.exc, ast.Call)
        and isinstance(first.exc.func, ast.Name)
        and first.exc.func.id == "RuntimeError"
        and len(first.exc.args) == 1
        and isinstance(first.exc.args[0], ast.Constant)
        and first.exc.args[0].value == "COUNTERFACTUAL_FORBIDDEN_RETRY_NO_RELEASE"
        and not first.exc.keywords
    )
    side_effect_free_signature = (
        not function.decorator_list
        and not function.args.defaults
        and all(default is None for default in function.args.kw_defaults)
    )
    return {
        "direct_unconditional_runtime_error_first": direct,
        "no_decorator_or_default_expression_before_guard": side_effect_free_signature,
        "guard_lineno": first.lineno if direct else None,
    }


def source_execution_guard_proof(source: ast.Module) -> dict[str, Any]:
    """Join every dormant base-import/install/predict path to its first guard."""

    load_base = find_function(source, "_load_base")
    install = find_function(source, "_install")
    predict = find_function(source, "predict_with_pair_tape")
    functions = {
        "_load_base": load_base,
        "_install": install,
        "predict_with_pair_tape": predict,
    }
    guards = {name: leading_runtime_error_guard(function) for name, function in functions.items()}

    d4_import_calls = [
        node
        for node in ast.walk(load_base)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"spec_from_file_location", "module_from_spec", "exec_module"}
    ]
    base_loads = [
        node
        for node in ast.walk(install)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id == "base"
    ]
    public_path_calls = [
        node
        for node in ast.walk(predict)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id in {"_load_base", "_install"})
            or (isinstance(node.func, ast.Attribute) and node.func.attr in {"Estimator", "predict"})
        )
    ]
    original_predict_calls = [
        node
        for node in ast.walk(install)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Subscript)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "originals"
        and isinstance(node.func.slice, ast.Constant)
        and node.func.slice.value == "predict"
    ]
    sys_modules_mutations = [
        node
        for function in (load_base, predict)
        for node in ast.walk(function)
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Attribute)
                and isinstance(target.value.value, ast.Name)
                and target.value.value.id == "sys"
                and target.value.attr == "modules"
                for target in node.targets
            )
        )
        or (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "pop"
            and isinstance(node.func.value, ast.Attribute)
            and isinstance(node.func.value.value, ast.Name)
            and node.func.value.value.id == "sys"
            and node.func.value.attr == "modules"
        )
    ]
    forbidden_dynamic_names = sorted(
        {
            node.func.id
            for node in ast.walk(source)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec", "compile", "__import__", "getattr"}
        }
    )
    sensitive = {
        "_load_base": d4_import_calls,
        "_install": base_loads,
        "predict_with_pair_tape": public_path_calls,
    }
    all_after_guard = {
        name: bool(nodes)
        and guards[name]["guard_lineno"] is not None
        and all(node.lineno > guards[name]["guard_lineno"] for node in nodes)
        for name, nodes in sensitive.items()
    }
    return {
        "entrypoint_guards": guards,
        "sensitive_node_counts": {name: len(nodes) for name, nodes in sensitive.items()},
        "all_sensitive_nodes_lexically_after_guard": all_after_guard,
        "all_entrypoints_fail_closed_before_sensitive_path": (
            all(
                item["direct_unconditional_runtime_error_first"]
                and item["no_decorator_or_default_expression_before_guard"]
                for item in guards.values()
            )
            and all(all_after_guard.values())
            and len(original_predict_calls) == 1
            and all(
                node.lineno > guards["_install"]["guard_lineno"]
                for node in original_predict_calls
            )
            and len(sys_modules_mutations) == 2
            and all(
                (
                    node.lineno > guards["_load_base"]["guard_lineno"]
                    if node.lineno < predict.lineno
                    else node.lineno > guards["predict_with_pair_tape"]["guard_lineno"]
                )
                for node in sys_modules_mutations
            )
            and not forbidden_dynamic_names
        ),
        "original_internal_predict_call_count": len(original_predict_calls),
        "guarded_sys_modules_mutation_count": len(sys_modules_mutations),
        "forbidden_dynamic_dispatch_names": forbidden_dynamic_names,
    }


def source_nonmutation_proof() -> dict[str, Any]:
    parent_text = PARENT_SOURCE.read_text()
    source_text = SOURCE.read_text()
    parent = ast.parse(parent_text)
    source = ast.parse(source_text)

    exact_functions = (
        "centered_rows_q12",
        "birth_source",
        "initialize_with_source",
        "reset_with_source",
        "predict_with_tape_capture",
        "transport_with_capture",
        "restore",
    )
    exact_function_parity = {
        name: ast_equal(find_function(parent, name), find_function(source, name))
        for name in exact_functions
    }

    parent_connected = find_function(parent, "assemble_connected_pairs")
    source_connected = find_function(source, "assemble_connected_pairs")
    parent_prefix, parent_capture, parent_suffix = split_around_sink(parent_connected, "capture_source")
    source_prefix, source_capture, source_suffix = split_around_sink(source_connected, "capture_source")
    connected_math_parity = ast_equal(parent_prefix, source_prefix) and ast_equal(parent_suffix, source_suffix)
    scalar_plane_capture = (
        len(source_capture.args) == 2
        and isinstance(source_capture.args[1], ast.Call)
        and isinstance(source_capture.args[1].func, ast.Name)
        and source_capture.args[1].func.id == "plane_scalar_witness"
        and len(source_capture.args[1].args) == 1
        and isinstance(source_capture.args[1].args[0], ast.Name)
        and source_capture.args[1].args[0].id == "planes"
    )
    parent_plane_capture = [ast.dump(arg, include_attributes=False) for arg in parent_capture.args]

    parent_action = find_function(parent, "cp_push_with_action")
    source_action = find_function(source, "cp_push_with_action")
    parent_action_prefix, _parent_action_capture, parent_action_suffix = split_around_sink(parent_action, "capture_action")
    source_action_prefix, source_action_capture, source_action_suffix = split_around_sink(source_action, "capture_action")
    action_byte_path_parity = ast_equal(parent_action_prefix, source_action_prefix) and ast_equal(
        parent_action_suffix, source_action_suffix
    )
    action_scalar_capture = (
        len(source_action_capture.args) == 4
        and isinstance(source_action_capture.args[1], ast.Name)
        and source_action_capture.args[1].id == "raw_by_sector"
        and isinstance(source_action_capture.args[2], ast.Name)
        and source_action_capture.args[2].id == "raw_total"
        and isinstance(source_action_capture.args[3], ast.Call)
        and isinstance(source_action_capture.args[3].func, ast.Name)
        and source_action_capture.args[3].func.id == "action_scalar_witness"
    )

    parent_predict = find_function(parent, "predict_with_pair_tape")
    source_predict = find_function(source, "predict_with_pair_tape")
    public_output_assembly_parity = ast_equal(
        function_assignment(parent_predict, "output"),
        function_assignment(source_predict, "output"),
    ) and ast_equal(parent_predict.body[-1], source_predict.body[-1])
    public_estimator_calls = [
        node
        for node in ast.walk(source_predict)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "predict"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Attribute)
        and node.func.value.func.attr == "Estimator"
    ]
    exact_public_estimator_call = (
        len(public_estimator_calls) == 1
        and len(public_estimator_calls[0].args) == 2
        and isinstance(public_estimator_calls[0].args[0], ast.Name)
        and public_estimator_calls[0].args[0].id == "mlp"
        and isinstance(public_estimator_calls[0].args[1], ast.Attribute)
        and isinstance(public_estimator_calls[0].args[1].value, ast.Name)
        and public_estimator_calls[0].args[1].value.id == "base"
        and public_estimator_calls[0].args[1].attr == "_PHASE2_FLOP_BUDGET"
        and not public_estimator_calls[0].keywords
    )

    plane_witness = find_function(source, "plane_scalar_witness")
    action_witness = find_function(source, "action_scalar_witness")
    plane_calls = fnp_calls(plane_witness)
    action_calls = fnp_calls(action_witness)
    witness_out_keywords = [
        keyword.arg
        for function in (plane_witness, action_witness)
        for call in ast.walk(function)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if keyword.arg == "out"
    ]
    source_imports_plain_numpy = any(
        (
            isinstance(node, ast.Import)
            and any(alias.name == "numpy" or alias.name.startswith("numpy.") for alias in node.names)
        )
        or (
            isinstance(node, ast.ImportFrom)
            and node.module is not None
            and (node.module == "numpy" or node.module.startswith("numpy."))
        )
        for node in ast.walk(source)
    )

    ledger = literal_assignment(source, "OPERATION_LEDGER")
    execution_guard = source_execution_guard_proof(source)
    return {
        "exact_unmodified_function_ast_parity": exact_function_parity,
        "connected_math_ast_parity_around_scalar_capture": connected_math_parity,
        "parent_capture_was_full_planes": parent_plane_capture,
        "new_capture_is_scalar_witness_only": scalar_plane_capture,
        "action_and_combined_construction_ast_parity": action_byte_path_parity,
        "new_action_capture_preserves_existing_arrays_and_adds_scalar_witness": action_scalar_capture,
        "public_output_assembly_ast_parity": public_output_assembly_parity,
        "full_public_estimator_predict_call_count": len(public_estimator_calls),
        "full_public_estimator_predict_call_is_exact": exact_public_estimator_call,
        "plane_witness_fnp_calls": dict(plane_calls),
        "action_witness_fnp_calls": dict(action_calls),
        "witness_has_no_out_mutation": not witness_out_keywords,
        "source_imports_plain_numpy": source_imports_plain_numpy,
        "complete_syntactic_fnp_calls": dict(fnp_calls(source)),
        "operation_ledger": ledger,
        "execution_guard_proof": execution_guard,
    }


def runner_boundary_proof() -> dict[str, Any]:
    text = RUNNER.read_text()
    tree = ast.parse(text)
    arguments = []
    for call in ast.walk(tree):
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and call.func.attr == "add_argument"
            and call.args
            and isinstance(call.args[0], ast.Constant)
        ):
            arguments.append(call.args[0].value)
    functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    capture_source = find_function(tree, "capture_source")
    capture_action = find_function(tree, "capture_action")
    capture_source_text = ast.get_source_segment(text, capture_source) or ""
    capture_action_text = ast.get_source_segment(text, capture_action) or ""
    finalize = find_function(tree, "finalize")
    retained_assignments = [
        node
        for node in ast.walk(capture_action)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Subscript)
            and isinstance(target.value, ast.Attribute)
            and isinstance(target.value.value, ast.Name)
            and target.value.value.id == "self"
            and target.value.attr == "combined"
            for target in node.targets
        )
    ]
    bank_allocations = [
        node
        for node in ast.walk(finalize)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "bank" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and isinstance(node.value.func.value, ast.Name)
        and node.value.func.value.id == "np"
        and node.value.func.attr == "empty"
    ]
    combined_clears = [
        node
        for node in ast.walk(finalize)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "clear"
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "combined"
    ]
    direct_bank_returns = [
        node
        for node in ast.walk(finalize)
        if isinstance(node, ast.Return)
        and isinstance(node.value, ast.Tuple)
        and len(node.value.elts) == 2
        and isinstance(node.value.elts[1], ast.Name)
        and node.value.elts[1].id == "bank"
    ]
    removed_tokens_present = sorted(
        token
        for token in REMOVED_DIAGNOSTIC_TOKENS
        if re.search(rf"\b{re.escape(token)}\b", text)
    )

    forbidden_runner_imports = sorted(
        {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if alias.name in {"importlib", "runpy", "subprocess"}
        }
        | {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module in {"importlib", "runpy", "subprocess"}
        }
    )
    forbidden_runner_calls = sorted(
        {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"eval", "exec", "compile", "__import__", "getattr"}
        }
    )
    runner_predict_attributes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in {"Estimator", "predict"}
    ]
    runner_source_or_base_imports = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            "k42_k51_scalar_witness_source" in ast.dump(node)
            or "estimator" in ast.dump(node).lower()
        )
    ]
    preimport_static_safety = (
        not forbidden_runner_imports
        and not forbidden_runner_calls
        and not runner_predict_attributes
        and not runner_source_or_base_imports
        and arguments == ["--selftest"]
        and "execute" not in functions
    )
    if not preimport_static_safety:
        raise AssertionError("runner failed static safety proof before verifier import")

    spec = importlib.util.spec_from_file_location("k4251_counterfactual_runner_static", RUNNER)
    if spec is None or spec.loader is None:
        raise AssertionError("cannot import counterfactual runner")
    module = importlib.util.module_from_spec(spec)
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
    selftest = module.counterfactual_selftest()
    hold = module.static_hold_payload()
    bytecode_artifacts_after_import = sorted(
        str(path.relative_to(HERE)) for path in HERE.rglob("*.pyc")
    )

    return {
        "cli_arguments": arguments,
        "function_names": sorted(functions),
        "capture_source_imports_or_mentions_numpy": "numpy" in capture_source_text or "np." in capture_source_text,
        "capture_source_mentions_full_plane_names": any(
            token in capture_source_text for token in ("k42", "k51", "planes", "asarray")
        ),
        "capture_action_fetches_sector_array": "np.asarray(_raw_by_sector_remote)" in capture_action_text,
        "capture_action_fetches_only_combined": "np.asarray(raw_total_remote)" in capture_action_text,
        "client_bank_lifetime_ast": {
            "capture_retains_combined_vector": len(retained_assignments) == 1,
            "final_bank_allocation_count": len(bank_allocations),
            "retained_vectors_cleared_after_final_bank_allocation": (
                len(bank_allocations) == 1
                and len(combined_clears) == 1
                and bank_allocations[0].lineno < combined_clears[0].lineno
            ),
            "preallocated_c_order_bank_returned_directly": len(direct_bank_returns) == 1,
        },
        "removed_diagnostic_tokens_present": removed_tokens_present,
        "preimport_static_safety": preimport_static_safety,
        "forbidden_runner_imports": forbidden_runner_imports,
        "forbidden_runner_calls": forbidden_runner_calls,
        "runner_predict_attribute_count": len(runner_predict_attributes),
        "runner_source_or_base_import_count": len(runner_source_or_base_imports),
        "bytecode_artifacts_after_import": bytecode_artifacts_after_import,
        "selftest": selftest,
        "hold": hold,
    }


def _propagating_min(values: tuple[float, ...]) -> float:
    return math.nan if any(math.isnan(value) for value in values) else min(values)


def _propagating_max(values: tuple[float, ...]) -> float:
    return math.nan if any(math.isnan(value) for value in values) else max(values)


def scalar_gate_equivalence_proof() -> dict[str, Any]:
    alphabet = (-math.inf, -2.0, -0.0, 0.0, 3.0, math.inf, math.nan)
    tested = 0
    finite_equivalent = True
    nonzero_equivalent_after_finite = True
    for length in (1, 2, 3):
        for values in itertools.product(alphabet, repeat=length):
            tested += 1
            minimum = _propagating_min(values)
            maximum = _propagating_max(values)
            direct_finite = all(math.isfinite(value) for value in values)
            witness_finite = math.isfinite(minimum) and math.isfinite(maximum)
            finite_equivalent &= direct_finite == witness_finite
            if direct_finite:
                direct_nonzero = any(value != 0.0 for value in values)
                witness_nonzero = minimum < 0.0 or maximum > 0.0
                nonzero_equivalent_after_finite &= direct_nonzero == witness_nonzero

    diagonals = ((0.0, -0.0), (0.0, 1.0), (-2.0, 0.0))
    diagonal_equivalent = all(
        (sum(value != 0.0 for value in values) == 0)
        == all(value == 0.0 for value in values)
        for values in diagonals
    )
    sum_pairs = ((1.0, 2.0), (-0.0, -0.0), (1.0e-200, -1.0e-200), (math.inf, 1.0))
    duplicate_uint64_equivalent = True
    for left, right in sum_pairs:
        duplicate_bytes = struct.pack("<d", left + right)
        duplicate_bits = struct.unpack("<Q", duplicate_bytes)[0]
        same = struct.unpack("<d", struct.pack("<Q", duplicate_bits))[0]
        changed = struct.unpack("<d", struct.pack("<Q", duplicate_bits ^ 1))[0]
        duplicate_uint64_equivalent &= struct.pack("<d", same) == duplicate_bytes
        duplicate_uint64_equivalent &= struct.pack("<d", changed) != duplicate_bytes
    return {
        "enumerated_nonempty_arrays": tested,
        "finite_extrema_equivalent": finite_equivalent,
        "nonzero_extrema_equivalent_after_finite": nonzero_equivalent_after_finite,
        "zero_diagonal_count_equivalent": diagonal_equivalent,
        "uint64_duplicate_sum_byte_identity_equivalent": duplicate_uint64_equivalent,
    }


def terminal_authority_proof() -> dict[str, Any]:
    observed_hashes = {
        "parent_source": sha256(PARENT_SOURCE),
        "parent_runner": sha256(PARENT_RUNNER),
        "parent_freeze": sha256(PARENT_FREEZE),
        "parent_closure": sha256(PARENT_CLOSURE),
        "base_source": sha256(BASE_SOURCE),
        "d29_pricing": sha256(D29_PRICING),
        "graph": sha256(GRAPH),
        "reconciler": sha256(RECONCILER),
        "markdown": sha256(MARKDOWN),
        "html": sha256(HTML),
    }
    closure = json.loads(PARENT_CLOSURE.read_text())
    graph = json.loads(GRAPH.read_text())
    nodes = [node for node in graph["nodes"] if node.get("id") == "P2R0C1D29-K42K51-S0"]
    if len(nodes) != 1:
        raise AssertionError("terminal graph node missing or duplicated")
    node = nodes[0]
    outgoing = [edge for edge in graph["edges"] if edge.get("from") == node["id"]]
    package_fragment = HERE.name
    referenced_by_graph = any(
        package_fragment in evidence
        for graph_node in graph["nodes"]
        for evidence in graph_node.get("evidence", [])
    )
    return {
        "observed_hashes": observed_hashes,
        "all_authority_hashes_match": observed_hashes == EXPECTED_PARENT_HASHES,
        "closure_status": closure.get("status"),
        "closure_gate_count": closure.get("gate_verdict", {}).get("gate_count"),
        "closure_passing_gate_count": closure.get("gate_verdict", {}).get("passing_gate_count"),
        "closure_failed_gates": closure.get("gate_verdict", {}).get("exact_failed_gates"),
        "closure_B_residual_seconds": closure.get("gate_verdict", {}).get("sole_failure", {}).get("B_seconds"),
        "closure_residual_limit_seconds": closure.get("gate_verdict", {}).get("sole_failure", {}).get("hard_limit_seconds_strictly_below"),
        "closure_retry_allowed": closure.get("finality", {}).get("retry_allowed"),
        "closure_rank643_allowed": closure.get("finality", {}).get("rank643_authorized"),
        "graph_node_status": node.get("status"),
        "graph_node_terminal_criterion": node.get("terminal_criterion"),
        "graph_node_decomposition_children": node.get("decomposition_children"),
        "graph_node_outgoing_edges": outgoing,
        "counterfactual_package_referenced_by_graph": referenced_by_graph,
    }


def package_binding_proof() -> dict[str, Any]:
    prereg = json.loads(PREREGISTRATION.read_text())
    bindings = prereg.get("package_bindings", {})
    observed = {
        "DERIVATION.md": sha256(DERIVATION),
        "k42_k51_scalar_witness_source.py": sha256(SOURCE),
        "counterfactual_runner_scaffold.py": sha256(RUNNER),
        "verify_static.py": sha256(Path(__file__)),
    }
    forbidden_runtime_files = sorted(
        path.name
        for path in HERE.iterdir()
        if path.is_file()
        and (
            "RELEASE" in path.name
            or "RUN.lock" in path.name
            or "WORKER_CONSUMED" in path.name
            or "BUNDLE" in path.name
            or path.suffix == ".npy"
        )
    )
    expected_package_files = {
        "DERIVATION.md",
        "PREREGISTRATION.json",
        "SOURCE_FREEZE_V1.json",
        "STATIC_VERIFICATION_V1.json",
        "counterfactual_runner_scaffold.py",
        "k42_k51_scalar_witness_source.py",
        "verify_static.py",
    }
    observed_package_files = sorted(
        str(path.relative_to(HERE)) for path in HERE.rglob("*") if path.is_file()
    )
    return {
        "preregistration_schema": prereg.get("schema"),
        "preregistration_status": prereg.get("status"),
        "observed_bindings": observed,
        "declared_bindings": bindings,
        "all_bindings_match": observed == bindings,
        "forbidden_runtime_files": forbidden_runtime_files,
        "expected_package_files": sorted(expected_package_files),
        "observed_package_files": observed_package_files,
        "unexpected_package_files": sorted(set(observed_package_files) - expected_package_files),
        "rank_selection_law_present": prereg.get("governance", {}).get("rank_selection_law_present"),
        "release_path_present": prereg.get("governance", {}).get("release_path_present"),
        "execute_path_present": prereg.get("governance", {}).get("execute_path_present"),
        "graph_mutation_authority": prereg.get("governance", {}).get("graph_mutation_authority"),
    }


def accounting_proof(source_proof: dict[str, Any]) -> dict[str, Any]:
    n = 1024
    per_layer = 8 * n * n + 16 * n - 16
    added = 15 * per_layer
    parent_integrated = 151_825_424_384
    counterfactual = parent_integrated + added
    cap = 152_200_000_000
    b_total = 1_510_343_464_319 + added
    phase2 = 2**41
    ledger = source_proof["operation_ledger"]
    return {
        "width": n,
        "per_layer_formula_value": per_layer,
        "added_witness_flops": added,
        "counterfactual_integrated_incremental_flops": counterfactual,
        "hard_incremental_cap_flops": cap,
        "cap_headroom_flops": cap - counterfactual,
        "counterfactual_standalone_dense_diagnostic_flops": 152_024_121_344 + added,
        "counterfactual_B_total_flops": b_total,
        "counterfactual_B_margin_below_2pow41_flops": phase2 - b_total,
        "added_operation_calls": ADDED_WITNESS_CALLS,
        "counterfactual_operation_calls": EXPECTED_COUNTERFACTUAL_CALLS,
        "ledger_matches": (
            ledger["scalar_witness_flops"] == added
            and ledger["counterfactual_predicted_integrated_incremental_flops"] == counterfactual
            and ledger["hard_incremental_flop_cap"] == cap
            and ledger["counterfactual_cap_headroom_flops"] == cap - counterfactual
            and ledger["counterfactual_B_total_flops"] == b_total
            and ledger["counterfactual_B_margin_below_2pow41_flops"] == phase2 - b_total
            and ledger["added_scalar_witness_operation_calls"] == ADDED_WITNESS_CALLS
            and ledger["official_operation_call_deltas"] == EXPECTED_COUNTERFACTUAL_CALLS
        ),
        "tiny_n3_formula_flops": 8 * 3 * 3 + 16 * 3 - 16,
        "parent_plus_added_call_identity": {
            key: PARENT_OPERATION_CALLS.get(key, 0) + ADDED_WITNESS_CALLS.get(key, 0)
            for key in EXPECTED_COUNTERFACTUAL_CALLS
        },
        "parent_plus_added_call_identity_matches": all(
            PARENT_OPERATION_CALLS.get(key, 0) + ADDED_WITNESS_CALLS.get(key, 0) == value
            for key, value in EXPECTED_COUNTERFACTUAL_CALLS.items()
        ),
        "retained_combined_vector_bank_bytes": 15 * n * 8,
        "final_combined_bank_bytes": 15 * n * 8,
        "simultaneous_client_combined_bank_bytes": 2 * 15 * n * 8,
        "resource_projection_bytes": 7_043_162_112 + 2 * 15 * n * 8,
        "resource_projection_margin_below_8gib_bytes": 2**33 - (7_043_162_112 + 2 * 15 * n * 8),
        "largest_scalar_witness_scratch_bytes": n * 8,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    terminal = terminal_authority_proof()
    source = source_nonmutation_proof()
    scalar = scalar_gate_equivalence_proof()
    bindings = package_binding_proof()
    if not bindings["all_bindings_match"]:
        raise AssertionError("package hashes failed before runner import")
    if not source["execution_guard_proof"]["all_entrypoints_fail_closed_before_sensitive_path"]:
        raise AssertionError("source execution guards failed before runner import")
    runner = runner_boundary_proof()
    accounting = accounting_proof(source)
    ledger = source["operation_ledger"]

    gates = {
        "terminal_parent_hashes_exact": terminal["all_authority_hashes_match"],
        "terminal_parent_closed_33_of_34_on_residual_only": (
            terminal["closure_gate_count"] == 34
            and terminal["closure_passing_gate_count"] == 33
            and terminal["closure_failed_gates"] == ["all_official_residual_seconds_strictly_below_0_4"]
            and terminal["closure_B_residual_seconds"] == 0.5109498272649944
            and terminal["closure_residual_limit_seconds"] == 0.4
            and terminal["closure_retry_allowed"] is False
            and terminal["closure_rank643_allowed"] is False
        ),
        "canonical_graph_closed_no_successor": (
            terminal["graph_node_status"] == "closed_measured"
            and terminal["graph_node_decomposition_children"] == []
            and terminal["graph_node_outgoing_edges"] == []
            and not terminal["counterfactual_package_referenced_by_graph"]
            and "resource-residual-only" in terminal["graph_node_terminal_criterion"]
            and "do not retry" in terminal["graph_node_terminal_criterion"]
            and "rank 643" in terminal["graph_node_terminal_criterion"]
        ),
        "scientific_core_ast_unchanged": (
            all(source["exact_unmodified_function_ast_parity"].values())
            and source["connected_math_ast_parity_around_scalar_capture"]
            and source["action_and_combined_construction_ast_parity"]
        ),
        "source_execution_unconditionally_guarded_before_base_import_or_predict": (
            source["execution_guard_proof"]["all_entrypoints_fail_closed_before_sensitive_path"]
            and source["execution_guard_proof"]["sensitive_node_counts"]
            == {"_load_base": 3, "_install": 24, "predict_with_pair_tape": 4}
            and source["full_public_estimator_predict_call_is_exact"]
        ),
        "full_planes_never_cross_plain_numpy_boundary": (
            source["new_capture_is_scalar_witness_only"]
            and not source["source_imports_plain_numpy"]
            and not runner["capture_source_imports_or_mentions_numpy"]
            and not runner["capture_source_mentions_full_plane_names"]
            and runner["selftest"]["full_plane_payload_rejected"]
        ),
        "source_action_and_combined_bytes_nonmutating": (
            source["action_and_combined_construction_ast_parity"]
            and source["new_action_capture_preserves_existing_arrays_and_adds_scalar_witness"]
            and source["witness_has_no_out_mutation"]
        ),
        "literal_witness_calls_exact": (
            source["plane_witness_fnp_calls"] == {"min": 1, "max": 1, "diagonal": 1, "count_nonzero": 1}
            and source["action_witness_fnp_calls"] == {"min": 1, "max": 1, "add": 1, "array_equal": 1}
            and Counter(source["complete_syntactic_fnp_calls"]) == EXPECTED_SYNTACTIC_FNP_CALLS
        ),
        "scalar_gate_equivalence_exhaustive_tiny": all(
            value for key, value in scalar.items() if key != "enumerated_nonempty_arrays"
        ),
        "removed_diagnostics_absent_from_source_and_runner": not runner["removed_diagnostic_tokens_present"]
        and not any(
            re.search(rf"\b{re.escape(token)}\b", SOURCE.read_text())
            for token in REMOVED_DIAGNOSTIC_TOKENS
        ),
        "exact_added_126074640F": accounting["added_witness_flops"] == 126_074_640
        and accounting["tiny_n3_formula_flops"] == 104,
        "exact_counterfactual_delta_and_cap": (
            accounting["counterfactual_integrated_incremental_flops"] == 151_951_499_024
            and accounting["hard_incremental_cap_flops"] == 152_200_000_000
            and accounting["cap_headroom_flops"] == 248_500_976
            and accounting["ledger_matches"]
        ),
        "exact_counterfactual_total_and_budget": (
            accounting["counterfactual_B_total_flops"] == 1_510_469_538_959
            and accounting["counterfactual_B_margin_below_2pow41_flops"] == 688_553_716_593
        ),
        "exact_added_and_complete_call_inventory": accounting["parent_plus_added_call_identity_matches"],
        "lean_resource_projection": (
            accounting["retained_combined_vector_bank_bytes"] == 122_880
            and accounting["final_combined_bank_bytes"] == 122_880
            and accounting["simultaneous_client_combined_bank_bytes"] == 245_760
            and accounting["resource_projection_bytes"] == 7_043_407_872
            and accounting["resource_projection_margin_below_8gib_bytes"] == 1_546_526_720
            and accounting["largest_scalar_witness_scratch_bytes"] == 8_192
            and ledger["simultaneous_client_combined_bank_bytes"] == 245_760
            and runner["hold"]["resource_projection"]
            == {
                "retained_combined_vector_bank_bytes": 122_880,
                "final_combined_bank_bytes": 122_880,
                "simultaneous_client_combined_bank_bytes": 245_760,
                "counterfactual_projected_peak_rss_bytes": 7_043_407_872,
            }
            and all(runner["client_bank_lifetime_ast"].values())
            and runner["selftest"]["simultaneous_client_bank_lifetime_exact"]
        ),
        "runner_is_nonexecuting_hold_only": (
            runner["cli_arguments"] == ["--selftest"]
            and runner["preimport_static_safety"]
            and not runner["bytecode_artifacts_after_import"]
            and "execute" not in runner["function_names"]
            and "require_release" not in runner["function_names"]
            and "load_sealed_row" not in runner["function_names"]
            and runner["hold"]["execute_path_present"] is False
            and runner["hold"]["release_path_present"] is False
            and runner["hold"]["rank_selection_law_present"] is False
            and runner["hold"]["source_execution_guard_contract"]
            == {
                "entrypoints": ["_load_base", "_install", "predict_with_pair_tape"],
                "exact_direct_runtime_error_message": "COUNTERFACTUAL_FORBIDDEN_RETRY_NO_RELEASE",
                "guard_precedes_base_import_install_and_predict": True,
                "verified_statically_not_executed": True,
            }
        ),
        "runner_fetches_only_combined_vector": (
            not runner["capture_action_fetches_sector_array"]
            and runner["capture_action_fetches_only_combined"]
        ),
        "full_public_parity_is_contract_not_claim": (
            runner["hold"]["full_public_parity_contract"]["components"]
            == ["host", "native_raw20", "native_sigma", "native_j_correction", "final_public_prediction"]
            and runner["hold"]["full_public_parity_contract"]["A1_B_A2_byte_parity_required"]
            and not runner["hold"]["full_public_parity_contract"]["claimed_or_executed_here"]
            and source["public_output_assembly_ast_parity"]
            and source["full_public_estimator_predict_call_count"] == 1
        ),
        "package_hash_bindings_exact": bindings["all_bindings_match"],
        "no_release_rank_runtime_or_graph_authority": (
            not bindings["forbidden_runtime_files"]
            and not bindings["unexpected_package_files"]
            and bindings["rank_selection_law_present"] is False
            and bindings["release_path_present"] is False
            and bindings["execute_path_present"] is False
            and bindings["graph_mutation_authority"] is False
        ),
    }
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise AssertionError(f"static counterfactual verification failed: {failed}")

    result = {
        "schema": "q24-k42-k51-source-scalar-witness-counterfactual-static-v1",
        "status": "PASS_STATIC_COUNTERFACTUAL_ONLY__INDEPENDENT_REVIEW_REQUIRED__NO_RELEASE_RANK_ROW_ESTIMATOR_OR_RUN",
        "terminal_authority": terminal,
        "source_nonmutation_proof": source,
        "runner_boundary_proof": runner,
        "scalar_gate_equivalence_proof": scalar,
        "accounting_proof": accounting,
        "package_binding_proof": bindings,
        "gates": gates,
        "firewall": {
            "row_opened": False,
            "weights_opened": False,
            "rank_selected": False,
            "release_created": False,
            "estimator_imported": False,
            "estimator_called": False,
            "source_called": False,
            "target_or_score_read": False,
            "graph_mutated": False,
            "submission_made": False,
        },
    }
    if args.out is None:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        if args.out.resolve() != STATIC_RESULT.resolve():
            raise AssertionError("static result may only be written to the canonical path")
        atomic_json(args.out, result)
        print(args.out)


if __name__ == "__main__":
    main()
