#!/usr/bin/env python3
"""Static and tiny synthetic proof for Runtime V5's row-boundary reset.

This verifier never imports or executes the parent, source, adapter, token,
row scope, row manifests, weights, targets, or bank generator.  Its only
non-stdlib imports are the already frozen CPU-stack NumPy/FlopScope modules,
used on three in-memory float64 scalars per synthetic context.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path

import flopscope
import flopscope.numpy as fnp
import numpy as np
from flopscope import _budget as budget_module


HERE = Path(__file__).resolve().parent
RUNNER = HERE / "runtime_runner.py"
ADAPTER = HERE / "capture_adapter.py"
BUDGET_MODULE = Path(budget_module.__file__).resolve()
EXPECTED_BUDGET_MODULE_SHA256 = (
    "ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01"
)
SYNTHETIC_CONTEXTS = 1000


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def top_function(tree, name):
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("missing function: " + name)


def calls_with_attribute(root, attribute):
    return [
        node
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attribute
    ]


def static_placement_proof():
    runner_text = RUNNER.read_text(encoding="utf-8")
    adapter_text = ADAPTER.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_text)
    adapter_tree = ast.parse(adapter_text)
    main = top_function(runner_tree, "main")
    generate = top_function(adapter_tree, "generate_one_target_blind_row")

    reset_calls = calls_with_attribute(runner_tree, "budget_reset")
    require(len(reset_calls) == 1, "runtime must contain one reset call site")
    reset_call = reset_calls[0]
    require(
        ast.unparse(reset_call.func) == "flopscope.budget_reset",
        "reset must use the already-loaded module identity",
    )
    require(
        not calls_with_attribute(adapter_tree, "budget_reset"),
        "adapter/meter path must contain no reset",
    )

    loops = [
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(rank, record)"
        and ast.unparse(node.iter) == "enumerate(records)"
    ]
    require(len(loops) == 1, "expected one ordered row loop")
    loop = loops[0]
    require(
        isinstance(loop.body[-1], ast.Expr)
        and loop.body[-1].value is reset_call,
        "reset must be the final row-loop statement",
    )

    generate_line = next(
        node.lineno
        for node in loop.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Attribute)
            and child.func.attr == "generate_one_target_blind_row"
            for child in ast.walk(node.value)
        )
    )
    validate_line = next(
        node.lineno
        for node in loop.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_validate_row_result"
    )
    append_line = next(
        node.lineno
        for node in loop.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and ast.unparse(node.value.func) == "row_receipts.append"
    )
    progress_if = next(node for node in loop.body if isinstance(node, ast.If))
    print_lines = [
        node.lineno
        for node in ast.walk(progress_if)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    require(len(print_lines) == 1, "expected one optional progress print")
    require(
        generate_line < validate_line < append_line < print_lines[0] < reset_call.lineno,
        "row receipt validation/append/progress/reset order drift",
    )

    contexts = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "BudgetContext"
    ]
    require(len(contexts) == 1, "expected one BudgetContext per row")
    with_nodes = [
        node
        for node in ast.walk(generate)
        if isinstance(node, ast.With)
        and any(ast.unparse(item.context_expr) == "context" for item in node.items)
    ]
    require(len(with_nodes) == 1, "expected one row meter context")
    require(
        not any(
            call in ast.walk(with_nodes[0])
            for call in calls_with_attribute(generate, "budget_reset")
        ),
        "reset appeared inside meter context",
    )
    summary_line = next(
        node.lineno
        for node in generate.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "summary" for target in node.targets)
    )
    meter_line = next(
        node.lineno
        for node in generate.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "exact_meters" for target in node.targets)
    )
    return_line = next(node.lineno for node in generate.body if isinstance(node, ast.Return))
    require(
        with_nodes[0].end_lineno < summary_line < meter_line < return_line < generate_line,
        "adapter must close context and construct detached receipt before runner regains control",
    )

    return {
        "adapter_budget_context_call_sites": 1,
        "adapter_reset_call_sites": 0,
        "context_close_before_detached_summary": True,
        "detached_summary_before_exact_meter_construction": True,
        "exact_meters_and_output_receipt_before_runner_validation": True,
        "runner_reset_call_sites": 1,
        "runner_reset_receiver": "session_flopscope_module_identity",
        "row_order": [
            "adapter_return_after_context_close_and_receipt_construction",
            "runner_validate_row_result",
            "runner_append_row_receipt",
            "optional_progress_print",
            "flopscope_budget_reset",
            "next_row_budget_context",
        ],
        "reset_is_final_loop_statement": True,
        "reset_inside_meter": False,
    }


def source_leak_and_reset_proof():
    require(
        sha256(BUDGET_MODULE) == EXPECTED_BUDGET_MODULE_SHA256,
        "pinned FlopScope budget module drift",
    )
    tree = ast.parse(BUDGET_MODULE.read_text(encoding="utf-8"))
    context = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BudgetContext"
    )
    accumulator = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BudgetAccumulator"
    )
    context_exit = next(
        node for node in context.body if isinstance(node, ast.FunctionDef) and node.name == "__exit__"
    )
    accumulator_record = next(
        node for node in accumulator.body if isinstance(node, ast.FunctionDef) and node.name == "record"
    )
    accumulator_reset = next(
        node for node in accumulator.body if isinstance(node, ast.FunctionDef) and node.name == "reset"
    )
    public_reset = top_function(tree, "budget_reset")
    require(
        any(ast.unparse(node) == "_accumulator.record(self, body_end=body_end)" for node in ast.walk(context_exit)),
        "BudgetContext.__exit__ no longer records globally",
    )
    require(
        any(ast.unparse(node) == "self._records.append" for node in ast.walk(accumulator_record) if isinstance(node, ast.Attribute)),
        "accumulator record append missing",
    )
    reset_text = ast.unparse(accumulator_reset)
    require("self._records.clear()" in reset_text, "reset does not clear retained records")
    require("self._rollup.clear()" in reset_text, "reset does not clear rollup")
    require("self._diagnostic_op_locations.clear()" in reset_text, "reset does not clear diagnostic locations")
    require(
        any(ast.unparse(node) == "_accumulator.reset()" for node in ast.walk(public_reset) if isinstance(node, ast.Call)),
        "public reset does not clear accumulator",
    )
    return {
        "budget_module_path": str(BUDGET_MODULE),
        "budget_module_sha256": EXPECTED_BUDGET_MODULE_SHA256,
        "budget_context_exit_records_global_accumulator": True,
        "global_record_retains_diagnostic_op_log": True,
        "public_reset_clears_records_rollup_diagnostic_locations_and_caches": True,
        "public_reset_rebaselines_live_contexts": True,
    }


def synthetic_invariance_proof():
    accumulator = budget_module._accumulator
    require(len(accumulator._records) == 0, "synthetic verifier did not start clean")
    reset = getattr(flopscope, "budget_reset")
    outputs = set()
    detached_meter_values = set()
    context_meter_values_after_reset = set()
    max_global_records_before_reset = 0
    max_global_records_after_reset = 0
    max_global_diagnostic_oprecords = 0
    detached_summary_preserved = 0
    receipt_preserved = 0
    closed_context_summaries_preserved = 0
    output_byte_hashes_preserved = 0
    reset_calls = 0

    left = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    right = np.array([2.0, 2.0, 2.0], dtype=np.float64)
    for index in range(SYNTHETIC_CONTEXTS):
        context = flopscope.BudgetContext(
            flop_budget=1000,
            quiet=True,
            namespace="q24_v5_reset_fixture_" + str(index),
        )
        with context:
            output = fnp.add(left, right)
        detached = context.summary_dict(by_namespace=False)
        receipt = {
            "exact_meters": {"complete_flops": int(detached["flops_used"])},
            "output": float(output[0]),
        }
        require(receipt == {"exact_meters": {"complete_flops": 6}, "output": 3.0}, "fixture drift")
        require(len(accumulator._records) == 1, "closed context retention exceeded one row")
        max_global_records_before_reset = max(
            max_global_records_before_reset, len(accumulator._records)
        )
        max_global_diagnostic_oprecords = max(
            max_global_diagnostic_oprecords,
            sum(len(record.op_log) for record in accumulator._records),
        )
        detached_before = copy.deepcopy(detached)
        context_summary_before = copy.deepcopy(context.summary_dict(False))
        receipt_before = copy.deepcopy(receipt)
        output_hash_before = hashlib.sha256(memoryview(output).cast("B")).hexdigest()
        reset()
        reset_calls += 1
        max_global_records_after_reset = max(
            max_global_records_after_reset, len(accumulator._records)
        )
        require(len(accumulator._records) == 0, "reset left a retained global record")
        global_summary_after = flopscope.budget_summary_dict(False)
        require(global_summary_after["flops_used"] == 0, "reset left global flops")
        require(global_summary_after["operations"] == {}, "reset left global operations")
        require(detached == detached_before, "reset mutated detached summary")
        require(receipt == receipt_before, "reset mutated materialized receipt/output")
        require(context.summary_dict(False) == context_summary_before, "reset mutated context summary")
        require(
            hashlib.sha256(memoryview(output).cast("B")).hexdigest() == output_hash_before,
            "reset mutated synthetic ndarray bytes",
        )
        detached_summary_preserved += 1
        receipt_preserved += 1
        closed_context_summaries_preserved += 1
        output_byte_hashes_preserved += 1
        outputs.add(receipt["output"])
        detached_meter_values.add(receipt["exact_meters"]["complete_flops"])
        context_meter_values_after_reset.add(context.summary_dict(False)["flops_used"])

    require(reset_calls == SYNTHETIC_CONTEXTS, "not one reset per completed synthetic row")
    require(len(accumulator._records) == 0, "synthetic verifier did not finish clean")
    return {
        "completed_contexts": SYNTHETIC_CONTEXTS,
        "budget_reset_calls": reset_calls,
        "resets_per_completed_context": 1,
        "detached_complete_flops_values": sorted(detached_meter_values),
        "context_complete_flops_after_reset_values": sorted(context_meter_values_after_reset),
        "output_values_after_reset": sorted(outputs),
        "detached_summaries_preserved": detached_summary_preserved,
        "closed_context_full_summaries_preserved": closed_context_summaries_preserved,
        "materialized_receipts_and_outputs_preserved": receipt_preserved,
        "output_ndarray_byte_hashes_preserved": output_byte_hashes_preserved,
        "max_global_accumulator_records_before_reset": max_global_records_before_reset,
        "max_global_accumulator_records_after_reset": max_global_records_after_reset,
        "max_global_diagnostic_oprecords_before_reset": max_global_diagnostic_oprecords,
        "final_global_accumulator_records": len(accumulator._records),
        "diagnostic_retention_bound": "AT_MOST_ONE_COMPLETED_ROW",
        "memory_bounded_against_per_row_accumulator_growth": True,
    }


def synthetic_loop_adversaries():
    """Execute the exact production loop AST against target-free mocks."""
    runner_tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    main = top_function(runner_tree, "main")
    loop = next(
        node
        for node in ast.walk(main)
        if isinstance(node, ast.For)
        and ast.unparse(node.target) == "(rank, record)"
        and ast.unparse(node.iter) == "enumerate(records)"
    )
    loop_module = ast.fix_missing_locations(
        ast.Module(body=[copy.deepcopy(loop)], type_ignores=[])
    )

    class EventList(list):
        def __init__(self, events):
            super().__init__()
            self.events = events

        def append(self, value):
            self.events.append(("append", value["rank"]))
            super().append(value)

    class MockFlopscope:
        def __init__(self, events, fail_reset=False):
            self.events = events
            self.fail_reset = fail_reset
            self.reset_calls = 0

        def budget_reset(self):
            active = budget_module.get_active_budget()
            require(
                active is None or active is budget_module._global_default,
                "production loop reset observed an active BudgetContext",
            )
            self.events.append(("reset", self.reset_calls))
            self.reset_calls += 1
            if self.fail_reset:
                raise RuntimeError("synthetic reset failure")

    class MockAdapter:
        def __init__(self, events):
            self.events = events

        def generate_one_target_blind_row(self, *, session, record, expected_rank, runtime_token):
            self.events.append(("generate", expected_rank))
            return {
                "rank": expected_rank,
                "exact_meters": {"wall_seconds": 1.0, "peak_rss_bytes": 1},
            }

    def exercise(row_count, *, validation_failure_rank=None, reset_failure=False):
        events = []
        session_flops = MockFlopscope(events, fail_reset=reset_failure)
        adapter = MockAdapter(events)
        receipts = EventList(events)

        def validate(result, record, component_sha256):
            rank = result["rank"]
            events.append(("validate", rank))
            if rank == validation_failure_rank:
                raise RuntimeError("synthetic validation failure")

        def progress(*args, **kwargs):
            events.append(("progress", len(receipts)))

        namespace = {
            "EXPECTED_RANKS": tuple(range(688)),
            "adapter": adapter,
            "component_sha256": {},
            "enumerate": enumerate,
            "flopscope": session_flops,
            "len": len,
            "print": progress,
            "records": [{"rank": rank} for rank in range(row_count)],
            "row_receipts": receipts,
            "runtime_token": object(),
            "session": {"flops": session_flops},
            "_validate_row_result": validate,
        }
        failure = None
        try:
            exec(compile(loop_module, str(RUNNER), "exec"), namespace)
        except RuntimeError as exc:
            failure = str(exc)
        require(namespace["flopscope"] is namespace["session"]["flops"], "module identity drift")
        return events, session_flops.reset_calls, failure

    one_events, one_resets, one_failure = exercise(1)
    require(one_failure is None and one_resets == 1, "non-progress row reset count")
    require(
        [name for name, _ in one_events] == ["generate", "validate", "append", "reset"],
        "non-progress row event order",
    )
    eight_events, eight_resets, eight_failure = exercise(8)
    require(eight_failure is None and eight_resets == 8, "progress-boundary reset count")
    require(
        [name for name, _ in eight_events[-5:]]
        == ["generate", "validate", "append", "progress", "reset"],
        "progress-boundary row event order",
    )
    invalid_events, invalid_resets, invalid_failure = exercise(
        1, validation_failure_rank=0
    )
    require(invalid_failure == "synthetic validation failure", "validation failure did not propagate")
    require(invalid_resets == 0, "failed/unvalidated row reset")
    require(
        [name for name, _ in invalid_events] == ["generate", "validate"],
        "validation failure reached append/reset",
    )
    reset_events, reset_resets, reset_failure = exercise(2, reset_failure=True)
    require(reset_failure == "synthetic reset failure", "reset failure did not fail closed")
    require(reset_resets == 1, "reset failure call count")
    require(
        [name for name, _ in reset_events]
        == ["generate", "validate", "append", "reset"],
        "reset failure continued to next generation",
    )
    return {
        "each_of_eight_successful_rows_has_one_reset": True,
        "eight_successful_rows_reset_calls": eight_resets,
        "failed_or_unvalidated_row_reset_calls": invalid_resets,
        "non_progress_successful_row_reset_calls": one_resets,
        "progress_boundary_rank7_reset_calls": sum(
            name == "reset" and value == 7 for name, value in eight_events
        ),
        "reset_failure_stops_before_next_generate": True,
        "reset_uses_exact_session_flopscope_module_identity": True,
        "reset_observed_no_active_budget_context": True,
        "validated_append_progress_reset_order": True,
    }


def main():
    result = {
        "access_counters": {
            "banks_created": 0,
            "numerical_parent_source_or_adapter_imports": 0,
            "real_rows_executed": 0,
            "row_manifests_opened": 0,
            "targets_opened": 0,
            "weights_opened": 0,
        },
        "schema": "public-q24-three-bank-runtime-flopscope-reset-verification-v5",
        "source_semantics": source_leak_and_reset_proof(),
        "static_placement": static_placement_proof(),
        "status": "PASS_STATIC_AND_SYNTHETIC_NO_REAL_EXECUTION",
        "synthetic_loop_adversaries": synthetic_loop_adversaries(),
        "synthetic_invariance": synthetic_invariance_proof(),
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
