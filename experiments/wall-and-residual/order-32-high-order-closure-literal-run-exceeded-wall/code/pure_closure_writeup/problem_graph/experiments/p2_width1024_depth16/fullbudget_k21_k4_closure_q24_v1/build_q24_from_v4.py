#!/usr/bin/env python3
"""Mechanically derive the Q24 source from the immutable full V4 closure."""

from __future__ import annotations

import ast
import hashlib
import pprint
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
OUTPUT = HERE / "estimator.py"
PARENT_SHA256 = "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
PARENT_BYTES = 68_698


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def assignment_node(tree: ast.Module, name: str) -> ast.Assign:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    if len(matches) != 1:
        raise AssertionError((name, len(matches)))
    return matches[0]


def replace_assignment(source: str, name: str, value: object) -> str:
    tree = ast.parse(source)
    node = assignment_node(tree, name)
    lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: node.lineno - 1])
    stop = sum(len(line) for line in lines[: node.end_lineno])
    rendered = f"{name} = {pprint.pformat(value, width=78, compact=True)}\n"
    return source[:start] + rendered + source[stop:]


def exact_imports(tree: ast.AST) -> list[str]:
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return sorted(imports)


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    assert len(parent_bytes) == PARENT_BYTES
    assert sha256_bytes(parent_bytes) == PARENT_SHA256
    source = parent_bytes.decode("utf-8")

    # Truthful symbol/documentation renames do not alter executable structure.
    source = source.replace("FB-EC4-32", "FB-EC4-Q24")
    source = source.replace("Q32", "Q24").replace("q32", "q24")
    source = source.replace("# 1/t!, t=0..32.", "# 1/t!, t=0..24.")
    assert source.count("_Q24_TERMS = 33") == 1
    source = source.replace("_Q24_TERMS = 33", "_Q24_TERMS = 25")

    renamed_tree = ast.parse(source)
    factorial = ast.literal_eval(assignment_node(renamed_tree, "_INV_FACTORIAL").value)
    assert len(factorial) == 33
    source = replace_assignment(source, "_INV_FACTORIAL", factorial[:25])
    q24_grid = tuple(tuple(range(derivative, derivative + 25)) for derivative in range(7))
    source = replace_assignment(source, "_Q24_ORDER_GRID", q24_grid)

    tree = ast.parse(source)
    assert exact_imports(tree) == sorted(
        [
            "flopscope",
            "flopscope.numpy",
            "whestbench.BaseEstimator",
            "whestbench.domain.MLP",
        ]
    )
    assert "Q32" not in source and "q32" not in source
    assert ast.literal_eval(assignment_node(tree, "_Q24_TERMS").value) == 25
    assert len(ast.literal_eval(assignment_node(tree, "_INV_FACTORIAL").value)) == 25
    grid = ast.literal_eval(assignment_node(tree, "_Q24_ORDER_GRID").value)
    assert grid == q24_grid
    assert "_PHASE2_FLOP_BUDGET = 2_199_023_255_552" in source
    assert source.endswith("\n")
    OUTPUT.write_text(source, encoding="utf-8")
    print(
        f"WROTE_Q24 estimator={OUTPUT} bytes={OUTPUT.stat().st_size} "
        f"sha256={sha256_bytes(OUTPUT.read_bytes())}"
    )


if __name__ == "__main__":
    main()
