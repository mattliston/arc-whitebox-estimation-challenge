#!/usr/bin/env python3
"""Mechanically derive the Q32 fp32 pair-kernel source from frozen full V4."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
OUTPUT = HERE / "estimator.py"
PARENT_SHA256 = "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"
PARENT_BYTES = 68_698

CANDIDATE_FUNCTION = '''def _g_four_all(context):
    """Evaluate all 17 four-moment kernels in fp32, then return fp64."""
    _features, pair_left, pair_right, rho_powers, inverse_factorial = context
    pair_left32 = fnp.asarray(pair_left, dtype=fnp.float32)
    pair_right32 = fnp.asarray(pair_right, dtype=fnp.float32)
    rho_powers32 = fnp.asarray(rho_powers, dtype=fnp.float32)
    inverse_factorial32 = fnp.asarray(inverse_factorial, dtype=fnp.float32)
    kernels32 = fnp.einsum(
        "mdti,mdtj,tij,t->mdij",
        pair_left32,
        pair_right32,
        rho_powers32,
        inverse_factorial32,
        optimize=[(0, 3), (0, 1, 2)],
    )
    return fnp.asarray(kernels32, dtype=fnp.float64)
'''


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError((name, len(matches)))
    return matches[0]


def imports(tree: ast.AST) -> list[str]:
    result: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            result.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return sorted(result)


def main() -> None:
    parent_bytes = PARENT.read_bytes()
    assert len(parent_bytes) == PARENT_BYTES
    assert sha256_bytes(parent_bytes) == PARENT_SHA256
    parent_source = parent_bytes.decode("utf-8")
    parent_tree = ast.parse(parent_source)
    old = function(parent_tree, "_g_four_all")

    lines = parent_source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[: old.lineno - 1])
    stop = sum(len(line) for line in lines[: old.end_lineno])
    candidate_source = parent_source[:start] + CANDIDATE_FUNCTION + parent_source[stop:]
    candidate_tree = ast.parse(candidate_source)

    assert imports(candidate_tree) == imports(parent_tree)
    normalized = copy.deepcopy(candidate_tree)
    normalized.body[normalized.body.index(function(normalized, "_g_four_all"))] = copy.deepcopy(old)
    assert ast.dump(normalized, include_attributes=False) == ast.dump(
        parent_tree, include_attributes=False
    )
    assert candidate_source.endswith("\n")

    OUTPUT.write_text(candidate_source, encoding="utf-8")
    candidate_bytes = OUTPUT.read_bytes()
    print(
        "WROTE_Q32_FP32_PAIR "
        f"bytes={len(candidate_bytes)} sha256={sha256_bytes(candidate_bytes)}"
    )


if __name__ == "__main__":
    main()
