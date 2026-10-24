#!/usr/bin/env python3
"""Read-only structural verifier for the additive full-V4 Q24 source."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "fullbudget_k21_k4_closure_v4" / "estimator.py"
CHILD = HERE / "estimator.py"
PARENT_SHA256 = "3ddddd87facd110d4f9a4bccaa865d667b59d919f6df8d7217534483e197847c"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def assignment(tree: ast.Module, name: str) -> ast.Assign:
    matches = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(matches) == 1, (name, len(matches))
    return matches[0]


def imports(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            found.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return sorted(found)


class NormalizeQ24(ast.NodeTransformer):
    names = {
        "_Q24_TERMS": "_Q32_TERMS",
        "_Q24_ORDER_GRID": "_Q32_ORDER_GRID",
        "_q24_context": "_q32_context",
    }

    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self.names.get(node.id, node.id)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node.name = self.names.get(node.name, node.name)
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if isinstance(node.value, str):
            node.value = node.value.replace("FB-EC4-Q24", "FB-EC4-32")
            node.value = node.value.replace("Q24", "Q32").replace("q24", "q32")
        return node


def main() -> None:
    checks: dict[str, bool] = {}
    checks["parent_hash"] = sha256(PARENT) == PARENT_SHA256
    parent_source = PARENT.read_text()
    child_source = CHILD.read_text()
    parent_tree = ast.parse(parent_source)
    child_tree = ast.parse(child_source)

    exact_import_set = sorted(
        [
            "flopscope",
            "flopscope.numpy",
            "whestbench.BaseEstimator",
            "whestbench.domain.MLP",
        ]
    )
    checks["imports_exact"] = imports(child_tree) == exact_import_set
    checks["q24_terms"] = ast.literal_eval(assignment(child_tree, "_Q24_TERMS").value) == 25
    child_factorial = ast.literal_eval(assignment(child_tree, "_INV_FACTORIAL").value)
    parent_factorial = ast.literal_eval(assignment(parent_tree, "_INV_FACTORIAL").value)
    checks["factorial_exact_prefix"] = child_factorial == parent_factorial[:25]
    child_grid = ast.literal_eval(assignment(child_tree, "_Q24_ORDER_GRID").value)
    checks["order_grid_exact"] = child_grid == tuple(
        tuple(range(derivative, derivative + 25)) for derivative in range(7)
    )
    checks["q38_retained"] = (
        len(ast.literal_eval(assignment(child_tree, "_Q38_HERMITE_GRID").value)) == 3
        and "q=0..38" in child_source
    )
    checks["all_17_channels_retained"] = len(
        ast.literal_eval(assignment(child_tree, "_DERIVATIVE_CHANNELS").value)
    ) == 17
    checks["guard_exact"] = all(
        token in child_source
        for token in (
            "_PRODUCTION_WIDTH = 1024",
            "_PRODUCTION_DEPTH = 16",
            "_PHASE2_FLOP_BUDGET = 2_199_023_255_552",
            "or int(budget) < _PHASE2_FLOP_BUDGET",
            "return fnp.zeros((smoke_depth, smoke_width), dtype=fnp.float32)",
        )
    )
    checks["truthful_tokens"] = (
        "Q32" not in child_source
        and "q32" not in child_source
        and "FB-EC4-Q24" in child_source
        and "# 1/t!, t=0..24." in child_source
    )

    # After normalizing names/docs and restoring the three frozen parent
    # payloads, the complete child AST must equal the immutable parent AST.
    normalized = NormalizeQ24().visit(copy.deepcopy(child_tree))
    ast.fix_missing_locations(normalized)
    for name in ("_Q32_TERMS", "_INV_FACTORIAL", "_Q32_ORDER_GRID"):
        assignment(normalized, name).value = copy.deepcopy(assignment(parent_tree, name).value)
    checks["whole_module_ast_delta_only_q_order"] = ast.dump(
        normalized, include_attributes=False
    ) == ast.dump(parent_tree, include_attributes=False)

    forbidden = (
        "numpy",
        "scipy",
        "ctypes",
        "cffi",
        "asyncio",
        "threading",
        "multiprocessing",
        "subprocess",
        "monkeypatch",
    )
    checks["forbidden_imports_absent"] = all(
        not any(item == token or item.startswith(token + ".") for token in forbidden)
        for item in imports(child_tree)
    )
    checks["no_cache"] = not any(HERE.rglob("__pycache__")) and not any(HERE.rglob("*.pyc"))

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise AssertionError(f"Q24 static verification failed: {failed}")
    print(
        json.dumps(
            {
                "status": "PASS_Q24_EXACT_PARENT_AST_AND_PHASE2_STATIC",
                "parent_sha256": PARENT_SHA256,
                "estimator_sha256": sha256(CHILD),
                "estimator_bytes": CHILD.stat().st_size,
                "checks": checks,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
