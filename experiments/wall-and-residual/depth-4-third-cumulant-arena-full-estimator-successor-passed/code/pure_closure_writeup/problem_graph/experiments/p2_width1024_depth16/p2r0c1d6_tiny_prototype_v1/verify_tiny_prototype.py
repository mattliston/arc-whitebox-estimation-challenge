#!/usr/bin/env python3
"""Independent verifier for the sealed D6 tiny-prototype receipt."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
FINAL_SEAL = HERE / "TINY_PROTOTYPE_FINAL_SEAL.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def meter_inventory(meter) -> dict:
    calls = Counter(record.op_name for record in meter.op_log)
    costs = Counter()
    for record in meter.op_log:
        costs[record.op_name] += int(record.flop_cost)
    return {
        "flops": int(meter.flops_used),
        "operation_count": len(meter.op_log),
        "operations": {
            name: {"calls": int(calls[name]), "flop_cost": int(costs[name])}
            for name in sorted(calls)
        },
    }


def expected_hot(m: int, k: int, n: int) -> dict:
    costs = Counter()
    for level in range(1, 5):
        parents = 7 ** (level - 1)
        ae = parents * (m // 2**level) * (k // 2**level)
        be = parents * (k // 2**level) * (n // 2**level)
        ce = parents * (m // 2**level) * (n // 2**level)
        costs["add"] += ae + 6 * ce
        costs["subtract"] += 3 * ae + 4 * be + ce
        costs["copyto"] += 7 * (ae + be)
    costs["matmul"] = 7**4 * (m // 16) * (n // 16) * (2 * (k // 16) - 1)
    calls = {"add": 28, "copyto": 56, "matmul": 1, "subtract": 32}
    return {
        "flops": int(sum(costs.values())),
        "operation_count": 117,
        "operations": {
            name: {"calls": calls[name], "flop_cost": int(costs[name])}
            for name in sorted(calls)
        },
    }


def recursive_reference(a: np.ndarray, b: np.ndarray, levels: int) -> np.ndarray:
    if levels == 0:
        return np.matmul(a, b)
    m2, k2, n2 = a.shape[-2] // 2, a.shape[-1] // 2, b.shape[-1] // 2
    a11, a12 = a[..., :m2, :k2], a[..., :m2, k2:]
    a21, a22 = a[..., m2:, :k2], a[..., m2:, k2:]
    b11, b12 = b[..., :k2, :n2], b[..., :k2, n2:]
    b21, b22 = b[..., k2:, :n2], b[..., k2:, n2:]
    s1 = a21 + a22
    s2 = s1 - a11
    s3 = a11 - a21
    s4 = a12 - s2
    t1 = b12 - b11
    t2 = b22 - t1
    t3 = b22 - b12
    t4 = t2 - b21
    p = recursive_reference(
        np.stack((a11, a12, s4, a22, s1, s2, s3), axis=0),
        np.stack((b11, b21, b22, t4, t1, t2, t3), axis=0),
        levels - 1,
    )
    u1 = p[0] + p[1]
    u2 = p[0] + p[5]
    u3 = u2 + p[6]
    u4 = u2 + p[4]
    u5 = u4 + p[2]
    u6 = u3 - p[3]
    u7 = u3 + p[4]
    return np.block([[u1, u5], [u6, u7]])


def function_node(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(name)


def fnp_calls(node: ast.AST) -> Counter:
    result = Counter()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call) or not isinstance(child.func, ast.Attribute):
            continue
        if isinstance(child.func.value, ast.Name) and child.func.value.id == "fnp":
            result[child.func.attr] += 1
            if child.func.attr in {"add", "subtract", "matmul"}:
                assert any(keyword.arg == "out" for keyword in child.keywords)
    return result


def verify_lifetimes(regions: dict[str, tuple[int, int]]) -> dict:
    # Inclusive phase intervals. Shared physical regions are legal only when
    # logical generations do not overlap in time.
    logical = []
    for phase, label in enumerate(
        [
            "dep_a_l1", "dep_b_l1", "dep_a_l2", "dep_b_l2",
            "dep_a_l3", "dep_b_l3", "dep_a_l4", "dep_b_l4",
        ],
        start=1,
    ):
        logical.append((label, "dependency", phase, phase))
    logical.extend(
        [
            ("leaf", "leaf_product", 9, 10),
            ("bank0_c3", "product_bank_0", 10, 11),
            ("bank1_c2", "product_bank_1", 11, 12),
            ("bank0_c1", "product_bank_0", 12, 13),
            ("u_l4", "u_scratch", 10, 10),
            ("u_l3", "u_scratch", 11, 11),
            ("u_l2", "u_scratch", 12, 12),
            ("u_l1", "u_scratch", 13, 13),
            ("output0", "output_bank_0", 13, 30),
            ("output1", "output_bank_1", 27, 30),
        ]
    )
    collisions = []
    for index, left in enumerate(logical):
        for right in logical[index + 1 :]:
            temporal = max(left[2], right[2]) <= min(left[3], right[3])
            ls, le = regions[left[1]]
            rs, re = regions[right[1]]
            spatial = max(ls, rs) < min(le, re)
            if temporal and spatial:
                collisions.append([left[0], right[0]])
    return {
        "logical_generation_count": len(logical),
        "live_alias_collisions": collisions,
        "pass": not collisions,
    }


def main() -> None:
    final_seal = json.loads(FINAL_SEAL.read_text())
    assert final_seal["status"] == "SEALED_PASS_D6_TINY_PROTOTYPE"
    for relative, expected in final_seal["sha256"].items():
        assert sha256(HERE / relative) == expected, relative

    preseal = json.loads((HERE / "TINY_PROTOTYPE_PRESEAL.json").read_text())
    assert preseal["status"] == "FROZEN_BEFORE_ONE_SHOT_TINY_PROTOTYPE"
    for relative, expected in preseal["sealed_files_sha256"].items():
        assert sha256(HERE / relative) == expected, relative
    receipt = json.loads((HERE / "TINY_PROTOTYPE_RECEIPT.json").read_text())
    algebra = json.loads((HERE / "FROZEN_D5_ALGEBRA.json").read_text())
    assert receipt["status"] == "PASS_SEALED_D6_TINY_PROTOTYPE"
    assert all(receipt["checks"].values())
    assert all(receipt["gapped_stride_and_guard_checks"].values())
    assert all(receipt["branch_order_proof"]["checks"].values())
    assert algebra["left_branch_tuple"] == ["a11", "a12", "s4", "a22", "s1", "s2", "s3"]
    assert algebra["right_branch_tuple"] == ["b11", "b21", "b22", "t4", "t1", "t2", "t3"]
    assert algebra["upstream_d5_binding"]["estimator_sha256"] == "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5"

    source = (HERE / "tiny_d6_kernel.py").read_text()
    tree = ast.parse(source)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert imports <= {"__future__", "dataclasses", "flopscope.numpy"}
    lowered = source.lower()
    for forbidden in ("estimator", "fixture", "whestbench", "importlib", "target", "score"):
        # These words are allowed in the module's negative-scope docstring only.
        assert forbidden not in "\n".join(source.splitlines()[8:]).lower(), forbidden
    forbidden_calls = {"stack", "block", "concatenate", "einsum", "dot", "tensordot"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in forbidden_calls
        if isinstance(node, ast.Subscript):
            assert not isinstance(node.ctx, ast.Store)

    forward_calls = fnp_calls(function_node(tree, "_forward_depth4"))
    reconstruct_calls = fnp_calls(function_node(tree, "_reconstruct_depth4"))
    kernel_calls = fnp_calls(function_node(tree, "winograd_depth4"))
    assert forward_calls == {"add": 1, "subtract": 7, "copyto": 14}
    assert reconstruct_calls == {"add": 6, "subtract": 1}
    assert kernel_calls == {"matmul": 1}
    for name in ("_forward_depth4", "_reconstruct_depth4", "winograd_depth4"):
        assert "reshape" not in fnp_calls(function_node(tree, name))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for call in ast.walk(node):
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                    assert call.func.id != node.name

    rows = receipt["arena"]["regions"]
    assert rows[0]["start"] == 0
    assert all(left["stop"] == right["start"] for left, right in zip(rows, rows[1:]))
    assert rows[-1]["stop"] == receipt["arena"]["total_elements"]
    regions = {row["name"]: (row["start"], row["stop"]) for row in rows}
    liveness = verify_lifetimes(regions)
    assert liveness["pass"]
    assert regions["output_bank_0"][1] <= regions["output_bank_1"][0]

    for case in receipt["cases"]:
        assert case["meter"] == expected_hot(*case["active_shape"])
        assert case["meter"] == case["expected_meter"]

    spec = importlib.util.spec_from_file_location("sealed_tiny_d6_kernel", HERE / "tiny_d6_kernel.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    layout = module.make_layout(32, 32, 64)
    arena = module.allocate_arena(layout)
    with flops.BudgetContext(flop_budget=1_000_000, quiet=True) as bind_meter:
        views = module.bind_arena_views(arena, layout)
    assert meter_inventory(bind_meter) == receipt["binding_meter"]
    rng = np.random.default_rng(906_602)
    left_plain = rng.standard_normal((32, 16)).astype(np.float32)
    right_plain = (rng.standard_normal((16, 48)) / 4.0).astype(np.float32)
    out = views.output_banks[1][:, :48]
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as hot_meter:
        observed = module.winograd_depth4(
            fnp.asarray(left_plain), fnp.asarray(right_plain), views, out=out
        )
    assert observed is out
    assert meter_inventory(hot_meter) == expected_hot(32, 16, 48)
    assert not np.asarray(out).flags.c_contiguous
    assert np.array_equal(np.asarray(out), recursive_reference(left_plain, right_plain, 4))
    assert np.isfinite(np.asarray(out)).all()

    report = {
        "status": "PASS_INDEPENDENT_D6_TINY_PROTOTYPE_VERIFIER",
        "receipt_sha256": sha256(HERE / "TINY_PROTOTYPE_RECEIPT.json"),
        "kernel_sha256": sha256(HERE / "tiny_d6_kernel.py"),
        "maximum_hot_flops": expected_hot(32, 32, 64)["flops"],
        "gapped_hot_flops": expected_hot(32, 16, 48)["flops"],
        "binding_flops": receipt["binding_meter"]["flops"],
        "liveness": liveness,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
