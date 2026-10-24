#!/usr/bin/env python3
"""Independent static and optional target-free micro check for P2R0C1D5."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from collections import Counter
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[5]
SOURCE = (
    ROOT
    / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16"
    / "public_noeigh_card_v1/estimator.py"
)

BOUND = {
    SOURCE: "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7",
    HERE / "PREREGISTER.md": "5b7d8d778c684082cc07b04040f7e9e4b9fc8d60c1ab6298aaa7703e36306740",
    HERE / "AUDIT_RECEIPT.md": "6ecae286367353d3c634c0718c20d5df212a539e8c7ab27f27c1c0c99691a3e1",
    HERE / "verify_receipt.py": "e2dfbb7a35f460e37a8d590cd773f0c6954a734221dc15872446febceb0f9bc3",
    HERE / "STATIC_VERIFICATION.json": "2f5147738874d637e9b1f7d4623d6ff87ca6dfd415a4fc529ab87e5d7a7aaf53",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def winograd(m: int, k: int, n: int, depth: int) -> int:
    leaf = 7**depth * mm(m // 2**depth, k // 2**depth, n // 2**depth)
    overhead = sum(
        7**level * 11 * (m * k + k * n + m * n) // 4 ** (level + 1)
        for level in range(depth)
    )
    return leaf + overhead


def extra_elements(m: int, k: int, n: int, depth: int, channels: int) -> int:
    q = Fraction(7, 4)
    value = channels * sum(q**j for j in range(1, depth + 1)) * (m * k + k * n)
    value += channels * q ** (depth - 1) * max(m * k, k * n)
    value += channels * q**depth * m * n
    value += channels * 3 * q ** (depth - 1) * m * n
    value += 2 * m * n
    assert value.denominator == 1
    return value.numerator


def static_check() -> dict[str, object]:
    for path, expected in BOUND.items():
        assert sha256(path) == expected, path

    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    push = functions["_cp_push"]
    read = functions["_cp_read"]
    k3 = functions["_cp_k3"]
    push_matmuls = [
        node for node in ast.walk(push)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "matmul"
    ]
    read_einsums = [
        node for node in ast.walk(read)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "einsum"
    ]
    assert len(push_matmuls) == 2
    assert len(read_einsums) == 2
    assert all(ast.literal_eval(node.args[0]) == "ir,jr->ij" for node in read_einsums)
    assert any(isinstance(node, ast.MatMult) for node in ast.walk(k3))

    ranks = [1_024 * layer for layer in range(1, 16)]
    push_bill = sum(2 * mm(1_024, 1_024, rank) for rank in ranks)
    read_bill = sum(2 * mm(1_024, rank, 1_024) for rank in ranks[:-1])
    ordinary = push_bill + read_bill
    kernel = sum(2 * winograd(1_024, 1_024, rank, 4) for rank in ranks)
    kernel += sum(2 * winograd(1_024, rank, 1_024, 4) for rank in ranks[:-1])
    pack = 10 * 2 * 1_024 * 1_024
    assert (push_bill, read_bill, ordinary) == (
        515_144_417_280,
        450_942_205_952,
        966_086_623_232,
    )
    assert kernel == 594_693_332_992
    assert ordinary - kernel - pack == 371_372_318_720

    # Each ordinary depth-4 read kernel is 4*(15 arithmetic + 2 stacks +
    # 1 block) + one leaf matmul = 73 calls.  A push writes its top result
    # directly into the destination with two concatenate(out=...) calls in
    # place of one top block, hence 74.  Ten paired pushes also need one
    # common-left stack each.
    push_kernels = 10 + 2 * 5
    read_kernels = 11 + 2 * 3
    push_ops = push_kernels * 74 + 10
    read_ops = read_kernels * 73
    assert (push_kernels, read_kernels, push_ops, read_ops) == (20, 17, 1_490, 1_241)
    assert push_ops + read_ops == 2_731

    incumbent = 3_421_503_488
    seq4 = incumbent + 4 * extra_elements(1_024, 1_024, 15_360, 4, 1)
    pair_push4 = (
        incumbent
        + 4 * extra_elements(1_024, 1_024, 10_240, 4, 2)
        + 4 * 2 * 1_024 * 1_024
    )
    pair_read4 = (
        incumbent
        + 4 * extra_elements(1_024, 11_264, 1_024, 4, 2)
        + 4 * 2 * 1_024 * 11_264
    )
    seq5 = incumbent + 4 * extra_elements(1_024, 1_024, 15_360, 5, 1)
    assert (seq4, pair_push4, pair_read4, seq5) == (
        6_798_163_968,
        7_902_887_936,
        7_838_334_976,
        9_353_728_000,
    )
    return {
        "status": "PASS_INDEPENDENT_STATIC",
        "replacement_calls": 58,
        "ordinary_bill": ordinary,
        "candidate_cp3_bill": kernel + pack,
        "net_saving": ordinary - kernel - pack,
        "kernel_related_operations": push_ops + read_ops,
        "maximum_algebraic_rss_bound_bytes": max(seq4, pair_push4, pair_read4),
        "rss_margin_to_8_gib_bytes": 8 * 1_024**3 - max(seq4, pair_push4, pair_read4),
    }


def micro_check() -> dict[str, object]:
    """Paired depth-4 push spelling; synthetic arrays only, no repo data."""
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np

    def kernel(a, b, depth, out=None):
        if depth == 0:
            return fnp.matmul(a, b)
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
        aa = fnp.stack((a11, a12, s4, a22, s1, s2, s3), axis=0)
        bb = fnp.stack((b11, b21, b22, t4, t1, t2, t3), axis=0)
        del s1, s2, s3, s4, t1, t2, t3, t4
        p = kernel(aa, bb, depth - 1)
        del aa, bb
        u1 = p[0] + p[1]
        u2 = p[0] + p[5]
        u3 = u2 + p[6]
        u4 = u2 + p[4]
        del u2
        u5 = u4 + p[2]
        del u4
        u6 = u3 - p[3]
        u7 = u3 + p[4]
        del u3, p
        if out is None:
            return fnp.block([[u1, u5], [u6, u7]])
        fnp.concatenate((u1, u5), axis=-1, out=out[..., :m2, :])
        fnp.concatenate((u6, u7), axis=-1, out=out[..., m2:, :])
        return out

    rng = np.random.default_rng(9_327)
    left_plain = rng.standard_normal((32, 32)).astype(np.float32)
    right_plain = (rng.standard_normal((2, 32, 64)) / np.sqrt(32)).astype(np.float32)
    left = fnp.asarray(left_plain)
    right = fnp.asarray(right_plain)
    output = fnp.empty((2, 32, 64), dtype=fnp.float32)
    with flops.BudgetContext(flop_budget=10**9, quiet=True) as meter:
        paired_left = fnp.stack((left, left), axis=0)
        result = kernel(paired_left, right, 4, output)
    expected = 2 * 32 * 32 + 2 * winograd(32, 32, 64, 4)
    observed = np.asarray(result).astype(np.float64)
    reference = left_plain.astype(np.float64)[None, ...] @ right_plain.astype(np.float64)
    delta = observed - reference
    names = Counter(record.op_name for record in meter.op_log)
    assert int(meter.flops_used) == expected
    assert len(meter.op_log) == 75
    assert names == Counter(
        {"subtract": 32, "add": 28, "stack": 9, "block": 3,
         "concatenate": 2, "matmul": 1}
    )
    assert np.isfinite(observed).all()
    return {
        "status": "PASS_INDEPENDENT_TARGET_FREE_D4_MICRO",
        "literal_flops": int(meter.flops_used),
        "formula_flops": expected,
        "operation_count": len(meter.op_log),
        "operation_inventory": dict(sorted(names.items())),
        "rms_delta_vs_float64": float(np.sqrt(np.mean(delta * delta))),
        "maximum_absolute_delta_vs_float64": float(np.max(np.abs(delta))),
        "maximum_relative_delta_vs_float64": float(
            np.max(np.abs(delta)) / np.max(np.abs(reference))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true")
    args = parser.parse_args()
    result = static_check()
    if args.micro:
        result["micro"] = micro_check()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
