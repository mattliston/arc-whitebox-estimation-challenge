#!/usr/bin/env python3
"""Static verifier and optional target-free micro-meter for P2R0C1D5."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import sys
from decimal import Decimal, getcontext
from fractions import Fraction
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[5]

BOUND = {
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/estimator.py": "373ad4d55edbc7fd0f6aa7e9bedafe41e283f2685e5002c8b46ac44c05ebcdd7",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/BUILD_RECEIPT.json": "3fd515406b6eec954523e7d553aba769999998318fb29fd987ce4637e2bbb0d3",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/SUBMIT_WATCH.log": "6f2339963c1272d9eee46e20897b06fc87f5b321126e2e1976a565df71f3c781",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/METER_RESULT.json": "b6aedc76b190702ac0b4bcf2c567d40e0f533b861ccde95bcb17ac21627eb365",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_trim_card_v1/PAIRED_RESIDUAL_RECEIPT.json": "2eb85dba23858669b532887c9c7d774bf8f9581fec5609ba4133b6976c28b777",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_adjusted_score_pareto_v1/RECEIPT.json": "12e82f5b952c6b2786c834f028360cb25bea483f6ee0261a6ee1b198d4ff81e2",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_adjusted_score_pareto_v1/ERRATUM_001_EXECUTABILITY.json": "042ec26fa003dbe2fda47ef69b801104d6e35e46198a6927a6eacc600ac8f8a0",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_host_compute_value_attribution_v1/V_ATTRIBUTION_RECEIPT.json": "7a44a817d3f5de9e2c7b5ee0c0b9e68277ceb3e0408d9750dae26db2437e3e06",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_flops.py": "bbdad83348b7ce7b63c9d8da66512c40c69e055c653d24687e62132bd34220c2",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_array_ops.py": "e45e927f18c54352503a4951d3d925a5f31f92ecea245e32360825d71a05da6c",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_pointwise.py": "7d376515446f1b174bef375c48116a552098bd514f5552450876f56822beb0c7",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_dtype_billing.py": "d6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_budget.py": "ad9e78172bd0e8175533d4aae75b67df749d96ecc1ca868135eb3d781e542d01",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_einsum.py": "0f25b8f5709c5a96404f20c38afc9b705568ab2654ab6c5fc81f3de292b73284",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_ndarray.py": "fa5bbecb48b123ab77f31ced3da877d2e333aa5b403a324c33da591b967d3e74",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/data/default_weights.json": "f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a",
    "phase_one_submissions/327723/estimator.py": "c3c6981b4687724e8bbabf55c8103586b589bfbf0b2755bf3e71a584fef64ec9",
    "autoresearch/experiments/cands/fable_clean_strassen_20260810/estimator.py": "c3c6981b4687724e8bbabf55c8103586b589bfbf0b2755bf3e71a584fef64ec9",
    "autoresearch/experiments/cands/fable_spend_strassen_20260810/estimator.py": "4dee5200cb0c1c72eb5e9c9d8c0cecc989423363fdcd08b232e0e6bf4ead7820",
    "autoresearch/experiments/cands/fable_spend7_strassen_20260810/estimator.py": "f6dc80663b9e35743e48141796e21087861bd1ea2e83015c3ebab38b9616b6d6",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b2_independent_gaussian_linked_cluster_grade3_v1/CONSTRAINT_RECEIPT.md": "d6275c2ace74294c4d9c6fcc6b5126dbbb9725e4dac51aeca1affba0e98c7cba",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b3_covariance_resummed_sparse_base_v1/CONSTRAINT_RECEIPT.md": "41cecf538bee6a3feadcea88c8d82c35ed4489756bc500084118d11b7b16f561",
    "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/p2r0f3b2b3_symmetric_cp_residual_prior_audit_v1/AUDIT_RECEIPT.md": "1683ddb0400767516f74f3eed68e3f23d43ed71f6685fd2493dbf55a99026594",
}

EXPECTED_KERNEL = {
    1: (847_842_377_728, 118_244_245_504),
    2: (746_611_408_896, 219_475_214_336),
    3: (661_941_616_640, 304_145_006_592),
    4: (594_693_332_992, 371_393_290_240),
    5: (547_817_207_808, 418_269_415_424),
    6: (527_741_313_536, 438_345_309_696),
    7: (546_821_157_760, 419_265_465_472),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def call_attr(node: ast.AST, attr: str) -> list[ast.Call]:
    return [
        item for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and isinstance(item.func, ast.Attribute)
        and item.func.attr == attr
    ]


def matmul_cost(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def winograd_cost(m: int, k: int, n: int, depth: int) -> int:
    q = 1 << depth
    leaf = 7**depth * matmul_cost(m // q, k // q, n // q)
    overhead = sum(
        7**level * 11 * (m * k + k * n + m * n) // (4 ** (level + 1))
        for level in range(depth)
    )
    return leaf + overhead


def memory_extra(m: int, k: int, n: int, depth: int, channels: int) -> int:
    q = Fraction(7, 4)
    stacks = channels * sum(q**j for j in range(1, depth + 1)) * (m * k + k * n)
    expressions = channels * q ** (depth - 1) * max(m * k, k * n)
    leaf = channels * q**depth * m * n
    recombine = channels * 3 * q ** (depth - 1) * m * n
    output = 2 * m * n
    total = stacks + expressions + leaf + recombine + output
    assert total.denominator == 1
    return total.numerator


def verify_static() -> dict:
    for relative, expected in BOUND.items():
        path = ROOT / relative
        assert path.is_file(), relative
        assert sha256(path) == expected, relative

    source_path = ROOT / next(iter(BOUND))
    source = source_path.read_text()
    tree = ast.parse(source)
    assigned = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body if isinstance(node, ast.Assign)
        for target in node.targets if isinstance(target, ast.Name)
        if isinstance(node.value, ast.Constant)
    }
    assert assigned["_CP3_CAP"] == 16_384
    assert assigned["_PRODUCTION_WIDTH"] == 1_024
    assert assigned["_PRODUCTION_DEPTH"] == 16
    push = function(tree, "_cp_push")
    read = function(tree, "_cp_read")
    k3 = function(tree, "_cp_k3")
    assert len(call_attr(push, "matmul")) == 2
    read_einsums = call_attr(read, "einsum")
    assert len(read_einsums) == 2
    assert all(ast.literal_eval(call.args[0]) == "ir,jr->ij" for call in read_einsums)
    assert any(isinstance(node, ast.MatMult) for node in ast.walk(k3))
    assert "for layer_index in range(1, mlp.depth):" in source
    assert "if layer_index + 1 == mlp.depth:" in source

    board = json.loads((ROOT / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_noeigh_card_v1/SUBMIT_WATCH.log").read_text())
    submission = board["submission"]
    assert submission["id"] == 328_439
    assert submission["grading_status_cd"] == "graded"
    assert Decimal(str(submission["score"])) == Decimal("2.2842888692472472e-8")
    assert Decimal(str(submission["score_secondary"])) == Decimal("2.9409159800763974e-8")

    meter = json.loads((ROOT / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/public_q24_native_transport_energy_final_workspace_v4/METER_RESULT.json").read_text())
    assert meter["call_order"] == [35, 279, 35]
    assert Decimal(str(meter["aggregate"]["maximum_residual_wall_seconds"])) == Decimal("0.35143496841192245")
    incumbent_rss = max(row["maximum_process_rss_bytes"] for row in meter["trial_results"])
    assert incumbent_rss == 3_421_503_488

    weights = json.loads((ROOT / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/data/default_weights.json").read_text())
    assert weights["weights"]["matmul"] == 1.0
    assert weights["weights"]["einsum"] == 1.0
    assert weights["weights"]["add"] == 1.0
    assert weights["weights"]["subtract"] == 1.0
    assert weights["weights"]["stack"] == 1.0
    assert weights["weights"]["block"] == 1.0
    assert weights["dtype_rates"]["float32"] == 1.0
    flops_source = (ROOT / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_flops.py").read_text()
    arrays_source = (ROOT / "pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/data_gate/.venv/lib/python3.11/site-packages/flopscope/_array_ops.py").read_text()
    assert "2 * m * k * n - m * n" in flops_source
    assert 'attach_docstring(stack, _np.stack, "counted_custom", "numel(output) FLOPs")' in arrays_source
    assert 'attach_docstring(block, _np.block, "counted_custom", "numel(output) FLOPs")' in arrays_source

    ranks = [1_024 * layer for layer in range(1, 16)]
    push_bill = sum(2 * matmul_cost(1_024, 1_024, rank) for rank in ranks)
    read_bill = sum(2 * matmul_cost(1_024, rank, 1_024) for rank in ranks[:-1])
    ordinary = push_bill + read_bill
    assert push_bill == 515_144_417_280
    assert read_bill == 450_942_205_952
    assert ordinary == 966_086_623_232
    table = {}
    for depth in range(1, 8):
        bill = sum(2 * winograd_cost(1_024, 1_024, rank, depth) for rank in ranks)
        bill += sum(2 * winograd_cost(1_024, rank, 1_024, depth) for rank in ranks[:-1])
        table[depth] = (bill, ordinary - bill)
    assert table == EXPECTED_KERNEL
    packing = 10 * 2 * 1_024 * 1_024
    assert packing == 20_971_520
    net_saving = EXPECTED_KERNEL[4][1] - packing
    assert net_saving == 371_372_318_720

    seq4 = incumbent_rss + 4 * memory_extra(1_024, 1_024, 15_360, 4, 1)
    fused_push4 = incumbent_rss + 4 * memory_extra(1_024, 1_024, 10_240, 4, 2) + 4 * 2 * 1_024 * 1_024
    fused_read4 = incumbent_rss + 4 * memory_extra(1_024, 11_264, 1_024, 4, 2) + 4 * 2 * 1_024 * 11_264
    seq5 = incumbent_rss + 4 * memory_extra(1_024, 1_024, 15_360, 5, 1)
    assert seq4 == 6_798_163_968
    assert fused_push4 == 7_902_887_936
    assert fused_read4 == 7_838_334_976
    assert seq5 == 9_353_728_000
    assert max(seq4, fused_push4, fused_read4) < 8 * 1_024**3 < seq5

    getcontext().prec = 50
    raw = Decimal("2.9409159800763974e-8")
    adjusted = Decimal("2.2842888692472472e-8")
    budget = Decimal(2**41)
    equivalent_f = adjusted / raw * budget
    projected_multiplier = (equivalent_f - Decimal(net_saving)) / budget
    projected_adjusted = raw * projected_multiplier
    break_even_raw = Decimal("1e-8") / projected_multiplier
    required_fraction = Decimal(1) - break_even_raw / raw
    assert projected_adjusted == Decimal("1.7876252782987673382085345685482025146484375000000e-8")
    assert break_even_raw == Decimal("1.6451523794042476068958092765120068276761532403050e-8")

    prereg = (HERE / "PREREGISTER.md").read_text()
    for phrase in (
        "58 float32 dense contractions",
        "371,372,318,720",
        "FROZEN, UNEXECUTED",
        "does **not** establish raw MSE",
    ):
        assert phrase in prereg

    return {
        "status": "PASS_STATIC",
        "push_calls": 30,
        "read_calls": 28,
        "ordinary_cp3_contraction_bill": ordinary,
        "depth4_kernel_bill": EXPECTED_KERNEL[4][0],
        "batch_pack_bill": packing,
        "net_saving": net_saving,
        "max_conservative_rss_bytes": max(seq4, fused_push4, fused_read4),
        "depth5_closed_rss_bytes": seq5,
        "planning_only_projected_adjusted": str(projected_adjusted),
        "planning_only_break_even_raw": str(break_even_raw),
        "planning_only_required_raw_reduction_fraction": str(required_fraction),
    }


def micro_meter() -> dict:
    """Verify the frozen scalar formula on synthetic arrays; opens no repo data."""
    import flopscope as flops
    import flopscope.numpy as fnp
    import numpy as np

    def kernel(a, b, levels):
        if levels == 0:
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
        p = kernel(aa, bb, levels - 1)
        u1 = p[0] + p[1]
        u2 = p[0] + p[5]
        u3 = u2 + p[6]
        u4 = u2 + p[4]
        u5 = u4 + p[2]
        u6 = u3 - p[3]
        u7 = u3 + p[4]
        return fnp.block([[u1, u5], [u6, u7]])

    a_plain = (np.arange(64, dtype=np.float32).reshape(8, 8) - 31.0) / 17.0
    b_plain = (np.arange(128, dtype=np.float32).reshape(8, 16) - 63.0) / 19.0
    with flops.BudgetContext(flop_budget=10_000_000, quiet=True) as budget:
        a = fnp.asarray(a_plain)
        b = fnp.asarray(b_plain)
        before = int(budget.flops_used)
        result = kernel(a, b, 2)
        billed = int(budget.flops_used) - before
    expected = winograd_cost(8, 8, 16, 2)
    delta = np.asarray(result) - (a_plain @ b_plain)
    max_abs = float(np.max(np.abs(delta)))
    assert billed == expected, (billed, expected)
    assert np.isfinite(np.asarray(result)).all()
    assert max_abs <= 1e-4, max_abs
    return {
        "status": "PASS_TARGET_FREE_MICRO",
        "shape": [[8, 8], [8, 16]],
        "depth": 2,
        "literal_flops": billed,
        "formula_flops": expected,
        "max_abs_vs_numpy_matmul": max_abs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--micro", action="store_true")
    args = parser.parse_args()
    result = verify_static()
    if args.micro:
        result["micro"] = micro_meter()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
