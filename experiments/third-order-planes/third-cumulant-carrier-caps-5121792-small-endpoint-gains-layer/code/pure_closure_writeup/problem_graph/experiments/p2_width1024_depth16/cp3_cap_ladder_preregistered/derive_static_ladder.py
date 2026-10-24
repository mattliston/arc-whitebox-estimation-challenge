#!/usr/bin/env python3
"""Target-blind source-equivalence and v0.12 static CP cap accounting."""

from __future__ import annotations

import ast
import hashlib
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WIDTH = 1024
DEPTH = 16
BUDGET = 2**41
SCORE_FLOOR_FLOPS = BUDGET / 10.0
PROVISIONAL_ANALYTIC_FLOPS = 103_923_320_832
CAP_PATHS = {
    512: ROOT / "cp3_cap512_k3_readout" / "estimator.py",
    1024: ROOT / "cp3_cap1024_k3_readout" / "estimator.py",
    1536: ROOT / "cp3_cap1536_k3_readout" / "estimator.py",
    1792: ROOT / "cp3_cap1792_k3_readout" / "estimator.py",
}


class _NormalizeCandidate(ast.NodeTransformer):
    """Erase docstrings and normalize only the immutable cap assignment."""

    def visit_Expr(self, node: ast.Expr):  # noqa: N802
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return None
        return self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign):  # noqa: N802
        node = self.generic_visit(node)
        if any(
            isinstance(target, ast.Name) and target.id == "_CP_CAP"
            for target in node.targets
        ):
            node.value = ast.Constant(value=0)
        return node


def _normalized_ast(source: str) -> str:
    tree = _NormalizeCandidate().visit(ast.parse(source))
    return ast.dump(ast.fix_missing_locations(tree), include_attributes=False)


def _rank_schedule(cap: int) -> tuple[list[int], list[tuple[int, int, bool]]]:
    rank = 0
    pushed_ranks: list[int] = []
    grow_rows: list[tuple[int, int, bool]] = []
    for layer0 in range(DEPTH):
        if layer0 > 0:
            pushed_ranks.append(rank)
        if layer0 + 1 < DEPTH:
            raw_rank = rank + WIDTH
            truncated = raw_rank > cap
            grow_rows.append((rank, raw_rank, truncated))
            rank = min(raw_rank, cap)
    return pushed_ranks, grow_rows


def _static_cost(cap: int) -> dict[str, object]:
    n = WIDTH
    pushed_ranks, grow_rows = _rank_schedule(cap)

    # Two factor pushes per nonempty layer. v0.12 matmul cost is
    # 2*m*k*n - m*n, with float32 dtype rate one.
    factor_push = sum(2 * n * rank * (2 * n - 1) for rank in pushed_ranks)

    # For each exact diagonal read: two n*rank pointwise products, one
    # matrix-vector contraction n*(2*rank-1), and one n-vector scale.
    k3_read = sum(4 * n * rank for rank in pushed_ranks)

    grow = 0
    argsort = 0
    argsort_calls = 0
    truncation_raw_ranks: list[int] = []
    for inherited_rank, raw_rank, truncated in grow_rows:
        # Caller f64->f32 casts plus gate, birth, divide, and concatenations.
        grow += 4 * n * inherited_rank + 5 * n * n + inherited_rank + 8 * n
        if not truncated:
            continue

        argsort_calls += 1
        truncation_raw_ranks.append(raw_rank)
        sort_cost = 4 * raw_rank * math.ceil(math.log2(raw_rank))
        argsort += sort_cost

        # Three squared/dot arrays and reductions, scalar energy algebra,
        # argsort, then three fancy gathers at the retained cap.
        grow += (
            6 * n * raw_rank
            + 7 * raw_rank
            + sort_cost
            + 8 * n * cap
            + 4 * cap
        )

    # One extra float32 eye plus the f64 open-loop readout delta versus the
    # frozen analytic host: 14*n billed FLOPs per layer at dtype rate two.
    fixed = n + 14 * n * DEPTH
    increment = factor_push + k3_read + grow + fixed
    projected_total = PROVISIONAL_ANALYTIC_FLOPS + increment

    return {
        "cap": cap,
        "pushed_rank_sum": sum(pushed_ranks),
        "pushed_ranks": pushed_ranks,
        "grow_raw_ranks": [row[1] for row in grow_rows],
        "truncation_raw_ranks": truncation_raw_ranks,
        "argsort_calls": argsort_calls,
        "argsort_flops": argsort,
        "factor_push_flops": factor_push,
        "k3_read_core_flops": k3_read,
        "grow_cast_gate_birth_truncate_flops": grow,
        "fixed_identity_and_open_loop_readout_flops": fixed,
        "static_accounted_cp_increment_flops": increment,
        "projected_total_using_provisional_analytic_base": projected_total,
        "projected_budget_fraction": projected_total / BUDGET,
        "projected_score_floor_headroom_flops": SCORE_FLOOR_FLOPS - projected_total,
    }


def derive() -> dict[str, object]:
    sources = {cap: path.read_text(encoding="utf-8") for cap, path in CAP_PATHS.items()}
    normalized = {cap: _normalized_ast(source) for cap, source in sources.items()}
    reference = normalized[512]
    equivalent = {str(cap): tree == reference for cap, tree in normalized.items()}
    if not all(equivalent.values()):
        raise RuntimeError(f"cap sources differ beyond _CP_CAP/docstrings: {equivalent}")

    source_hashes = {
        str(cap): hashlib.sha256(CAP_PATHS[cap].read_bytes()).hexdigest()
        for cap in CAP_PATHS
    }
    rows = [_static_cost(cap) for cap in CAP_PATHS]
    cap512_accounted = rows[0]["static_accounted_cp_increment_flops"]
    cap512_provisional_observed = 32_538_849_282

    return {
        "schema": "phase2-cp3-preregistered-cap-ladder-static-v1",
        "status": "TARGET_BLIND_STATIC_SCREEN__COMMON_DRIVER_RECEIPTS_REQUIRED",
        "shape": {"width": WIDTH, "depth": DEPTH, "budget_flops": BUDGET},
        "score_floor_flops": SCORE_FLOOR_FLOPS,
        "provisional_analytic_base_flops": PROVISIONAL_ANALYTIC_FLOPS,
        "source_hashes": source_hashes,
        "normalized_ast_equal_except_cap_and_docstrings": equivalent,
        "rows": rows,
        "cap512_crosscheck": {
            "static_accounted_increment_flops": cap512_accounted,
            "provisionally_observed_increment_flops": cap512_provisional_observed,
            "absolute_difference_flops": abs(
                cap512_provisional_observed - int(cap512_accounted)
            ),
            "qualification": "raw exploratory stdout was not retained; this is not an authoritative receipt",
        },
        "ladder_policy": {
            "frozen_unconditional_caps": [512, 1024, 1536],
            "conditional_meter_only_cap": 1792,
            "cap1792_admission": "before any target opening, common driver total must be strictly below 0.1*2^41 and all resource gates must pass",
            "excluded_cap": 2048,
        },
        "access": {"target_or_truth": 0, "teacher": 0, "score": 0},
    }


if __name__ == "__main__":
    print(json.dumps(derive(), indent=2, sort_keys=True))
