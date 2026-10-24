#!/usr/bin/env python3
"""Isolated physical audit for the exact Tichavsky <6,3,3:40> primitive.

This file does not import or execute an estimator, fixture, target, package,
or submission path.  It verifies the recovered integer additive circuit as a
bilinear tensor, meters one tiny physical float32 primitive, and produces a
static Q24 CP3 push/read bill for a prebound multi-axis carrier.

The additive schedule is loaded from the separately recovered public source;
no participant estimator code or source coefficients are copied here.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import resource
import time
from typing import Any

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


DEFAULT_SCHEDULE = Path(
    "/private/tmp/r40_public_extract_20260826/"
    "tichavsky_r40_schedule_c142_v1.py"
)
SCHEDULE_SHA256 = (
    "ccf2cc0ff7ff62784899a51f801e0827c5307f57ace5898a35167467c6a49b41"
)
UPSTREAM_COMMIT = "8bb354d63061504d1a712efafc8d06a0e8fa3f07"
UPSTREAM_FMMA_SHA256 = (
    "0522e007ea759cc96949f7b504c3a1c76eddc3f0c353410e8018f79abc44ccec"
)

WIDTH = 1024
RANKS = tuple(WIDTH * index for index in range(1, 16))
CAP = 1 << 41
NON_CP3_HOST = 425_860_164_095
OPTIMISTIC_MAPPED_HOST = 893_495_871_479
PRIOR_CP3_R40_ARITHMETIC = 467_635_707_384
CURRENT_D1_CP3 = 847_561_359_360
CURRENT_LITERAL_HOST = 2_132_366_833_051
CURRENT_HOST_RSS_BYTES = 3_065_937_920


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_schedule(path: Path) -> Any:
    digest = _sha256(path)
    if digest != SCHEDULE_SHA256:
        raise RuntimeError(("schedule SHA256 drift", digest, SCHEDULE_SHA256))
    spec = importlib.util.spec_from_file_location("r40_c142_authority", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(("cannot load schedule", str(path)))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    expected = {
        "candidate_sizes": {"A": 74, "B": 53, "C": 142},
        "live_sizes": {"A": 108, "B": 54, "C": 132},
        "tensor_scale": 4,
        "pair_scale": 16.0,
    }
    for key, value in expected.items():
        if module.METADATA[key] != value:
            raise RuntimeError(("schedule metadata drift", key, module.METADATA[key], value))
    return module


def _run_integer(row: dict[str, Any], inputs: list[Any]) -> list[Any]:
    env = dict(zip(row["input_keys"], inputs, strict=True))
    for node in row["nodes"]:
        left, right = env[node["left"]], env[node["right"]]
        env[node["key"]] = (
            left + right if node["op"] == "add" else left - right
        )
    return [
        env[key] * int(sign)
        for key, sign in zip(row["output_keys"], row["output_signs"], strict=True)
    ]


def _linear_map(row: dict[str, Any]) -> np.ndarray:
    count = len(row["input_keys"])
    values = []
    for index in range(count):
        basis = [0] * count
        basis[index] = 1
        values.append(_run_integer(row, basis))
    return np.asarray(values, dtype=np.int64).T


def _tensor_certificate(schedule: Any) -> dict[str, Any]:
    a = _linear_map(schedule.FROZEN["A"])
    b = _linear_map(schedule.FROZEN["B"])
    c = _linear_map(schedule.FROZEN["C"])
    failures = []
    maximum = 0
    checked = 0
    for ai in range(18):
        a_row, a_inner = divmod(ai, 3)
        for bi in range(9):
            b_inner, b_column = divmod(bi, 3)
            observed = c @ (a[:, ai] * b[:, bi])
            expected = np.zeros(18, dtype=np.int64)
            if a_inner == b_inner:
                expected[3 * a_row + b_column] = 4
            delta = observed - expected
            error = int(np.max(np.abs(delta)))
            maximum = max(maximum, error)
            checked += 1
            if error:
                failures.append(
                    {
                        "a_input": ai,
                        "b_input": bi,
                        "max_abs": error,
                        "observed": observed.tolist(),
                        "expected": expected.tolist(),
                    }
                )
    return {
        "basis_pairs_checked": checked,
        "exact": not failures,
        "failure_count": len(failures),
        "max_abs_integer_tensor_error": maximum,
        "tensor_scale": 4,
        "matrix_shapes": {"A": list(a.shape), "B": list(b.shape), "C": list(c.shape)},
        "coefficient_ranges": {
            "A": [int(a.min()), int(a.max())],
            "B": [int(b.min()), int(b.max())],
            "C": [int(c.min()), int(c.max())],
        },
        "failures": failures[:3],
    }


def _scalar_views_2d(value: Any) -> list[Any]:
    rows, columns = map(int, value.shape)
    # Length-one basic slices remain arrays (unlike a scalar index) and are
    # zero-FLOP affine views in FlopScope 0.9.1.
    return [
        value[row : row + 1, column : column + 1]
        for row in range(rows)
        for column in range(columns)
    ]


def _run_physical_slp(
    row: dict[str, Any],
    inputs: list[Any],
    output_bank: Any,
) -> tuple[Any, ...]:
    output_views = tuple(fnp.unstack(output_bank, axis=0))
    output_locations = dict(zip(row["output_keys"], output_views, strict=True))
    env = dict(zip(row["input_keys"], inputs, strict=True))
    for node in row["nodes"]:
        operation = fnp.add if node["op"] == "add" else fnp.subtract
        destination = output_locations.get(node["key"])
        if destination is None:
            value = operation(env[node["left"]], env[node["right"]])
        else:
            value = operation(
                env[node["left"]], env[node["right"]], out=destination
            )
        env[node["key"]] = value
    for key, sign in zip(row["output_keys"], row["output_signs"], strict=True):
        if int(sign) < 0:
            fnp.negative(env[key], out=env[key])
    return tuple(env[key] for key in row["output_keys"])


def _physical_633(schedule: Any, left: Any, right: Any) -> Any:
    """Literal tiny circuit with prebound terminal banks and one leaf call."""

    if tuple(map(int, left.shape)) != (6, 3) or tuple(map(int, right.shape)) != (3, 3):
        raise ValueError((left.shape, right.shape))
    a_bank = fnp.empty((40, 1, 1), dtype="float32")
    b_bank = fnp.empty((40, 1, 1), dtype="float32")
    _run_physical_slp(schedule.FROZEN["A"], _scalar_views_2d(left), a_bank)
    _run_physical_slp(schedule.FROZEN["B"], _scalar_views_2d(right), b_bank)
    products = fnp.multiply(a_bank, b_bank)
    output = fnp.empty((6, 3), dtype="float32")
    product_views = list(fnp.unstack(products, axis=0))
    c_row = schedule.FROZEN["C"]
    output_views = _scalar_views_2d(output)
    # C outputs are deliberately bound to the already-natural 6x3 bank, so
    # there is no terminal stack, reshape, copy, or crop hidden in the kernel.
    output_by_key = dict(zip(c_row["output_keys"], output_views, strict=True))
    env = dict(zip(c_row["input_keys"], product_views, strict=True))
    for node in c_row["nodes"]:
        operation = fnp.add if node["op"] == "add" else fnp.subtract
        destination = output_by_key.get(node["key"])
        if destination is None:
            value = operation(env[node["left"]], env[node["right"]])
        else:
            value = operation(
                env[node["left"]], env[node["right"]], out=destination
            )
        env[node["key"]] = value
    fnp.true_divide(output, 4.0, out=output)
    return output


def _rss_bytes() -> int:
    try:
        import psutil  # type: ignore

        return int(psutil.Process().memory_info().rss)
    except Exception:
        raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return raw if platform.system() == "Darwin" else raw * 1024


def _tiny_meter(schedule: Any) -> dict[str, Any]:
    rng = np.random.default_rng(40_633_142)
    left_np = rng.normal(0.0, 0.75, size=(6, 3)).astype(np.float32)
    right_np = rng.normal(0.0, 0.75, size=(3, 3)).astype(np.float32)
    left = fnp.asarray(left_np)
    right = fnp.asarray(right_np)
    before_rss = _rss_bytes()
    context = flops.BudgetContext(
        flop_budget=10_000_000,
        wall_time_limit_s=30.0,
        quiet=True,
    )
    started = time.perf_counter()
    with context:
        observed_value = _physical_633(schedule, left, right)
    observed_wall = time.perf_counter() - started
    after_rss = _rss_bytes()
    observed = np.asarray(observed_value, dtype=np.float32)
    direct64 = left_np.astype(np.float64) @ right_np.astype(np.float64)
    direct32 = left_np @ right_np
    delta64 = observed.astype(np.float64) - direct64
    delta32 = observed - direct32
    direct32_delta64 = direct32.astype(np.float64) - direct64
    calls = Counter()
    costs = Counter()
    for operation in context._op_log:
        calls[operation.op_name] += 1
        costs[operation.op_name] += int(operation.flop_cost)
    expected_flops = 74 + 53 + 142 + 15 + 40 + 18
    if int(context.flops_used) != expected_flops:
        raise AssertionError(("tiny physical count", context.flops_used, expected_flops))
    return {
        "shape": [6, 3, 3],
        "dtype": "float32",
        "flops": int(context.flops_used),
        "expected_flops": expected_flops,
        "wrapper_calls": len(context._op_log),
        "call_histogram": dict(sorted(calls.items())),
        "flop_histogram": dict(sorted(costs.items())),
        "backend_s": float(context.flopscope_backend_time_s),
        "overhead_s": float(context.flopscope_overhead_time_s),
        "residual_s": float(context.residual_wall_time_s or 0.0),
        "context_wall_s": float(context.wall_time_s or observed_wall),
        "observed_wall_s": float(observed_wall),
        "rss_before_bytes": before_rss,
        "rss_after_bytes": after_rss,
        "rss_delta_bytes": after_rss - before_rss,
        "numerical": {
            "finite": bool(np.all(np.isfinite(observed))),
            "r40_vs_direct64_max_abs": float(np.max(np.abs(delta64))),
            "r40_vs_direct64_rmse": float(np.sqrt(np.mean(delta64 * delta64))),
            "direct32_vs_direct64_max_abs": float(
                np.max(np.abs(direct32_delta64))
            ),
            "direct32_vs_direct64_rmse": float(
                np.sqrt(np.mean(direct32_delta64 * direct32_delta64))
            ),
            "r40_vs_direct32_max_abs": float(np.max(np.abs(delta32))),
            "r40_vs_direct32_rmse": float(np.sqrt(np.mean(delta32 * delta32))),
        },
        "inventory": {
            "A_binary": 74,
            "B_binary": 53,
            "C_binary": 142,
            "terminal_sign_elements": 15,
            "rank_products": 40,
            "scale_elements": 18,
            "padding": 0,
            "stack": 0,
            "reshape": 0,
            "copy": 0,
            "crop": 0,
            "coefficient_materialization": 0,
        },
    }


def _ceil_multiple(value: int, modulus: int) -> int:
    return ((value + modulus - 1) // modulus) * modulus


def _matrix_bill(
    m: int,
    k: int,
    n: int,
    depth: int,
    *,
    nodes: tuple[int, int, int],
    literal_signs: bool,
) -> dict[str, Any]:
    pm = _ceil_multiple(m, 2 * 6**depth)
    pk = _ceil_multiple(k, 2 * 3**depth)
    pn = _ceil_multiple(n, 2 * 3**depth)
    transform_a = transform_b = transform_c = sign = 0
    for level in range(depth):
        cells = 40**level
        bm = pm // 6 ** (level + 1)
        bk = pk // 3 ** (level + 1)
        bn = pn // 3 ** (level + 1)
        transform_a += cells * nodes[0] * bm * bk
        transform_b += cells * nodes[1] * bk * bn
        transform_c += cells * nodes[2] * bm * bn
        if literal_signs:
            sign += cells * (4 * bm * bk + 11 * bk * bn)
    lm = pm // 6**depth
    lk = pk // 3**depth
    ln = pn // 3**depth
    # Exact one-level Winograd arithmetic.  Its seven output additions cancel
    # the -1 term in seven exact matmuls, giving this integer expression.
    leaf = 40**depth * (lm * lk + lk * ln + 7 * lm * ln * lk // 4)
    padded = (pm, pk, pn) != (m, k, n)
    pad_copy = m * k + k * n if padded else 0
    # One charged reshape/view-materialization for padded A, B, and C.  All
    # recurrent digits are maintained as explicit axes and use free unstack,
    # transpose, and basic-slice views thereafter.
    layout = pm * pk + pk * pn + pm * pn if depth else 0
    # The valid crop and 4^-depth normalization share one multiply/out call.
    scale_crop = m * n if depth else 0
    total = (
        transform_a
        + transform_b
        + transform_c
        + leaf
        + pad_copy
        + layout
        + sign
        + scale_crop
    )
    return {
        "depth": depth,
        "padded": [pm, pk, pn],
        "total": total,
        "transform_A": transform_a,
        "transform_B": transform_b,
        "transform_C": transform_c,
        "winograd_leaf": leaf,
        "padding_copy": pad_copy,
        "layout_reshape": layout,
        "stack": 0,
        "coefficient_materialization": 0,
        "terminal_sign": sign,
        "fused_scale_crop": scale_crop,
    }


def _roster() -> list[tuple[str, int, int, int, int]]:
    result = [
        ("push", rank, 2 * rank, WIDTH, WIDTH) for rank in RANKS
    ]
    for rank in RANKS[:-1]:
        result.append(("read_first", rank, WIDTH, rank, WIDTH))
        result.append(("read_second", rank, WIDTH, rank, WIDTH))
    return result


def _aggregate_static(*, literal_signs: bool) -> dict[str, Any]:
    rows = []
    aggregate: defaultdict[str, int] = defaultdict(int)
    for kind, rank, m, k, n in _roster():
        choices = [
            _matrix_bill(
                m,
                k,
                n,
                depth,
                nodes=(74, 53, 142),
                literal_signs=literal_signs,
            )
            for depth in range(5)
        ]
        best = min(choices, key=lambda row: (row["total"], row["depth"]))
        rows.append({"kind": kind, "rank": rank, "shape": [m, k, n], **best})
        for key, value in best.items():
            if key not in {"depth"} and isinstance(value, int):
                aggregate[key] += value
    total = int(aggregate["total"])
    host = NON_CP3_HOST + total
    return {
        "literal_signs": literal_signs,
        "aggregate": dict(sorted(aggregate.items())),
        "depth_histogram": dict(
            sorted(Counter(row["depth"] for row in rows).items())
        ),
        "cp3_flops": total,
        "host_flops": host,
        "headroom_to_2pow41": CAP - host,
        "cp3_savings_vs_current_D1": CURRENT_D1_CP3 - total,
        "host_savings_vs_current_literal": CURRENT_LITERAL_HOST - host,
        "host_delta_vs_optimistic_893496B": host - OPTIMISTIC_MAPPED_HOST,
        "calls": rows,
    }


def _memory_bound(m: int, k: int, n: int, depth: int) -> dict[str, Any]:
    pm = _ceil_multiple(m, 2 * 6**depth)
    pk = _ceil_multiple(k, 2 * 3**depth)
    pn = _ceil_multiple(n, 2 * 3**depth)
    a_leaf = 40**depth * (pm // 6**depth) * (pk // 3**depth)
    b_leaf = 40**depth * (pk // 3**depth) * (pn // 3**depth)
    c_leaf = 40**depth * (pm // 6**depth) * (pn // 3**depth)
    stages: list[tuple[str, int]] = []
    for level in range(depth):
        cells = 40**level
        ae = cells * (pm // 6 ** (level + 1)) * (pk // 3 ** (level + 1))
        be = cells * (pk // 3 ** (level + 1)) * (pn // 3 ** (level + 1))
        stages.append((f"A_level_{level}", (18 + 40 + 24) * ae))
        stages.append((f"B_level_{level}", a_leaf + (9 + 40 + 12) * be))
    for level in reversed(range(depth)):
        cells = 40**level
        ce = cells * (pm // 6 ** (level + 1)) * (pn // 3 ** (level + 1))
        stages.append((f"C_level_{level}", (40 + 18 + 53) * ce))
    # Seven Winograd products are streamed directly into four output
    # accumulators.  This retains A/B/C plus at most one quarter-sized A/B/C
    # product triple rather than materializing seven transformed banks.
    stages.append(("streamed_W1", 5 * (a_leaf + b_leaf + c_leaf) // 4))
    name, elements = max(stages, key=lambda row: row[1])
    return {
        "shape": [m, k, n],
        "depth": depth,
        "padded": [pm, pk, pn],
        "peak_stage": name,
        "peak_float32_elements_upper_bound": elements,
        "peak_float32_bytes_upper_bound": 4 * elements,
        "peak_gib_upper_bound": 4 * elements / 2**30,
        "additive_with_current_host_rss_gib_upper_bound": (
            4 * elements + CURRENT_HOST_RSS_BYTES
        ) / 2**30,
    }


def _dispatch_estimate() -> dict[str, Any]:
    digit_count = 15 * 3 + 28 * 2
    products = 43
    nodewise_transform = digit_count * (74 + 53 + 142)
    frozen_transform = digit_count * (8 + 6 + 14)
    leaf_calls = products * 22
    # Two zero pads, 2-3 copy calls, three charged layouts, one fused scale;
    # this deliberately counts the push pair as two copies rather than a stack.
    setup_charged_calls = 15 * 9 + 28 * 8
    return {
        "r40_digits": digit_count,
        "matrix_products": products,
        "nodewise": {
            "transform_calls": nodewise_transform,
            "leaf_calls": leaf_calls,
            "setup_scale_calls": setup_charged_calls,
            "charged_calls_total": nodewise_transform + leaf_calls + setup_charged_calls,
        },
        "frozen_affine_target": {
            "logical_transform_calls": frozen_transform,
            "leaf_calls": leaf_calls,
            "setup_scale_calls": setup_charged_calls,
            "charged_calls_total_before_any_affine_splits": (
                frozen_transform + leaf_calls + setup_charged_calls
            ),
            "status": "requires full-geometry affine-bind certificate",
        },
        "free_view_calls_not_in_charged_total": ["unstack", "transpose", "basic slice"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, default=DEFAULT_SCHEDULE)
    parser.add_argument("--skip-meter", action="store_true")
    args = parser.parse_args()
    schedule = _load_schedule(args.schedule)
    tensor = _tensor_certificate(schedule)
    if not tensor["exact"]:
        raise AssertionError(("R40 tensor identity failed", tensor))
    sign_folded = _aggregate_static(literal_signs=False)
    literal_signs = _aggregate_static(literal_signs=True)
    if sign_folded["cp3_flops"] >= CURRENT_D1_CP3:
        raise AssertionError("physical static plan does not beat D1")
    report = {
        "schema": "q24-r40-physical-kernel-audit-v1",
        "scope": {
            "estimator_executed": False,
            "fixture_opened": False,
            "target_opened": False,
            "package_changed": False,
            "submission_attempted": False,
        },
        "authority": {
            "schedule_path": str(args.schedule),
            "schedule_sha256": _sha256(args.schedule),
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_FMMa_6_3_3_sha256": UPSTREAM_FMMA_SHA256,
            "schedule_nodes": {key: row["nodes_count"] for key, row in schedule.FROZEN.items()},
            "schedule_logical_calls": {key: row["calls"] for key, row in schedule.FROZEN.items()},
            "schedule_workspace_slots": {
                key: row["workspace_peak"] for key, row in schedule.FROZEN.items()
            },
        },
        "tensor_certificate": tensor,
        "static": {
            "literal_terminal_signs": literal_signs,
            "symbolically_folded_terminal_signs": sign_folded,
            "prior_optimistic_cp3_arithmetic": PRIOR_CP3_R40_ARITHMETIC,
            "current_D1_cp3": CURRENT_D1_CP3,
            "non_cp3_host": NON_CP3_HOST,
            "cap": CAP,
            "contract": {
                "padding_zeros": "charged 0 FLOPs by FlopScope; calls retained in dispatch",
                "padding_copy": "all original A/B elements charged once",
                "layout": "one charged reshape each for padded A, B, and C",
                "recurrent_stack": "none; explicit digit axes and prebound terminal destinations",
                "signs": "literal bill shown; folded bill carries signs through references and fuses final signs with scale/crop",
                "decode": "all 142 C nodes charged at every R40 digit",
                "scale_crop": "one arithmetic pass over valid output; no separate crop copy",
                "coefficients": "integer +/-1 circuit; no coefficient array materialization",
                "leaf": "exact one-level Winograd arithmetic with seven batched-rank products",
            },
        },
        "dispatch": _dispatch_estimate(),
        "memory": {
            "largest_push": _memory_bound(2 * RANKS[-1], WIDTH, WIDTH, 3),
            "largest_read": _memory_bound(WIDTH, RANKS[-2], WIDTH, 2),
            "status": "static streamed-W1 upper bound; full-shape RSS not executed",
        },
        "tiny_meter": None if args.skip_meter else _tiny_meter(schedule),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
