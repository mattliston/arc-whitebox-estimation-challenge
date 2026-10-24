#!/usr/bin/env python3
"""One-shot target-free literal-meter smoke of the frozen candidate kernel."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import flopscope as flops
import flopscope.numpy as fnp
import numpy as np


HERE = Path(__file__).resolve().parent
CANDIDATE = HERE / "estimator.py"
RESULT = HERE / "KERNEL_SMOKE_RESULT.json"
SEAL = HERE / "SMOKE_SEAL.json"
EXPECTED_CANDIDATE_SHA256 = "69a46c68c18a30e4d4fa3e786a6c742a7183b2f3eef9bb58ec1328df149556d5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mm(m: int, k: int, n: int) -> int:
    return m * n * (2 * k - 1)


def winograd(m: int, k: int, n: int, depth: int) -> int:
    leaf = 7**depth * mm(m // 2**depth, k // 2**depth, n // 2**depth)
    overhead = sum(
        7**level * 11 * (m * k + k * n + m * n) // 4 ** (level + 1)
        for level in range(depth)
    )
    return leaf + overhead


if RESULT.exists():
    raise RuntimeError("one-shot smoke result already exists; no retry allowed")
seal = json.loads(SEAL.read_text())
if seal.get("status") != "FROZEN_BEFORE_ONE_SHOT_SMOKE":
    raise RuntimeError("smoke seal status mismatch")
for relative, expected in seal["sha256"].items():
    observed = sha256(HERE / relative)
    if observed != expected:
        raise RuntimeError(f"sealed hash mismatch for {relative}: {observed}")
if sha256(CANDIDATE) != EXPECTED_CANDIDATE_SHA256:
    raise RuntimeError("candidate hash mismatch")
spec = importlib.util.spec_from_file_location("p2r0c1d5_candidate", CANDIDATE)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

rng = np.random.default_rng(50_241)
left_plain = rng.standard_normal((32, 32)).astype(np.float32)
right_plain = (rng.standard_normal((2, 32, 64)) / np.sqrt(32)).astype(np.float32)
left = fnp.asarray(left_plain)
right = fnp.asarray(right_plain)
output = fnp.empty((2, 32, 64), dtype=fnp.float32)
with flops.BudgetContext(flop_budget=1_000_000_000, quiet=True) as meter:
    paired_left = fnp.stack((left, left), axis=0)
    observed = module._cp3_winograd(paired_left, right, 4, out=output)
expected_flops = 2 * 32 * 32 + 2 * winograd(32, 32, 64, 4)
plain = np.asarray(observed)
reference = left_plain.astype(np.float64)[None, ...] @ right_plain.astype(np.float64)
delta = plain.astype(np.float64) - reference
inventory = Counter(record.op_name for record in meter.op_log)
expected_inventory = Counter({
    "subtract": 32,
    "add": 28,
    "stack": 9,
    "block": 3,
    "concatenate": 2,
    "matmul": 1,
})
checks = {
    "literal_formula_parity": int(meter.flops_used) == expected_flops,
    "operation_count": len(meter.op_log) == 75,
    "operation_inventory": inventory == expected_inventory,
    "shape": tuple(plain.shape) == (2, 32, 64),
    "dtype": str(plain.dtype) == "float32",
    "finite": bool(np.isfinite(plain).all()),
    "max_abs_vs_float64": float(np.max(np.abs(delta))) <= 1.0e-4,
}
result = {
    "status": "PASS_TARGET_FREE_KERNEL_SMOKE" if all(checks.values()) else "FAIL_TARGET_FREE_KERNEL_SMOKE",
    "candidate_sha256": sha256(CANDIDATE),
    "seed": 50241,
    "input_shapes": [[32, 32], [2, 32, 64]],
    "depth": 4,
    "literal_flops": int(meter.flops_used),
    "formula_flops": expected_flops,
    "operation_count": len(meter.op_log),
    "operation_inventory": dict(sorted(inventory.items())),
    "rms_delta_vs_float64": float(np.sqrt(np.mean(delta * delta))),
    "max_abs_delta_vs_float64": float(np.max(np.abs(delta))),
    "checks": checks,
}
RESULT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
if not all(checks.values()):
    raise SystemExit(1)
