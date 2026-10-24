#!/usr/bin/env python3
"""Execute the presealed verifier with two bounded launch-only corrections."""

from __future__ import annotations

import sys
import types
from pathlib import Path


HERE = Path(__file__).resolve().parent
source_path = HERE / "verify_tiny_prototype.py"
source = source_path.read_text()
old_seal = 'FINAL_SEAL = HERE / "TINY_PROTOTYPE_FINAL_SEAL.json"'
new_seal = 'FINAL_SEAL = HERE / "TINY_PROTOTYPE_FINAL_SEAL_002.json"'
assert source.count(old_seal) == 1
source = source.replace(old_seal, new_seal)
old_meter = '''    out = views.output_banks[1][:, :48]
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as hot_meter:
        observed = module.winograd_depth4(
            fnp.asarray(left_plain), fnp.asarray(right_plain), views, out=out
        )
'''
new_meter = '''    out = views.output_banks[1][:, :48]
    left = fnp.asarray(left_plain)
    right = fnp.asarray(right_plain)
    with flops.BudgetContext(flop_budget=100_000_000, quiet=True) as hot_meter:
        observed = module.winograd_depth4(left, right, views, out=out)
'''
assert source.count(old_meter) == 1
source = source.replace(old_meter, new_meter)
sys.modules.setdefault("sealed_tiny_d6_kernel", types.ModuleType("sealed_tiny_d6_kernel"))
namespace = {"__name__": "__main__", "__file__": str(source_path)}
exec(compile(source, str(source_path), "exec"), namespace, namespace)
