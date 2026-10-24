#!/usr/bin/env python3
"""Verify the frozen public-prelaunch Phase 2 authority and scaling arithmetic."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
authority = json.loads((HERE / "OFFICIAL_SOURCE_AUDIT.json").read_text())
scaling = json.loads((HERE / "STATIC_SCALING_GATE.json").read_text())

assert authority["status"] == "PASS_PUBLIC_ASSET_AUTHORITY__OFFICIAL_LAUNCH_PENDING"
assert authority["launch_status"]["submissions_open"] is False
assert authority["dataset"]["revision"] == "v2-phase2"
assert authority["dataset"]["weight_shape"] == [16, 1024, 1024]
assert authority["dataset"]["prediction_shape"] == [16, 1024]
assert authority["round"]["flop_budget_per_mlp"] == 2**41
assert authority["software"]["whestbench"]["version"] == "0.16.0"
assert authority["software"]["flopscope"]["version"] == "0.12.0"

p1 = scaling["phase1"]
p2 = scaling["phase2"]
ratios = scaling["ratios"]
assert p1["forward_flops"] == 2 * p1["depth"] * p1["width"] ** 2
assert p2["forward_flops"] == 2 * p2["depth"] * p2["width"] ** 2
assert p2["budget"] == 2**41
assert abs(ratios["budget"] - p2["budget"] / p1["budget"]) < 1e-15
assert ratios["fixed_rank_depth_width_squared_geometry"] == (
    p2["depth"] * p2["width"] ** 2 / (p1["depth"] * p1["width"] ** 2)
)

screen = scaling["old_r22_screen"]
formula = lambda n: 20 * n**3 + 64 * n**2 + 1024 * n + 32768
assert screen["phase2_dense_psd_per_transition"] == formula(p2["width"])
assert screen["phase2_dense_psd_total_15_transitions"] == (
    p2["transitions_after_first_layer"] * formula(p2["width"])
)
assert screen["eight_x_non_psd_remainder_plus_phase2_psd"] > p2["budget"]

print("PASS_PHASE2_PUBLIC_PRELAUNCH_STATIC_AUTHORITY")

