#!/usr/bin/env python3
"""TAG fable-engineer2 (E2-EW): the analyst's EXPECTED WORST roundtrip (fable_analyst_v1/timeout_risk.py AN-45) for one eval page:
OLS of predict wall on 1 + standardized flops + cold (cold = MLP right after a worker setup: participant_setup_wall_s present),
expected worst = max(fit) + median(roundtrip - predict). Also mean kernel and the MLP count used. usage: expected_worst.py <sid> [...]"""
import json, sys
from pathlib import Path
import numpy as np
H = Path(__file__).resolve().parent; F = H.parent.parent / "fable_fire_v1"
for sid in sys.argv[1:]:
    p = H / f"eval_{sid}.json"
    if not p.exists(): p = F / f"eval_{sid}.json"
    per = [m for m in json.load(open(p))["evaluation"]["results"]["per_mlp"] if m.get("telemetry")]
    T = [m["telemetry"] for m in per]
    pred = np.array([t["participant_predict_wall_s"] for t in T]); rt = np.array([t["participant_roundtrip_wall_s"] for t in T])
    flops = np.array([t["flops_used"] for t in T]); cold = np.array([t.get("participant_setup_wall_s") is not None for t in T], float)
    kern = np.array([t["flopscope_server_kernel_wall_s"] for t in T])
    X = np.column_stack([np.ones(len(pred)), (flops - flops.mean()) / max(flops.std(), 1.0), cold])
    beta = np.linalg.lstsq(X, pred, rcond=None)[0]; fit = X @ beta; O = float(np.median(rt - pred))
    print(f"{sid}: n={len(T)} cold={int(cold.sum())} expected worst roundtrip {fit.max() + O:.1f} s (fit max {fit.max():.1f} + O {O:.2f}); "
          f"predict mean {pred.mean():.2f} max {pred.max():.1f}; kernel mean {kern.mean():.2f} s; cold coef {beta[2]:+.2f} s")
