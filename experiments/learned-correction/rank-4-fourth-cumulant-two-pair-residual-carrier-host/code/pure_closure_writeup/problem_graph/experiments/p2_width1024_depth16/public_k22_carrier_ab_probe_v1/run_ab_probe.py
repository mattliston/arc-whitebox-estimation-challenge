#!/usr/bin/env python3
"""Target-free A/B probe: K22-residual carrier (rank 4) vs frozen base host.

TAG=k22r-ab  rows=[0,1,2,55,100,201,400,643]  targets_read=0
Gate 1: with K22R_RANK=0 the patched module must reproduce the base host
bit-for-bit at equal metered FLOPs on row 0 (any added branch is inert).
Gate 2: rank-4 A/B on the 8-row roster; report per-row final-output delta
RMS against the S-receipt population error RMS 3.019e-4, plus FLOP deltas.
"""
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_condrepair_deployment_card_v3/estimator.py"
K22R = HERE / "estimator_k22r.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
PROBE_RANKS = [0, 1, 2, 55, 100, 201, 400, 643]
ERROR_RMS = 3.019e-4  # sqrt of the S-receipt population final-layer MSE


def load(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[tag] = module
    spec.loader.exec_module(module)
    return module


def predict(module, weights, seed, name):
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(seed)), name=name)
    mlp.validate()
    estimator = module.Estimator()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=600.0,
                                 quiet=True) as context:
            out = estimator.predict(mlp, 2**41)
    return np.asarray(out, dtype=np.float64), int(context.flops_used)


def main():
    flops.configure(symmetry_warnings=False)
    base = load("k22r_probe_base", BASE)
    k22r = load("k22r_probe_cand", K22R)

    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]

    def row_inputs(rank):
        entry = entries[rank]
        row_manifest_path = STORE / entry["path"]
        row = json.loads(row_manifest_path.read_text())
        weights = np.load(row_manifest_path.parent / "weights.npy",
                          mmap_mode="r", allow_pickle=False)
        return weights, int(row["mlp_seed"])

    # Gate 1: rank-0 bit parity on row 0.
    weights0, seed0 = row_inputs(PROBE_RANKS[0])
    k22r.K22R_RANK = 0
    p_base, f_base = predict(base, weights0, seed0, "k22r-parity-base")
    p_zero, f_zero = predict(k22r, weights0, seed0, "k22r-parity-zero")
    parity = {
        "max_abs_delta": float(np.max(np.abs(p_zero - p_base))),
        "flops_base": f_base,
        "flops_rank0": f_zero,
        "bit_exact": bool(np.array_equal(p_zero, p_base)),
        "flop_equal": f_base == f_zero,
    }
    print("PARITY", json.dumps(parity), flush=True)
    if not (parity["bit_exact"] and parity["flop_equal"]):
        raise SystemExit("rank-0 parity gate FAILED; carrier not inert")

    # Gate 2: rank-4 A/B.
    k22r.K22R_RANK = 4
    rows_report = []
    for rank in PROBE_RANKS:
        weights, seed = row_inputs(rank)
        started = time.perf_counter()
        pb, fb = predict(base, weights, seed, f"k22r-base-{rank}")
        pc, fc = predict(k22r, weights, seed, f"k22r-r4-{rank}")
        delta = pc - pb
        final_delta_rms = float(np.sqrt(np.mean(np.square(delta[-1]))))
        rows_report.append({
            "row": rank,
            "final_delta_rms": final_delta_rms,
            "delta_over_error_rms": final_delta_rms / ERROR_RMS,
            "flops_base": fb,
            "flops_k22r": fc,
            "flops_added": fc - fb,
            "finite": bool(np.all(np.isfinite(pc))),
            "wall_seconds": round(time.perf_counter() - started, 1),
        })
        print("ROW", json.dumps(rows_report[-1]), flush=True)

    ratios = [r["delta_over_error_rms"] for r in rows_report]
    payload = {
        "schema": "k22r-ab-probe-v1",
        "node": "P2R0F3A",
        "status": "COMPLETE_TARGET_FREE",
        "parity_gate": parity,
        "rows": rows_report,
        "summary": {
            "min_delta_over_error": min(ratios),
            "median_delta_over_error": float(np.median(ratios)),
            "max_delta_over_error": max(ratios),
            "mean_flops_added": float(np.mean([r["flops_added"] for r in rows_report])),
        },
        "error_rms_reference": ERROR_RMS,
        "estimator_k22r_sha256": hashlib.sha256(K22R.read_bytes()).hexdigest(),
        "targets_read": 0,
    }
    payload["base_host"] = "public_condrepair_deployment_card_v3/estimator.py (W3: conditional repair + failure guard; the repair dividend funds the carrier)"
    payload["supersedes"] = "AB_K22R_PROBE.json (carrier sha 0ecbbd49... whose k31 channel was exactly 2x the true transported entry; audit-corrected to 3.0 coefficients)"
    out = HERE / "AB_K22R_PROBE_v2.json"
    assert not out.exists(), "append-only"
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print("RECEIPT", hashlib.sha256(out.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
