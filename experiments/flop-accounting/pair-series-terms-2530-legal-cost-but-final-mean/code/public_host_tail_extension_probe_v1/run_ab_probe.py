#!/usr/bin/env python3
"""Target-free A/B probe: T30 tail-extended host vs frozen base host on 8 rows."""
import hashlib, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "fullbudget_k21_k4_closure_q24_manual_rho_hybrid_near_floor_min_v1/estimator.py"
T30 = HERE / "estimator_t30.py"
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
PROBE_RANKS = [0, 1, 2, 55, 100, 201, 400, 643]
# S-receipt layer error RMS (population, banked, no target read here):
S_LAYER_MSE_FINAL = 9.115992600417398e-08

def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as h:
        for b in iter(lambda: h.read(1 << 20), b""):
            d.update(b)
    return d.hexdigest()

def load(tag, path):
    spec = importlib.util.spec_from_file_location(tag, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[tag] = m
    spec.loader.exec_module(m)
    return m

def predict(module, weights, seed, name):
    mlp = MLP(width=1024, depth=16, weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(seed)), name=name)
    mlp.validate()
    est = module.Estimator()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=300.0, quiet=True) as ctx:
            out = est.predict(mlp, 2**41)
    return np.asarray(out, dtype=np.float64), int(ctx.flops_used)

def main():
    flops.configure(symmetry_warnings=False)
    base = load("probe_base", BASE)
    t30 = load("probe_t30", T30)
    manifest = json.loads((STORE / "manifest.json").read_text())
    entries = manifest["roles"]["observer_fit"]["row_manifests"]
    rows = []
    for rank in PROBE_RANKS:
        entry = entries[rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        weights = np.load(rmp.parent / "weights.npy", mmap_mode="r", allow_pickle=False)
        t0 = time.perf_counter()
        pb, fb = predict(base, weights, int(row["mlp_seed"]), f"probe-base-{rank}")
        pt, ft = predict(t30, weights, int(row["mlp_seed"]), f"probe-t30-{rank}")
        delta = pt - pb
        per_layer_rms = np.sqrt(np.mean(np.square(delta), axis=1))
        rows.append({
            "rank": rank,
            "finite": bool(np.isfinite(pt).all()),
            "base_flops": fb,
            "t30_flops": ft,
            "added_flops": ft - fb,
            "delta_final_rms": float(per_layer_rms[-1]),
            "delta_max_abs": float(np.abs(delta).max()),
            "delta_layer_rms": per_layer_rms.tolist(),
            "wall_seconds": time.perf_counter() - t0,
        })
        print(json.dumps({k: rows[-1][k] for k in ("rank","finite","added_flops","delta_final_rms","delta_max_abs")}), flush=True)
    error_rms_final = float(np.sqrt(S_LAYER_MSE_FINAL))
    summary = {
        "schema": "p2r0f2t-ab-effect-probe-v1",
        "status": "COMPLETE_TARGET_FREE",
        "base_sha256": sha(BASE),
        "t30_sha256": sha(T30),
        "probe_ranks": PROBE_RANKS,
        "rows": rows,
        "population_final_error_rms_from_s_receipt": error_rms_final,
        "median_delta_final_rms": float(np.median([r["delta_final_rms"] for r in rows])),
        "max_delta_final_rms": float(max(r["delta_final_rms"] for r in rows)),
        "median_effect_to_error_ratio": float(np.median([r["delta_final_rms"] for r in rows]) / error_rms_final),
        "max_added_flops": max(r["added_flops"] for r in rows),
        "all_finite": all(r["finite"] for r in rows),
        "targets_read": 0,
    }
    Path(HERE / "AB_EFFECT_PROBE.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("median_delta_final_rms","max_delta_final_rms","population_final_error_rms_from_s_receipt","median_effect_to_error_ratio","max_added_flops","all_finite")}, indent=1))

if __name__ == "__main__":
    main()
