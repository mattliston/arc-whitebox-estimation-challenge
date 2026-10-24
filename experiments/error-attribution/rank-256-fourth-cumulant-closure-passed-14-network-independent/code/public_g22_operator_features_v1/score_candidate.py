#!/usr/bin/env python3
"""Score a candidate estimator file on one net against the independent 4M truth (noise-corrected final-layer MSE),
with the phase-2 budget, exactly as the grader calls it. Usage: score_candidate.py <candidate.py> <rank>"""
import importlib.util, json, sys, time
from pathlib import Path
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
cand, rank = Path(sys.argv[1]), int(sys.argv[2])
G19 = P2 / "public_g19_attribution_v1"
if rank >= 3016500:   # virgin fresh instance (seed = rank): weights file by seed, truth fresh_{seed}_truth4M.npz
    W = np.load(G19 / f"fresh_{rank}_weights.npy", mmap_mode="r", allow_pickle=False); mlp_seed = rank
else:
    man = json.loads((STORE / "manifest.json").read_text()); e = man["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / e["path"]; row = json.loads(rmp.read_text()); W = np.load(rmp.parent / "weights.npy", mmap_mode="r"); mlp_seed = int(row["mlp_seed"])
mlp = MLP(width=1024, depth=16, weights=[W[l] for l in range(16)], seed=int(derive_estimator_seed(mlp_seed)), name=f"cand-{rank}")
spec = importlib.util.spec_from_file_location(f"cand_{rank}", cand); m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
t0 = time.time()
with threadpool_limits(limits=4):
    with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=900.0, quiet=True) as ctx:
        out = np.asarray(m.Estimator().predict(mlp, 2**41), dtype=np.float64)
T = np.load(G19 / (f"fresh_{rank}_truth4M.npz" if rank >= 3016500 else f"mps_truth_rank{rank:04d}_indep4M.npz"))
tr = T["mc_post"][15].astype(np.float64); se = T["mc_se"][15].astype(np.float64)
mse = float(np.mean((out[15] - tr) ** 2) - np.mean(se ** 2)); util = ctx.flops_used / 2**41
print(json.dumps({"candidate": str(cand.name), "rank": rank, "final_mse_nc": mse, "util": round(util, 4), "adjusted": mse * max(util, 0.1), "wall_s": round(time.time() - t0, 1), "finite": bool(np.isfinite(out).all())}))
