#!/usr/bin/env python3
"""TAG d11-cap-screen: run the CP3_CAP-gated D11 variant on stored panel
nets; report billed flops + noise-corrected final-layer MSE vs truth and
vs the deployed outputs. Usage: (venv) CP3_CAP=4096 run_cap_screen.py 35"""
import importlib.util as ilu
import json, os, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
P2 = HERE.parent
G19 = P2 / "public_g19_attribution_v1"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
def main():
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    import flopscope as flops
    from threadpoolctl import threadpool_limits
    flops.configure(symmetry_warnings=False)
    manifest = json.loads((STORE / "manifest.json").read_text())
    spec = ilu.spec_from_file_location("capcard", HERE / "estimator_cap.py")
    mod = ilu.module_from_spec(spec); sys.modules["capcard"] = mod
    spec.loader.exec_module(mod)
    out = {}
    for a in sys.argv[1:]:
        rank = int(a)
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        W = np.load(rmp.parent / "weights.npy", mmap_mode="r")
        mlp = MLP(width=1024, depth=16, weights=[W[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"cap-{rank}")
        mlp.validate()
        with threadpool_limits(limits=2):
            ctx = flops.BudgetContext(flop_budget=2**41,
                                      wall_time_limit_s=600.0, quiet=True)
            with ctx:
                pred_all = np.asarray(mod.Estimator().predict(mlp, 2**41),
                                      dtype=np.float64)
        billed = int(ctx.flops_used)
        pred = pred_all[15]
        tr = np.load(G19 / f"mps_truth_rank{rank:04d}_P8M.npz")
        dep = np.load(G19 / f"deployed_rank{rank:04d}.npy")
        truth = tr["mc_post"][15].astype(np.float64)
        noise = float(np.sum(tr["mc_se"][15] ** 2))
        mse = (float(np.sum((truth - pred) ** 2)) - noise) / 1024
        mse_dep = (float(np.sum((truth - dep[15]) ** 2)) - noise) / 1024
        out[rank] = {"billed": billed, "u_local": round(billed / 2**41, 4),
                     "mse_nc": mse, "dep_mse_nc": mse_dep,
                     "raw_delta_pct": round(100*(mse/mse_dep-1), 2)}
        print(rank, json.dumps(out[rank]))
    cap = os.environ.get("CP3_CAP", "16384")
    (HERE / f"CAP_SCREEN_{cap}.json").write_text(
        json.dumps(out, indent=1) + "\n")
if __name__ == "__main__":
    main()
