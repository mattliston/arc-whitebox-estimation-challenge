#!/usr/bin/env python3
"""TAG g19g-collective-card-screen: run the collective card on panel
nets, report per-net final-layer noise-corrected MSE vs the DEPLOYED
outputs (same truth files). Pure raw comparison - the correction is
deterministic analytics with ~zero bill, so raw delta = adj delta.
Usage: (data_gate venv) run_card_screen.py <rank> [rank ...]"""
import importlib.util as ilu
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G19 = HERE.parent / "public_g19_attribution_v1"
STORE = (HERE.parent / "learned_residual_observer_design/"
         "v4_feature_store_v3/sealed_target_blind_feature_store_v3")


def load_card(tag):
    spec = ilu.spec_from_file_location(tag, HERE / "estimator.py")
    mod = ilu.module_from_spec(spec)
    sys.modules[tag] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    import flopscope as flops
    from threadpoolctl import threadpool_limits
    manifest = json.loads((STORE / "manifest.json").read_text())
    mod = load_card("coll_card")
    out = {}
    for a in sys.argv[1:]:
        rank = int(a)
        entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
        rmp = STORE / entry["path"]
        row = json.loads(rmp.read_text())
        W = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                    allow_pickle=False)
        mlp = MLP(width=1024, depth=16,
                  weights=[W[l] for l in range(16)],
                  seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
                  name=f"scr-{rank}")
        mlp.validate()
        with threadpool_limits(limits=2):
            with flops.BudgetContext(flop_budget=2**43,
                                     wall_time_limit_s=600.0,
                                     quiet=True):
                pred_all = np.asarray(
                    mod.Estimator().predict(mlp, 2**43),
                    dtype=np.float64)
        pred = pred_all[15]
        tr = np.load(G19 / f"mps_truth_rank{rank:04d}_P8M.npz")
        dep = np.load(G19 / f"deployed_rank{rank:04d}.npy")
        truth = tr["mc_post"][15].astype(np.float64)
        noise = float(np.sum(tr["mc_se"][15] ** 2))
        mse_new = (float(np.sum((truth - pred) ** 2)) - noise) / 1024
        mse_dep = (float(np.sum((truth - dep[15]) ** 2)) - noise) / 1024
        out[rank] = {"deployed": mse_dep, "collective_card": mse_new,
                     "delta_pct": round(100 * (1 - mse_new / mse_dep), 2)}
        print(rank, json.dumps(out[rank]))
    if out:
        ds = [v["delta_pct"] for v in out.values()]
        print(json.dumps({"mean_delta_pct": round(float(np.mean(ds)), 2),
                          "sd": round(float(np.std(ds)), 2)}))


if __name__ == "__main__":
    main()
