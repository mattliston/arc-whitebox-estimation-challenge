#!/usr/bin/env python3
"""TAG g19g-f2q-export: run the collective-instrumented card (injection
OFF via COLL_SCALE=0, tracker ON) on one store rank and save the F2Q
tape-augmentation features from the card's OWN states (train/deploy
consistent by construction):
  coll_feats[l, i, :] = [K3S_l * w_i^2, w_i, (K3S_l/sigma_l) * b_i]
Saved to rows/observer_fit/<rank>/coll_feats.npy.
Usage: (venv) export_coll_features.py <rank>"""
import importlib.util as ilu
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STORE = (HERE.parent / "learned_residual_observer_design/"
         "v4_feature_store_v3/sealed_target_blind_feature_store_v3")
ROWS = STORE / "rows/observer_fit"
OUT = (HERE.parent / "public_f2q_coll_features_v1/rows")


def main():
    rank = int(sys.argv[1])
    os.environ["COLL_SCALE"] = "0.0"
    os.environ["COLL_EXPORT"] = "1"
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    import flopscope as flops
    from threadpoolctl import threadpool_limits
    spec = ilu.spec_from_file_location(f"cexp{rank}",
                                       HERE / "estimator.py")
    mod = ilu.module_from_spec(spec)
    sys.modules[f"cexp{rank}"] = mod
    spec.loader.exec_module(mod)
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    W = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                allow_pickle=False)
    mlp = MLP(width=1024, depth=16,
              weights=[W[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"cx-{rank}")
    mlp.validate()
    mod._COLL_EXPORT_ROWS.clear()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2 ** 43,
                                 wall_time_limit_s=600.0, quiet=True):
            mod._predict(mlp)
    feats = np.zeros((16, 1024, 3), dtype=np.float32)
    for (l, w_e, b_e, k3s, sig) in mod._COLL_EXPORT_ROWS:
        feats[l, :, 0] = k3s * w_e * w_e
        feats[l, :, 1] = w_e
        feats[l, :, 2] = (k3s / max(sig, 1e-30)) * b_e
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{rank:04d}.npy", feats)
    print(json.dumps({"rank": rank, "rows": len(mod._COLL_EXPORT_ROWS),
                      "saved": True}))


if __name__ == "__main__":
    main()
