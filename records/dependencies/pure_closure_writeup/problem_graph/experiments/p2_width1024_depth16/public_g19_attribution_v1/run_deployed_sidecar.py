#!/usr/bin/env python3
"""TAG g19b-deployed-sidecar-v1: deployed (host+J) plus post-hoc sidecar
correction, alpha overlap scan, scored vs MC truth. Usage: <rank>"""
import importlib.util, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean, load_host
from run_sidecar_supply import one_forward

def main():
    rank = int(sys.argv[1])
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    s = np.load(HERE / f"sidecar_states_rank{rank:04d}.npz")
    mc_post, mc_se = d["mc_post"], d["mc_se"]
    hm, hv, hk3, hk4 = d["host_m"], d["host_v"], d["host_k3"], d["host_k4"]
    raw20 = s["raw20"]
    P2 = HERE.parent
    STORE = P2/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
    manifest = json.loads((STORE/"manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE/entry["path"]
    row = json.loads(rmp.read_text())
    W = np.load(rmp.parent/"weights.npy", mmap_mode="r", allow_pickle=False)
    Wd = [np.asarray(W[l], dtype=np.float64) for l in range(16)]

    r = np.zeros(1024)
    for l in range(1, 16):
        i = l - 1
        base = edg2_relu_mean(hm[i], hv[i], hk3[i], hk4[i])
        delta = edg2_relu_mean(hm[i], s["s_v"][i], s["s_k3"][i], s["s_k4"][i]) - base
        r = raw20[l, :, 5] * (r @ Wd[l]) + delta

    import flopscope as flops
    from threadpoolctl import threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    mod = load_host(f"dep_{rank}")
    mlp = MLP(width=1024, depth=16, weights=[W[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))), name=f"dep-{rank}")
    mlp.validate()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2**43, wall_time_limit_s=600.0, quiet=True):
            dep = np.asarray(mod.Estimator().predict(mlp, 2**43), dtype=np.float64)
    assert dep.shape == (16, 1024)
    truth = mc_post[15]
    noise = float(np.sum(mc_se[15]**2))/1024.0
    out = {"rank": rank, "noise_mse": noise}
    out["deployed_mse"] = float(np.mean((truth - dep[15])**2))
    for a in (0.25, 0.5, 0.75, 1.0):
        out[f"deployed_plus_{a}_sidecar"] = float(np.mean((truth - dep[15] - a*r)**2))
    host_mse = float(np.mean((truth - d["host_post"][15])**2))
    out["host_mse"] = host_mse
    out["host_plus_sidecar"] = float(np.mean((truth - d["host_post"][15] - r)**2))
    np.save(HERE / f"deployed_rank{rank:04d}.npy", dep)
    print(json.dumps(out, indent=1))
    (HERE / f"DEPLOYED_SIDECAR_rank{rank:04d}.json").write_text(json.dumps(out, indent=1) + "\n")

if __name__ == "__main__":
    main()
