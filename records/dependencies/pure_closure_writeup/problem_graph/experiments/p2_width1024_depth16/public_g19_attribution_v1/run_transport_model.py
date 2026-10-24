#!/usr/bin/env python3
"""TAG g19b-transport-model-v1: verify the sidecar-correction identity.
Claim: the host's per-layer mean error follows e_l = Phi_l * (e_{l-1} @ W_l)
+ delta_l with delta_l = Edg2(true state_l) - Edg2(host state_l), so a
correction built by transporting delta's OUTSIDE the recursion and adding at
the output recovers the error. Truth-fed ceiling: rebuild e_16 from the
delta's; report recovered fraction of endpoint error energy, plus the same
with delta's from ONLY (k4,k3) differences (the buildable sidecar scope).
Usage: run_transport_model.py <rank>"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
BASE = P2 / "public_q24_native_transport_energy_final_workspace_v4/estimator.py"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean, load_host, tap_host_state


def main():
    rank = int(sys.argv[1])
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    mc_m, mc_v = d["mc_pre_m"], d["mc_pre_v"]
    mc_k3, mc_k4 = d["mc_pre_k3"], d["mc_pre_k4"]
    mc_post, mc_se, host_post = d["mc_post"], d["mc_se"], d["host_post"]
    hm, hv, hk3, hk4 = d["host_m"], d["host_v"], d["host_k3"], d["host_k4"]

    # gates Phi_l = raw20[l,:,5] from the native tape + weights
    import flopscope as flops
    from threadpoolctl import threadpool_limits
    from whestbench.domain import MLP
    from whestbench.seeds import derive_estimator_seed
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    rmp = STORE / entry["path"]
    row = json.loads(rmp.read_text())
    weights = np.load(rmp.parent / "weights.npy", mmap_mode="r",
                      allow_pickle=False)
    mod = load_host(f"tm_{rank}")
    mlp = MLP(width=1024, depth=16,
              weights=[weights[l] for l in range(16)],
              seed=int(derive_estimator_seed(int(row["mlp_seed"]))),
              name=f"tm-{rank}")
    mlp.validate()
    with threadpool_limits(limits=2):
        with flops.BudgetContext(flop_budget=2 ** 43,
                                 wall_time_limit_s=600.0, quiet=True):
            _hv, raw_v, _sv = mod.predict_with_native_tape(mlp)
    raw20 = np.asarray(raw_v, dtype=np.float64)
    assert raw20.shape[0] == 16 and raw20.shape[1] == 1024, raw20.shape
    W = [np.asarray(weights[l], dtype=np.float64) for l in range(16)]

    e_true = mc_post - host_post

    def run_variant(name, deltas):
        # r_l: transported correction, r_1 = delta at index1
        r = np.zeros((16, 1024))
        for l in range(1, 16):
            phi = raw20[l, :, 5]
            r[l] = phi * (r[l - 1] @ W[l]) + deltas[l]
        rec = {}
        for scope, idx in (("L16", [15]), ("late", range(10, 16))):
            num = 0.0
            den = 0.0
            noi = 0.0
            for l in idx:
                num += float(np.sum((e_true[l] - r[l]) ** 2))
                den += float(np.sum(e_true[l] ** 2))
                noi += float(np.sum(mc_se[l] ** 2))
            rec[scope] = {
                "residual_over_error": num / den,
                "recovered_fraction_nc": 1.0 - max(num - noi, 0.0)
                                               / max(den - noi, 1e-30),
            }
        return {name: rec}

    deltas_full = np.zeros((16, 1024))
    deltas_k43 = np.zeros((16, 1024))
    for l in range(1, 16):
        i = l - 1
        base = edg2_relu_mean(hm[i], hv[i], hk3[i], hk4[i])
        deltas_full[l] = edg2_relu_mean(mc_m[l], mc_v[l], mc_k3[l],
                                        mc_k4[l]) - base
        deltas_k43[l] = edg2_relu_mean(hm[i], hv[i], mc_k3[l],
                                       mc_k4[l]) - base
    # the m-part of delta_full double counts the transport; the injection
    # form uses per-layer FRESH terms: delta with TRUE m replaced by the
    # host m plus the transported previous error is not directly available,
    # so test both spellings:
    out = {"rank": rank}
    out.update(run_variant("full_delta", deltas_full))
    out.update(run_variant("k4k3_delta_only", deltas_k43))
    # fresh-injection spelling: v/k3/k4 swaps only (m handled by transport)
    deltas_fresh = np.zeros((16, 1024))
    for l in range(1, 16):
        i = l - 1
        base = edg2_relu_mean(hm[i], hv[i], hk3[i], hk4[i])
        deltas_fresh[l] = edg2_relu_mean(hm[i], mc_v[l], mc_k3[l],
                                         mc_k4[l]) - base
    out.update(run_variant("fresh_vk3k4_delta", deltas_fresh))
    (HERE / f"TRANSPORT_MODEL_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
