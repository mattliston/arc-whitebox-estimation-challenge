#!/usr/bin/env python3
"""TAG g19b-perfect-ceiling-v1: 10-net leave-one-net-out post-J ceiling with
PERFECT state gaps (oracle capacity receipt). Universal betas; subset ladder.
Writes PERFECT_STATE_CEILING.json."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, '.')
from run_attribution import edg2_relu_mean
HERE = Path('.').resolve()
NETS = [150, 200, 250, 300, 400, 500, 600, 650, 35, 100]

def load_net(rank):
    lite = HERE / f"states_rank{rank:04d}_lite.npz"
    if lite.exists(): st = dict(np.load(lite))
    else:
        d = np.load(f"states_rank{rank:04d}.npz")
        st = {k: d[k] for k in ("host_post","host_m","host_v","host_k3","host_k4")}
    s = np.load(f"sidecar_states_rank{rank:04d}.npz")
    for k in ("s_v","s_k3","s_k4","raw20"): st[k] = s[k]
    tr = np.load(f"mps_truth_rank{rank:04d}_P8M.npz")
    dep = np.load(f"deployed_rank{rank:04d}.npy")
    P2 = HERE.parent
    STORE = P2/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
    manifest = json.loads((STORE/"manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE/entry["path"]).parent/"weights.npy", mmap_mode="r")
    Wd = [np.asarray(W[l], dtype=np.float64) for l in range(16)]
    return st, tr, dep, Wd

def cols(rank):
    st, tr, dep, Wd = load_net(rank)
    r = {k: np.zeros(1024) for k in ("k4","k3","v","m")}
    for l in range(1, 16):
        i = l-1
        hm,hv,hk3,hk4 = st["host_m"][i],st["host_v"][i],st["host_k3"][i],st["host_k4"][i]
        base = edg2_relu_mean(hm,hv,hk3,hk4)
        dds = {"k4": edg2_relu_mean(hm,hv,hk3,tr["mc_pre_k4"][l])-base,
               "k3": edg2_relu_mean(hm,hv,tr["mc_pre_k3"][l],hk4)-base,
               "v": edg2_relu_mean(hm,tr["mc_pre_v"][l],hk3,hk4)-base,
               "m": edg2_relu_mean(tr["mc_pre_m"][l],hv,hk3,hk4)-base}
        phi = st["raw20"][l,:,5]
        for k in r: r[k] = phi*(r[k]@Wd[l]) + dds[k]
    resid = tr["mc_post"][15]-dep[15]
    noise = float(np.sum(tr["mc_se"][15]**2))
    return r, resid, noise

cache = {r: cols(r) for r in NETS}
out = {"schema": "g19b-perfect-state-ceiling-v1", "nets": NETS,
       "scope": "oracle_capacity", "subsets": {}}
for subset in (("v",),("v","k3"),("k4","k3","v"),("m",),("m","v"),("m","v","k3"),("k4","k3","v","m")):
    news, deps, betas = [], [], []
    for hold in NETS:
        trn = [x for x in NETS if x != hold]
        GT = np.concatenate([np.stack([cache[r][0][k] for k in subset],axis=1) for r in trn])
        RT = np.concatenate([cache[r][1] for r in trn])
        beta,*_ = np.linalg.lstsq(GT, RT, rcond=None)
        betas.append([float(b) for b in beta])
        G = np.stack([cache[hold][0][k] for k in subset],axis=1)
        resid, noise = cache[hold][1], cache[hold][2]
        deps.append(float(np.sum(resid**2))/1024 - noise/1024)
        news.append(float(np.sum((resid-G@beta)**2))/1024 - noise/1024)
    out["subsets"]["+".join(subset)] = {
        "mean_deployed_nc": float(np.mean(deps)),
        "mean_corrected_nc": float(np.mean(news)),
        "per_net_corrected_nc": {str(n): float(v) for n, v in zip(NETS, news)},
        "loo_betas": betas}
Path("PERFECT_STATE_CEILING_M.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps({k: {"mean": v["mean_corrected_nc"]}
                  for k, v in out["subsets"].items()}, indent=1))
