#!/usr/bin/env python3
"""TAG residue-analysis: exact per-step decomposition
  dK21_{l+1} = T1 (E-transport of old delta) + T2 (map gap on host)
             + T3 (quotient read R[Q_l]),
with T1 = ktfe-hseed, T2 = hseed-host, T3 = full-ktfe (all banked/new).
Identity check is ALGEBRAIC (must hold to fp error). Then: Frobenius
norms, values through the C1 adjoint M_l, and structure of the
per-step generator G_eff = T2+T3."""
import json
import numpy as np
import sys
sys.path.insert(0, ".")
from run_tangent_v2 import c1_std_factors

rank = 35
d = np.load(f"states_rank{rank:04d}_lite.npz") if __import__(
    "pathlib").Path(f"states_rank{rank:04d}_lite.npz").exists() \
    else np.load(f"states_rank{rank:04d}.npz")
z = np.load(f"host_cov_rank{rank:04d}_base.npz")
tp = np.load(f"tensor_planes_rank{rank:04d}.npz")
hs = np.load(f"tensor_planes_rank{rank:04d}_d2_hseed.npz")
kf = np.load(f"tensor_planes_rank{rank:04d}_d2_ktfe.npz")
s = np.load(f"sidecar_states_rank{rank:04d}.npz")
tr = np.load(f"mps_truth_rank{rank:04d}_P8M.npz")
dep = np.load(f"deployed_rank{rank:04d}.npy")
raw20 = s["raw20"]
P2 = __import__("pathlib").Path(".").resolve().parent
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
manifest = json.loads((STORE / "manifest.json").read_text())
entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
Wd = [np.asarray(W[k], dtype=np.float64) for k in range(16)]
resid = (tr["mc_post"][15] - dep[15]).astype(np.float64)
e1 = {}
a2 = {}
Sv = {}
for l in range(1, 16):
    i = l - 1
    m_ = d["host_m"][i].astype(np.float64)
    v_ = np.maximum(d["host_v"][i].astype(np.float64), 1e-30)
    c1, e, sd = c1_std_factors(m_, v_)
    e1[l] = c1[1] / sd
    a2[l] = c1[2] / sd ** 2
    al = m_ / sd
    Sv[l] = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi) / (2 * sd)
g = {}
acc = resid.copy()
for L in range(15, 0, -1):
    g[L] = acc.copy()
    acc = Wd[L] @ (raw20[L, :, 5].astype(np.float64) * acc)
Mcore = None
Ms = {}
for l in range(14, 0, -1):
    A = e1[l][:, None] * Wd[l + 1]
    core = (Mcore if Mcore is not None else np.zeros((1024, 1024)))
    core = core + np.diag(Sv[l + 1] * g[l + 1])
    Mcore = A @ core @ A.T
    Ms[l] = (a2[l][:, None] * Mcore) * e1[l][None, :]
rows = []
tots = np.zeros(4)
for l in range(4, 14):
    L1 = l + 1
    if f"pre_{L1}" not in tp.files:
        continue
    h1 = z["k21"][L1 - 1].astype(np.float64)
    full1 = tp[f"pre_{L1}"].astype(np.float64)
    hs1 = hs[f"pre_{L1}"].astype(np.float64)
    kf1 = kf[f"pre_{L1}"].astype(np.float64)
    dK1 = full1 - h1
    T1 = kf1 - hs1
    T2 = hs1 - h1
    T3 = full1 - kf1
    idcheck = float(np.sqrt(np.mean((T1 + T2 + T3 - dK1) ** 2))
                    / max(np.sqrt(np.mean(dK1 ** 2)), 1e-300))
    M = Ms[L1] if L1 in Ms else None
    def val(X):
        return float(np.sum(X * M)) * 1e7 if M is not None else 0.0
    G = T2 + T3
    U, S, Vt = np.linalg.svd(G)
    r95 = int(np.searchsorted(np.cumsum(S**2)/np.sum(S**2), 0.95)) + 1
    rows.append({"L": L1, "idcheck": round(idcheck, 6),
                 "n_dK": float(np.sqrt(np.mean(dK1**2))),
                 "n_T1": float(np.sqrt(np.mean(T1**2))),
                 "n_T2": float(np.sqrt(np.mean(T2**2))),
                 "n_T3": float(np.sqrt(np.mean(T3**2))),
                 "n_Geff": float(np.sqrt(np.mean(G**2))),
                 "v_dK": round(val(dK1), 2), "v_T1": round(val(T1), 2),
                 "v_T2": round(val(T2), 2), "v_T3": round(val(T3), 2),
                 "v_Geff": round(val(G), 2), "Geff_r95": r95})
    tots += np.array([val(dK1), val(T1), val(T2), val(G)])
    print(json.dumps(rows[-1]))
print(json.dumps({"totals_x1e7": {"dK": round(tots[0],2),
    "T1_transport": round(tots[1],2), "T2_mapgap": round(tots[2],2),
    "Geff": round(tots[3],2)}}))
