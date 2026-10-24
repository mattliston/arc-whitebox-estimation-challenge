#!/usr/bin/env python3
"""TAG g22-final-cliff-probe (T-115b): why is the final layer's teacher-forced removal 71% when the interior is 99%?
On the oracle rows (rows_meanleak_mech15), fit per-unit RIDGE models of the layer-15 one-step target on
 (a) the 51 features at layer 15,
 (b) (a) + the previous layer's one-step target pushed through W15 (W d14, W^2 d14^2, |W| |d14|) = the transported
     previous-layer error (a proxy for the unobservable covariance residual at 14, which co-moves with the mean error, T-100),
 (c) (a) + the same for layer 13 pushed through W14 then W15,
and report held-out (valid-40) removal for each. If (b) jumps, the cliff is transported error the features cannot see and
the fix is a message channel that carries the previous layer's correction. Also reports the same for layer 12 as a control."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ROWS = HERE / "rows_meanleak_mech15"
VALID = {37, 39, 42, 48, 59, 63, 66, 67, 71, 73, 75, 79, 93, 101, 122, 130, 132, 135, 139, 145, 146, 149, 150, 151, 155, 159, 163, 169, 173, 175, 181, 192, 198, 199, 204, 219, 224, 227, 233, 234}
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]

def load(L):
    Xa, Xb, Xc, Y, V = [], [], [], [], []
    for d in sorted(ROWS.glob("row_*")):
        if not (d / "result.json").exists(): continue
        rank = int(d.name.split("_")[1])
        tape = np.load(d / "raw20.npy").astype(np.float64); mech = np.load(d / "mech.npy").astype(np.float64)
        host = np.load(d / "host.npy").astype(np.float64); sigma = np.load(d / "sigma.npy").astype(np.float64)
        truth = np.load(HERE / "truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
        t = truth - host
        W = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy", mmap_mode="r")
        WL = np.asarray(W[L], dtype=np.float64); WL1 = np.asarray(W[L - 1], dtype=np.float64)
        d1 = t[L - 1]; d2 = t[L - 2]
        fa = np.concatenate([tape[L], mech[L]], axis=1)
        tb = np.stack([WL @ d1, (WL * WL) @ (d1 * d1), np.abs(WL) @ np.abs(d1), WL @ (sigma[L - 1] * d1)], axis=1)
        p2 = WL1 @ d2
        tc = np.stack([WL @ p2, (WL * WL) @ (p2 * p2), np.abs(WL) @ np.abs(p2)], axis=1)
        s = sigma[L][:, None]
        Xa.append(fa / s); Xb.append(np.concatenate([fa, tb], axis=1) / s); Xc.append(np.concatenate([fa, tb, tc], axis=1) / s); Y.append(t[L] / sigma[L]); V.append(np.full(1024, rank in VALID))
    return [np.concatenate(z) for z in (Xa, Xb, Xc, Y, V)]

def ridge_removal(X, y, v, lam=1e-3):
    # per-feature standardization on train, ridge, held-out removal; quadratic expansion of the top-30 features by |corr|
    mu = X[~v].mean(0); sd = X[~v].std(0) + 1e-12; Z = (X - mu) / sd
    c = np.abs((Z[~v] * (y[~v] - y[~v].mean())[:, None]).mean(0)); top = np.argsort(-c)[:30]
    Q = np.concatenate([Z, Z[:, top] ** 2, np.tanh(Z[:, top])], axis=1); Q = np.concatenate([Q, np.ones((len(Q), 1))], axis=1)
    A = Q[~v]; b = y[~v]; w = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ b)
    r = y[v] - Q[v] @ w
    return 1 - np.mean(r ** 2) / np.mean(y[v] ** 2)

for L in (15, 12):
    Xa, Xb, Xc, Y, V = load(L)
    print(f"layer {L}: rows {len(Y)//1024}; held-out removal (ridge+quadratic): features only {100*ridge_removal(Xa, Y, V):.1f}% | + transported d{L-1} {100*ridge_removal(Xb, Y, V):.1f}% | + transported d{L-2} {100*ridge_removal(Xc, Y, V):.1f}%", flush=True)
