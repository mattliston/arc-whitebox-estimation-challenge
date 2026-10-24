#!/usr/bin/env python3
"""TAG g19e-f2q-probe-v1: linear probe for tape augmentation value.
Cross-unit, deployable features from carried planes (K21 contractions,
K22-residual star trace, tangent-chain dv_hat), transported to L16 through
the gate recursion; ridge against the DEPLOYED (post-J) residual; scored
cross-net 35<->100 noise-corrected. Signal here justifies the F2Q retrain.
Usage: run_f2q_probe.py"""
import json
import sys
from math import erf, factorial
from pathlib import Path

import numpy as np

_erf = np.vectorize(erf)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_attribution import edg2_relu_mean

P2 = HERE.parent
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")


def build_features(rank):
    d = np.load(HERE / f"states_rank{rank:04d}.npz")
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    K = np.load(HERE / f"sidecar_K_rank{rank:04d}.npz")["K"].astype(np.float64)
    s = np.load(HERE / f"sidecar_states_rank{rank:04d}.npz")
    tr = np.load(HERE / f"mps_truth_rank{rank:04d}_P8M.npz")
    dep = np.load(HERE / f"deployed_rank{rank:04d}.npy").astype(np.float64)
    raw20 = s["raw20"]
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")
    Wd = [np.asarray(W[l], dtype=np.float64) for l in range(16)]

    def transport_from(l0, vec):
        r = vec
        for l in range(l0 + 1, 16):
            r = raw20[l, :, 5] * (r @ Wd[l])
        return r

    cols, names = [], []
    eps = np.zeros((1024, 1024))
    for l in range(1, 15):
        i = l - 1
        m_, v_ = d["host_m"][i], d["host_v"][i]
        k3, k4 = d["host_k3"][i], d["host_k4"][i]
        C = z["cov"][i].astype(np.float64)
        K21 = z["k21"][i].astype(np.float64)
        sd = np.sqrt(np.maximum(v_, 1e-30))
        al = m_ / sd
        pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
        cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
        # cross-unit contraction features at this layer
        f_local = {
            "K21phi": K21 @ (pdf / 1024.0),
            "K21Tphi": K21.T @ (pdf / 1024.0),
            "K21cdf": K21 @ (cdf / 1024.0),
            "K21row": np.sqrt(np.mean(K21 ** 2, axis=1)),
            "Kres_phi": (K[i] @ (pdf / 1024.0)) if i < K.shape[0]
            else np.zeros(1024),
        }
        # tangent-chain step (deployable supplyK mode)
        c1cols = [sd * pdf + m_ * cdf, sd * cdf]
        ma = -al
        He = [np.ones_like(al), ma]
        for q in range(2, 14):
            He.append(ma * He[-1] - (q - 1) * He[-2])
        for q in range(2, 12):
            c1cols.append(sd * He[q - 2] * pdf)
        c1 = np.stack(c1cols)
        rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
        D = np.zeros_like(rho)
        rp = np.ones_like(rho)
        for t in range(1, 9):
            if t > 1:
                rp = rp * rho
            D += t * rp / factorial(t) * np.outer(c1[t], c1[t])
        D = D / np.outer(sd, sd)

        def Gker(a, b, tmax=8):
            acc = np.zeros_like(rho)
            rp2 = np.ones_like(rho)
            for t in range(tmax + 1):
                if t > 0:
                    rp2 = rp2 * rho
                acc += rp2 / factorial(t) * np.outer(
                    c1[a + t] / sd ** a, c1[b + t] / sd ** b)
            return acc

        Kk = K[i] if i < K.shape[0] else np.zeros_like(rho)
        inj = (Kk / 4.0) * Gker(2, 2)
        core = D * eps + inj
        Wn = Wd[l + 1]
        eps = Wn.T @ core @ Wn
        f_local["dv_hat"] = np.diag(eps).copy()
        for nm, vec in f_local.items():
            cols.append(transport_from(l, vec))
            names.append(f"L{l}_{nm}")
    X = np.stack(cols, axis=1)
    y = tr["mc_post"][15].astype(np.float64) - dep[15]
    noise = float(np.sum(tr["mc_se"][15].astype(np.float64) ** 2))
    return X, y, noise, names


def main():
    data = {r: build_features(r) for r in (35, 100)}
    names = data[35][3]
    out = {}
    for lam in (1e-4, 1e-3, 1e-2):
        res = {}
        for fit_r, test_r in ((35, 100), (100, 35)):
            Xf, yf, _, _ = data[fit_r]
            G = Xf.T @ Xf
            G += lam * np.trace(G) / G.shape[0] * np.eye(G.shape[0])
            beta = np.linalg.solve(G, Xf.T @ yf)
            Xt, yt, noise, _ = data[test_r]
            den = float(np.sum(yt ** 2))
            num = float(np.sum((yt - Xt @ beta) ** 2))
            res[f"{fit_r}->{test_r}"] = {
                "deployed_nc": (den - noise) / 1024.0,
                "corrected_nc": (num - noise) / 1024.0,
                "reduction_nc": round(1.0 - max(num - noise, 0.0)
                                      / max(den - noise, 1e-30), 4)}
        out[str(lam)] = res
        print(lam, json.dumps(res))
    (HERE / "F2Q_PROBE_RESULT.json").write_text(json.dumps(
        {"names_per_layer": ["K21phi", "K21Tphi", "K21cdf", "K21row",
                             "Kres_phi", "dv_hat"], "results": out},
        indent=1) + "\n")


if __name__ == "__main__":
    main()
