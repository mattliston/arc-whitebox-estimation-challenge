#!/usr/bin/env python3
"""TAG h2h3-C4v1: closed-form birth-accumulation vs banked dK21 value.
Hypothesis: the tensor-borne plane error is the multi-step-pushed
Gaussian-completion DISTINCT birth content (C-determined, chain motifs
only per the DTOT=2 measurement), transported with diagonal (cdf)
dressing. Compute Slice_l = plane slice of sum_{l'<l} Push[Birth_{l'}]
and compare, per layer: corr(Slice_l, dK21_l) and the value ratio
<Slice_l, M_l> / <dK21_l, M_l> (M from C1). All offline, banked r35.
Usage: run_c4v1_closed_form.py <rank>"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_tangent_v2 import c1_std_factors


def main():
    rank = int(sys.argv[1])
    d = (np.load(HERE / f"states_rank{rank:04d}_lite.npz")
         if (HERE / f"states_rank{rank:04d}_lite.npz").exists()
         and not (HERE / f"states_rank{rank:04d}.npz").exists()
         else np.load(HERE / f"states_rank{rank:04d}.npz"))
    z = np.load(HERE / f"host_cov_rank{rank:04d}_base.npz")
    tp = np.load(HERE / f"tensor_planes_rank{rank:04d}.npz")
    P2 = HERE.parent
    STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
             "sealed_target_blind_feature_store_v3")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    W = np.load((STORE / entry["path"]).parent / "weights.npy",
                mmap_mode="r")
    Wd = [np.asarray(W[k], dtype=np.float64) for k in range(16)]

    # per-layer host factors
    F = {}
    for l in range(1, 15):
        i = l - 1
        m_ = d["host_m"][i].astype(np.float64)
        v_ = d["host_v"][i].astype(np.float64)
        c1, e, sd = c1_std_factors(m_, v_)
        C = z["cov"][i].astype(np.float64)
        rho = np.clip(C / np.outer(sd, sd), -0.9999, 0.9999)
        np.fill_diagonal(rho, 0.0)
        al = m_ / sd
        cdf = 0.5 * (1 + np.vectorize(__import__("math").erf)(
            al / np.sqrt(2)))
        F[l] = {"c1": c1, "rho": rho, "cdf": cdf}

    # births at l' (distinct completion, DTOT=2 chain patterns):
    # patterns (di,dj,dk) in {(2,1,1),(1,2,1),(1,1,2)} with bonds
    # rho_ij rho_ik / rho_ij rho_jk / rho_ik rho_jk resp.
    # Accumulate pushed content; P maps birth-frame -> current frame.
    live = {}       # l' -> P (current push matrix)
    out = {}
    slices = {}
    for l in range(1, 15):
        # advance all live births to this frame
        if l > 1:
            Wl = Wd[l]
            cdfprev = F[l - 1]["cdf"]
            for lp in list(live):
                live[lp] = Wl.T @ (cdfprev[:, None] * live[lp])
        # compute the plane slice of accumulated content at this frame
        # (only births strictly older than 1 step: the 1-step content
        # is inside the host's own emission map)
        S = np.zeros((1024, 1024))
        for lp, P in live.items():
            if lp >= l:            # safety
                continue
            c1b = F[lp]["c1"]
            rho = F[lp]["rho"]
            for (di, dj, dk, bond) in ((2, 1, 1, "ij_ik"),
                                       (1, 2, 1, "ij_jk"),
                                       (1, 1, 2, "ik_jk")):
                f, g, h = c1b[di], c1b[dj], c1b[dk]
                if bond == "ij_ik":
                    A1 = (P * g[None, :]) @ rho.T
                    A2 = (P * h[None, :]) @ rho.T
                    S += ((P * f[None, :]) * A1) @ A2.T
                elif bond == "ij_jk":
                    A1 = (P * f[None, :]) @ rho
                    B2 = (P * h[None, :]) @ rho.T
                    S += (A1 * (P * g[None, :])) @ B2.T
                else:
                    A1 = (P * f[None, :]) @ rho
                    S += ((P * g[None, :]) * A1) @ (P * h[None, :]).T
        slices[l] = S
        # register the birth at this layer (post-frame content pushed
        # by W_{l+1} on next advance): P starts as W_{l+1}^T after the
        # next loop iteration; initialize as identity-with-first-push:
        if l < 14:
            live[l] = Wd[l + 1].T
        if f"pre_{l}" in tp.files and l >= 3:
            dK = (tp[f"pre_{l}"].astype(np.float64)
                  - z["k21"][l - 1].astype(np.float64))
            num = float(np.sum(S * dK))
            cc = num / max(np.sqrt(np.sum(S * S) * np.sum(dK * dK)),
                           1e-300)
            beta = num / max(np.sum(S * S), 1e-300)
            out[l] = {"corr": round(cc, 3), "beta": round(beta, 3),
                      "snorm": float(np.sqrt(np.mean(S ** 2))),
                      "dknorm": float(np.sqrt(np.mean(dK ** 2)))}
            print(rank, f"L{l}", json.dumps(out[l]))
    (HERE / f"C4V1_CLOSED_FORM_rank{rank:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
