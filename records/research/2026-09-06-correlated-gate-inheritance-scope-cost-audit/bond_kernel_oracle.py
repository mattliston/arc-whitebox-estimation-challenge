#!/usr/bin/env python3
"""TAG g29-bond-kernel-oracle-v1: is the card's unit-specific deep k3 error the
GATE-ONLY inheritance kernel of its CP3 carrier?

The card gates every pushed CP3 term with the ReLU-derivative expectation only
(post = g_i g_j g_k * pre, g = Phi(alpha); estimator_cap.py `_cp_grow`), i.e. the
independent-orthant approximation of the first-order Edgeworth response.  The
full-tensor engine (run_tensor_recursion_v4.py, F3) multiplies the pre-tensor
by the correlated kernel: F = g g g + [rho_ij f_i f_j g_k + rho_ik f_i g_j f_k
+ rho_jk g_i f_j f_k] + (bond order 2: rho^2 b2 b2 g, rho rho b2 f f) + ...,
f = pdf(alpha), b2 = -alpha pdf(alpha).  For a rank-1 term the correction is
NOT rank-1 (Hadamard product with rho on a bond), so the card cannot carry it;
this oracle computes, per layer, its EFFECT on the next layer's per-unit k3
read (a 1024^4-class contraction done on the Metal GPU) and compares it with
the card's residual dk3 = truth k3 - card k3 at that layer.

Per term r with ungated pushed factors (u, w) at layer l (pre-tensor
T = lam * sym(u,u,w)), next weights W' (in, out), read unit m, W ≡ W'[:, m]:
  gate-only read   R0_m = 3 lam (W.gu)^2 (W.gw)
  bond-1           D1_m = 3 lam [ (W.gw) Q_m(fu) + 2 (W.gu) B_m(fu, fw) ]
                   Q_m(x) = sum_ij W_i W_j rho_ij x_i x_j,  B_m(x,y) = sum_ik W_i W_k rho_ik x_i y_k
  bond-2           D2_m = 3 lam [ (W.gw) Q2_m(b2u)/2 + (W.gu) B2_m(b2u, b2w)
                                  + 2 P_m(b2u; fu, fw) + P_m(b2w; fu, fu) ]
                   Q2/B2 as Q/B with rho^2;  P_m(z; x, y) = sum_i W_i z_i (rho(W∘x))_i (rho(W∘y))_i
Inputs: bank_L{l}.npz (lam, U, Wq, pre_m, pre_v, pre_cov) from estimator_retro.py
RETRO_DUMP_BANK; weights from the sealed store; targets
public_g19_attribution_v1/dk3_truth32M_minus_card_rank{rank}.npz (dk3_{l+1}) and
the 32M truth k3.  Usage: bond_kernel_oracle.py <rank> <bank_dir> [--truth-rho]
Outputs: BOND_KERNEL_rank{rank}[_truthrho].json + bond_kernel_vectors_rank{rank}.npz
"""
import json
import os
import sys
import time
from math import erf
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
G19 = P2 / "public_g19_attribution_v1"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
BATCH = int(os.environ.get("BOND_BATCH", "48"))
_erf = np.vectorize(erf)


def load_weights(rank):
    if rank >= 3016500:
        return np.load(G19 / f"fresh_{rank}_weights.npy", mmap_mode="r")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    return np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")


def stats(vec, target):
    vec = np.asarray(vec, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    vv = float(vec @ vec)
    beta = float(vec @ target) / vv if vv > 0 else 0.0
    corr = (float(vec @ target) / np.sqrt(vv * float(target @ target))
            if vv > 0 and float(target @ target) > 0 else 0.0)
    resid = target - beta * vec
    explained = 1.0 - float(resid @ resid) / float(target @ target)
    return {"beta": round(beta, 4), "corr": round(corr, 4),
            "explained_at_beta": round(explained, 4),
            "explained_at_1": round(1.0 - float((target - vec) @ (target - vec))
                                    / float(target @ target), 4)}


def main():
    rank = int(sys.argv[1])
    bank_dir = Path(sys.argv[2])
    truth_rho = "--truth-rho" in sys.argv
    assert torch.backends.mps.is_available()
    dev = torch.device("mps")
    weights = load_weights(rank)
    dk3 = np.load(G19 / f"dk3_truth32M_minus_card_rank{rank:04d}.npz")
    truth = np.load(G19 / f"mps_truth_rank{rank:04d}_P32M.npz")
    cov_truth = np.load(G19 / f"mps_cov2_rank{rank:04d}.npz") if truth_rho else None
    out = {"tag": "g29-bond-kernel-oracle-v1", "rank": rank,
           "rho_source": "truth 8M pre_cov" if truth_rho else "card's own pre_cov",
           "layers": {}}
    vectors = {}
    t0 = time.time()
    for l in range(1, 15):
        f = bank_dir / f"bank_L{l}.npz"
        if not f.is_file():
            print(f"missing {f}", flush=True)
            continue
        z = np.load(f)
        lam = z["lam"].astype(np.float32)
        U = z["U"].astype(np.float32)          # (1024, r) ungated pushed p
        Wq = z["Wq"].astype(np.float32)        # (1024, r) ungated pushed q
        r_n = int(lam.shape[0])
        if r_n == 0:
            continue
        m_ = z["pre_m"].astype(np.float64)
        v_ = np.maximum(z["pre_v"].astype(np.float64), 1e-30)
        sd = np.sqrt(v_)
        alpha = m_ / sd
        pdf = np.exp(-0.5 * alpha * alpha) / np.sqrt(2 * np.pi)
        cdf = 0.5 * (1 + _erf(alpha / np.sqrt(2)))
        b2 = -alpha * pdf
        if truth_rho:
            C = cov_truth["mc_pre_cov"][l].astype(np.float64)
            sdt = np.sqrt(np.maximum(np.diag(C), 1e-30))
            rho = C / np.outer(sdt, sdt)
        else:
            C = z["pre_cov"].astype(np.float64)
            rho = C / np.outer(sd, sd)
        np.fill_diagonal(rho, 0.0)
        Wn = np.asarray(weights[l + 1], dtype=np.float32)   # (in, out): read at pre-layer l+1
        Wt = torch.from_numpy(Wn).to(dev)                    # (1024_in, 1024_out)
        rho_t = torch.from_numpy(rho.astype(np.float32)).to(dev)
        rho2_t = rho_t * rho_t
        g_t = torch.from_numpy(cdf.astype(np.float32)).to(dev)
        f_t = torch.from_numpy(pdf.astype(np.float32)).to(dev)
        b2_t = torch.from_numpy(b2.astype(np.float32)).to(dev)
        lam_t = torch.from_numpy(lam).to(dev)
        R0 = torch.zeros(1024, dtype=torch.float64, device="cpu")
        D1 = torch.zeros(1024, dtype=torch.float64, device="cpu")
        D2 = torch.zeros(1024, dtype=torch.float64, device="cpu")
        for s0 in range(0, r_n, BATCH):
            s1 = min(s0 + BATCH, r_n)
            u = torch.from_numpy(U[:, s0:s1].T.copy()).to(dev)     # (R, 1024)
            w = torch.from_numpy(Wq[:, s0:s1].T.copy()).to(dev)
            lb = lam_t[s0:s1].to(torch.float32)                      # (R,)
            gu = (g_t[None, :] * u) @ Wt                             # (R, 1024_out) = W'^T (g∘u)
            gw = (g_t[None, :] * w) @ Wt
            R0 += (lb[:, None] * 3.0 * gu * gu * gw).sum(0).cpu().double()
            # bond-1: Q_m(fu) and B_m(fu, fw) via Y = W' ∘ x[:, None] (R, in, out)
            Yfu = Wt[None, :, :] * (f_t[None, :] * u)[:, :, None]
            Yfw = Wt[None, :, :] * (f_t[None, :] * w)[:, :, None]
            RYfu = torch.matmul(rho_t, Yfu)                          # (R, in, out)
            RYfw = torch.matmul(rho_t, Yfw)
            Q = (RYfu * Yfu).sum(1)                                  # (R, out)
            B = (RYfu * Yfw).sum(1)
            D1 += (lb[:, None] * 3.0 * (gw * Q + 2.0 * gu * B)).sum(0).cpu().double()
            # bond-2: double bonds with rho^2 on b2-columns; two-bond paths
            Yb2u = Wt[None, :, :] * (b2_t[None, :] * u)[:, :, None]
            Yb2w = Wt[None, :, :] * (b2_t[None, :] * w)[:, :, None]
            R2Yb2u = torch.matmul(rho2_t, Yb2u)
            Q2 = (R2Yb2u * Yb2u).sum(1)
            B2 = (R2Yb2u * Yb2w).sum(1)
            del R2Yb2u
            P_uu_w = (Yb2u * RYfu * RYfw).sum(1)                     # center on a u-slot: b2u, bonds to fu and fw
            P_w_uu = (Yb2w * RYfu * RYfu).sum(1)                     # center on the w-slot: b2w, bonds to fu, fu
            D2 += (lb[:, None] * 3.0 * (0.5 * gw * Q2 + gu * B2 + 2.0 * P_uu_w
                                        + P_w_uu)).sum(0).cpu().double()
            del Yfu, Yfw, RYfu, RYfw, Yb2u, Yb2w, Q, B, Q2, B2, P_uu_w, P_w_uu
        torch.mps.synchronize()
        R0n, D1n, D2n = R0.numpy(), D1.numpy(), D2.numpy()
        target = dk3[f"dk3_{l + 1}"].astype(np.float64)
        tk3 = truth["mc_pre_k3"][l + 1].astype(np.float64)
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
        rec = {
            "terms": r_n,
            "rms_truth_k3": rms(tk3), "rms_dk3": rms(target), "rms_R0_gateonly_cp_read": rms(R0n),
            "rms_D1": rms(D1n), "rms_D2": rms(D2n),
            "D1_vs_dk3": stats(D1n, target),
            "D1plusD2_vs_dk3": stats(D1n + D2n, target),
            "D2_vs_dk3": stats(D2n, target),
            "D1_vs_R0_corr": round(float(D1n @ R0n) / np.sqrt(float(D1n @ D1n) * float(R0n @ R0n)), 4),
            "wall_s": round(time.time() - t0, 1),
        }
        out["layers"][str(l + 1)] = rec
        vectors[f"R0_{l + 1}"] = R0n
        vectors[f"D1_{l + 1}"] = D1n
        vectors[f"D2_{l + 1}"] = D2n
        vectors[f"dk3_{l + 1}"] = target
        print(json.dumps({"read_layer": l + 1, **{k: rec[k] for k in ("terms", "rms_dk3", "rms_D1", "rms_D2")},
                          "D1": rec["D1_vs_dk3"], "D12": rec["D1plusD2_vs_dk3"], "wall_s": rec["wall_s"]}),
              flush=True)
    suffix = "_truthrho" if truth_rho else ""
    (HERE / f"BOND_KERNEL_rank{rank:04d}{suffix}.json").write_text(json.dumps(out, indent=1))
    np.savez(HERE / f"bond_kernel_vectors_rank{rank:04d}{suffix}.npz", **vectors)
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
