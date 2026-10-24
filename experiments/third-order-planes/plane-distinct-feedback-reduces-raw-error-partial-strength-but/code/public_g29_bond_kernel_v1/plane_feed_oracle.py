#!/usr/bin/env python3
"""TAG g29-plane-feed-oracle-v1: is the card's deep k3 residual the missing
PLANE-TO-DISTINCT feed (first order in the carried pre-plane)?

The engine (run_tensor_recursion_v4.py, `pats`/`feed_factor` loop) adds to the
post-tensor, for every ordered choice (u, w, s) of the index triple, the
first-order Edgeworth response of the DISTINCT entries to the pre-plane
K[u,w] = cum3pre[u,u,w]:  (1/2) K[u,w] * E_conn[phi''_u phi'_w phi_s], expanded
as sum over bond patterns (tuw, tus, tws), tus + tws >= 1, total <= 4:
  cf = 1/2 / (tuw! tus! tws!),  d_{tuw+tus}[u] b_{tuw+tws}[w] a_{tus+tws}[s]
  rho_uw^tuw rho_us^tus rho_ws^tws
with d_t = He_t(-alpha) pdf/sd, b_0 = cdf, b_t = He_{t-1}(-alpha) pdf,
a_0 = sd pdf + m cdf, a_1 = sd cdf, a_q = sd He_{q-2}(-alpha) pdf.
The card has no such channel (its planes are exact pair series into planes;
its star is pushed once into planes/diagonal).  This oracle computes, per
layer, the feed's EFFECT on the next layer's per-unit k3 read restricted to
distinct output entries (u != w, s not in {u, w}); the six orientations give
six equal reads, so  read_m = 3 * sum_{u!=w!=s} W_u W_w W_s K_uw kern(u,w,s).
One-sided patterns (tus*tws = 0) are 2 GEMMs each (deployable class);
two-sided patterns are 1024^4-class (offline only).  Compared with the card's
residual dk3 = truth32M - card at the read layer.
Usage: plane_feed_oracle.py <rank> <bank_dir>   (bank_L{l}.npz + plane_L{l}.npz)
Outputs: PLANE_FEED_rank{rank}.json, plane_feed_vectors_rank{rank}.npz
"""
import json
import os
import sys
import time
from math import erf, factorial
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
G19 = P2 / "public_g19_attribution_v1"
STORE = (P2 / "learned_residual_observer_design/v4_feature_store_v3/"
         "sealed_target_blind_feature_store_v3")
TMAX = int(os.environ.get("FEED_TMAX", "4"))
MBATCH = int(os.environ.get("FEED_MBATCH", "16"))
_erf = np.vectorize(erf)


def load_weights(rank):
    if rank >= 3016500:
        return np.load(G19 / f"fresh_{rank}_weights.npy", mmap_mode="r")
    manifest = json.loads((STORE / "manifest.json").read_text())
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    return np.load((STORE / entry["path"]).parent / "weights.npy", mmap_mode="r")


def columns(m, v, tmax):
    sd = np.sqrt(np.maximum(v, 1e-30))
    al = m / sd
    pdf = np.exp(-0.5 * al * al) / np.sqrt(2 * np.pi)
    cdf = 0.5 * (1 + _erf(al / np.sqrt(2)))
    ma = -al
    He = [np.ones_like(al), ma]
    for k in range(2, tmax + 3):
        He.append(ma * He[-1] - (k - 1) * He[-2])
    d = [He[t] * pdf / sd for t in range(0, tmax + 1)]
    b = [cdf] + [He[t - 1] * pdf for t in range(1, tmax + 1)]
    a = [sd * pdf + m * cdf, sd * cdf] + [sd * He[q - 2] * pdf for q in range(2, tmax + 1)]
    return sd, d, b, a


def patterns(tmax):
    out = []
    for tuw in range(0, tmax + 1):
        for tus in range(0, tmax + 1 - tuw):
            for tws in range(0, tmax + 1 - tuw - tus):
                if tus + tws < 1:
                    continue
                out.append((tuw, tus, tws, 1.0 / (factorial(tuw) * factorial(tus) * factorial(tws))))
    return out


def stats(vec, target):
    vec = np.asarray(vec, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    vv = float(vec @ vec)
    tt = float(target @ target)
    beta = float(vec @ target) / vv if vv > 0 else 0.0
    corr = float(vec @ target) / np.sqrt(vv * tt) if vv > 0 and tt > 0 else 0.0
    return {"beta": round(beta, 4), "corr": round(corr, 4),
            "explained_at_beta": round(1.0 - float((target - beta * vec) @ (target - beta * vec)) / tt, 4),
            "explained_at_1": round(1.0 - float((target - vec) @ (target - vec)) / tt, 4)}


def feed_reads(Wt, K_t, rho_pows, d, b, a, pats, dev):
    """Per-pattern read vectors (1024,) as float64 numpy. Wt: (in, out) f32 on dev."""
    n = Wt.shape[0]
    reads = {}
    W2 = Wt * Wt
    for (tuw, tus, tws, cf) in pats:
        Kt = K_t * rho_pows[tuw] if tuw else K_t                    # (K ∘ rho^tuw), u rows, w cols
        du, dw, ds = tuw + tus, tuw + tws, tus + tws
        dcol, bcol, acol = d[du], b[dw], a[ds]
        if tws == 0:
            # bracket_u = sum_{s != w} W_s a_s rho_us^tus = (rho^tus (W∘a))_u - W_w a_w rho_uw^tus
            S = rho_pows[tus] @ (Wt * acol[:, None])                 # (in, out): S[u, m]
            G = Kt @ (Wt * bcol[:, None])                            # G[u, m] = sum_w K~_uw b_w W_wm
            main = ((Wt * dcol[:, None]) * S * G).sum(0)             # sum_u W_um d_u S_um G_um
            Kx = K_t * rho_pows[tuw + tus]                           # exclusion s = w: K ∘ rho^(tuw+tus)
            excl = ((Wt * dcol[:, None]) * (Kx @ (W2 * (bcol * acol)[:, None]))).sum(0)
            reads[(tuw, tus, tws)] = 3.0 * cf * (main - excl)
        elif tus == 0:
            # bracket_w = (rho^tws (W∘a))_w - W_u a_u rho_uw^tws ; exclusion s = u
            S = rho_pows[tws] @ (Wt * acol[:, None])                 # S[w, m]
            G = Kt.T @ (Wt * dcol[:, None])                          # G[w, m] = sum_u K~_uw d_u W_um
            main = ((Wt * bcol[:, None]) * S * G).sum(0)
            Kx = K_t * rho_pows[tuw + tws]
            excl = ((Wt * bcol[:, None]) * (Kx.T @ (W2 * (dcol * acol)[:, None]))).sum(0)
            reads[(tuw, tus, tws)] = 3.0 * cf * (main - excl)
        else:
            # two-sided: read_m = 3 cf sum_uw (W_u d_u) K~_uw (W_w b_w) [A diag(W∘a) B]_uw,
            # A = rho^tus, B = rho^tws (s = u, w excluded automatically: rho diag = 0)
            A, B = rho_pows[tus], rho_pows[tws]
            vec = torch.zeros(n, dtype=torch.float32, device=dev)
            for m0 in range(0, n, MBATCH):
                m1 = min(m0 + MBATCH, n)
                Wa = (Wt[:, m0:m1] * acol[:, None]).T                # (M, in)
                C = torch.matmul(A[None, :, :] * Wa[:, None, :], B)  # (M, in, in): A diag(W_m∘a) B
                Y = ((Wt[:, m0:m1] * dcol[:, None]).T[:, :, None] * Kt[None, :, :]
                     * (Wt[:, m0:m1] * bcol[:, None]).T[:, None, :])
                vec[m0:m1] = (C * Y).sum((1, 2))
                del C, Y
            reads[(tuw, tus, tws)] = 3.0 * cf * vec
    return {k: v.cpu().double().numpy() for k, v in reads.items()}


def main():
    rank = int(sys.argv[1])
    bank_dir = Path(sys.argv[2])
    assert torch.backends.mps.is_available()
    dev = torch.device("mps")
    weights = load_weights(rank)
    _dk3p = G19 / f"dk3_truth32M_minus_card_rank{rank:04d}.npz"
    _trp = G19 / f"mps_truth_rank{rank:04d}_P32M.npz"
    if _dk3p.is_file() and _trp.is_file():
        dk3 = np.load(_dk3p); truth = np.load(_trp)
    else:   # panel nets without a 32M residual: reads are still saved, stats against zeros (ignored)
        dk3 = {f"dk3_{L}": np.zeros(1024) for L in range(1, 17)}
        truth = {"mc_pre_k3": np.ones((16, 1024))}
    pats = patterns(TMAX)
    out = {"tag": "g29-plane-feed-oracle-v1", "rank": rank, "tmax": TMAX,
           "patterns": [list(p[:3]) for p in pats], "layers": {}}
    vectors = {}
    t0 = time.time()
    for l in range(1, 15):
        fb, fp = bank_dir / f"bank_L{l}.npz", bank_dir / f"plane_L{l}.npz"
        if not (fb.is_file() and fp.is_file()):
            print(f"missing layer {l}", flush=True)
            continue
        z = np.load(fb)
        K = np.load(fp)["pre_k21"].astype(np.float64)
        np.fill_diagonal(K, 0.0)                                     # distinct output only (u != w)
        m_ = z["pre_m"].astype(np.float64)
        v_ = np.maximum(z["pre_v"].astype(np.float64), 1e-30)
        sd, d, b, a = columns(m_, v_, TMAX)
        rho = z["pre_cov"].astype(np.float64) / np.outer(sd, sd)
        np.fill_diagonal(rho, 0.0)
        f32 = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(dev)
        Wt = f32(np.asarray(weights[l + 1]))
        rho_t = f32(rho)
        rho_pows = {0: None, 1: rho_t}
        for t in range(2, TMAX + 1):
            rho_pows[t] = rho_pows[t - 1] * rho_t
        reads = feed_reads(Wt, f32(K), rho_pows, [f32(x) for x in d], [f32(x) for x in b],
                           [f32(x) for x in a], pats, dev)
        torch.mps.synchronize()
        target = dk3[f"dk3_{l + 1}"].astype(np.float64)
        tk3 = truth["mc_pre_k3"][l + 1].astype(np.float64)
        rms = lambda x: float(np.sqrt(np.mean(np.square(x))))
        total = sum(reads.values())
        one_sided = sum(v for k, v in reads.items() if k[1] * k[2] == 0)
        bond2 = sum(v for k, v in reads.items() if sum(k) <= 2)
        lead = reads[(0, 1, 0)] + reads[(0, 0, 1)]
        rec = {"rms_truth_k3": rms(tk3), "rms_dk3": rms(target), "rms_K21pre": rms(K),
               "rms_feed_total": rms(total), "rms_feed_one_sided": rms(one_sided),
               "rms_feed_bond2": rms(bond2), "rms_feed_leading": rms(lead),
               "total_vs_dk3": stats(total, target), "one_sided_vs_dk3": stats(one_sided, target),
               "bond2_vs_dk3": stats(bond2, target), "leading_vs_dk3": stats(lead, target),
               "per_pattern_rms": {str(k): rms(v) for k, v in reads.items()},
               "wall_s": round(time.time() - t0, 1)}
        out["layers"][str(l + 1)] = rec
        vectors[f"feed_total_{l + 1}"] = total
        vectors[f"feed_one_sided_{l + 1}"] = one_sided
        vectors[f"feed_bond2_{l + 1}"] = bond2
        vectors[f"feed_leading_{l + 1}"] = lead
        vectors[f"dk3_{l + 1}"] = target
        print(json.dumps({"read_layer": l + 1, "rms_dk3": rec["rms_dk3"], "rms_feed_total": rec["rms_feed_total"],
                          "rms_one_sided": rec["rms_feed_one_sided"], "total": rec["total_vs_dk3"],
                          "one_sided": rec["one_sided_vs_dk3"], "wall_s": rec["wall_s"]}), flush=True)
    (HERE / f"PLANE_FEED_rank{rank:04d}.json").write_text(json.dumps(out, indent=1))
    np.savez(HERE / f"plane_feed_vectors_rank{rank:04d}.npz", **vectors)
    print(json.dumps({"done": True, "wall_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
