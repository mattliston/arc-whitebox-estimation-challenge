#!/usr/bin/env python3
"""TAG k3push-hermite: Hermite 3-point completion of the card's k3 push.

The card's one-step k3-push errs 6-11% (MAP_GAP_HOST); its emission
uses diag k3post + K21 pair planes only. This test computes the next
layer's k3 from the FULL Gaussian 3-point function of the post
activations: E[ph_a ph_b ph_c] = sum_{pqr} al_p al_q al_r
C(pqr; rho_ab, rho_ac, rho_bc), with per-unit Hermite coefficients
al_k(c) from the unit's own (m, v) marginal (Gaussian-level; GC
corrections later), rho from the CARD'S OWN carried covariance.
k3_next_c = sum_abk w_a w_b w_k E3~_abk (centered) with w = the push
column of unit c. All (s1,s2,s3) term classes with a zero factorize
into all-units GEMMs; the single (1,1,1) triangle class (from
p=q=r=2) is computed exactly per probe unit (one GEMM each).
Grades vs truth mc_pre_k3[l+1] against the card's carried error.
Inputs are all banked (host_m/v, host cov, mps truth, weights).
Usage: run_k3push_hermite.py <rank> [layers...]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import load_weights

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
LAYERS = [int(a) for a in sys.argv[2:]] or [2, 5, 8, 11, 13]
KH = 6            # Hermite order kept per unit
NPROBE = 96       # probe outputs for the exact triangle term


def unit_hermite(m, v):
    """alpha_k(c) for relu under N(m_c, v_c), k=0..KH; vectorized."""
    from math import factorial, sqrt
    s = np.linspace(-10, 10, 20001)
    w = np.exp(-s * s / 2)
    w /= w.sum()
    He = [np.ones_like(s), s]
    for k in range(2, KH + 1):
        He.append(s * He[-1] - (k - 1) * He[-2])
    sd = np.sqrt(np.maximum(v, 1e-30))
    z = m[:, None] + sd[:, None] * s[None, :]      # n x S
    ph = np.maximum(z, 0.0)
    al = np.stack([(ph * (w * He[k])[None, :]).sum(1)
                   / sqrt(factorial(k)) for k in range(KH + 1)])
    return al                                       # (KH+1) x n


def triple_coef(p, q, r):
    """(s1,s2,s3) legs and coefficient for E[He_p He_q He_r]."""
    from math import factorial
    s3 = (p + q - r)
    s2 = (p + r - q)
    s1 = (q + r - p)
    if s3 < 0 or s2 < 0 or s1 < 0 or s3 % 2 or s2 % 2 or s1 % 2:
        return None
    s3, s2, s1 = s3 // 2, s2 // 2, s1 // 2
    coef = (factorial(p) * factorial(q) * factorial(r)
            / (factorial(s1) * factorial(s2) * factorial(s3)))
    return s1, s2, s3, coef


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    from math import factorial, sqrt
    out = {}
    for l in LAYERS:
        i = l - 1                       # host arrays: index i <-> layer l
        m_ = st["host_m"][i].astype(np.float64)
        v_ = st["host_v"][i].astype(np.float64)
        C = hc["cov"][i].astype(np.float64)
        sd = np.sqrt(np.maximum(np.diag(C), 1e-30))
        rho = C / np.outer(sd, sd)
        np.fill_diagonal(rho, 1.0)
        al = unit_hermite(m_, np.diag(C))
        # normalized coefficient includes sqrt(k!) basis: covariance of
        # ph uses sum al_k(a) al_k(b) rho^k  (orthonormal He basis)
        tk3 = tr["mc_pre_k3"][l + 1].astype(np.float64)
        hk3 = st["host_k3"][l].astype(np.float64)
        # pick push matrix empirically: try W[l], W[l+1], both orient
        tv = tr["mc_pre_v"][l + 1].astype(np.float64)
        var_post = np.zeros(1024)
        for k in range(1, KH + 1):
            var_post += al[k] ** 2
        best = None
        for wi in (l, l + 1):
            if wi >= W.shape[0]:
                continue
            Wm = np.asarray(W[wi], dtype=np.float64)
            for tag, Mx in (("T", Wm.T), ("N", Wm)):
                pred = (Mx ** 2).T @ var_post
                c = np.corrcoef(pred, tv)[0, 1]
                if best is None or c > best[0]:
                    best = (c, wi, tag, Mx)
        vcorr, wi, tag, Mx = best
        Wp = Mx                        # z_{l+1} = Wp^T ph(z_l)
        # ---- 3-point sum, factorized classes
        # centered E3~ = E3 - parts already in mean/cov push; k3 truth
        # is the third CUMULANT of z_next: k3_c = sum www kappa3(ph)
        # kappa3(ph_a,ph_b,ph_c) = sum_{pqr>=1} al al al C(...) with
        # at least ... exactly: joint 3rd cumulant of a Gaussian
        # functional triple = sum over CONNECTED Hermite pairings =
        # all (p,q,r>=1) terms with s1,s2,s3 not forming a
        # disconnected split; for three variables connectivity fails
        # only when two of the s's are zero (then the third variable
        # factors). Terms with exactly one s=0 remain connected.
        # Build classes:
        classes = {}
        for p in range(1, KH + 1):
            for q in range(1, KH + 1):
                for r in range(1, KH + 1):
                    if p + q + r > 6:
                        continue
                    tc = triple_coef(p, q, r)
                    if tc is None:
                        continue
                    s1, s2, s3, coef = tc
                    nz = (s1 > 0) + (s2 > 0) + (s3 > 0)
                    if nz <= 1:
                        continue        # disconnected (or degenerate)
                    classes.setdefault((s1, s2, s3), []).append(
                        (p, q, r, coef))
        # rho Hadamard powers cache
        maxs = max(max(k) for k in classes)
        rp = {0: np.ones_like(rho), 1: rho}
        for k in range(2, maxs + 1):
            rp[k] = rp[k - 1] * rho
        # normalized alpha with 1/sqrt(k!) already in al; the triple
        # formula for orthonormal He: E[He_p/sp He_q/sq He_r/sr] =
        # coef/(sp sq sr) rho-legs with sp=sqrt(p!) -> al_p already
        # divided; multiply coef/(sqrt(p!q!r!)) accordingly:
        k3_fact = np.zeros(1024)        # factorized classes, all units
        tri_terms = []                  # (1,1,1) member list
        for (s1, s2, s3), members in classes.items():
            # weight matrix legs: ab leg rho^s3, ak leg rho^s2, bk leg rho^s1
            A3 = np.zeros((1024, 1024))
            for (p, q, r, coef) in members:
                w_abc = coef / (sqrt(factorial(p)) * sqrt(factorial(q))
                                * sqrt(factorial(r)))
                A3 += w_abc * np.outer(al[p], al[q])
            if min(s1, s2, s3) >= 1:
                tri_terms.append(((s1, s2, s3), members))
                continue
            # exactly one s is zero -> factorizes.
            if s3 == 0:
                # no ab leg: E3 term = [al_p al_q] rho^s2_ak rho^s1_bk
                # sum_ab w_a w_b -> u = Wp^T-weighted:
                # k3_c += sum_k w_k (rho^s2^T (w*alp))_k-ish; build:
                # t_c = sum_abk w_a w_b w_k alp_a alq_b alr_k
                #        rho^s2_ak rho^s1_bk
                # = sum_k w_k alr_k * (rho^s2^T (w∘alp))_k
                #                   * (rho^s1^T (w∘alq))_k
                for (p, q, r, coef) in members:
                    w_abc = coef / (sqrt(factorial(p))
                                    * sqrt(factorial(q))
                                    * sqrt(factorial(r)))
                    P2 = rp[s2].T @ (Wp * al[p][:, None])   # n x c
                    P1 = rp[s1].T @ (Wp * al[q][:, None])
                    k3_fact += w_abc * ((Wp * al[r][:, None])
                                        * P2 * P1).sum(0)
            elif s2 == 0:
                for (p, q, r, coef) in members:
                    w_abc = coef / (sqrt(factorial(p))
                                    * sqrt(factorial(q))
                                    * sqrt(factorial(r)))
                    P3 = rp[s3].T @ (Wp * al[p][:, None])   # over a
                    P1 = rp[s1].T @ (Wp * al[r][:, None])   # over k
                    k3_fact += w_abc * ((Wp * al[q][:, None])
                                        * P3 * P1).sum(0)
            else:  # s1 == 0
                for (p, q, r, coef) in members:
                    w_abc = coef / (sqrt(factorial(p))
                                    * sqrt(factorial(q))
                                    * sqrt(factorial(r)))
                    P3 = rp[s3].T @ (Wp * al[q][:, None])   # over b
                    P2 = rp[s2].T @ (Wp * al[r][:, None])   # over k
                    k3_fact += w_abc * ((Wp * al[p][:, None])
                                        * P3 * P2).sum(0)
        # ---- triangle classes, exact on probe units
        probes = np.linspace(0, 1023, NPROBE).astype(int)
        k3_tri = np.zeros(NPROBE)
        for (s123, members) in tri_terms:
            s1, s2, s3 = s123
            for (p, q, r, coef) in members:
                w_abc = coef / (sqrt(factorial(p)) * sqrt(factorial(q))
                                * sqrt(factorial(r)))
                for jj, c in enumerate(probes):
                    w = Wp[:, c]
                    # t = sum_ab (w alp)_a (w alq)_b rho^s3_ab *
                    #     [rho^s2 diag(w alr) rho^s1]_ab
                    K = (rp[s2] * 1.0) @ ((al[r] * w)[:, None]
                                          * rp[s1])
                    t = float(((al[p] * w)[:, None] * rp[s3]
                               * (al[q] * w)[None, :] * K).sum())
                    k3_tri[jj] += w_abc * t
        pred_all = k3_fact.copy()
        pred_probe = k3_fact[probes] + k3_tri
        s3t = max(np.sqrt(np.mean(tk3 ** 2)), 1e-30)
        rec = {
            "push_pick": f"W[{wi}]{tag} vcorr {round(vcorr, 5)}",
            "card_k3_rel": round(float(np.sqrt(np.mean(
                (hk3 - tk3) ** 2)) / s3t), 4),
            "hermite_k3_rel_nontri": round(float(np.sqrt(np.mean(
                (pred_all - tk3) ** 2)) / s3t), 4),
            "hermite_k3_rel_probe_withtri": round(float(np.sqrt(np.mean(
                (pred_probe - tk3[probes]) ** 2))
                / max(np.sqrt(np.mean(tk3[probes] ** 2)), 1e-30)), 4),
            "card_k3_rel_probe": round(float(np.sqrt(np.mean(
                (hk3[probes] - tk3[probes]) ** 2))
                / max(np.sqrt(np.mean(tk3[probes] ** 2)), 1e-30)), 4),
        }
        out[l] = rec
        print(RANK, f"L{l}->L{l + 1}", json.dumps(rec), flush=True)
    (HERE / f"K3PUSH_HERMITE_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
