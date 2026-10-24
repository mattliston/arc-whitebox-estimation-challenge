#!/usr/bin/env python3
"""TAG tracker-T1: dictionary-factored depth transport, L1 -> L3.

The representation: per unit c at layer l the quadratic state is
  M_c = sum_d Dic_d^T diag(w_c^{(d)}) Dic_d
over a GROWING family of shared dictionaries (layer 0: ahat rows;
each subsequent layer adds its own linear-coefficient vectors as a
new dictionary). The engine's post-quadratic B = c1*M + c2*aa^T (+
dropped word spillover, measured) keeps this family exactly closed
under W-mixing: diag weights mix linearly; the new aa^T terms join
the next dictionary. Linear state a_c stays dense-exact. All pair
Grams between dictionaries are GEMM-class.
Cell T1: transport L1->L2->L3 on rank35, measure the pair-field
supply (C and the K21-relevant Grams) vs banked truth at each level;
grade the difference-first supply vs the banked R-fields.
Usage: run_tracker_t1.py <rank>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import hermite_relu_coefs, load_weights
from run_chaos_supply_shallow import unit_scalars_and_var
from run_vresid_field import engine_post_cov

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
n = 1024


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    v0 = tr["mc_pre_v"][0].astype(np.float64)
    W0 = np.asarray(W[0], dtype=np.float64)
    if np.corrcoef(np.sum(W0 ** 2, 0), v0)[0, 1] > \
            np.corrcoef(np.sum(W0 ** 2, 1), v0)[0, 1]:
        W0 = W0.T
    sig = np.linalg.norm(W0, axis=1)
    ahat = W0 / sig[:, None]
    al = hermite_relu_coefs(12)
    # ---- L1 pre-state (exact): dictionaries D0 = ahat
    W1 = np.asarray(W[1], dtype=np.float64).T
    Wsig = W1 * sig[None, :]
    m1 = al[0] * Wsig.sum(1)
    A1 = al[1] * (Wsig @ ahat)                # dense linear state
    Wq1 = np.sqrt(2.0) * al[2] * Wsig         # weights on D0
    dicts = [ahat]                            # list of (n x 1024)
    wts = [Wq1]                               # per-unit diag weights
    # exact-variance completion per layer
    rho00 = ahat @ ahat.T
    Cz = np.zeros((n, n))
    rp = np.ones_like(rho00)
    for k in range(1, 13):
        rp = rp * rho00
        Cz += (al[k] ** 2) * (Wsig @ rp @ Wsig.T)
    out = {}
    A, mv, Cexact = A1, m1, Cz
    for l in [1, 2]:
        # ---- measure the carried pair field at this level
        # C_model = A A^T + 0.5 <M_a, M_b> + vres-diag correction
        G_dd = [[d1 @ d2.T for d2 in dicts] for d1 in dicts]
        MM = np.zeros((n, n))
        for i, wi in enumerate(wts):
            for j, wj in enumerate(wts):
                Gij = G_dd[i][j]
                MM += (wi @ (Gij * Gij)) @ wj.T \
                    if i <= j else 0.0
        MM = MM + MM.T - np.diag(np.diag(MM))
        # careful diagonal double count: recompute exactly
        MM = np.zeros((n, n))
        for i, wi in enumerate(wts):
            for j, wj in enumerate(wts):
                Gij = G_dd[i][j]
                MM += (wi @ (Gij * Gij)) @ wj.T
        C_model = A @ A.T + 0.5 * MM
        vres = np.maximum(np.diag(Cexact) - np.diag(C_model), 0.0)
        np.fill_diagonal(C_model, np.diag(Cexact))
        tC = tr["mc_pre_cov"][l].astype(np.float64)
        rec = {"C_rel": round(float(np.linalg.norm(C_model - tC)
                                    / np.linalg.norm(tC)), 5),
               "Cexact_rel": round(float(np.linalg.norm(Cexact - tC)
                                         / np.linalg.norm(tC)), 5),
               "ndicts": len(dicts)}
        out[l] = rec
        print(RANK, f"L{l}", json.dumps(rec), flush=True)
        if l == 2:
            break
        # ---- engines per unit: scalars on the (topK+tail) law
        S = np.zeros((n, 4))
        E1 = np.zeros(n)
        for c in range(n):
            # eigen through the dictionary form (single dict at l=1)
            Mc = (dicts[0].T * wts[0][c][None, :]) @ dicts[0]
            lam, U = np.linalg.eigh(Mc)
            at = U.T @ A[c]
            vr = max(float(Cexact[c, c]) - float(A[c] @ A[c])
                     - 0.5 * float((lam ** 2).sum()), 0.0)
            at = np.concatenate([at, [np.sqrt(vr)]])
            lam = np.concatenate([lam, [0.0]])
            e1, vd, pz, p0, p1, p2 = unit_scalars_and_var(
                float(mv[c]), lam, at)
            E1[c] = e1
            S[c] = (pz, p0, p1, p2)
            if (c + 1) % 256 == 0:
                print(f"  engines {c + 1}/{n}", flush=True)
        # ---- post objects in factored form:
        # alpha_c = pz*a + p0*Ma + ... (Stein words, all factored)
        AhA = dicts[0] @ A.T
        Ma = ((wts[0].T * AhA).T @ dicts[0])
        AhMa = dicts[0] @ Ma.T
        M2a = ((wts[0].T * AhMa).T @ dicts[0])
        alpha = (S[:, 0:1] * A + S[:, 1:2] * Ma - S[:, 2:3] * M2a)
        # B_c = pz*M + p0*aa^T  (leading words; spillover dropped v1)
        # ---- exact linear mixing to the next layer
        Wn = np.asarray(W[l + 1], dtype=np.float64)
        mv = Wn.T @ E1
        A = Wn.T @ alpha
        # M-part: new weights on D0: w' = Wn^T (pz*w); new dictionary
        # D1 = the previous layer's a-vectors with weights p0-mixed
        w0new = Wn.T @ (S[:, 0:1] * wts[0])
        # aa^T-terms: sum_i Wn_ic p0_i a_i a_i^T -> dictionary of a_i
        # rows with per-unit weights Wn^T diag-mixed:
        anorm = np.linalg.norm(
            np.asarray(A1 if l == 1 else A), axis=1) + 1e-30
        # dictionary must be UNIT rows for the weight algebra:
        Dnew = (A1 if l == 1 else A) / anorm[:, None]
        wnew = Wn.T @ (S[:, 1:2] * (anorm ** 2)[:, None]
                       * np.eye(n)) if False else \
            (Wn.T * (S[:, 1] * anorm ** 2)[None, :])
        dicts = [dicts[0], Dnew]
        wts = [w0new, wnew]
        # exact next-C for the completion (arccos-free): push of the
        # engine-level pair model is NOT available exactly; use the
        # banked truth C as the vres anchor (instrument-grade for T1;
        # the deployable version anchors on the card's carried C per
        # the delta law):
        Cexact = tr["mc_pre_cov"][l + 1].astype(np.float64)
    (HERE / f"TRACKER_T1_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
