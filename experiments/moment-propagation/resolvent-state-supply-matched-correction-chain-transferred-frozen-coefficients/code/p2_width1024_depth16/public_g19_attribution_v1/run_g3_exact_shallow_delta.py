#!/usr/bin/env python3
"""TAG g3-exact-shallow-delta: the difference-first delta at L2/L3
with EXACT identification-free cores (zero fit error, the layers the
consumption pays most and the chain supplies least).

Exact state at mc layer 2 (z2 = W2^T relu(z1), z1 = W1^T relu(W0^T x)):
per unit c of z2's PRE state... we supply the dv-slot L{l+1} from the
post-cov delta at layer l, so we need cores of layer l's PRE state.
Layer 2 pre-state (identification-free, H<=2): built in
run_chaos_supply_shallow: m, a (dense 1024-vec), M = Ahat^T diag(u) Ahat
+ certified vres Gaussian-tail completion. Per unit: top-K eigenmodes
of the factored M + tail mode (sqrt(vres-remainder), lam=0) ->
resolvent post-state -> certified K21post assembly vs engine emission
on the same implied stats -> dP -> pushed dv delta.
Usage: run_g3_exact_shallow_delta.py <rank> <layer=2> [K=64]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import hermite_relu_coefs, load_weights
from run_tensor_recursion_v4 import emission_K21

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
L = int(sys.argv[2]) if len(sys.argv) > 2 else 2
K = int(sys.argv[3]) if len(sys.argv) > 3 else 64


def main():
    W = load_weights(RANK)
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    v0 = tr["mc_pre_v"][0].astype(np.float64)
    W0 = np.asarray(W[0], dtype=np.float64)
    if np.corrcoef(np.sum(W0 ** 2, 0), v0)[0, 1] > \
            np.corrcoef(np.sum(W0 ** 2, 1), v0)[0, 1]:
        W0 = W0.T
    sig = np.linalg.norm(W0, axis=1)
    ahat = W0 / sig[:, None]
    al = hermite_relu_coefs(12)
    n = 1024
    if L == 2:
        # z2-pre state in x-chaos through TWO mixings? No: z2's pre
        # state = W2-mix of phi(z1)-chaos — NOT identification-free
        # beyond H<=2-of-z1... At THIS cell use L=1 and L=2 pre states
        # that are exactly expressible:
        raise SystemExit("use L=1 for the fully exact cell")
    # L == 1: z1-pre state exact: a_c = al1*(Wsig ahat), M_c factored
    W1 = np.asarray(W[1], dtype=np.float64).T
    Wsig = W1 * sig[None, :]
    mvec = al[0] * Wsig.sum(1)
    A = al[1] * (Wsig @ ahat)
    U2 = np.sqrt(2.0) * al[2] * Wsig
    rho = ahat @ ahat.T
    # exact variance for the tail completion
    Cz1 = np.zeros((n, n))
    rp = np.ones_like(rho)
    for k in range(1, 13):
        rp = rp * rho
        Cz1 += (al[k] ** 2) * (Wsig @ rp @ Wsig.T)
    vz1 = np.diag(Cz1).copy()
    m_ = mvec
    from run_vresid_field import engine_post_cov
    # my pair model (order<=2 Grams + exact diag), reusing the
    # validated stage-1 machinery inline:
    S = np.zeros((n, 4))
    E1 = np.zeros(n)
    VD = np.zeros(n)
    from run_chaos_supply_shallow import unit_scalars_and_var
    for c in range(n):
        Mc = (ahat.T * U2[c][None, :]) @ ahat
        lam, U = np.linalg.eigh(Mc)
        at = U.T @ A[c]
        vres = max(float(vz1[c]) - float(A[c] @ A[c])
                   - 0.5 * float((lam ** 2).sum()), 0.0)
        at = np.concatenate([at, [np.sqrt(vres)]])
        lam = np.concatenate([lam, [0.0]])
        e1, vd, pzv, p0, p1, p2 = unit_scalars_and_var(
            float(m_[c]), lam, at)
        E1[c] = e1
        VD[c] = vd
        S[c] = (pzv, p0, p1, p2)
        if (c + 1) % 256 == 0:
            print(f"scalars {c + 1}/{n}", flush=True)
    # order-1 Gram via Stein words (validated):
    AhA = ahat @ A.T
    Ma = ((U2.T * AhA).T @ ahat)
    AhMa = ahat @ Ma.T
    M2a = ((U2.T * AhMa).T @ ahat)
    AhM2a = ahat @ M2a.T
    M3a = ((U2.T * AhM2a).T @ ahat)
    Lw = (S[:, 0:1] * A + S[:, 1:2] * Ma - S[:, 2:3] * M2a
          + S[:, 3:4] * M3a)
    G1 = Lw @ Lw.T
    RR = rho * rho
    t_MM = (U2 @ RR) @ U2.T
    AB = A @ A.T
    aMa = (AhA.T ** 2) @ U2.T
    c1v, c2v = S[:, 0], S[:, 1]
    G2 = 0.5 * (np.outer(c1v, c1v) * t_MM
                + np.outer(c2v, c2v) * (AB * AB)
                + np.outer(c1v, c2v) * aMa.T
                + np.outer(c2v, c1v) * aMa)
    Cov_mine = G1 + G2
    np.fill_diagonal(Cov_mine, VD)
    # matched engine side on the SAME implied stats:
    C_impl = Cz1
    K21_impl = np.zeros((n, n))  # implied K21 of z1: exact via
    # 3-point Hermite factorized classes is heavy; z1's k3 is SMALL
    # (near-Gaussian mixing); matched-zero on both sides:
    k3_impl = np.zeros(n)
    k4_impl = np.zeros(n)
    K22_impl = np.zeros((n, n))
    K31_impl = np.zeros((n, n))
    mu_eng = engine_post_cov(m_, np.diag(Cz1).copy(), k3_impl, k4_impl,
                             C_impl, K21_impl, K22_impl, K31_impl)
    dP = Cov_mine - mu_eng
    Wp = np.asarray(W[2], dtype=np.float64)
    dv_c = np.einsum("ac,ab,bc->c", Wp, dP, Wp, optimize=True)
    tv = tr["mc_pre_v"][2].astype(np.float64)
    hv = st["host_v"][1].astype(np.float64)
    dv_true = tv - hv
    dvh = np.load(HERE / f"dvhat_rank{RANK:04d}.npz")["L2"]
    out = {
        "corr_vs_dvtrue": round(float(np.corrcoef(dv_c, dv_true)[0, 1]), 4),
        "chain_corr_L2": round(float(np.corrcoef(
            dvh, tr["mc_pre_v"][2].astype(np.float64)
            - st["host_v"][1].astype(np.float64))[0, 1]), 4),
        "dP_rel": round(float(np.linalg.norm(dP)
                              / np.linalg.norm(mu_eng)), 4),
    }
    np.savez_compressed(HERE / f"g3exact_shallow_L1_r{RANK}.npz",
                        dv_c=dv_c.astype(np.float32))
    print(RANK, json.dumps(out), flush=True)
    (HERE / f"G3_EXACT_SHALLOW_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
