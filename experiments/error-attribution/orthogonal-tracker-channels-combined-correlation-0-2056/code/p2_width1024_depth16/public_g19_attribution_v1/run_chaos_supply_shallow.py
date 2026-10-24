#!/usr/bin/env python3
"""TAG chaos-supply-shallow: deployable analytic Δv at shallow layers.

Stage 1: v(L2) from the identification-free chaos frame.
z1 = W1^T relu(z0), z0 = W0^T x Gaussian. Per unit c the H<=2 state
of z1 in x-chaos is exactly (m_c, a_c, M_c) (factored; ReLU odd He
coefficients vanish so there is no cubic sector; Q4 dropped at this
stage). Covpost(z1)_ab = sum_k <coef_k(phi_a), coef_k(phi_b)>:
  order-1: E[x phi] = sum_j M^j a E[phi^(j+1)]   (Stein words, j<=3)
  order-2: E[(xx^T - I)phi] via iterated Stein (words M, aa^T,
           sym(a (Ma)^T), M^2-cross), scalars E[phi^(k)] from the
           certified per-unit CF machinery (P(z>0), p(0), p'(0)).
  diagonal: EXACT per-unit E[phi^2]-E[phi]^2 by 1D quadrature on the
           unit's own quadratic-chaos law (density via corrected CF).
v(L2) = diag(W2^T Covpost W2), graded vs banked truth (shared-draw,
1e-5) and vs the card. Usage: run_chaos_supply_shallow.py <rank>
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_chaos1024_shallow import hermite_relu_coefs, load_weights
from run_engine1024_probe import log_cf, emean_direct, unit_state

RANK = int(sys.argv[1]) if len(sys.argv) > 1 else 35
NT_D, TMAX_D = 4096, 100.0
NX_D, XW_D = 2048, 13.0


def unit_scalars_and_var(m, lam, at):
    """CF-only per-unit scalars (no density; all integrals certified):
    E[phi], Var(phi), P(z>0), p(0), p'(0), p''(0)."""
    ts = np.concatenate([np.linspace(1e-6, 1.0, 4000, endpoint=False),
                         np.linspace(1.0, 300.0, 24000)])
    den = 1.0 - 1j * ts[:, None] * lam[None, :]
    a2 = at[None, :] ** 2
    lcf = (1j * ts * m - 1j * ts * lam.sum() / 2
           - 0.5 * ts ** 2 * (a2 / den).sum(1)
           - 0.5 * np.log(den).sum(1))
    cf = np.exp(lcf)
    d1 = (1j * m - 1j * lam.sum() / 2 - ts * (a2 / den).sum(1)
          - 0.5j * ts ** 2 * (a2 * lam[None, :] / den ** 2).sum(1)
          + 0.5j * (lam[None, :] / den).sum(1))
    d2 = (-(a2 / den).sum(1)
          - 2j * ts * (a2 * lam[None, :] / den ** 2).sum(1)
          + ts ** 2 * (a2 * lam[None, :] ** 2 / den ** 3).sum(1)
          - 0.5 * (lam[None, :] ** 2 / den ** 2).sum(1))
    cf2 = cf * (d1 * d1 + d2)
    Ez2 = m * m + float((at ** 2).sum()) + 0.5 * float((lam ** 2).sum())
    e1 = 0.5 * (m + (2 / np.pi) * (np.trapz((1 - cf.real) / ts ** 2, ts)
                                   + 1.0 / 300.0))
    e2 = 0.5 * (Ez2 - (2 / np.pi) * np.trapz(cf2.imag / ts, ts))
    pz = 0.5 + (1 / np.pi) * np.trapz(cf.imag / ts, ts)
    p0 = (1 / np.pi) * np.trapz(cf.real, ts)
    p1 = (1 / np.pi) * np.trapz(ts * cf.imag, ts)
    p2 = -(1 / np.pi) * np.trapz(ts * ts * cf.real, ts)
    return float(e1), float(e2 - e1 * e1), float(pz), float(p0), \
        float(p1), float(p2)


def main():
    W = load_weights(RANK)
    tr = np.load(HERE / f"mps_cov2_rank{RANK:04d}.npz")
    st = np.load(HERE / f"states_rank{RANK:04d}.npz")
    v0 = tr["mc_pre_v"][0].astype(np.float64)
    W0 = np.asarray(W[0], dtype=np.float64)
    if np.corrcoef(np.sum(W0 ** 2, 0), v0)[0, 1] > \
            np.corrcoef(np.sum(W0 ** 2, 1), v0)[0, 1]:
        W0 = W0.T
    W1 = np.asarray(W[1], dtype=np.float64).T   # rows = units of z1
    W2 = np.asarray(W[2], dtype=np.float64)     # push z1-post -> z2
    sig = np.linalg.norm(W0, axis=1)
    ahat = W0 / sig[:, None]
    al = hermite_relu_coefs(12)
    # factored H<=2 state of every z1 unit:
    Wsig = W1 * sig[None, :]
    mvec = al[0] * Wsig.sum(1)
    A = al[1] * (Wsig @ ahat)                   # a_c row-vectors
    U2 = np.sqrt(2.0) * al[2] * Wsig            # M_c = Ahat^T diag(u) Ahat
    rho = ahat @ ahat.T
    n = 1024
    # EXACT z1 pair covariance (arccos family, all orders k<=12)
    Cz1 = np.zeros((n, n))
    rp = np.ones_like(rho)
    for k in range(1, 13):
        rp = rp * rho
        Cz1 += (al[k] ** 2) * ((Wsig @ rp) * Wsig).sum(1)[:, None] * 0
        Cz1 += (al[k] ** 2) * (Wsig @ rp @ Wsig.T)
    vz1_exact = np.diag(Cz1).copy()
    # per-unit eigen data + scalars (exact, certified machinery)
    E1 = np.zeros(n)
    VD = np.zeros(n)
    S = np.zeros((n, 4))                        # E[phi'],p0,p'0,p''0
    for c in range(n):
        Mc_diagform = U2[c]
        # eigen via the factored form: M = Ahat^T diag(u) Ahat
        Mc = (ahat.T * Mc_diagform[None, :]) @ ahat
        lam, Uc = np.linalg.eigh(Mc)
        at = Uc.T @ A[c]
        # exact-variance completion: H>=4 sector of z1 as an
        # independent Gaussian remainder (CLT over many tiny modes)
        vres = max(float(vz1_exact[c]) - float(A[c] @ A[c])
                   - 0.5 * float((lam ** 2).sum()), 0.0)
        at = np.concatenate([at, [np.sqrt(vres)]])
        lam = np.concatenate([lam, [0.0]])
        e1, vd, pz, p0, p1, p2 = unit_scalars_and_var(
            mvec[c], lam, at)
        E1[c] = e1
        VD[c] = vd
        S[c] = (pz, p0, p1, p2)
        if (c + 1) % 128 == 0:
            print(f"unit scalars {c + 1}/1024", flush=True)
    # order-1 Gram: L_c = sum_j M^j a * E[phi^(j+1)], j<=3
    # E[phi^(j+1)]: j=0 -> P(z>0); j=1 -> p(0); j=2 -> -p'(0);
    # j=3 -> p''(0)  (phi'' = delta_0; each further derivative
    # alternates sign via integration by parts)
    # build words in FACTORED form: a, Ma, M2a per unit
    # Ma_c = Ahat^T diag(u_c) Ahat a_c; do it batched:
    AhA = ahat @ A.T                            # s-products: (i, c)
    Ma = ((U2.T * AhA) .T @ ahat)               # c x 1024: rows M_c a_c
    AhMa = ahat @ Ma.T
    M2a = ((U2.T * AhMa).T @ ahat)
    AhM2a = ahat @ M2a.T
    M3a = ((U2.T * AhM2a).T @ ahat)
    Lw = (S[:, 0:1] * A + S[:, 1:2] * Ma - S[:, 2:3] * M2a
          + S[:, 3:4] * M3a)                    # c x 1024
    G1 = Lw @ Lw.T
    # order-2 Gram: B_c ~ E[phi']M + E[phi'']aa^T
    #   + (-E[phi'''}) sym(a (Ma)^T) + E[phi''']-class M2 cross...
    # keep the two dominant words (validated basis {M, aa^T} carries
    # ~99% at perturbative scale) + sym(a,Ma):
    # <B_a, B_b> expands into trace products, all GEMM-computable:
    # <M_a, M_b> = u_a^T (rho*rho) u_b
    # <M_a, x y^T> = x^T M_a y (factored)
    # <x x^T, y y^T> = (x.y)^2
    RR = rho * rho
    t_MM = (U2 @ RR) @ U2.T                     # <M_a, M_b>
    AB = A @ A.T                                # a_a . a_b
    t_aaa = AB * AB
    # x^T M_b y with x=a_a,y=a_a: a_a^T M_b a_a:
    aMa = (AhA.T ** 2) @ U2.T                   # [c1, c2]: a_c1^T M_c2 a_c1
    c1v = S[:, 0]
    c2v = S[:, 1]
    c3v = -S[:, 2]
    B_MM = np.outer(c1v, c1v) * t_MM
    B_aa = np.outer(c2v, c2v) * t_aaa
    B_cross = np.outer(c1v, c2v) * aMa.T + np.outer(c2v, c1v) * aMa
    # sym(a,(Ma)) word ~ c3: <sym(a Ma^T)_a, ...> small; drop stage 1
    G2 = 0.5 * (B_MM + B_aa + B_cross)
    # >=3-sector Gram: copula-shaped tail pinned to the exact per-unit
    # residual: T_ab = sum_{k>=3} bt_k(a) bt_k(b) rt^k with bt from
    # the Gaussian-marginal Hermite coefficients rescaled so that
    # sum_{k>=3} bt_k^2 = VD - diag(G1+G2)  (exact diag by design).
    from run_k3push_hermite import unit_hermite
    alh = unit_hermite(mvec, vz1_exact)          # (KH+1) x n, KH=6
    tail2 = np.maximum(VD - np.diag(G1 + G2), 0.0)
    base2 = np.maximum((alh[3:] ** 2).sum(0), 1e-30)
    scl = np.sqrt(tail2 / base2)
    rt = Cz1 / np.sqrt(np.outer(vz1_exact, vz1_exact))
    np.fill_diagonal(rt, 1.0)
    Ttail = np.zeros_like(Cz1)
    rk = rt * rt
    for k in range(3, alh.shape[0]):
        rk = rk * rt
        bt = alh[k] * scl
        Ttail += np.outer(bt, bt) * rk
    Cov = G1 + G2 + Ttail
    np.fill_diagonal(Cov, VD)
    v2 = np.einsum("ac,ab,bc->c", W2, Cov, W2, optimize=True)
    tv = tr["mc_pre_v"][2].astype(np.float64)
    hv = st["host_v"][1].astype(np.float64)
    out = {
        "v2_rel": round(float(np.sqrt(np.mean((v2 - tv) ** 2))
                              / np.sqrt(np.mean(tv ** 2))), 5),
        "card_v2_rel": round(float(np.sqrt(np.mean((hv - tv) ** 2))
                                   / np.sqrt(np.mean(tv ** 2))), 5),
        "dv_corr": round(float(np.corrcoef(v2 - hv, tv - hv)[0, 1]), 4),
        "dv_beta": round(float(np.dot(v2 - hv, tv - hv)
                               / np.dot(tv - hv, tv - hv)), 4),
        "diag_exact_med_ratio": round(float(np.median(
            VD / np.maximum(tr["mc_post_var"][1].astype(np.float64),
                            1e-30))), 5),
    }
    # DELTA-MODEL grade: myR = chaos readout - closure(card L1 state),
    # vs banked R_1 (same reference). Shared systematics cancel.
    from run_vresid_field import engine_post_cov
    hc = np.load(HERE / f"host_cov_rank{RANK:04d}_base.npz")
    i = 0
    mu11c = engine_post_cov(
        st["host_m"][i].astype(np.float64),
        st["host_v"][i].astype(np.float64),
        st["host_k3"][i].astype(np.float64),
        st["host_k4"][i].astype(np.float64),
        hc["cov"][i].astype(np.float64),
        hc["k21"][i].astype(np.float64),
        hc["k22"][i].astype(np.float64),
        hc["k31"][i].astype(np.float64))
    RF = np.load(HERE / f"vresid_field_rank{RANK:04d}.npz")
    R1 = RF["R_1"].astype(np.float64)
    myR = Cov - mu11c
    nR = np.linalg.norm(R1)
    pushR = np.einsum("ac,ab,bc->c", W2, R1, W2, optimize=True)
    pushRes = np.einsum("ac,ab,bc->c", W2, R1 - myR, W2, optimize=True)
    out["deltaR_corr"] = round(float(
        np.dot(myR.ravel(), R1.ravel())
        / max(np.linalg.norm(myR) * nR, 1e-30)), 4)
    out["deltaR_frob_closure"] = round(
        1 - float(np.linalg.norm(R1 - myR) / nR), 4)
    out["deltaR_pushedv_closure"] = round(
        1 - float(np.sqrt(np.mean(pushRes ** 2))
                  / np.sqrt(np.mean(pushR ** 2))), 4)
    print(RANK, json.dumps(out), flush=True)
    (HERE / f"CHAOS_SUPPLY_L2_rank{RANK:04d}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
