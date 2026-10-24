#!/usr/bin/env python3
"""TAG recurrence-3b-v3: exact-engine depth iteration, CORRECTED CF.

Tiny exact world (n=32, r=8). State per unit: (m_i, a_i, M_i) --
degree-2 chaos in the SHARED latent g. Truth: MC-sample z1 from the
chaos rep, then iterate the actual net z' = W^T relu(z). Model: the
fused-stream recurrence Xi_{l+1} = W^T F(Xi_l) with EXACT engines:
  mean   : E[phi] = (m + E|z|)/2, E|z| = (2/pi) int (1-Re CF)/t^2
           + analytic tail 1/T (CF -> 0), CORRECTED CF
           (den 1 - i t lam, NOT 1 - 2 i t lam; lane spec had the
           factor-2 bug -- certified 0.18% med on direct test,
           tail-corrected here).
  linear : leave-one-mode-out F_rest engine (validated 0.9998/0.989)
  quad   : diag via (h^2-1)-weighted GH on F_rest; off-diag p0*outer
           (v2, validated 0.998/1.017)
Baseline arm (--gaussian): v1 Gaussian couplings (c1=Phi, c2=pdf/sd)
in the SAME world for the honest in-world comparison.
Gate (preregistered, lane 2026-09-01): C_rel and K21_rel <= 15% at
depth 8; v1 baseline was mean 5-8%, C 23-48%, K21 44-69% (old world).
Usage: exact_recurrence_v3.py [--gaussian] [seed]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
GAUSS = "--gaussian" in sys.argv
SEED = int([a for a in sys.argv[1:] if not a.startswith("-")][0]) \
    if [a for a in sys.argv[1:] if not a.startswith("-")] else 9
N, n, r, DEPTH = 4_000_000, 32, 8, 8
NT, TMAX = 24_000, 300.0        # E|z| grid (dense head handled below)
NTD, TMAXD = 4096, 80.0         # density grid for F_rest
NX, XW = 2048, 14.0             # x-grid half-width in rest-sd units
GH_K = 64


def full_cf(ts, m, lam, at):
    """CF of z = m + at.h + sum lam(h^2-1)/2 on t-grid (CORRECTED)."""
    den = 1.0 - 1j * ts[:, None] * lam[None, :]
    return (np.exp(1j * ts * m - 1j * ts * lam.sum() / 2
                   - 0.5 * ts ** 2 * ((at[None, :] ** 2) / den).sum(1))
            / np.sqrt(np.prod(den, axis=1)))


def emean_direct(m, lam, at):
    """E[relu(z)] = (m + E|z|)/2, tail-corrected integral."""
    ts = np.concatenate([np.linspace(1e-6, 1.0, 4000, endpoint=False),
                         np.linspace(1.0, TMAX, NT)])
    cf = full_cf(ts, m, lam, at)
    eabs = (2 / np.pi) * (np.trapz((1 - np.real(cf)) / ts ** 2, ts)
                          + 1.0 / TMAX)
    return 0.5 * (m + eabs)


def unit_engine(m, lam, at):
    """Exact per-unit emission: (Emean, alpha[r], B[r,r]) eigencoords.

    F_rest per mode from the shared eigen-CF; couplings by GH."""
    ts = np.linspace(1e-6, TMAXD, NTD)
    den = 1.0 - 1j * ts[:, None] * lam[None, :]
    fac = (np.exp(-1j * ts[:, None] * lam[None, :] / 2
                  - 0.5 * ts[:, None] ** 2 * (at[None, :] ** 2) / den)
           / np.sqrt(den))                       # per-mode CF factors
    base = np.exp(1j * ts * m)
    cf_full = base * fac.prod(axis=1)
    cf_rest = cf_full[None, :] / np.where(np.abs(fac.T) < 1e-300,
                                          1e-300, fac.T)  # r x nt
    sd_r = np.sqrt(max((at ** 2).sum() + (lam ** 2).sum() / 2, 1e-12))
    xs = np.linspace(m - XW * sd_r, m + XW * sd_r, NX)
    dt = ts[1] - ts[0]
    # p_rest(mu, x) = (1/pi) Re int CF_rest(t) e^{-itx} dt
    ker = np.exp(-1j * np.outer(ts, xs))
    dens = np.maximum((cf_rest @ ker).real * dt / np.pi, 0.0)
    dens /= np.maximum(np.trapz(dens, xs, axis=1), 1e-300)[:, None]
    dx = xs[1] - xs[0]
    # G(c) = E[(c+x)^+] on the c-grid cg = -xs reversed; build via
    # right cumulative sums of p and xp:
    P_up = np.cumsum(dens[:, ::-1], axis=1)[:, ::-1] * dx    # P(x>xs)
    XP_up = np.cumsum((dens * xs[None, :])[:, ::-1],
                      axis=1)[:, ::-1] * dx                  # E[x;x>xs]
    gh_x, gh_w = np.polynomial.hermite_e.hermegauss(GH_K)
    gh_w = gh_w / gh_w.sum()
    alpha = np.zeros(r)
    Bdiag = np.zeros(r)
    for mu in range(r):
        c = at[mu] * gh_x + lam[mu] * (gh_x ** 2 - 1) / 2
        # F_rest(c) = c*P(x>-c) + E[x; x>-c]
        idx = np.clip(np.searchsorted(xs, -c), 0, NX - 1)
        F = c * P_up[mu, idx] + XP_up[mu, idx]
        alpha[mu] = np.sum(gh_w * gh_x * F)
        Bdiag[mu] = np.sum(gh_w * (gh_x ** 2 - 1) * F)
    p0 = max((cf_full.real * dt).sum() / np.pi, 0.0)
    B = p0 * np.outer(at, at)
    np.fill_diagonal(B, Bdiag)
    return alpha, B


def gaussian_engine(m, lam, at):
    """v1 baseline: Gaussian couplings c1=Phi(alpha), c2=pdf/sd."""
    from math import erf
    v = (at ** 2).sum() + (lam ** 2).sum() / 2
    sd = np.sqrt(max(v, 1e-12))
    al = m / sd
    cdf = 0.5 * (1 + erf(al / np.sqrt(2)))
    pdf = np.exp(-al ** 2 / 2) / np.sqrt(2 * np.pi)
    em = m * cdf + sd * pdf
    alpha = cdf * at
    B = cdf * np.diag(lam) + (pdf / sd) * np.outer(at, at)
    return em, alpha, B


def model_moments(mv, A, Mt):
    C = A @ A.T + 0.5 * np.einsum("iab,jab->ij", Mt, Mt)
    K21 = (np.einsum("iab,ibc,jca->ij", Mt, Mt, Mt)
           + np.einsum("ia,jab,ib->ij", A, Mt, A)
           + 2.0 * np.einsum("ia,iab,jb->ij", A, Mt, A))
    return C, K21


def truth_moments(Z):
    zc = Z - Z.mean(0)
    C = (zc.T @ zc) / len(zc)
    K21 = ((zc ** 2).T @ zc) / len(zc)
    return Z.mean(0).astype(np.float64), C.astype(np.float64), \
        K21.astype(np.float64)


def main():
    rng = np.random.default_rng(SEED)
    A0 = rng.standard_normal((n, r)) * 0.3
    M0 = rng.standard_normal((n, r, r)) * 0.25
    M0 = 0.5 * (M0 + np.transpose(M0, (0, 2, 1)))
    m0 = rng.standard_normal(n) * 0.1
    Ws = [rng.standard_normal((n, n)) * np.sqrt(2.0 / n)
          for _ in range(DEPTH)]
    g = rng.standard_normal((N, r)).astype(np.float32)
    Z = (m0[None, :] + g @ A0.T.astype(np.float32)
         + 0.5 * (np.einsum("na,iab,nb->ni", g, M0.astype(np.float32),
                            g, optimize=True)
                  - np.einsum("iaa->i", M0)[None, :].astype(np.float32)))
    del g
    mv, A, Mt = m0.copy(), A0.copy(), M0.copy()
    tag = "gauss" if GAUSS else "exact"
    out = {}
    for L in range(1, DEPTH + 1):
        tm, tC, tK = truth_moments(Z)
        mC, mK = model_moments(mv, A, Mt)
        rec = {
            "mu_rel": float(np.linalg.norm(mv - tm)
                            / max(np.linalg.norm(tm), 1e-30)),
            "C_rel": float(np.linalg.norm(mC - tC)
                           / max(np.linalg.norm(tC), 1e-30)),
            "K21_rel": float(np.linalg.norm(mK - tK)
                             / max(np.linalg.norm(tK), 1e-30)),
        }
        out[L] = {k: round(v, 4) for k, v in rec.items()}
        print(f"[{tag}] L{L}", json.dumps(out[L]), flush=True)
        if L == DEPTH:
            break
        # ---- truth push
        Z = np.maximum(Z, 0.0) @ Ws[L - 1].astype(np.float32)
        # ---- model push
        E_new = np.zeros(n)
        Al_new = np.zeros((n, r))
        B_new = np.zeros((n, r, r))
        for i in range(n):
            lam, U = np.linalg.eigh(Mt[i])
            at = U.T @ A[i]
            if GAUSS:
                em, al_e, B_e = gaussian_engine(mv[i], lam, at)
            else:
                em = emean_direct(mv[i], lam, at)
                al_e, B_e = unit_engine(mv[i], lam, at)
            E_new[i] = em
            Al_new[i] = U @ al_e
            B_new[i] = U @ B_e @ U.T
        W = Ws[L - 1]
        mv = W.T @ E_new
        A = W.T @ Al_new.reshape(n, r)
        Mt = np.einsum("ic,iab->cab", W, B_new)
    (HERE / f"RECURRENCE_V3_{tag}_seed{SEED}.json").write_text(
        json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
