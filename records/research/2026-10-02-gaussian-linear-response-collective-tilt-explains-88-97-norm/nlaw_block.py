#!/usr/bin/env python3
"""TAG opus-physicist NLAW_BLOCK: the one-layer law of the second-chaos matrix as MATRIX-VECTOR products (for a carrier that holds a few
vectors: the lead's O/ncar.py, a native port). Same law as nlaw.law (parity certificate below), never forming an n x n matrix except the
ones the card already holds (C = pre-activation covariance, K21 = third-cumulant plane, row index doubled).

  M x = 2 W' N (W x) + 2 kappa C_next x          (Ntilde of the next layer applied to a vector x; W = weights into the next layer)
  N y, y = W x, with z1 = Co (g o y)   [Co = C with its diagonal removed: Co u = C u - v o u]:
   1 transport     g o gbar Nt(g o y)                                 Nt(.) = the carried matrix of this layer applied to a vector
   2 births        g o Co (S2 o z1)
   3 tilts, own    a o z1 + g o Co (a o y),   a = d o (Co V) + (e/2) o Pv + S3 + d o (K21' gamma) + (e/2) o (K21 V),      [Co V = C V - v o V: CORRECTED 11:05Z, was C V]
                   Pv = gbar diag(Nt) + 2 (Co o Co) gamma          (mean tilt, variance tilt, the units' own (3,1) cumulant, K21 parts of the two tilts)
   4 conversion    g o K3V(g o y),   K3V u = pp (c o (pp' u)),  c_r = lam_r (V . pp_r)     (pp = the bank's pushed columns of this layer)
   5 K21 x C       g o K21' (q1 o z1) + g o Co (q1 o K21 (g o y)) + d o K21 ((S2/2) o z1) + g o Co ((S2/2) o K21' (d o y))
                   + (1/2) [d o K21 ((d o CV) o y) + (d o CV) o K21' (d o y)] + ((e/2) o CV) o K21 (g o y) + g o K21' (((e/2) o CV) o y)
                   (K21 with its diagonal removed; q1 = d (1 - g + m alpha / sigma))
   6 homogeneity   - kappa g o z1                                     kappa = the multiple of C in the tilt P = gbar Nt + 2 C Gamma C + K3V
   7 diagonal      [S4 + V o (C V - V o v) + 2 gamma o ((Co o Co) gamma) - 2 g^2 o ((Co o Co) g^2) + gamma gbar diag(Nt) + V o (K21' gamma) + gamma o (K21 V)
                    - 2 g d o ((K21 o Co) g^2) - 2 g^2 o ((K21' o Co)(g d)) - kappa VarG] o y
   8 the operators of 1, 2, 4, 5 are OFF-DIAGONAL matrices in the law; applied as products they bring their own diagonal entries, which are removed:
                   - [gbar g^2 diag(Nt) + g^2 (Co o Co) S2 + g^2 diag(K3V) + 2 g^2 ((K21' o Co) q1) + d g ((K21 o Co) S2)] o y,   diag(K3V)_a = sum_r c_r pp_r(a)^2
  Not a matrix-vector product and left out here: (d d') o Co o K3V (an elementwise product of two matrices; it is in nlaw.law / ncarry.py).
  Products per application to a block of b vectors: 2 with W, 3 with C (z1, births, tilts), 8 with K21 (4) and C (2) and K21' (4) in term 5
  [two of them are extra C products], 2 over the bank columns for the conversion. The per-unit vectors of terms 3 and 7 cost matrix-vector products once per layer.
usage (certificate): nlaw_block.py <rank> <layer>   -> max relative difference to nlaw.law on a random block, with and without the conversion matrix"""
import sys, math
import numpy as np
import nlaw


def apply_N(Y, C, mu, m, Nt_apply, Nt_diag, kappa, K21=None, K3V_apply=None, K3V_diag=None, skip=()):
    u = nlaw.unit_stats(C, mu, m); v, s, al, g, d, gam, V, S2, S3, S4, e3 = (u[k] for k in ('v', 's', 'al', 'g', 'd', 'gam', 'V', 'S2', 'S3', 'S4', 'e3'))
    col = lambda x: x[:, None]; Co = lambda X: C @ X - col(v) * X; gb = float(gam.mean()); CoCo = C * C - np.diag(v * v); CV = C @ V
    gY = col(g) * Y; z1 = Co(gY)
    a = d * (CV - v * V) + (e3 / 2) * (gb * Nt_diag + 2 * (CoCo @ gam)) + S3          # mean tilt over c != a only (CORRECTED 11:05Z: C V - v o V = Co V)
    dg = S4 + V * (CV - V * v) + 2 * gam * (CoCo @ gam) - 2 * g * g * (CoCo @ (g * g)) + gam * gb * Nt_diag - kappa * u['varG']
    out = col(g) * (gb * Nt_apply(gY)) + col(g) * Co(col(S2) * z1) - kappa * col(g) * z1
    own = gb * g * g * Nt_diag + g * g * (CoCo @ S2)          # the diagonal entries of the off-diagonal operators: taken out (term 8), the diagonal is term 7
    if K3V_apply is not None: out = out + col(g) * K3V_apply(gY); own = own + g * g * K3V_diag
    if K21 is not None and 'k3c' not in skip:
        Ko = K21 - np.diag(np.diag(K21)); q1 = d * (1 - g + m * al / s); a = a + d * (Ko.T @ gam) + (e3 / 2) * (Ko @ V)
        KgY = Ko @ gY; KtdY = Ko.T @ (col(d) * Y)
        out = out + col(g) * (Ko.T @ (col(q1) * z1)) + col(g) * Co(col(q1) * KgY) + col(d) * (Ko @ (col(S2 / 2) * z1)) + col(g) * Co(col(S2 / 2) * KtdY)
        out = out + 0.5 * (col(d) * (Ko @ (col(d * CV) * Y)) + col(d * CV) * KtdY) + col(e3 / 2 * CV) * KgY + col(g) * (Ko.T @ (col(e3 / 2 * CV) * Y))
        Cz = C - np.diag(v); dg = dg + V * (Ko.T @ gam) + gam * (Ko @ V) - 2 * g * d * ((Ko * Cz) @ (g * g)) - 2 * g * g * ((Ko.T * Cz) @ (g * d))
        own = own + 2 * g * g * ((Ko.T * Cz) @ q1) + d * g * ((Ko * Cz) @ S2)
    out = out + col(a) * z1 + col(g) * Co(col(a) * Y)
    return out + col(dg - own) * Y


if __name__ == '__main__':
    import json
    from pathlib import Path
    HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank = int(sys.argv[1]); l = int(sys.argv[2]); n = 1024
    Z = np.load(HERE / f't19g_rank{rank:04d}.npz'); A = np.load(HERE / f't19m_rank{rank:04d}.npz')
    tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64); symm = lambda X: (X + X.T) / 2
    avg = lambda k: (A[f'{k}_L{l}_b0'].astype(np.float64) + A[f'{k}_L{l}_b1'].astype(np.float64)) / 2
    C = (Z[f'M11_L{l}_b0'].astype(np.float64) + Z[f'M11_L{l}_b1'].astype(np.float64)) / 2; mu = Z[f'mu_L{l}'].astype(np.float64); m = tp[l]
    M11 = Z[f'M11_L{l}_b0'].astype(np.float64); Nt = symm(Z[f'T_L{l}_b0'].astype(np.float64) - M11 * float(Z[f'S_L{l}_b0']) - 2 * M11 @ M11); K21 = avg('M21'); K3V = symm(avg('T_L'))
    Y = np.random.default_rng(0).standard_normal((n, 8))
    for name, K3 in (('without the conversion', None), ('with the conversion matrix', K3V)):
        N, kap = nlaw.law(C, mu, m, Nt, K21=K21, K3V=K3)
        if K3 is not None:      # nlaw.law also holds the elementwise term (d d') o Co o K3V, which is not a matrix-vector product: take it out for the comparison
            u = nlaw.unit_stats(C, mu, m); N = N - np.outer(u['d'], u['d']) * (C - np.diag(np.diag(C))) * K3
        ref = N @ Y; got = apply_N(Y, C, mu, m, lambda X: Nt @ X, np.diag(Nt).copy(), kap, K21=K21, K3V_apply=None if K3 is None else (lambda X: K3 @ X), K3V_diag=None if K3 is None else np.diag(K3).copy())
        print(json.dumps({'rank': rank, 'layer': l, 'case': name, 'max_abs_difference_over_max_abs': float(np.abs(ref - got).max() / np.abs(ref).max()), 'kappa_over_trC': float(kap / np.trace(C))}))
