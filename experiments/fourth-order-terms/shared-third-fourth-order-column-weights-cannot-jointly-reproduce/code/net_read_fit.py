#!/usr/bin/env python3
"""TAG fable-physicist (FO-2): the NET covariance change of the three truth planes (8M K21/K22/K31 errors vs the card at layer L, r35 bank
dump) through the pair map's leading-order covariance read (r=0 Hermite terms), on response rows x live columns, symmetrized:
  dC_ac = 1/2 [E21_ac pdf_a g_c/s_a + E21_ca g_a pdf_c/s_c] + 1/4 E22_ac pdf_a pdf_c/(s_a s_c) - 1/6 [E31_ac al_a pdf_a g_c/v_a + E31_ca al_c pdf_c g_a/v_c]
Cross-check of the linear read vs the actual pair map is printed. Families fitted to the NET change by CG (oracle, free weights):
  'col'   : per-column K21 re-weight dlam_j (p_j^3 family) + per-column fourth-order lam4_j (p_j^4 family), same columns (2N unknowns)
  'col4'  : lam4_j only; 'col3': dlam_j only.
Reports the explained fraction of the net change (Frobenius, response x live) and of its mean-orthogonal part.
usage: net_read_fit.py <layer> <families> <iters> <subset all|young>"""
import sys, json
from pathlib import Path
import numpy as np
from math import erf
HERE = Path(__file__).resolve().parent; P2 = HERE.parents[1]; G19 = P2 / 'public_g19_attribution_v1'
D = P2 / 'fable_oldtier_economics_v1' / 'bank_r35'; MP = np.load(G19 / 'mps_planes_rank0035.npz')
L = int(sys.argv[1]); fams = sys.argv[2].split(','); iters = int(sys.argv[3]); subset = sys.argv[4]
verf = np.vectorize(erf)
z = np.load(D / f'layer_{L:02d}.npz'); st = np.load(D / f'state_{L:02d}.npz')
resp, live = z['response'], z['live']; no = int(z['old_columns'])
C = st['Cpre'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); mu = st['mu'].astype(np.float64); rad = float(st['radial'])
K = st['K21'].astype(np.float64)
al = mu / s; pdf = np.exp(-al * al / 2) / np.sqrt(2 * np.pi); g = 0.5 * (1 + verf(al / np.sqrt(2)))
R, Lv = resp, live
off = (R[:, None] != Lv[None, :])
# weights on the response x live block; for transposed terms we need the (c,a) entries -> use full planes
a_pdf_s = pdf / s; a_g = g; a_w31 = al * pdf / v
def read(E21, E22, E31):
    t1 = 0.5 * (E21[np.ix_(R, Lv)] * a_pdf_s[R][:, None] * a_g[Lv][None, :] + E21[np.ix_(Lv, R)].T * a_g[R][:, None] * a_pdf_s[Lv][None, :])
    t2 = 0.25 * E22[np.ix_(R, Lv)] * a_pdf_s[R][:, None] * a_pdf_s[Lv][None, :]
    t3 = -(1 / 6) * (E31[np.ix_(R, Lv)] * a_w31[R][:, None] * a_g[Lv][None, :] + E31[np.ix_(Lv, R)].T * a_g[R][:, None] * a_w31[Lv][None, :])
    return (t1 + t2 + t3) * off
E21 = MP[f'K21_{L}'].astype(np.float64) - K; np.fill_diagonal(E21, 0)
E22 = MP[f'K22_{L}'].astype(np.float64) - rad * (np.outer(v, v) + 2 * C * C) / 3; np.fill_diagonal(E22, 0)
E31 = MP[f'K31_{L}'].astype(np.float64) - rad * v[:, None] * C; np.fill_diagonal(E31, 0)
Dt = read(E21, E22, E31)
parts = dict(r21=read(E21, 0 * E22, 0 * E31), r22=read(0 * E21, E22, 0 * E31), r31=read(0 * E21, 0 * E22, E31))
nrm = lambda X: float(np.linalg.norm(X))
print(json.dumps(dict(layer=L, net=nrm(Dt), **{k: nrm(x) for k, x in parts.items()},
                      cos21_2231=float((parts['r21'] * (parts['r22'] + parts['r31'])).sum() / nrm(parts['r21']) / nrm(parts['r22'] + parts['r31'])))), flush=True)
cols = slice(no, None) if subset == 'young' else slice(None)
P = z['pp'][:, cols].astype(np.float64); N = P.shape[1]
P2_, P3_ = P * P, P ** 3
def fam_read(dl, l4):   # planes from the families, then the read
    e21 = (P2_ * dl) @ P.T if dl is not None else 0 * C
    e22 = (P2_ * l4) @ P2_.T if l4 is not None else 0 * C
    e31 = (P3_ * l4) @ P.T if l4 is not None else 0 * C
    for M in (e21, e22, e31):
        if not np.isscalar(M): np.fill_diagonal(M, 0)
    return read(e21, e22, e31)
def fam_readT(Rm, want3, want4):  # adjoint of fam_read at the block Rm (response x live)
    # build full-size weighted planes G21, G22, G31 whose Frobenius inner product with a plane equals <Rm, read(plane)>
    Z = np.zeros_like(C); G21 = Z.copy(); G22 = Z.copy(); G31 = Z.copy()
    Rw = Rm * off
    G21[np.ix_(R, Lv)] += 0.5 * Rw * a_pdf_s[R][:, None] * a_g[Lv][None, :]
    G21[np.ix_(Lv, R)] += 0.5 * (Rw * a_g[R][:, None] * a_pdf_s[Lv][None, :]).T
    G22[np.ix_(R, Lv)] += 0.25 * Rw * a_pdf_s[R][:, None] * a_pdf_s[Lv][None, :]
    G31[np.ix_(R, Lv)] += -(1 / 6) * Rw * a_w31[R][:, None] * a_g[Lv][None, :]
    G31[np.ix_(Lv, R)] += -(1 / 6) * (Rw * a_g[R][:, None] * a_w31[Lv][None, :]).T
    for M in (G21, G22, G31): np.fill_diagonal(M, 0)
    g3 = ((G21 @ P) * P2_).sum(0) if want3 else None
    g4 = (((G22 @ P2_) * P2_).sum(0) + ((G31 @ P) * P3_).sum(0)) if want4 else None
    return g3, g4
for fam in fams:
    w3, w4 = fam in ('col', 'col3'), fam in ('col', 'col4')
    def unpack(x): return (x[:N] if w3 else None), (x[N:] if (w3 and w4) else (x if w4 else None))
    def pack(a, b): return np.concatenate([u for u in (a, b) if u is not None])
    def apply_N(x):
        dl, l4 = unpack(x); r = fam_read(dl, l4); a, b = fam_readT(r, w3, w4); return pack(a, b)
    a, b = fam_readT(Dt, w3, w4); rhs = pack(a, b)
    x = np.zeros_like(rhs); r = rhs - apply_N(x); pv = r.copy(); rs = r @ r
    for it in range(iters):
        Ap = apply_N(pv); alpha = rs / (pv @ Ap); x += alpha * pv; r -= alpha * Ap; rsn = r @ r
        pv = r + (rsn / rs) * pv; rs = rsn
    dl, l4 = unpack(x); Df = fam_read(dl, l4)
    print(json.dumps(dict(layer=L, family=fam, subset=subset, n_cols=N, explained=round(1 - nrm(Dt - Df) ** 2 / nrm(Dt) ** 2, 4))), flush=True)
