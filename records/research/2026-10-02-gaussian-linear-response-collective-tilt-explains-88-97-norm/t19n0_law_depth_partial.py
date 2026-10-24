#!/usr/bin/env python3
"""TAG opus-physicist T19n0 (numpy): PARTIAL test of the one-layer law at depth with what is already stored (before t19m): transitions
5->6, 9->10, 11->12, 13->14 of network 35. Inputs at layer l: Ntilde and the covariance from t19g block b (the truth of layer l+1 is taken
from the OTHER block), K21 plane from t2mc block 1 (samples 8M-16M, independent of t19g), exact means, true post means.
Transport by the matrix closure gbar x Ntilde (the lead's E35: 98-99.6% of the exact contraction). NOT included (need t19m): the conversion
from the third cumulant sum_c V_c kappa3(a,b,c), the V-contraction of K31, the tensor term of k3c. Terms as in t19n_law_depth.py.
usage: t19n0_law_depth_partial.py <rank>"""
import sys, json, math
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; n = 1024; off = ~np.eye(n, dtype=bool); dgm = np.eye(n, dtype=bool)
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rank = int(sys.argv[1]); man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r'); Z = np.load(HERE / f't19g_rank{rank:04d}.npz'); MCF = np.load(HERE / f't2mc_rank{rank:04d}.npz')
tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64); symm = lambda X: (X + X.T) / 2
def nt(l, b):
    M11 = Z[f'M11_L{l}_b{b}'].astype(np.float64); return symm(Z[f'T_L{l}_b{b}'].astype(np.float64) - M11 * float(Z[f'S_L{l}_b{b}']) - 2 * M11 @ M11), M11
for l in (5, 9, 11, 13):
    mu = Z[f'mu_L{l}'].astype(np.float64); m = tp[l]; q = lambda k_: np.asarray(MCF[f'{k_}_L{l}_b1'], dtype=np.float64)
    m1 = q('S1_z'); S2z = q('S2_z'); Cz = q('M11_z') - np.outer(m1, m1); K21 = q('M21_z') - 2 * m1[:, None] * Cz - S2z[:, None] * m1[None, :]; K21o = K21 - np.diag(np.diag(K21))
    Wn = W[l + 1].astype(np.float64); push = lambda X: 2 * (Wn.T @ X @ Wn)
    C1 = (Z[f'M11_L{l + 1}_b0'].astype(np.float64) + Z[f'M11_L{l + 1}_b1'].astype(np.float64)) / 2; mu1 = Z[f'mu_L{l + 1}'].astype(np.float64); Cmu = C1 @ mu1
    forms = dict(trC_C=np.trace(C1) * C1, C2=C1 @ C1, mumu=np.outer(mu1, mu1), muCmu=np.outer(mu1, Cmu) + np.outer(Cmu, mu1), eye=np.eye(n))
    def specific(N):
        R = N.copy()
        for mask, names in ((off, ('trC_C', 'C2', 'mumu', 'muCmu')), (dgm, ('trC_C', 'C2', 'mumu', 'eye'))):
            X = np.stack([forms[k][mask] for k in names], 1); R[mask] = N[mask] - X @ np.linalg.lstsq(X, N[mask], rcond=None)[0]
        return R
    Rb = [specific(nt(l + 1, b)[0]) for b in (0, 1)]; RR = float((Rb[0] * Rb[1]).sum()); res = {}
    order = ['Tu', 'births', 'rows', 'vtilt', 'own31', 'diag', 'k3c', 'diag_ng', 'gain']; PB = []; kaps = []
    for b in (0, 1):                      # law inputs from block b, truth of layer l+1 from block 1-b
        Nu, C = nt(l, b); v = np.diag(C).copy(); s = np.sqrt(v); al = mu / s; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a / math.sqrt(2)) / 2 for a in al]); d = ph / s
        gam = g - m * d; V = 2 * m * (1 - g); S2 = 2 * (gam - g * g); e3 = -al * ph / v
        r2 = v * ((1 + al * al) * g + al * ph); r3 = v * s * ((al ** 3 + 3 * al) * g + (al * al + 2) * ph); r4 = v * v * ((al ** 4 + 6 * al * al + 3) * g + (al ** 3 + 5 * al) * ph)
        S3 = 3 * (1 - g) * (r2 - 2 * m * m); S4 = r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2
        gg = np.outer(g, g); Co = C - np.diag(v); CV = Co @ V + v * V; gb = float(gam.mean()); T = {}
        T['Tu'] = gb * gg * Nu; T['births'] = gg * ((Co * S2[None, :]) @ Co)
        T['rows'] = Co * (d * CV)[:, None] * g[None, :] + Co * g[:, None] * (d * CV)[None, :]
        Pv = gb * np.diag(Nu) + 2 * ((Co * Co) @ gam); x = Co * ((e3 / 2) * Pv)[:, None] * g[None, :]; T['vtilt'] = x + x.T
        T['own31'] = Co * (S3[:, None] * g[None, :] + g[:, None] * S3[None, :])
        q1 = d * (1 - g + m * al / s); a1 = gg * ((K21o.T * q1[None, :]) @ Co + (Co * q1[None, :]) @ K21o)
        x = d[:, None] * ((K21o * (S2 / 2)[None, :]) @ Co + Co * (K21o.T @ gam)[:, None]) * g[None, :]; a2 = x + x.T
        a3 = 0.5 * np.outer(d, d) * (K21o * CV[None, :] + K21o.T * CV[:, None])
        x = (e3 / 2)[:, None] * (K21o * CV[:, None] + Co * (K21o @ V)[:, None]) * g[None, :]; a4 = x + x.T
        T['k3c'] = a1 + a2 + a3 + a4
        for k in T:
            if k != 'gain': np.fill_diagonal(T[k], 0.0)
        # GAIN part: the ReLU is homogeneous, so the pure-gain component kappa (mu/2, C) of the tilt maps to kappa x (post covariance), a FORM of the next
        # layer after the push; gate-linear transport of it leaves kappa x (post covariance - g C g) behind. Remove it: add - kappa x (g C g off the diagonal,
        # Gaussian variance on it), whose push differs from kappa x C(l+1) only by the non-linear parts of the post covariance.
        Pfull = gb * Nu + 2 * (C * gam[None, :]) @ C; kap = float((Pfull * C).sum() / (C * C).sum()); varG = r2 - (s * ph + mu * g) ** 2
        Tg_ = -kap * gg * C; np.fill_diagonal(Tg_, -kap * varG); T['gain'] = Tg_; kaps.append(kap / float(np.trace(C)))
        T['diag'] = np.diag(S4 + V * (CV - V * v) + ((2 * gam[:, None] * gam[None, :] - 2 * (g * g)[:, None] * (g * g)[None, :]) * Co * Co).sum(1))
        T['diag_ng'] = np.diag(gam * gb * np.diag(Nu) + V * (K21o.T @ gam) + gam * (K21o @ V) - 2 * g * d * ((K21o * Co) @ (g * g)) - 2 * g * g * ((K21o.T * Co) @ (g * d)))
        PB.append({k: specific(push(X)) for k, X in T.items()})
    share = lambda Xb: round(float(np.mean([(2 * (Xb[b] * Rb[1 - b]).sum() - (Xb[b] * Xb[1 - b]).sum()) / RR for b in (0, 1)])), 4)
    cum = {}; acc = [np.zeros((n, n)), np.zeros((n, n))]
    for k in order:
        acc = [acc[b] + PB[b][k] for b in (0, 1)]; cum['+ ' + k] = share(acc)
    wo = lambda drop: share([sum(PB[b][k] for k in order if k not in drop) for b in (0, 1)])
    def fit(keys):
        sc = []; cs_ = []
        for b in (0, 1):
            X = np.stack([PB[1 - b][k].ravel() for k in keys], 1); co = np.linalg.lstsq(X, Rb[b].ravel(), rcond=None)[0]
            f_s = np.stack([PB[b][k].ravel() for k in keys], 1) @ co; f_o = X @ co; sc.append((2 * float(f_s @ Rb[1 - b].ravel()) - float(f_s @ f_o)) / RR); cs_.append(co)
        return round(float(np.mean(sc)), 4), {k: round(float(c), 3) for k, c in zip(keys, np.mean(cs_, 0))}
    print(json.dumps(dict(rank=rank, transition=f'{l}->{l + 1}', truth_block_agreement=round(RR / float(np.sqrt((Rb[0] ** 2).sum() * (Rb[1] ** 2).sum())), 4), cumulative_nothing_fitted=cum,
                          without_vtilt=wo(['vtilt']), without_rows_and_vtilt=wo(['rows', 'vtilt']), without_gain=wo(['gain']), kappa_over_trC=[round(x, 5) for x in kaps], all_fitted=fit(order))), flush=True)
