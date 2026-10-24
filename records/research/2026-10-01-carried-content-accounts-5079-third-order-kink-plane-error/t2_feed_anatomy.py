#!/usr/bin/env python3
"""TAG opus-physicist T2b: anatomy of the TRUE fourth-cumulant feed into births, from t2mc (two independent 8M blocks, network <rank>).
The feed pushes (d_c/2) g_i g_j Q_ijcc to the next plane through two kernels:
   V[a,c]  = kappa4(y_a, y_a, z_c, z_c)                  (y = gate-linear next pre-activation; term1[a,b] = sum_c W'_cb (d_c/2) V[a,c])
   T2[a,b] = sum_c W'_ca d_c kappa4(y_a, y_b, z_c, z_c)   (term2)
Questions: (1) how much of V is the isotropic form c22 Var(y_a) v_c + c11 Cov(y_a,z_c)^2 ; (2) is the rest low-rank, and along which
unit vectors (cross-block validated SVD of the d-weighted residual on the kink rows of L) ; (3) how much of the pushed feed D_F4 on the
kink block do simple model fourth cumulants explain (least squares on one block, scored on the other):
   iso22 C_ij v_c | iso11 C_ic C_jc | m1 mu_i mu_j mu_c^2 | m2 mu_i mu_j v_c | m3 C_ij mu_c^2 | m4 (C_ic mu_j + C_jc mu_i) mu_c
   k2 mu_j K21[c,i] + mu_i K21[c,j] | k2b  C_ic K21[c,j]/v_c + (i<->j)  | a22 C_ij alpha_c^2 v_c
usage: t2_feed_anatomy.py <rank> [layers]"""
import sys, json, math
from pathlib import Path
import numpy as np
import t2_lib as L
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
rank = int(sys.argv[1])
MCF = np.load(HERE / f't2mc_rank{rank:04d}.npz'); CAP = np.load(HERE / f't2cap_rank{rank:04d}.npz')
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
BP = np.load(HERE / f't2_budget_rank{rank:04d}_planes.npz')
layers = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [int(x) for x in MCF['layers']]
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
WALL = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')
N = 1024; eye = np.eye(N, dtype=bool); nrm = np.linalg.norm; f8 = lambda x: np.asarray(x, dtype=np.float64)
erfc = np.vectorize(math.erfc)


def model_feed(W, d, g, terms):
    """pushed distinct feed (full push - Pi(plane)) of Q(i,j,c,c) = sum_A A_ij w_c + sum_B (B_ic B'_jc + B_jc B'_ic).
    terms: list of ('A', A, w) or ('B', B, B')."""
    G = g[:, None] * W; n = W.shape[0]
    full = np.zeros((n, n)); K22 = np.zeros((n, n)); K31 = np.zeros((n, n)); k4 = np.zeros(n)
    for t in terms:
        if t[0] == 'A':
            _, A, w = t; GAG = G.T @ A @ G
            full += np.outer(np.diag(GAG), (d * w / 2) @ W) + ((d * w) @ W)[:, None] * GAG
            K22 += np.outer(np.diag(A), w); K31 += A * w[:, None]; k4 += np.diag(A) * w
        else:
            _, B, Bp = t; X, Xp = G.T @ B, G.T @ Bp                      # [a, c]
            full += (2 * X * Xp) @ ((d / 2)[:, None] * W)
            Wd = W.T * d[None, :]                                          # [a, c] = W_ca d_c
            full += (Wd * X) @ Xp.T + (Wd * Xp) @ X.T
            K22 += 2 * B * Bp; K31 += np.diag(B)[:, None] * Bp.T + B.T * np.diag(Bp)[:, None]; k4 += 2 * np.diag(B) * np.diag(Bp)
    return full - L.Pi(W, L.feed_plane(d, g, K22, K31, k4))


if len(sys.argv) > 3 and sys.argv[3] == 'cert':          # brute-force certificate of model_feed (n = 6)
    rng = np.random.default_rng(3); n = 6; W = rng.standard_normal((n, n)); d = rng.random(n) + .2; g = rng.random(n)
    A = rng.standard_normal((n, n)); A = A + A.T; w = rng.standard_normal(n); B = rng.standard_normal((n, n)); Bp = rng.standard_normal((n, n))
    Q = np.einsum('ij,c->ijc', A, w) + np.einsum('ic,jc->ijc', B, Bp) + np.einsum('jc,ic->ijc', B, Bp)      # Q(i,j,c,c)
    F = 0.5 * np.einsum('i,j,c,ijc->ijc', g, g, d, Q); T = F + F.transpose(0, 2, 1) + F.transpose(2, 1, 0)
    mask = np.ones((n, n, n))
    for a in range(n):
        for b in range(n): mask[a, a, b] = mask[a, b, a] = mask[b, a, a] = 0.0
    bf = np.einsum('ia,ja,kb,ijk->ab', W, W, W, T * mask)
    # the coincident entries Q(i,k,i,i) etc. are only defined through the model: K31[i,k] = Q(i,k,i,i) = A_ik w_i + B_ii B'_ki + B_ki B'_ii
    print(json.dumps(dict(cert_model_feed=float(np.abs(model_feed(W, d, g, [('A', A, w), ('B', B, Bp)]) - bf).max() / np.abs(bf).max())))); sys.exit(0)

for l in layers:
    Lx = l + 1; W = f8(WALL[Lx]); mu = f8(MC[f'mean_pre_{l}']); C = f8(MC[f'cov_pre_{l}']); v = np.diag(C).copy(); s = np.sqrt(v); al = mu / s
    pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = erfc(-al / math.sqrt(2)) / 2; d = pdf / s
    resp = CAP[f'resp_{Lx}'] > .5; alive = CAP[f'alive_{Lx}'] > .5; blk = BP[f'blk_L{l}']
    G = g[:, None] * W; CG = C @ G; covy = G.T @ CG; vy0 = np.diag(covy); A2m = CG.T
    V = []; T2 = []; KZ = []
    for b in (0, 1):
        q = lambda k: f8(MCF[f'{k}_L{l}_b{b}'])
        V.append(q('A1') - np.outer(q('S2_yl'), q('S2_z')) - 2 * q('A2') ** 2)
        T2.append(q('A3') - covy * q('Sq')[:, None] - 2 * (q('A2') * (W.T * d[None, :])) @ q('A2').T)
        KZ.append(L.k21_from_sums(q('M21_z'), q('S1_z'), q('S2_z'), C))
    x2 = lambda A, B, m=None: (float((A[0] * B[1]).sum()) + float((A[1] * B[0]).sum())) / 2 if m is None else (float((A[0][m] * B[1][m]).sum()) + float((A[1][m] * B[0][m]).sum())) / 2
    # (1) isotropic content of V, all (a, c) and d-weighted on the kink rows of L
    I22, I11 = np.outer(vy0, v), A2m * A2m
    wgt = resp[:, None] * d[None, :]
    rec = dict(layer=l)
    for name, wm in (('all', np.ones((N, N))), ('kink rows x d_c', wgt)):
        X = np.stack([(I22 * wm).ravel(), (I11 * wm).ravel()], 1)
        co = [np.linalg.lstsq(X, (V[b] * wm).ravel(), rcond=None)[0] for b in (0, 1)]
        R = [V[b] * wm - (co[1 - b][0] * I22 + co[1 - b][1] * I11) * wm for b in (0, 1)]
        tot = x2([V[0] * wm, V[1] * wm], [V[0] * wm, V[1] * wm]); res = x2(R, R)
        rec[name] = dict(c22=[float(c[0]) for c in co], c11=[float(c[1]) for c in co], iso_share=1 - res / tot,
                         noise_share=float(((V[0] - V[1]) * wm).ravel() @ ((V[0] - V[1]) * wm).ravel()) / 2 / tot)
        if name != 'all':
            # (2) cross-block SVD of the weighted residual
            U0, S0, Vt0 = np.linalg.svd(R[0][resp], full_matrices=False); R1 = R[1][resp]
            cap = np.array([U0[:, k] @ R1 @ Vt0[k] for k in range(64)])
            cum = np.cumsum(cap * S0[:64]) / res
            rec['svd_cum_share'] = {k: float(cum[k - 1]) for k in (1, 2, 4, 8, 16, 32, 64)}
            cands = dict(v=v, mu2=mu * mu, mu=mu, al=al, al2=al * al, g=g, pdf=pdf, d=d, k3=np.diag(KZ[0]), one=np.ones(N), alpdf=al * pdf, s=s, mu_s=mu * s)
            cors = {}
            for k in range(3):
                r_ = Vt0[k] / np.maximum(d, 1e-12)          # undo the weight: the unit-space pattern of the residual
                m = d > 0.05 * d.max()
                Xc = np.stack([cands[c][m] * d[m] for c in cands], 1); Xc = Xc / nrm(Xc, axis=0)
                y_ = Vt0[k][m]
                cors[k] = dict(single={c: round(float(abs(Xc[:, i] @ y_) / nrm(y_)), 3) for i, c in enumerate(cands)},
                               all_R2=float(1 - nrm(y_ - Xc @ np.linalg.lstsq(Xc, y_, rcond=None)[0]) ** 2 / nrm(y_) ** 2))
            rec['svd_right_vectors_vs_unit_patterns'] = cors
    # (3) model fourth cumulants against the pushed truth feed on the kink block
    F = [f8(BP[f't_F4_L{l}_b{b}']) for b in (0, 1)]; nK = math.sqrt(x2([f8(BP[f't_K_L{l}_b{b}']) for b in (0, 1)], [f8(BP[f't_K_L{l}_b{b}']) for b in (0, 1)], blk))
    mm = np.outer(mu, mu); K21c = (KZ[0] + KZ[1]) / 2
    models = dict(iso22=[('A', C, v)], iso11=[('B', C, C / 2)], m1=[('A', mm, mu * mu)], m2=[('A', mm, v)], m3=[('A', C, mu * mu)],
                  m4=[('B', C, np.outer(mu, mu))], k2=[('B', K21c.T, np.outer(mu, np.ones(N)))], k2b=[('B', C, K21c.T / v[None, :])],
                  a22=[('A', C, al * al * v)])
    MF = {k: model_feed(W, d, g, t) for k, t in models.items()}
    FF = x2(F, F, blk); rec['feed_norm_pct_of_plane'] = 100 * math.sqrt(FF) / nK

    def fit(keys):
        X = np.stack([MF[k][blk] for k in keys], 1)
        co = [np.linalg.lstsq(X, F[b][blk], rcond=None)[0] for b in (0, 1)]
        R = [F[b][blk] - X @ co[1 - b] for b in (0, 1)]
        return dict(explained=1 - float(R[0] @ R[1]) / FF, coef={k: [float(co[0][i]), float(co[1][i])] for i, k in enumerate(keys)})
    rec['fits'] = {'+'.join(ks): fit(ks) for ks in (['iso22'], ['iso22', 'iso11'], ['iso22', 'iso11', 'm3'], ['iso22', 'iso11', 'm3', 'm4'],
                                                     ['iso22', 'iso11', 'm1', 'm2', 'm3', 'm4'], ['iso22', 'iso11', 'm1', 'm2', 'm3', 'm4', 'k2', 'k2b'],
                                                     ['iso22', 'iso11', 'm1', 'm2', 'm3', 'm4', 'k2', 'k2b', 'a22'])}
    rec['model_norms_pct'] = {k: 100 * float(nrm(MF[k][blk])) / nK for k in MF}
    print(json.dumps(rec), flush=True)
