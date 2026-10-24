#!/usr/bin/env python3
"""TAG opus-physicist T19b (numpy): the lead's second-chaos diagnostic (08:25Z) from t19mc (Ntilde, Nmean; 2 x 8M) against the truths.
   Ntilde_ij = sum_k kappa4(z_i, z_j, z_k, z_k),  Nmean_ij = kappa4(z_i, z_j, p, p), p = mhat . z~.
Predictions (iid Gaussian weights): k4_i = const + (6/n) Ntilde_ii + fourth chaos;  K31_ij = (3/n) Ntilde_ij + ...;
feed kappa4(z_i, z_j, z_k, z_k) = (1/n) Ntilde_ij for every hub k + ....
Independence: the Monte Carlo blocks 0 / 1 of t19mc are the blocks of t2mc (K31 truth M31, feed truth t_F4); every score pairs a
predictor from one block with a truth from the OTHER block; coefficients are fitted the same way. Per-unit truth: indep32M (independent
of both blocks); truth noise from |k4(P32M) - k4(indep32M)|^2 / 2.
 (1) per unit: share of the residual of the true k4 after the isotropic family (v^2 alone = the lead's reference; and 1, v, v^2) explained
     by Q_i = Ntilde_ii and by Q_i, Qm_i; coefficient fitted and at 6/n. Weights: the kurtosis weight of the mean formula.
 (2) K31 on the kink rows (|alpha| < 1.5), off-diagonal, T12's convention R2 = 1 - |y - fit|^2 / |y|^2 on the other block:
     v_i C_ij (the card's form), Ntilde_ij, both, + Nmean_ij; Ntilde at 3/n with nothing fitted.
 (3) feed: share of the true distinct-index feed energy on the kink block of the next layer (t4_cp4_ceiling.py's score) for the model
     Q(i,j,k,k) = M_ij w_k: isotropic M = C, w = v (0.21 at layer 9); M = Ntilde, w = 1/n (the lead's form) or w = v / tr C; with the
     exchange pairing C_ik C_jk added; fitted and with nothing fitted.
usage: t19b_analysis.py <rank> [parts, default 123]"""
import sys, json, math
from pathlib import Path
import numpy as np
import pcard_light as pl
import t2_lib as L
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
rank = int(sys.argv[1]); parts = sys.argv[2] if len(sys.argv) > 2 else '123'; n = 1024
Z = np.load(HERE / f't19mc_rank{rank:04d}.npz'); MCF = np.load(HERE / f't2mc_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
A, B = pl.pu(rank, 'P32M'), pl.pu(rank, 'indep32M'); eye = np.eye(n, dtype=bool)


def second_chaos(l, b):
    q = lambda k: np.asarray(Z[f'{k}_L{l}_b{b}'], dtype=np.float64)
    m1 = q('S1'); M11 = q('M11'); C = M11 - np.outer(m1, m1); mu = MC[f'mean_pre_{l}'].astype(np.float64); mh = mu / np.linalg.norm(mu); Cm = C @ mh
    Nt = q('T') - M11 * float(q('S')) - 2 * C @ C; Nm = q('Tm') - M11 * float(q('P2')) - 2 * np.outer(Cm, Cm)
    return Nt, Nm, C


for l in (int(x) for x in Z['layers']):
    N = [second_chaos(l, b) for b in (0, 1)]; mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); sd = np.sqrt(v); al = mu / sd
    out = dict(rank=rank, layer=l, tr_Ntilde_over_n2_v2=round(float(np.trace(N[0][0]) / (n * (v ** 2).sum())), 5),
               block_reliability_of_Ntilde_offdiag=round(float(np.corrcoef(N[0][0][~eye], N[1][0][~eye])[0, 1]), 4))
    if '1' in parts:
        kA, kB = A['mc_pre_k4'][l].astype(np.float64), B['mc_pre_k4'][l].astype(np.float64); vt = B['mc_pre_v'][l].astype(np.float64); mt = B['mc_pre_m'][l].astype(np.float64)
        at = mt / np.sqrt(vt); w = ((at * at - 1) * np.exp(-at * at / 2)) ** 2 + 1e-12; iso = vt ** 2; one = np.ones(n); sw = np.sqrt(w)
        noise = float((w * (kA - kB) ** 2).sum()) / 2; tot = float((w * kB * kB).sum()) - noise
        # Q without the self term kappa4(i,i,i,i) (taken from the 32M truth that holds the blocks)
        Q = [np.diag(N[b][0]) - kA for b in (0, 1)]; Qm = [np.diag(N[b][1]).copy() for b in (0, 1)]
        def resid(fixed_feats, block_feats, target=None):
            r = []; co = []
            for b in (0, 1):
                X = np.stack(fixed_feats + [f[b] for f in block_feats], 1); y = kB if target is None else target[b]
                c = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)[0]; co.append(c); r.append(y - X @ c)
            return float((w * r[0] * r[1]).sum()) - noise, (co[0] + co[1]) / 2
        res = {}
        for fam_name, fam in (('v^2', [iso]), ('1, v, v^2', [one, vt, iso])):
            r0, _ = resid(fam, []); rq, cq = resid(fam, [Q]); rqm, cqm = resid(fam, [Q, Qm]); rm, cm_ = resid(fam, [Qm])
            rfix, _ = resid(fam, [], target=[kB - (6.0 / n) * Q[b] for b in (0, 1)])
            res[fam_name] = dict(unit_specific_rel_rms=round(math.sqrt(max(r0, 0) / tot), 4), Q_fitted=round(1 - rq / r0, 4), Q_coef_times_n=round(float(cq[-1]) * n, 3),
                                 Q_at_6_over_n=round(1 - rfix / r0, 4), Qm_alone=round(1 - rm / r0, 4), Q_and_Qm=round(1 - rqm / r0, 4), Q_Qm_coefs=[round(float(cqm[-2]) * n, 3), round(float(cqm[-1]), 4)])
        out['per_unit_k4_share_explained'] = res
    if '2' in parts and f'M31_L{l}_b0' in MCF.files:
        K31 = []
        for b in (0, 1):
            q = lambda k_: np.asarray(MCF[f'{k_}_L{l}_b{b}'], dtype=np.float64)
            m1 = q('S1_z'); Cs = q('M11_z') - np.outer(m1, m1); K31.append(q('M31') - 3 * np.diag(Cs)[:, None] * Cs)
        kink = np.abs(al) < 1.5; m = kink[:, None] & ~eye; isoF = v[:, None] * C
        Fb = {'v_i C_ij': [isoF, isoF], 'Ntilde_ij': [N[0][0], N[1][0]], 'Nmean_ij': [N[0][1], N[1][1]], 'v_i mu_i mu_j': [(v * mu)[:, None] * mu[None, :]] * 2}
        def r2(keys, fixed=None):
            sc = []; cs_ = []
            for b in (0, 1):                      # predictor of block 1-b fitted to the truth of block b; scored: predictor of block b against the truth of block 1-b
                if fixed is None:
                    X = np.stack([Fb[k][1 - b][m] for k in keys], 1); co = np.linalg.lstsq(X, K31[b][m], rcond=None)[0]
                else: co = np.array(fixed)
                f = np.stack([Fb[k][b][m] for k in keys], 1) @ co; y = K31[1 - b][m]; sc.append(1 - float(((y - f) ** 2).sum()) / float((y ** 2).sum())); cs_.append(co)
            return round(float(np.mean(sc)), 4), [round(float(c), 6) for c in np.mean(cs_, 0)]
        y0, y1 = K31[0][m], K31[1][m]
        out['K31_kink_rows_R2'] = dict(kink_rows=int(kink.sum()), noise_ceiling=round(1 - float(((y0 - y1) ** 2).sum()) / 2 / float((0.5 * (y0 ** 2 + y1 ** 2)).sum()), 4),
                                       isotropic=r2(['v_i C_ij']), Ntilde_fitted=r2(['Ntilde_ij']), Ntilde_at_3_over_n=r2(['Ntilde_ij'], fixed=[3.0 / n]),
                                       isotropic_and_Ntilde=r2(['v_i C_ij', 'Ntilde_ij']), Ntilde_and_Nmean=r2(['Ntilde_ij', 'Nmean_ij']),
                                       isotropic_Ntilde_Nmean=r2(['v_i C_ij', 'Ntilde_ij', 'Nmean_ij']), three_over_n=round(3.0 / n, 6))
    if '3' in parts and f't_F4_L{l}_b0' in (BP := np.load(HERE / f't2_budget_rank{rank:04d}_planes.npz')).files:
        STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
        man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
        W = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')[l + 1].astype(np.float64)
        pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a_ / math.sqrt(2)) / 2 for a_ in al]); d = pdf / sd
        blk = BP[f'blk_L{l}']; F = [BP[f't_F4_L{l}_b{b}'].astype(np.float64)[blk] for b in (0, 1)]; FF = float(F[0] @ F[1]); G = g[:, None] * W
        def push_M(M, om):                      # distinct-index content of the model feed Q(i,j,k,k) = M_ij om_k, pushed to the next plane (t2_lib conventions)
            MG = G.T @ M @ G; full = np.outer(np.diag(MG), (om * d / 2) @ W) + MG * ((om * d) @ W)[:, None]
            K22 = np.outer(np.diag(M), om); K31_ = M * om[:, None]; k4 = np.diag(M) * om
            return (full - L.Pi(W, L.feed_plane(d, g, K22, K31_, k4)))[blk]
        iso22 = (L.feed_push_model(W, d, g, C, 1.0, 0.0) - L.Pi(W, L.feed_plane_model(d, g, C, 1.0, 0.0)))[blk]
        out['push_M_certificate_vs_feed_push_model'] = float(np.abs(push_M(C, v) - iso22).max() / np.abs(iso22).max())
        iso11 = (L.feed_push_model(W, d, g, C, 0.0, 1.0) - L.Pi(W, L.feed_plane_model(d, g, C, 0.0, 1.0)))[blk]
        X = {'isotropic C_ij v_k': [iso22, iso22], 'exchange pairing C_ik C_jk': [iso11, iso11],
             'Ntilde_ij / n': [push_M(N[b][0], np.full(n, 1.0 / n)) for b in (0, 1)], 'Ntilde_ij v_k / tr C': [push_M(N[b][0], v / v.sum()) for b in (0, 1)],
             'Nmean_ij v_k': [push_M(N[b][1], v) for b in (0, 1)]}
        def share(keys, fixed=None):
            sc = []; cs_ = []
            for b in (0, 1):
                if fixed is None:
                    Xf = np.stack([X[k][1 - b] for k in keys], 1); co = np.linalg.lstsq(Xf, F[b], rcond=None)[0]
                else: co = np.array(fixed)
                f_s = np.stack([X[k][b] for k in keys], 1) @ co; f_o = np.stack([X[k][1 - b] for k in keys], 1) @ co      # the same model from the two blocks
                sc.append((2 * float(f_s @ F[1 - b]) - float(f_s @ f_o)) / FF); cs_.append(co)
            return round(float(np.mean(sc)), 4), [round(float(c), 5) for c in np.mean(cs_, 0)]
        out['feed_share_explained'] = {'isotropic C_ij v_k': share(['isotropic C_ij v_k']), 'isotropic + exchange pairing': share(['isotropic C_ij v_k', 'exchange pairing C_ik C_jk']),
                                       'Ntilde / n, nothing fitted': share(['Ntilde_ij / n'], fixed=[1.0]), 'Ntilde / n, fitted': share(['Ntilde_ij / n']),
                                       'Ntilde v_k / tr C, nothing fitted': share(['Ntilde_ij v_k / tr C'], fixed=[1.0]), 'Ntilde v_k / tr C, fitted': share(['Ntilde_ij v_k / tr C']),
                                       'Ntilde v_k / tr C + exchange pairing': share(['Ntilde_ij v_k / tr C', 'exchange pairing C_ik C_jk']),
                                       'Ntilde v_k / tr C + exchange + Nmean': share(['Ntilde_ij v_k / tr C', 'exchange pairing C_ik C_jk', 'Nmean_ij v_k']),
                                       'isotropic + Ntilde v_k / tr C + exchange': share(['isotropic C_ij v_k', 'Ntilde_ij v_k / tr C', 'exchange pairing C_ik C_jk'])}
    print(json.dumps(out), flush=True)
