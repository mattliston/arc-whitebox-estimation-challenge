#!/usr/bin/env python3
"""TAG opus-physicist T19c (numpy): does the second-chaos matrix ADD to the hub-local terms in the feed, and is it what the free column
weights hold?  Share of the true distinct-index feed energy on the next layer's kink block (the score of t4_cp4_ceiling.py / the lead's
E15), network <rank>, layers with stored columns (t4cols) and feed truth (t2_budget planes):
   iso    isotropic C_ij v_k                          N      second-chaos form Ntilde_ij v_k / tr C (t19mc, per block)
   LOCAL  the exact hub-local terms S1..S4 on all pairs, at their physical values (the lead's E15 / E15b code, in numpy)
   COLL   the lead's collective scalar per column (E15b), where the columns one layer earlier are stored (O/e15cols)
   FREE   one free weight per bank column (t4_lam4_*_none.npz, fitted on truth block 0: scored on block 1 only)
Block-dependent predictors (N) are paired with the truth of the other block; coefficients are fitted the same way.
usage: t19c_feed_combined.py <rank> <layer> [<layer> ...]"""
import sys, json, math, time
from pathlib import Path
import numpy as np
import t2_lib as L
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; n = 1024
rank = int(sys.argv[1])
Z = np.load(HERE / f't19mc_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
BP = np.load(HERE / f't2_budget_rank{rank:04d}_planes.npz'); BI = np.load(HERE / f't4births_rank{rank:04d}.npz')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
WALL = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')


class Op:                                               # t4_lib.Op in numpy: T = sum_r lam_r Sym(u_r, u_r, om_r), fwd = its distinct-index content pushed to the next plane
    def __init__(self, W, u, om):
        self.W, self.u, self.om = W, u, om; self.U = W.T @ u; self.Om = W.T @ om
    def fwd(self, lam):
        X = (self.U * self.U) @ (lam[:, None] * self.Om.T) + 2 * (self.U * self.Om) @ (lam[:, None] * self.U.T)
        plane = (self.u * self.u) @ (lam[:, None] * self.om.T) + 2 * (self.u * self.om) @ (lam[:, None] * self.u.T)
        return X - L.Pi(self.W, plane)


for l in (int(x) for x in sys.argv[2:]):
    t0 = time.time(); W = WALL[l + 1].astype(np.float64)
    mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); sd = np.sqrt(v); al = mu / sd
    pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a_ / math.sqrt(2)) / 2 for a_ in al]); d = pdf / sd
    COL = np.load(HERE / f't4cols_rank{rank:04d}_L{l}.npz'); pp = COL['pp'].astype(np.float64); lam3 = COL['lam'].astype(np.float64); alive = COL['alive'].astype(np.float64)
    stops = [0] + [int(x) for x in COL['stops']]; Gm = (alive * g)[:, None]; dd = (d / 2)[:, None]; G = g[:, None] * W
    blk = BP[f'blk_L{l}']; F = [BP[f't_F4_L{l}_b{b}'].astype(np.float64)[blk] for b in (0, 1)]; FF = float(F[0] @ F[1])
    def second_chaos(b):
        q = lambda k: np.asarray(Z[f'{k}_L{l}_b{b}'], dtype=np.float64)
        m1 = q('S1'); M11 = q('M11'); Cb = M11 - np.outer(m1, m1); return q('T') - M11 * float(q('S')) - 2 * Cb @ Cb
    def push_M(M, om):
        MG = G.T @ M @ G; full = np.outer(np.diag(MG), (om * d / 2) @ W) + MG * ((om * d) @ W)[:, None]
        return (full - L.Pi(W, L.feed_plane(d, g, np.outer(np.diag(M), om), M * om[:, None], np.diag(M) * om)))[blk]
    X = {'iso': [(L.feed_push_model(W, d, g, C, 1.0, 0.0) - L.Pi(W, L.feed_plane_model(d, g, C, 1.0, 0.0)))[blk]] * 2}
    X['N'] = [push_M(second_chaos(b), v / v.sum()) for b in (0, 1)]
    # hub-local terms, all cohorts (E15b's code)
    loc = 0.0
    for j in range(l):
        bq = lambda k: BI[f'{k}_{j}']
        s_ = bq('sel'); a, s, gt, ph, m = (bq(k)[s_].astype(np.float64) for k in ('alpha', 'sigma', 'gate', 'pdf', 'host')); tp, tm = (bq(k)[s_].astype(np.float64) for k in ('plus', 'minus'))
        vv = s * s; dj = ph / s; r2 = vv * ((1 + a * a) * gt + a * ph); r3 = vv * s * ((a ** 3 + 3 * a) * gt + (a * a + 2) * ph); r4 = vv * vv * ((a ** 4 + 6 * a * a + 3) * gt + (a ** 3 + 5 * a) * ph)
        S1, S2, S3, S4 = (x[None, :] for x in (-a * ph / vv, 2 * (gt - gt * gt - m * dj), 3 * (1 - gt) * (r2 - 2 * m * m), r4 - 4 * m * r3 + 6 * m * m * r2 - 3 * m ** 4 - 3 * (r2 - m * m) ** 2))
        k0 = stops[j]; Kj = (stops[j + 1] - k0) // 2; pplus = pp[:, k0:k0 + Kj]; pminus = pp[:, k0 + Kj:k0 + 2 * Kj]; gap = (tp - tm)[None, :]
        Wt = (pplus - pminus) / gap; Y = (tp[None, :] * pminus - tm[None, :] * pplus) / gap
        aa = Gm * Y; bb = Gm * Wt; one = np.ones(Kj); yy, yw, ww = dd * Y * Y, dd * Y * Wt, dd * Wt * Wt
        h1 = 2 * S1 * yw + S2 * ww; h2 = S1 * yy + 2 * S2 * yw + S3 * ww; h3 = S2 * yy + 2 * S3 * yw + S4 * ww
        loc = loc + Op(W, np.concatenate((aa, aa + bb, aa - bb, bb), 1), np.concatenate((h1, h2, h2, h3), 1)).fwd(np.concatenate((one, one / 2, -one / 2, one)))[blk]
    X['LOCAL'] = [loc, loc]
    u = Gm * pp; om = dd * pp * pp; opA = Op(W, u, om)
    pre = O / f'e15cols_rank{rank:04d}_L{l - 1}.npz'
    if pre.exists():
        PRE = np.load(pre); ppm = PRE['pp'].astype(np.float64); Cm = MC[f'cov_pre_{l - 1}'].astype(np.float64); sel = BI[f'sel_{l - 1}']
        V21 = 2 * BI[f'host_{l - 1}'].astype(np.float64) * (1 - BI[f'gate_{l - 1}'].astype(np.float64)); ung = np.concatenate((ppm, Cm[:, sel], Cm[:, sel]), 1)
        c = (2.0 / n) * (V21[:, None] * ung).sum(0); coll = Op(W, u, dd * np.ones_like(pp)).fwd(lam3 * c)[blk]; X['COLL'] = [coll, coll]
    fw = HERE / f't4_lam4_rank{rank:04d}_L{l}_none.npz'
    if fw.exists(): X['FREE'] = [opA.fwd(np.load(fw)['lam4'].astype(np.float64))[blk]] * 2
    def share(keys, only_b=None):
        sc = []; cs_ = []
        for b in ((0, 1) if only_b is None else (only_b,)):                  # fitted to truth block b with the predictors of block 1-b; scored on truth block 1-b with the predictors of block b
            Xf = np.stack([X[k][1 - b] for k in keys], 1); co = np.linalg.lstsq(Xf, F[b], rcond=None)[0]
            f_s = np.stack([X[k][b] for k in keys], 1) @ co; f_o = Xf @ co
            sc.append((2 * float(f_s @ F[1 - b]) - float(f_s @ f_o)) / FF); cs_.append(co)
        return round(float(np.mean(sc)), 4), [round(float(x), 4) for x in np.mean(cs_, 0)]
    out = dict(rank=rank, layer=l, columns=int(pp.shape[1]), wall=None,
               norms_pct_of_feed={k: round(100 * float(np.linalg.norm(x[0])) / math.sqrt(FF), 1) for k, x in X.items()},
               cos_N_LOCAL=round(float(X['N'][0] @ loc) / float(np.linalg.norm(X['N'][0]) * np.linalg.norm(loc)), 3), cos_iso_N=round(float(X['N'][0] @ X['iso'][0]) / float(np.linalg.norm(X['N'][0]) * np.linalg.norm(X['iso'][0])), 3))
    res = {'iso': share(['iso']), 'iso + LOCAL': share(['iso', 'LOCAL']), 'N': share(['N']), 'N + LOCAL': share(['N', 'LOCAL']), 'iso + N + LOCAL': share(['iso', 'N', 'LOCAL'])}
    if 'COLL' in X: res.update({'iso + LOCAL + COLL': share(['iso', 'LOCAL', 'COLL']), 'N + LOCAL + COLL': share(['N', 'LOCAL', 'COLL'])})
    if 'FREE' in X:
        res.update({'FREE alone (block 0 fit on block 1)': share(['FREE'], only_b=0), 'FREE + iso': share(['FREE', 'iso'], only_b=0), 'FREE + N': share(['FREE', 'N'], only_b=0),
                    'FREE + N + LOCAL': share(['FREE', 'N', 'LOCAL'], only_b=0), 'N + LOCAL (same single pairing)': share(['N', 'LOCAL'], only_b=0)})
        out['cos_FREE_N'] = round(float(X['FREE'][0] @ X['N'][0]) / float(np.linalg.norm(X['FREE'][0]) * np.linalg.norm(X['N'][0])), 3)
    out['feed_share_explained'] = res; out['wall'] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)
