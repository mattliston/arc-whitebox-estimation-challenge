#!/usr/bin/env python3
"""TAG opus-physicist T2c: how OLD is the fourth-order content that feeds births and the pair map? (t2_mc_restart.py against t2_mc.py)
For target layer l and restart depth k (Gaussian with the true mean and covariance at layer l-k, true network for k layers) the script
compares, with the truth at layer l (two independent blocks each, cross-block products):
   feed   the pushed distinct fourth-cumulant feed D_F4 on the kink block of l+1: restart norm / truth norm, cosine, and the share of the
          truth's energy explained by the restart at coefficient 1 and at the best coefficient; the same after removing from both the
          isotropic form (c22 C_ij v_c + c11 C_ic C_jc fitted to each)
   K31    the pre plane K31[a,c] in the covariance-read weighting |alpha_a| pdf_a g_c / v_a (rows a, off-diagonal), same numbers, raw
          and after removing the isotropic form (k4_a / v_a) C_ac fitted per plane
   K21    the pre (2,1) plane on the kink block of l ; k3, k4 per unit
usage: t2_restart_analysis.py <rank>"""
import sys, json, math
from pathlib import Path
import numpy as np
import t2_lib as L
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent
rank = int(sys.argv[1])
R = np.load(HERE / f't2mcR_rank{rank:04d}.npz'); T = np.load(HERE / f't2mc_rank{rank:04d}.npz'); CAP = np.load(HERE / f't2cap_rank{rank:04d}.npz')
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); BP = np.load(HERE / f't2_budget_rank{rank:04d}_planes.npz')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
WALL = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r')
N = 1024; eye = np.eye(N, dtype=bool); nrm = np.linalg.norm; f8 = lambda x: np.asarray(x, dtype=np.float64); erfc = np.vectorize(math.erfc)
print(json.dumps(dict(TAG='opus-physicist T2c restart analysis', rank=rank, restart_counts=[int(x) for x in R['counts']], truth_counts=[int(x) for x in T['counts']])))


def x2(A, B, m):
    return (float((A[0][m] * B[1][m]).sum()) + float((A[1][m] * B[0][m]).sum())) / 2


def cmp(Rs, Ts, m, label):
    rr, tt, rt = x2(Rs, Rs, m), x2(Ts, Ts, m), (float((Rs[0][m] * Ts[0][m]).sum()) + float((Rs[0][m] * Ts[1][m]).sum()) + float((Rs[1][m] * Ts[0][m]).sum()) + float((Rs[1][m] * Ts[1][m]).sum())) / 4
    rr, tt = max(rr, 1e-300), max(tt, 1e-300)
    return {label + '_norm_ratio': math.sqrt(rr / tt), label + '_cos': rt / math.sqrt(rr * tt), label + '_explained_at_1': 1 - (tt - 2 * rt + rr) / tt,
            label + '_explained_best': rt * rt / (rr * tt)}


for l in [int(x) for x in R['layers']]:
    Lx = l + 1; W = f8(WALL[Lx]); mu = f8(MC[f'mean_pre_{l}']); C = f8(MC[f'cov_pre_{l}']); v = np.diag(C).copy(); s = np.sqrt(v); al = mu / s
    pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = erfc(-al / math.sqrt(2)) / 2; d = pdf / s
    G = g[:, None] * W; covy = G.T @ C @ G
    blk = BP[f'blk_L{l}']; respl = CAP[f'resp_{l}'] > .5; alivel = CAP[f'alive_{l}'] > .5; blkl = np.outer(respl, alivel) & ~eye
    w31 = (np.abs(al) * pdf / v)[:, None] * g[None, :] * (~eye)          # covariance-read weight of K31[a, c]
    Fiso = [L.feed_push_model(W, d, g, C, 1.0, 0.0) - L.Pi(W, L.feed_plane_model(d, g, C, 1.0, 0.0)),
            L.feed_push_model(W, d, g, C, 0.0, 1.0) - L.Pi(W, L.feed_plane_model(d, g, C, 0.0, 1.0))]
    Xiso = np.stack([Fiso[0][blk], Fiso[1][blk]], 1)

    def stats(src, tag):
        out = []
        for b in (0, 1):
            q = lambda k_: f8(src[f'{k_}_{tag}_b{b}'])
            m1 = q('S1_z'); Cs = q('M11_z') - np.outer(m1, m1)
            K22 = q('M22') - np.outer(q('S2_z'), q('S2_z')) - 2 * Cs * Cs; K31 = q('M31') - 3 * q('S2_z')[:, None] * Cs
            k4 = np.diag(q('M22')) - 3 * q('S2_z') ** 2
            K21 = L.k21_from_sums(q('M21_z'), q('S1_z'), q('S2_z'), C)
            F = L.feed_push(W, d, q('A1'), q('A2'), q('A3'), q('S2_yl'), q('S2_z'), q('Sq'), covy) - L.Pi(W, L.feed_plane(d, g, K22, K31, k4))
            co = np.linalg.lstsq(Xiso, F[blk], rcond=None)[0]; Fres = F - co[0] * Fiso[0] - co[1] * Fiso[1]
            iso31 = (k4 / q('S2_z'))[:, None] * Cs; a31 = float((K31 * iso31 * w31 ** 2).sum() / (iso31 * iso31 * w31 ** 2).sum())
            out.append(dict(F=F, Fres=Fres, c22=float(co[0]), c11=float(co[1]), K31w=K31 * w31, K31resw=(K31 - a31 * iso31) * w31, a31=a31, K21=K21,
                            k3=np.diag(K21).copy(), k4=k4, radial=float(np.mean(k4 / v ** 2)), vz=q('S2_z'), mres=float(np.sqrt(np.mean((m1 / s) ** 2)))))
        return out
    Tr = stats(T, f'L{l}'); allm = np.ones((N, N), dtype=bool); am = alivel
    print(json.dumps(dict(layer=l, truth=dict(radial=[t['radial'] for t in Tr], c22=[t['c22'] for t in Tr], c11=[t['c11'] for t in Tr], a31=[t['a31'] for t in Tr],
                                              feed_noniso_share=x2([t['Fres'] for t in Tr], [t['Fres'] for t in Tr], blk) / x2([t['F'] for t in Tr], [t['F'] for t in Tr], blk),
                                              K31_noniso_share=x2([t['K31resw'] for t in Tr], [t['K31resw'] for t in Tr], allm) / x2([t['K31w'] for t in Tr], [t['K31w'] for t in Tr], allm)))), flush=True)
    for k in [int(x) for x in R['depths']]:
        Rr = stats(R, f'L{l}_k{k}')
        rec = dict(layer=l, restart_depth=k, radial=[r['radial'] for r in Rr], c22=[r['c22'] for r in Rr], c11=[r['c11'] for r in Rr], a31=[r['a31'] for r in Rr],
                   residual_mean_over_sigma=[r['mres'] for r in Rr], var_ratio=float(np.mean(Rr[0]['vz'] / Tr[0]['vz'])))
        rec.update(cmp([r['F'] for r in Rr], [t['F'] for t in Tr], blk, 'feed'))
        rec.update(cmp([r['Fres'] for r in Rr], [t['Fres'] for t in Tr], blk, 'feed_noniso'))
        rec.update(cmp([r['K31w'] for r in Rr], [t['K31w'] for t in Tr], allm, 'K31'))
        rec.update(cmp([r['K31resw'] for r in Rr], [t['K31resw'] for t in Tr], allm, 'K31_noniso'))
        rec.update(cmp([r['K21'] for r in Rr], [t['K21'] for t in Tr], blkl, 'K21'))
        rec.update(cmp([r['k3'][:, None] for r in Rr], [t['k3'][:, None] for t in Tr], am[:, None], 'k3'))
        rec.update(cmp([(r['k4'] - r['radial'] * v * v)[:, None] for r in Rr], [(t['k4'] - t['radial'] * v * v)[:, None] for t in Tr], am[:, None], 'k4_spread'))
        print(json.dumps({a: (round(b_, 4) if isinstance(b_, float) else b_) for a, b_ in rec.items()}), flush=True)
