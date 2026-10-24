#!/usr/bin/env python3
"""TAG opus-physicist T19d (numpy): is the second-chaos matrix, beyond its isotropic part, diagonal plus low rank?  (lead's item 3)
Ntilde = a tr C x C + b C^2 (best isotropic fit) + D. Per layer, from the two 8M blocks of t19mc:
  - size of D against Ntilde, block-to-block agreement of D, share of D's energy on its diagonal;
  - eigen-decomposition of D of one block; a rank-r piece is scored on the OTHER block: energy kept <P_r(D_0), D_1> / <D_0, D_1>;
  - what a carrier of rank r would deliver: K31 R2 on the kink rows at 3/n, the feed share (with the hub-local terms, fitted) and the
    per-unit kurtosis share at 6/n, for Ntilde_r = isotropic fit + diagonal of D (optional) + the rank-r piece of D (off-diagonal).
usage: t19d_spectrum.py <rank> <layer> [<layer> ...]"""
import sys, json, math
from pathlib import Path
import numpy as np
import pcard_light as pl
import t2_lib as L
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; n = 1024; eye = np.eye(n, dtype=bool)
rank = int(sys.argv[1]); Z = np.load(HERE / f't19mc_rank{rank:04d}.npz'); MCF = np.load(HERE / f't2mc_rank{rank:04d}.npz'); MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz')
BP = np.load(HERE / f't2_budget_rank{rank:04d}_planes.npz'); A, B = pl.pu(rank, 'P32M'), pl.pu(rank, 'indep32M')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
WALL = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r'); RANKS = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 1024)
for l in (int(x) for x in sys.argv[2:]):
    mu = MC[f'mean_pre_{l}'].astype(np.float64); C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); sd = np.sqrt(v); al = mu / sd; C2 = C @ C
    Nt = []
    for b in (0, 1):
        q = lambda k: np.asarray(Z[f'{k}_L{l}_b{b}'], dtype=np.float64)
        m1 = q('S1'); M11 = q('M11'); Cb = M11 - np.outer(m1, m1); x = q('T') - M11 * float(q('S')) - 2 * Cb @ Cb; Nt.append((x + x.T) / 2)
    Xi = np.stack([C.ravel() * np.trace(C), C2.ravel()], 1); co = np.linalg.lstsq(Xi, ((Nt[0] + Nt[1]) / 2).ravel(), rcond=None)[0]; iso = co[0] * np.trace(C) * C + co[1] * C2
    D = [x - iso for x in Nt]; DD = float((D[0] * D[1]).sum()); NN = float((Nt[0] * Nt[1]).sum())
    out = dict(rank=rank, layer=l, isotropic_fit=dict(a_times_3=round(3 * float(co[0]), 5), b_times_3_over_2=round(1.5 * float(co[1]), 5)),
               D_energy_share_of_Ntilde=round(DD / NN, 4), D_block_agreement=round(DD / math.sqrt(float((D[0] ** 2).sum()) * float((D[1] ** 2).sum())), 4),
               D_energy_on_diagonal=round(float((np.diag(D[0]) * np.diag(D[1])).sum()) / DD, 4),
               top_eigvec_of_D_vs_mean_direction=None)
    E = []
    for b in (0, 1):
        Do = D[b] - np.diag(np.diag(D[b])); w_, V = np.linalg.eigh(Do); o_ = np.argsort(-np.abs(w_)); E.append((w_[o_], V[:, o_], np.diag(D[b]).copy()))
    out['top_eigvec_of_D_vs_mean_direction'] = [round(abs(float(E[0][1][:, k] @ mu)) / float(np.linalg.norm(mu)), 3) for k in range(3)]
    out['top_eigenvalues_over_rms_eigenvalue'] = [round(float(x) / float(np.sqrt((E[0][0] ** 2).mean())), 2) for x in E[0][0][:8]]
    def piece(b, r, diag=True):       # isotropic fit + diagonal of D + rank-r part of the off-diagonal D, all from block b
        w_, V, dg = E[b]; X = (V[:, :r] * w_[None, :r]) @ V[:, :r].T if r else np.zeros((n, n))
        return iso + X + (np.diag(dg) if diag else 0.0)
    Do = [D[b] - np.diag(np.diag(D[b])) for b in (0, 1)]; DDo = float((Do[0] * Do[1]).sum())
    out['offdiag_energy_kept_by_rank_r'] = {r: round(float((((E[0][1][:, :r] * E[0][0][None, :r]) @ E[0][1][:, :r].T) * Do[1]).sum()) / DDo, 4) for r in RANKS if r}
    # K31
    if f'M31_L{l}_b0' in MCF.files:
        K31 = []
        for b in (0, 1):
            q = lambda k_: np.asarray(MCF[f'{k_}_L{l}_b{b}'], dtype=np.float64)
            m1 = q('S1_z'); Cs = q('M11_z') - np.outer(m1, m1); K31.append(q('M31') - 3 * np.diag(Cs)[:, None] * Cs)
        m = (np.abs(al) < 1.5)[:, None] & ~eye
        r2 = lambda Mb: round(float(np.mean([1 - float(((K31[1 - b][m] - (3.0 / n) * Mb[b][m]) ** 2).sum()) / float((K31[1 - b][m] ** 2).sum()) for b in (0, 1)])), 4)
        out['K31_R2_at_3_over_n'] = {'rank %d' % r: r2([piece(b, r) for b in (0, 1)]) for r in RANKS}; out['K31_R2_at_3_over_n']['full matrix'] = r2(Nt)
    # per unit
    kA, kB = A['mc_pre_k4'][l].astype(np.float64), B['mc_pre_k4'][l].astype(np.float64); vt = B['mc_pre_v'][l].astype(np.float64); at = B['mc_pre_m'][l].astype(np.float64) / np.sqrt(vt)
    w = ((at * at - 1) * np.exp(-at * at / 2)) ** 2 + 1e-12; sw = np.sqrt(w); noise = float((w * (kA - kB) ** 2).sum()) / 2; fam = np.stack([np.ones(n), vt, vt ** 2], 1)
    def unit_share(Qb):
        r = [];
        for y in [kB] * 2 + [kB - (6.0 / n) * Qb[0], kB - (6.0 / n) * Qb[1]]:
            c = np.linalg.lstsq(fam * sw[:, None], y * sw, rcond=None)[0]; r.append(y - fam @ c)
        return round(1 - (float((w * r[2] * r[3]).sum()) - noise) / (float((w * r[0] * r[1]).sum()) - noise), 4)
    out['per_unit_k4_share_at_6_over_n'] = {'diagonal of D only (rank 0)': unit_share([np.diag(piece(b, 0)) - kA for b in (0, 1)]), 'full': unit_share([np.diag(Nt[b]) - kA for b in (0, 1)]),
                                             'isotropic fit alone': unit_share([np.diag(iso) - kA for b in (0, 1)])}
    # feed (with the isotropic C_ij v_k form fitted beside it, as the carrier would sit beside the card's own isotropic term)
    if f't_F4_L{l}_b0' in BP.files:
        W = WALL[l + 1].astype(np.float64); pdf = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = np.array([math.erfc(-a_ / math.sqrt(2)) / 2 for a_ in al]); d = pdf / sd; G = g[:, None] * W
        blk = BP[f'blk_L{l}']; F = [BP[f't_F4_L{l}_b{b}'].astype(np.float64)[blk] for b in (0, 1)]; FF = float(F[0] @ F[1]); om = v / v.sum()
        def push_M(M):
            MG = G.T @ M @ G; full = np.outer(np.diag(MG), (om * d / 2) @ W) + MG * ((om * d) @ W)[:, None]
            return (full - L.Pi(W, L.feed_plane(d, g, np.outer(np.diag(M), om), M * om[:, None], np.diag(M) * om)))[blk]
        def share(Mb):
            Xb = [push_M(Mb[b]) for b in (0, 1)]; return round(float(np.mean([(2 * float(Xb[b] @ F[1 - b]) - float(Xb[b] @ Xb[1 - b])) / FF for b in (0, 1)])), 4)
        out['feed_share_nothing_fitted'] = {'rank %d' % r: share([piece(b, r) for b in (0, 1)]) for r in RANKS if r in (0, 1, 4, 16, 64, 256, 1024)}
        out['feed_share_nothing_fitted']['full matrix'] = share(Nt); out['feed_share_nothing_fitted']['isotropic fit, no diagonal of D'] = share([iso, iso])
    print(json.dumps(out), flush=True)
