#!/usr/bin/env python3
"""TAG opus-physicist T111 (numpy): THE CARD'S K21 ERROR BY AGE OF THE CONTENT (= by carrier: ages 1-3 the young cohorts, 4+ the old tier after its joins)
at S47e (lead 10-11 18:4xZ), network 35, layers 10-12, K21 as the covariance uses it (rows pdf(alpha_i) / 2, columns g_j, per sigma_i^2 sigma_j).
Truth: the 8M-sample planes (two draws); age increments from the Gaussian-restart planes of T24 (t24_rank0035_r<k>_s0.npz, two draws each): I_d = what layer
l-d made. Card: the pair map's own K21 input at S47e (physicist/t99_pindump_v2 dump: the Nd point on the v11 simulator, pinned otr_m) with today's
covariance-output knobs at 10-14 (iso -0.064 on kink rows, k21n -0.048 off them; T28's form). Same arithmetic as t28_age_profile.py (eff), plus the
error's own fit: the share of |truth - card|^2 that the best per-age multiples of the increments remove (= the part of the error that is a per-carrier scale).
usage: t111_age_error_s47e.py"""
import math, os
os.environ.setdefault('OMP_NUM_THREADS', '2')
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; P2 = HERE.parent.parent; rank = 35; n = 1024; eye = np.eye(n, dtype=bool); dot = lambda x, y: float((x * y).sum())
T1 = np.load(P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}.npz'); p2 = P2 / f'public_g19_attribution_v1/mps_planes_rank{rank:04d}_noise2.npz'; T2 = np.load(p2) if p2.exists() else None
MC = np.load(P2 / f'fable_covhunter_v1/mcall_rank{rank:04d}_P32M.npz'); D = np.load(HERE / 't99_pindump_v2' / 'dumps' / f'P_{rank}.npz')
for l in (10, 11, 12):
    C = MC[f'cov_pre_{l}'].astype(np.float64); v = np.diag(C).copy(); s = np.sqrt(v); al = MC[f'mean_pre_{l}'].astype(np.float64) / s
    S21 = v[:, None] * s[None, :]; ph = np.exp(-al * al / 2) / math.sqrt(2 * math.pi); gt = np.array([0.5 * math.erfc(-a / math.sqrt(2)) for a in al])
    q = lambda X: np.where(eye, 0.0, X.astype(np.float64) / S21) * ph[:, None] * gt[None, :] / 2
    kc = D[f'K21_{l}'].astype(np.float64); Cc = D[f'C_{l}'].astype(np.float64); vc = np.diag(Cc).clip(1e-12); kink = (np.abs(D[f'mu_{l}'] / np.sqrt(vc)) < 1.5).astype(np.float64)
    card = q(kc + ((-0.064 * kink) * np.diag(kc) / vc)[:, None] * Cc + (-0.048 * (1 - kink))[:, None] * kc)
    t = [q(T1[f'K21_{l}'])] + ([q(T2[f'K21_{l}'])] if (T2 is not None and f'K21_{l}' in T2.files) else []); tm = sum(t) / len(t); tt = dot(t[0], t[-1])
    E = tm - card; ee = dot(E, E)
    print('\nlayer %d net %d, K21 as the covariance uses it: card holds share %.4f of the truth; error energy / truth %.4f' % (l, rank, (2 * dot(card, tm) - dot(card, card)) / tt, ee / tt))
    print('   age | increment energy / truth | card / truth on it | share of the ERROR energy its best multiple removes')
    prev = [np.zeros_like(tm), np.zeros_like(tm)]; INC = []
    for d_ in range(1, l):
        f = HERE / f't24_rank{rank:04d}_r{l - d_}_s0.npz'
        if not f.exists(): break
        Z = np.load(f)
        if f'K21_L{l}_b0' not in Z.files: break
        cur = [q(Z[f'K21_L{l}_b{b}']) for b in (0, 1)]; inc = [cur[0] - prev[0], cur[1] - prev[1]]; ii = dot(inc[0], inc[1]); im = (inc[0] + inc[1]) / 2
        ct, cc = dot(im, tm) / ii, dot(im, card) / ii
        print('   %3d | %8.4f | %6.3f | %6.4f' % (d_, ii / tt, cc / ct, dot(im, E) ** 2 / (ii * ee))); prev = cur; INC.append(inc)
    K = len(INC); G = np.zeros((K, K)); re = np.zeros(K)
    for a in range(K):
        im = (INC[a][0] + INC[a][1]) / 2; re[a] = dot(im, E)
        for b in range(K): G[a, b] = (dot(INC[a][0], INC[b][1]) + dot(INC[a][1], INC[b][0])) / 2
    ke = np.linalg.solve(G, re)
    young = sum(ke[a] * sum(G[a, b] * ke[b] for b in range(K)) for a in range(min(3, K)))
    print('   joint per-age fit of the ERROR: removes %.1f%% of its energy (ages 1-3 alone in that fit %.1f%%); the rest is content inside each age the card misplaces' % (
        100 * (ke @ re) / ee, 100 * young / ee))
