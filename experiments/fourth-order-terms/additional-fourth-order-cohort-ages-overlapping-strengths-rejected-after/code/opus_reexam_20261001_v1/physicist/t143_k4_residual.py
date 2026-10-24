#!/usr/bin/env python3
"""TAG opus-physicist T143 analysis (lead 10-13 11:1xZ): what per-unit k4 is still missing after C.
Truth: 32M per-unit MC, two independent draws (analyst/fresh/pu/mps_truth_rank*_A32M / B32M: mc_pre_m, mc_pre_v, mc_pre_k4); per-unit excess kurtosis
t = k4 / v^2 (mean of the two draws; draw noise = half their difference). Card: physicist/t143_k4dump/<run>/dump_L<l>.npz (otr_m_ph143: the per-unit
k4 the card uses, v, the K4L hub sum of every cohort age at unit strength, joined flags, (Cpre^2)_ii). Per network and layer L8-15:
  (1) corr(card k4 / v^2, truth) WITHOUT C (noC = s51k4lf34h) vs WITH C, and the k4 error weighted as the mean formula weights k4,
      w = (sigma phi(alpha) |alpha^2 - 1|)^2;
  (2) the residual r = truth - card(C), regressed on the sources the card leaves out (each over v^2): G5 = the young unjoined cohorts aged >= 5,
      GJ = the joined (old-tier) cohorts, C2 = (Cpre^2)_ii (L8-14), plus an intercept; share of the weighted residual energy explained, in sample
      (per network-layer) and leave-one-network-out (coefficients fitted per layer on the other networks).
usage: t143_k4_residual.py"""
import json
from pathlib import Path
import numpy as np
from scipy.stats import norm
HERE = Path(__file__).resolve().parent; O = HERE.parent; R = HERE / 't143_k4dump'
RANKS = (16, 35, 150, 200, 250); CARD_AGES = 4      # the card (C) uses K4L ages 1-4 at L8-15


def truth(rank):
    A = np.load(O / ('analyst/fresh/pu/mps_truth_rank%04d_A32M.npz' % rank)); B = np.load(O / ('analyst/fresh/pu/mps_truth_rank%04d_B32M.npz' % rank))
    return A, B


def card(rank, cfg, L):
    f = R / ('r%04d_%s' % (rank, cfg)) / ('dump_L%d.npz' % L)
    return np.load(f) if f.exists() else None


rows = {}
for rank in RANKS:
    A, B = truth(rank)
    for L in range(8, 16):
        d = card(rank, 'C', L)
        if d is None:
            continue
        tA = A['mc_pre_k4'][L] / A['mc_pre_v'][L] ** 2; tB = B['mc_pre_k4'][L] / B['mc_pre_v'][L] ** 2; t = (tA + tB) / 2
        tv = (A['mc_pre_v'][L] + B['mc_pre_v'][L]) / 2; tm = (A['mc_pre_m'][L] + B['mc_pre_m'][L]) / 2
        al = tm / np.sqrt(tv); w = (np.sqrt(tv) * norm.pdf(al) * np.abs(al ** 2 - 1)) ** 2
        v = d['v']; c = d['k4_card'] / v ** 2
        ages, joined, parts = d['ages'], d['joined'], d['parts']
        G5 = sum((parts[i] for i, a in enumerate(ages) if a > CARD_AGES and not joined[i]), np.zeros(1024)) / v ** 2
        GJ = sum((parts[i] for i, a in enumerate(ages) if joined[i]), np.zeros(1024)) / v ** 2
        C2 = d['c2'] / v ** 2 if 'c2' in d.files else np.zeros(1024)
        dn = card(rank, 'noC', L)
        cn = dn['k4_card'] / dn['v'] ** 2 if dn is not None else None
        rows[(rank, L)] = dict(t=t, noise=(tA - tB) / 2, w=w, c=c, cn=cn, r=t - c, G5=G5, GJ=GJ, C2=C2,
                               n5=int(sum(1 for i, a in enumerate(ages) if a > CARD_AGES and not joined[i])), nj=int(joined.sum()))

wsum = lambda D, x: float((D['w'] * x * x).sum())
print('(1) per-unit excess kurtosis vs the 32M truth: corr without C / with C; weighted k4 error with C as a share of without C (16 and 35)')
for (rank, L), D in sorted(rows.items()):
    line = 'rank %3d L%2d: truth mean %.4f sd %.4f (draw noise sd %.4f) | card(C) mean %.4f corr %.3f' % (
        rank, L, D['t'].mean(), D['t'].std(), D['noise'].std(), D['c'].mean(), np.corrcoef(D['c'], D['t'])[0, 1])
    if D['cn'] is not None:
        line += ' | noC corr %.3f | weighted error C / noC %.3f' % (np.corrcoef(D['cn'], D['t'])[0, 1], wsum(D, D['t'] - D['c']) / wsum(D, D['t'] - D['cn']))
    line += ' | young ages > 4 present: %d, joined ages: %d' % (D['n5'], D['nj'])
    print(line)


def design(D, keys):
    return np.column_stack([np.ones(1024)] + [D[k] for k in keys])


print('\n(2) residual (truth - card with C) explained, weighted energy share: in sample (per network-layer) | leave-one-network-out (per-layer fit on the others)')
for keys in (('G5',), ('GJ',), ('C2',), ('G5', 'GJ'), ('G5', 'GJ', 'C2')):
    ins, loo = [], []
    for (rank, L), D in sorted(rows.items()):
        X = design(D, keys); sw = np.sqrt(D['w'])
        b = np.linalg.lstsq(X * sw[:, None], D['r'] * sw, rcond=None)[0]
        ins.append(1 - wsum(D, D['r'] - X @ b) / wsum(D, D['r']))
        others = [(o, Lo) for (o, Lo) in rows if Lo == L and o != rank]
        if others:
            Xo = np.vstack([design(rows[k], keys) * np.sqrt(rows[k]['w'])[:, None] for k in others])
            yo = np.concatenate([rows[k]['r'] * np.sqrt(rows[k]['w']) for k in others])
            bo = np.linalg.lstsq(Xo, yo, rcond=None)[0]
            loo.append(1 - wsum(D, D['r'] - X @ bo) / wsum(D, D['r']))
    print('  %-12s in sample: mean %+.3f (min %+.3f, max %+.3f) | leave-one-network-out: mean %+.3f (min %+.3f, max %+.3f) over %d network-layers' % (
        '+'.join(keys), np.mean(ins), np.min(ins), np.max(ins), np.mean(loo) if loo else float('nan'), np.min(loo) if loo else float('nan'),
        np.max(loo) if loo else float('nan'), len(ins)))
