#!/usr/bin/env python3
"""TAG opus-physicist T128 analysis (lead 15:1xZ idea sprint, candidate (a)): is the true per-unit excess kurtosis at layers 13-15 explained by how each unit's
variance splits over the leading modes of the card's pre-activation covariance (a multi-radial scale-mixture model: k4_i = sum_m r_m (lam_m u_mi^2)^2 + r_0 v_i^2),
instead of the card's single radial (k4_i = radial v_i^2)? Card state from t128_worker_cdump.py (post covariance entering layer l), truths = 32M per-unit
MC (analyst/fresh/pu/mps_truth_rank*_A32M / B32M: mc_pre_v, mc_pre_k4). Reports per network / layer: true kurtosis mean / sd and its A/B noise, the card's
radial, R^2 of the mode-share regression (in sample, and fitted on the other network), and the k4-error energy weighted the way the mean formula weights k4
((sigma phi(alpha) |alpha^2 - 1|)^2) left after the fit. usage: t128_kurt_modes.py"""
import importlib.util
import numpy as np
from pathlib import Path
from scipy.stats import norm
HERE = Path(__file__).resolve().parent; O = HERE.parent
spec = importlib.util.spec_from_file_location('oc', O / 'trainer/otr_common.py'); oc = importlib.util.module_from_spec(spec); spec.loader.exec_module(oc)
K = 8
data = {}
for rank in (35, 16):
    W = np.load(oc.data_paths(rank)[0], mmap_mode='r')
    A = np.load(O / ('analyst/fresh/pu/mps_truth_rank%04d_A32M.npz' % rank)); B = np.load(O / ('analyst/fresh/pu/mps_truth_rank%04d_B32M.npz' % rank))
    for L in (13, 14, 15):
        d = np.load(HERE / 't128_cdump' / ('t128_C_%d_L%d.npz' % (rank, L)))
        w = np.asarray(W[L], dtype=np.float64); Cp = w.T @ d['C'] @ w; v = np.diag(Cp).copy()
        mu = w.T @ d['mu'] if d['mu'].shape[0] == 1024 else None
        tv = (A['mc_pre_v'][L] + B['mc_pre_v'][L]) / 2
        kA, kB = A['mc_pre_k4'][L] / A['mc_pre_v'][L] ** 2, B['mc_pre_k4'][L] / B['mc_pre_v'][L] ** 2
        kt = (kA + kB) / 2
        ev, U = np.linalg.eigh(Cp); ev, U = ev[::-1], U[:, ::-1]
        F = (U[:, :K] ** 2 * ev[None, :K]) / v[:, None]                     # share of each unit's variance in mode m
        X = np.column_stack([np.ones(1024), F ** 2, F])
        tm = (A['mc_pre_m'][L] + B['mc_pre_m'][L]) / 2
        al = tm / np.sqrt(tv); wt = (np.sqrt(tv) * norm.pdf(al) * np.abs(al ** 2 - 1)) ** 2
        data[(rank, L)] = dict(X=X, kt=kt, noise=(kA - kB) / 2, radial=float(d['radial']), wt=wt, vlog=np.log(v / tv), top=ev[:3] / ev.sum())
for (rank, L), D in sorted(data.items()):
    X, kt, wt = D['X'], D['kt'], D['wt']
    b, *_ = np.linalg.lstsq(X, kt, rcond=None); r = kt - X @ b
    other = data.get((16 if rank == 35 else 35, L))
    bo, *_ = np.linalg.lstsq(other['X'], other['kt'], rcond=None); ro = kt - X @ bo
    e_card = (wt * (kt - D['radial']) ** 2).sum()
    print('rank %d L%d: true kurt mean %.4f sd %.4f (A/B noise sd %.4f) | card radial %.4f | v log err sd %.5f | top-3 mode shares of trace %s | R^2 in-sample %.3f, '
          'fit-on-other %.3f | weighted k4 error left: in-sample fit %.1f%%, other-network fit %.1f%%, constant-at-true-mean %.1f%%' % (
        rank, L, kt.mean(), kt.std(), D['noise'].std(), D['radial'], D['vlog'].std(), np.round(D['top'], 3), 1 - r.var() / kt.var(), 1 - ro.var() / kt.var(),
        100 * (wt * r ** 2).sum() / e_card, 100 * (wt * ro ** 2).sum() / e_card, 100 * (wt * (kt - kt.mean()) ** 2).sum() / e_card))
