#!/usr/bin/env python3
"""TAG opus-physicist T128b (idea sprint, candidate (b)): does the card's per-unit variance error at layer 15 (log card / 32M truth) have LOW-RANK structure in the
eigenbasis of the card's own pre-activation covariance (a cheap structured correction), or is it unit-specific? Regresses the error on the variance shares of
the top 16 modes (and their squares) + |alpha|, log v; in sample and fitted on the other network. Card state from t128_worker_cdump.py. usage: t128_v_modes.py"""
import importlib.util
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent; O = HERE.parent
spec = importlib.util.spec_from_file_location('oc', O / 'trainer/otr_common.py'); oc = importlib.util.module_from_spec(spec); spec.loader.exec_module(oc)
D = {}
for rank in (35, 16):
    W = np.load(oc.data_paths(rank)[0], mmap_mode='r')
    A = np.load(O / ('analyst/fresh/pu/mps_truth_rank%04d_A32M.npz' % rank)); B = np.load(O / ('analyst/fresh/pu/mps_truth_rank%04d_B32M.npz' % rank))
    L = 15
    d = np.load(HERE / 't128_cdump' / ('t128_C_%d_L%d.npz' % (rank, L)))
    w = np.asarray(W[L], dtype=np.float64); Cp = w.T @ d['C'] @ w; v = np.diag(Cp).copy()
    tv = (A['mc_pre_v'][L] + B['mc_pre_v'][L]) / 2; tm = (A['mc_pre_m'][L] + B['mc_pre_m'][L]) / 2
    e = np.log(v / tv); noise = np.log(A['mc_pre_v'][L] / B['mc_pre_v'][L]) / 2
    ev, U = np.linalg.eigh(Cp); ev, U = ev[::-1], U[:, ::-1]
    F = (U[:, :16] ** 2 * ev[None, :16]) / v[:, None]
    X = np.column_stack([np.ones(1024), F, F ** 2, np.abs(tm / np.sqrt(tv)), np.log(v)])
    D[rank] = (X, e, noise, U[:, :16], ev[:16])
for rank, (X, e, noise, U, ev) in D.items():
    b, *_ = np.linalg.lstsq(X, e, rcond=None); r = e - X @ b
    Xo, eo = D[16 if rank == 35 else 35][:2]; bo, *_ = np.linalg.lstsq(Xo, eo, rcond=None); ro = e - X @ bo
    # projection of the error vector (as a diagonal) onto the top modes: is the error's energy concentrated on units dominated by the top modes?
    print('rank %d L15: log v error sd %.5f (truth A/B noise %.5f) | mean %.5f | R^2 modes+features in-sample %.3f, fitted on the other network %.3f | corr(error, top-mode share) %.3f' % (
        rank, e.std(), noise.std(), e.mean(), 1 - r.var() / e.var(), 1 - ro.var() / e.var(), np.corrcoef(e, X[:, 1])[0, 1]))
