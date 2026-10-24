#!/usr/bin/env python3
"""TAG opus-physicist T19q (numpy): which terms and which state does a CHEAP carrier need? Free run of the law (nlaw.py) from layer 0 on true
auxiliary inputs (as t19p_freerun.py), with terms dropped or the state cut at every layer; share of the true network-specific part held per layer.
  full      all terms (conversion, K21)                     noprod   the two K21 x C product groups dropped
  nooff     all off-diagonal K21 terms dropped              nox      the elementwise (d d') o Co o K3V dropped
  noown     the own (3,1) term dropped                      r32/r64  state = multiple of C + rank 32 / 64 at every layer
  m8/m16    state = multiple of C + partner form on the top 8 / 16 covariance eigenvectors (the lead's structure)
  cheap     noprod + nox + m16
usage: t19q_ablate.py <rank>"""
import sys, json, math
from pathlib import Path
import numpy as np
import nlaw
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; n = 1024
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rank = int(sys.argv[1]); man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r'); Z = np.load(HERE / f't19g_rank{rank:04d}.npz'); A = np.load(HERE / f't19m_rank{rank:04d}.npz')
tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64); symm = lambda X: (X + X.T) / 2
avg = lambda k, l: (A[f'{k}_L{l}_b0'].astype(np.float64) + A[f'{k}_L{l}_b1'].astype(np.float64)) / 2
def truth(l, b):
    M11 = Z[f'M11_L{l}_b{b}'].astype(np.float64); return symm(Z[f'T_L{l}_b{b}'].astype(np.float64) - M11 * float(Z[f'S_L{l}_b{b}']) - 2 * M11 @ M11)
w0 = W[0].astype(np.float64); C0 = w0.T @ w0
AUX = {0: dict(C=C0, mu=np.zeros(n), m=np.sqrt(np.diag(C0) / (2 * math.pi)), K21=None, K3V=None)}
for l in range(1, 16):
    AUX[l] = dict(C=(Z[f'M11_L{l}_b0'].astype(np.float64) + Z[f'M11_L{l}_b1'].astype(np.float64)) / 2, mu=Z[f'mu_L{l}'].astype(np.float64), m=tp[l], K21=avg('M21', l), K3V=symm(avg('T_L', l)))
def cut(N, C, mode):
    if not mode: return N
    a = float((N * C).sum() / (C * C).sum()); D = N - a * C
    if mode[0] == 'r':
        lam, V = np.linalg.eigh(symm(D)); o = np.argsort(-np.abs(lam))[:int(mode[1:])]; return a * C + (V[:, o] * lam[o][None, :]) @ V[:, o].T
    ev, V = np.linalg.eigh(C); U = V[:, -int(mode[1:]):]; UD = U @ (U.T @ D); return a * C + UD + UD.T - (UD @ U) @ U.T
VAR = {'full': ((), ''), 'noprod': (('k3prod',), ''), 'nooff': (('k3off',), ''), 'nox': (('k3x',), ''), 'noown': (('own31',), ''), 'r32': ((), 'r32'), 'r64': ((), 'r64'),
       'm8': ((), 'm8'), 'm16': ((), 'm16'), 'cheap': (('k3prod', 'k3x'), 'm16')}
state = {k: np.zeros((n, n)) for k in VAR}
for l in range(0, 15):
    a = AUX[l]; Wn = W[l + 1].astype(np.float64); Cn, mun = AUX[l + 1]['C'], AUX[l + 1]['mu']
    Rb = [nlaw.specific(truth(l + 1, b), Cn, mun) for b in (0, 1)]; RR = float((Rb[0] * Rb[1]).sum()); Rm = (Rb[0] + Rb[1]) / 2; row = dict(rank=rank, layer=l + 1, share={})
    for name, (skip, mode) in VAR.items():
        state[name], _ = nlaw.step(Wn, a['C'], a['mu'], a['m'], state[name], Cn, K21=a['K21'], K3V=a['K3V'], skip=skip); state[name] = cut(state[name], Cn, mode)
        s = nlaw.specific(state[name], Cn, mun); row['share'][name] = round(float((2 * (s * Rm).sum() - (s * s).sum()) / RR), 3)
    print(json.dumps(row), flush=True)
