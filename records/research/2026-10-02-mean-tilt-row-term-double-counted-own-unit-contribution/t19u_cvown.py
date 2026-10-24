#!/usr/bin/env python3
"""TAG opus-physicist T19u (numpy): IS THE UNIT'S OWN TERM IN THE ROW FORM A DOUBLE COUNT? nlaw.law has the mean-tilt row form with (C V)_a including c = a;
the layer 0 -> 1 certificate (t19k) has it with c != a, the c = a part being the own (3,1) term. One step from the true matrix (cross-block: the law fed with
block b of layer l against the truth of block 1-b of layer l+1), true auxiliary inputs, network-specific part; and the free run from layer 0.
variants: law (as is), cvown (row form without c = a), and the K21-row and elementwise K21 terms that also use (C V) with the fix applied to them too (cvown2).
usage: t19u_cvown.py <rank>"""
import sys, json, math
from pathlib import Path
import numpy as np
import nlaw_v3 as nlaw          # the version with the cvown flag (nlaw.py itself now has the row form fixed by default)
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
VAR = {'law': (), 'cvown': ('cvown',), 'lean': ('k3prod', 'k3x', 'k3el'), 'lean_cvown': ('k3prod', 'k3x', 'k3el', 'cvown'), 'lean_cvown_nobirth': ('k3prod', 'k3x', 'k3el', 'cvown', 'nobirth'), 'lean_nobirth': ('k3prod', 'k3x', 'k3el', 'nobirth')}
state = {k: np.zeros((n, n)) for k in VAR}
for l in range(0, 15):
    a = AUX[l]; Wn = W[l + 1].astype(np.float64); Cn, mun = AUX[l + 1]['C'], AUX[l + 1]['mu']
    Rb = [nlaw.specific(truth(l + 1, b), Cn, mun) for b in (0, 1)]; RR = float((Rb[0] * Rb[1]).sum()); Rm = (Rb[0] + Rb[1]) / 2; row = dict(rank=rank, layer=l + 1, free_run={}, one_step={})
    for name, skip in VAR.items():
        state[name], _ = nlaw.step(Wn, a['C'], a['mu'], a['m'], state[name], Cn, K21=a['K21'], K3V=a['K3V'], skip=skip); s = nlaw.specific(state[name], Cn, mun)
        row['free_run'][name] = round(float((2 * (s * Rm).sum() - (s * s).sum()) / RR), 3)
        if l >= 1 and l in (1, 2, 4, 6, 8, 10, 12, 14):
            sb = [nlaw.specific(nlaw.step(Wn, a['C'], a['mu'], a['m'], truth(l, b), Cn, K21=a['K21'], K3V=a['K3V'], skip=skip)[0], Cn, mun) for b in (0, 1)]
            row['one_step'][name] = round(float(np.mean([(2 * (sb[b] * Rb[1 - b]).sum() - (sb[0] * sb[1]).sum()) / RR for b in (0, 1)])), 3)
    print(json.dumps(row), flush=True)
