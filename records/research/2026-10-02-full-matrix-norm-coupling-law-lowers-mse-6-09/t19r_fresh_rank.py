#!/usr/bin/env python3
"""TAG opus-physicist T19r (numpy): WHAT A NATIVE CARRIER MUST HOLD (lead's request of 10:39Z, item a). Lean law (nlaw.py without block products
with K21), true auxiliary inputs (covariance, means, post means, K21, conversion matrix: as t19p_freerun.py), networks with t19m files (35, 651).

PART A, one step from the TRUE matrix of layer l (l = 1..14). The true matrix is split into its forms and its network-specific part R_l. Then
   fresh      = the law applied to the forms alone (carried network-specific part set to zero): sources + the image of the forms
   transport  = 2 gbar W' (g g' o R_l) W                    exact on a rank-r state (r columns pushed through the gates and W)
   rows       = the rest of the image of R_l (the variance tilt and the diagonal driven by diag R_l): not low rank, cheap as an operator
 each reduced to its network-specific part at layer l+1. Reported: their energies relative to the whole, the share of the TRUE network-specific
 part of layer l+1 they hold (cross-block energy, as everywhere in T19), the share of the energy of 'fresh' (and of the truth) in its top-k
 eigen-directions, and the share of the truth held when 'fresh + rows' is cut to its best rank k with the transport kept exact.

PART B, FREE RUN from layer 0 of the state a native carrier would hold: four form coefficients + a rank-r symmetric matrix.
   exact_r      the whole pushed matrix minus its multiple of C, cut to rank r by eigenvalue (the cut of ncarry.py state_rank)
   two_r_k      transport exact on the r columns; the rest sketched with k Gaussian probes: form coefficients fitted on the probe products,
                Q = orth(D Omega), B = Q' D Q (a second application), then (transport + Q B Q') cut to rank r
   nys_r_k      the same with the single-pass form (D Omega)(Omega' D Omega)^+ (D Omega)'
 Reported per layer: share of the true network-specific part held.
usage: t19r_fresh_rank.py <rank> [A|B|AB]"""
import sys, json, math
from pathlib import Path
import numpy as np
import nlaw
HERE = Path(__file__).resolve().parent; O = HERE.parent; P2 = O.parent; n = 1024
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rank = int(sys.argv[1]); PARTS = sys.argv[2] if len(sys.argv) > 2 else 'AB'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][rank]
W = np.load((STORE / man['path']).parent / 'weights.npy', mmap_mode='r'); Z = np.load(HERE / f't19g_rank{rank:04d}.npz'); A = np.load(HERE / f't19m_rank{rank:04d}.npz')
tp = np.load(P2 / f'fable_refit_v1/truth688/truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64); symm = lambda X: (X + X.T) / 2
avg = lambda k, l: (A[f'{k}_L{l}_b0'].astype(np.float64) + A[f'{k}_L{l}_b1'].astype(np.float64)) / 2
def truth(l, b):
    M11 = Z[f'M11_L{l}_b{b}'].astype(np.float64); return symm(Z[f'T_L{l}_b{b}'].astype(np.float64) - M11 * float(Z[f'S_L{l}_b{b}']) - 2 * M11 @ M11)
w0 = W[0].astype(np.float64); C0 = w0.T @ w0
AUX = {0: dict(C=C0, mu=np.zeros(n), m=np.sqrt(np.diag(C0) / (2 * math.pi)), K21=None, K3V=None)}
for l in range(1, 16):
    AUX[l] = dict(C=(Z[f'M11_L{l}_b0'].astype(np.float64) + Z[f'M11_L{l}_b1'].astype(np.float64)) / 2, mu=Z[f'mu_L{l}'].astype(np.float64), m=tp[l], K21=avg('M21', l), K3V=symm(avg('T_L', l)))
LEAN = ('k3prod', 'k3x', 'k3el'); KS = (8, 16, 20, 32, 48, 64)
def step(l, Nt):
    a = AUX[l]; return nlaw.step(W[l + 1].astype(np.float64), a['C'], a['mu'], a['m'], Nt, AUX[l + 1]['C'], K21=a['K21'], K3V=a['K3V'], skip=LEAN)[0]
def transport(l, R):
    a = AUX[l]; u = nlaw.unit_stats(a['C'], a['mu'], a['m']); Wn = W[l + 1].astype(np.float64); return 2 * float(u['gam'].mean()) * (Wn.T @ (np.outer(u['g'], u['g']) * R) @ Wn)
def cut_rank(D, r):
    lam, V = np.linalg.eigh(symm(D)); o = np.argsort(-np.abs(lam))[:r]; return (V[:, o] * lam[o][None, :]) @ V[:, o].T
def top_shares(D):
    lam = np.linalg.eigvalsh(symm(D)); e = np.sort(lam * lam)[::-1]; return {str(k): round(float(e[:k].sum() / e.sum()), 3) for k in KS}

if 'A' in PARTS:
    for l in range(1, 15):
        a = AUX[l]; Cn, mun = AUX[l + 1]['C'], AUX[l + 1]['mu']; Nl = (truth(l, 0) + truth(l, 1)) / 2; Rl = nlaw.specific(Nl, a['C'], a['mu'])
        Rb = [nlaw.specific(truth(l + 1, b), Cn, mun) for b in (0, 1)]; RR = float((Rb[0] * Rb[1]).sum()); Rm = (Rb[0] + Rb[1]) / 2
        share = lambda s: round(float((2 * (s * Rm).sum() - (s * s).sum()) / RR), 3); spec = lambda X: nlaw.specific(X, Cn, mun)
        S_full = spec(step(l, Nl)); S_fresh = spec(step(l, Nl - Rl)); S_T = spec(transport(l, Rl)); S_rows = S_full - S_fresh - S_T; e = lambda X: float((X * X).sum()); ef = e(S_full)
        row = dict(rank=rank, part='A', transition=f'{l}->{l + 1}',
                   energy_over_whole=dict(fresh=round(e(S_fresh) / ef, 3), transport=round(e(S_T) / ef, 3), rows=round(e(S_rows) / ef, 3)),
                   share_of_truth=dict(whole=share(S_full), fresh_alone=share(S_fresh), transport_alone=share(S_T), fresh_plus_transport=share(S_fresh + S_T)),
                   top_k_energy=dict(fresh=top_shares(S_fresh), fresh_plus_rows=top_shares(S_fresh + S_rows), transport=top_shares(S_T), truth=top_shares(Rm)),
                   share_of_truth_with_fresh_plus_rows_cut_to_rank_k_and_exact_transport={str(k): share(S_T + cut_rank(S_fresh + S_rows, k)) for k in (16, 20, 32, 48)})
        print(json.dumps(row), flush=True)

if 'B' in PARTS:
    def forms_of(C, mu):
        Cmu = C @ mu; return [np.trace(C) * C, C @ C, np.outer(mu, mu), np.outer(mu, Cmu) + np.outer(Cmu, mu)]
    VAR = {'full': None, 'exact_32': ('exact', 32, 0), 'exact_48': ('exact', 48, 0), 'two_32_16': ('two', 32, 16), 'two_32_32': ('two', 32, 32), 'two_48_20': ('two', 48, 20),
           'nys_32_20': ('nys', 32, 20), 'nys_32_32': ('nys', 32, 32)}
    state = {k: dict(F=np.zeros((n, n)), R=np.zeros((n, n))) for k in VAR}          # F = the form part (full matrix here, four numbers natively), R = the rank-r part
    for l in range(0, 15):
        Cn, mun = AUX[l + 1]['C'], AUX[l + 1]['mu']; Rb = [nlaw.specific(truth(l + 1, b), Cn, mun) for b in (0, 1)]; RR = float((Rb[0] * Rb[1]).sum()); Rm = (Rb[0] + Rb[1]) / 2
        row = dict(rank=rank, part='B', layer=l + 1, share={}); FM = forms_of(Cn, mun); rng = np.random.default_rng(1000 + l)
        for name, mode in VAR.items():
            st = state[name]; X = step(l, st['F'] + st['R'])
            if mode is None: st['F'], st['R'] = X, np.zeros((n, n))
            elif mode[0] == 'exact':
                a_ = float((X * Cn).sum() / (Cn * Cn).sum()); st['F'] = a_ * Cn; st['R'] = cut_rank(X - a_ * Cn, mode[1])
            else:
                kind, r, k = mode; T = transport(l, st['R']) if l >= 1 else np.zeros((n, n)); Fo = X - T; Om = rng.standard_normal((n, k)); FO = Fo @ Om
                G = np.stack([f @ Om for f in FM], 0); co = np.linalg.lstsq(G.reshape(4, -1).T, FO.ravel(), rcond=None)[0]; Fm = sum(c * f for c, f in zip(co, FM)); DO = FO - Fm @ Om
                if kind == 'two':
                    Q = np.linalg.qr(DO)[0]; Sk = Q @ symm(Q.T @ ((Fo - Fm) @ Q)) @ Q.T
                else:
                    Sk = DO @ np.linalg.pinv(symm(Om.T @ DO), rcond=1e-3, hermitian=True) @ DO.T
                st['F'] = Fm; st['R'] = cut_rank(T + Sk, r)
            s = nlaw.specific(st['F'] + st['R'], Cn, mun); row['share'][name] = round(float((2 * (s * Rm).sum() - (s * s).sum()) / RR), 3)
        print(json.dumps(row), flush=True)
