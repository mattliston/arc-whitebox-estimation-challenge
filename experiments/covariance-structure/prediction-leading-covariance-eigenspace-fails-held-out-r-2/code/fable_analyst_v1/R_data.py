#!/usr/bin/env python3
"""TAG fable-analyst (AN-27): predict the carried-covariance error INSIDE the card's top-k eigen-subspace (one-sided slice).
Torch research card (unmodified, learned ON), CPU. After the step at layer l (1..14): C = the carried post covariance, m = post mean,
P = I - m m^T/|m|^2, U = top-k eigenvectors of P C P (k = 100; the first 30 give k = 30). Target R = P (C - C_truth) P U (n x k),
C_truth = 32M MC post covariance (fable_covhunter_v1/mcall_rank*_P32M.npz). Features (n x k, rows projected by P), from the card's own
pair-map inputs at l (pre-activation K21 plane, K22 plane, K31 = t4*rho form, gates g, d = pdf/sd = E f'', d3 = -alpha pdf/v = E f'''):
  K21: d o (K21 (g o U)),  K21T: g o (K21^T (d o U)),  K31: d3 o (K31 (g o U)),  K31T: g o (K31^T (d3 o U)),  K22: d o (K22 (d o U)),
  plus U, U Lambda (= P C U), g o U, v o U (post variance), m o U, d o U. Saves per layer and k the sufficient statistics (X^T X, X^T y, y^T y).
usage: R_data.py <rank> <threads>"""
import sys, json, time, math
from pathlib import Path
HERE0 = Path(__file__).resolve().parent; P2 = HERE0.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
HERE = HERE0
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
MCALL = np.load(P2 / 'fable_covhunter_v1' / f'mcall_rank{rank:04d}_P32M.npz')
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
card = model.Card(torch, np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy'), p, a, c, e)
NAMES = ['K21', 'K21T', 'K31', 'K31T', 'K22', 'U', 'ULam', 'gU', 'vU', 'mU', 'dU']
def offz(X): return X - torch.diag(torch.diagonal(X))
def features(pin, C, m, U, lam):
    mu, Cpre, K21, k4, k22 = pin
    v = torch.diagonal(Cpre); s = v.sqrt(); al = mu / s; pdf = torch.exp(-al * al / 2) / math.sqrt(2 * math.pi)
    g = torch.erfc(-al / math.sqrt(2)) / 2; d = pdf / s; d3 = -al * pdf / v; rad = float(k4[0] / (v[0] * v[0]))
    K21o = offz(K21); K22o = offz(k22); K31o = offz(rad * v[:, None] * Cpre)
    vp = torch.diagonal(C)
    F = {'K21': d[:, None] * (K21o @ (g[:, None] * U)), 'K21T': g[:, None] * (K21o.T @ (d[:, None] * U)),
         'K31': d3[:, None] * (K31o @ (g[:, None] * U)), 'K31T': g[:, None] * (K31o.T @ (d3[:, None] * U)),
         'K22': d[:, None] * (K22o @ (d[:, None] * U)), 'U': U, 'ULam': U * lam[None, :], 'gU': g[:, None] * U,
         'vU': vp[:, None] * U, 'mU': m[:, None] * U, 'dU': d[:, None] * U}
    return F
def projP(X, hh): return X - torch.outer(hh, hh @ X)
stats = {}; pin = {}
orig_pair = card.env['_nonlinear_pair_batch']
def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
    pin[card.call_layer] = (mu.double(), C.double(), K21.double(), k4.double(), k22_override.double())
    return orig_pair(mu, C, K21, k4, k22_override, metric)
card.env['_nonlinear_pair_batch'] = wpair
orig_step = card.step
def step(layer, state):
    out = orig_step(layer, state)
    if 1 <= layer <= 14:
        t0 = time.time()
        with torch.no_grad():
            m = out[0].double(); C = out[1].double(); hh = m / m.norm()
            PCP = projP(projP(C, hh).T, hh).T
            lam, V = torch.linalg.eigh((PCP + PCP.T) / 2); U = V[:, -100:].flip(1); lam = lam[-100:].flip(0)
            Ct = torch.as_tensor(MCALL[f'cov_post_{layer}'], dtype=torch.float64)
            dC = C - Ct; R = projP(projP(dC, hh).T, hh).T @ U
            F = features(pin[layer], C, m, U, lam)
            for k in (30, 100):
                X = torch.stack([projP(F[n_][:, :k], hh).reshape(-1) for n_ in NAMES], 1); y = R[:, :k].reshape(-1)
                stats[(layer, k)] = dict(G=(X.T @ X).numpy(), b=(X.T @ y).numpy(), yy=float(y @ y))
            nA = torch.as_tensor(MCALL[f'cov_post_A_{layer}'], dtype=torch.float64); nB = torch.as_tensor(MCALL[f'cov_post_B_{layer}'], dtype=torch.float64)
            N = (nA - nB) / 2; RN = projP(projP(N, hh).T, hh).T @ U
            stats[(layer, 'noise')] = dict(R_norm30=float(R[:, :30].norm()), R_norm100=float(R.norm()), N_norm30=float(RN[:, :30].norm()), N_norm100=float(RN.norm()))
        print(json.dumps(dict(rank=rank, layer=layer, R30=round(stats[(layer, 'noise')]['R_norm30'], 5), noise30=round(stats[(layer, 'noise')]['N_norm30'], 5), wall=round(time.time() - t0, 1))), flush=True)
    return out
card.step = step
with torch.no_grad(): card.run(False)
out = dict(names=np.array(NAMES))
for (l, k), d_ in stats.items():
    for kk, vv in d_.items(): out[f'{kk}_{l}_{k}'] = np.asarray(vv)
np.savez(HERE / f'R_data_r{rank}.npz', **out); print('saved', flush=True)
