#!/usr/bin/env python3
"""TAG fable-analyst (AN-29): closed-loop value of the PREDICTED one-sided subspace correction (R_fit.py band ridge coefficients, cross-fitted:
net 35 uses the fit on 100, net 100 the fit on 35, other nets the mean of both; bands L2-6 / L7-10 / L11-14). After the step at l = 2..14:
U = top-k eigenvectors of the card's P C P, R_hat = sum_j beta_j P F_j (R_data.py features), C -= eps (R_hat U^T + U R_hat^T - U sym(U^T R_hat) U^T).
Learned parts ON, no refit. usage: R_closedloop.py <ranks> <k> <eps list> <threads>"""
import sys, json, time, math
from pathlib import Path
HERE0 = Path(__file__).resolve().parent; P2 = HERE0.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1')); sys.path.insert(1, str(HERE0))
import numpy as np, torch
from common import *
import model_bank_research as model
HERE = HERE0
ranks = [int(x) for x in sys.argv[1].split(',')]; K = int(sys.argv[2]); EPS = [float(x) for x in sys.argv[3].split(',')]; torch.set_num_threads(int(sys.argv[4]))
D = {r: np.load(HERE / f'R_data_r{r}.npz') for r in (35, 100)}; names = list(D[35]['names'])
BANDS = ((2, 6), (7, 10), (11, 14))
def fit(r, band, lam=1e-6):
    Ls = range(band[0], band[1] + 1); G = sum(D[r][f'G_{l}_{K}'] for l in Ls); b = sum(D[r][f'b_{l}_{K}'] for l in Ls)
    s = np.sqrt(np.diag(G)); return np.linalg.solve(G / np.outer(s, s) + lam * np.eye(len(b)), b / s) / s
COEF = {35: {bd: fit(100, bd) for bd in BANDS}, 100: {bd: fit(35, bd) for bd in BANDS}}
COEF['mean'] = {bd: (fit(35, bd) + fit(100, bd)) / 2 for bd in BANDS}
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
def offz(X): return X - torch.diag(torch.diagonal(X))
def projP(X, hh): return X - torch.outer(hh, hh @ X)
def run(rank, eps):
    coef = COEF.get(rank, COEF['mean'])
    card = model.Card(torch, np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy'), p, a, c, e)
    truth = np.load(P2 / 'public_g22_operator_features_v1/truthpost' / f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
    pin = {}; orig_pair = card.env['_nonlinear_pair_batch']
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
        pin[card.call_layer] = (mu.double(), C.double(), K21.double(), k4.double(), k22_override.double())
        return orig_pair(mu, C, K21, k4, k22_override, metric)
    card.env['_nonlinear_pair_batch'] = wpair
    orig_step = card.step
    def step(layer, state):
        out = orig_step(layer, state)
        if eps == 0 or not (2 <= layer <= 14): return out
        band = [bd for bd in BANDS if bd[0] <= layer <= bd[1]][0]; beta = coef[band]
        m = out[0].double(); C = out[1].double(); hh = m / m.norm()
        PCP = projP(projP(C, hh).T, hh).T; lam, V = torch.linalg.eigh((PCP + PCP.T) / 2); U = V[:, -K:].flip(1); lam = lam[-K:].flip(0)
        mu, Cpre, K21, k4, k22 = pin[layer]
        v = torch.diagonal(Cpre); s = v.sqrt(); al = mu / s; pdf = torch.exp(-al * al / 2) / math.sqrt(2 * math.pi)
        g = torch.erfc(-al / math.sqrt(2)) / 2; d = pdf / s; d3 = -al * pdf / v; rad = float(k4[0] / (v[0] * v[0]))
        K21o = offz(K21); K22o = offz(k22); K31o = offz(rad * v[:, None] * Cpre); vp = torch.diagonal(C)
        F = [d[:, None] * (K21o @ (g[:, None] * U)), g[:, None] * (K21o.T @ (d[:, None] * U)), d3[:, None] * (K31o @ (g[:, None] * U)),
             g[:, None] * (K31o.T @ (d3[:, None] * U)), d[:, None] * (K22o @ (d[:, None] * U)), U, U * lam[None, :], g[:, None] * U,
             vp[:, None] * U, m[:, None] * U, d[:, None] * U]
        Rh = sum(float(bj) * projP(Fj, hh) for bj, Fj in zip(beta, F))
        S = U.T @ Rh; S = (S + S.T) / 2
        X = Rh @ U.T + U @ Rh.T - U @ S @ U.T
        return (out[0], (C - eps * X).to(out[1].dtype)) + tuple(out[2:])
    card.step = step
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth) ** 2).mean())
for rk in ranks:
    base = run(rk, 0.0)
    for ep in EPS:
        t0 = time.time(); mse = run(rk, ep)
        print(json.dumps(dict(rank=rk, k=K, eps=ep, mse=mse, ratio=round(mse / base, 4), wall=round(time.time() - t0, 1))), flush=True)
