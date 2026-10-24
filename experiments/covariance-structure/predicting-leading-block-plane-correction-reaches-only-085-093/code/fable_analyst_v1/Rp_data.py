#!/usr/bin/env python3
"""TAG fable-analyst (AN-30): is the ONE-SIDED top-k block of the plane-driven correction predictable? At every pair-map call l = 2..14 of the
torch research card (unmodified trajectory, learned ON): D = cov_pair(8M truth K21 off-diag + card k3, truth K22 off-diag, truth K31) -
cov_pair(card planes) (the lead's planes_covonly construction), P = I - hh^T/|h|^2 (h = the pair map's host), U = top-30 eigenvectors of
P cov0 P (cov0 = the card's pair covariance output). Target R_k = P D P U_k (n x k), k = 10, 30. Features (n x k, rows projected by P),
the lead's list: plane-vector products Z in {K21 (g o U), K21^T (d o U), K31 (g o U), K31^T (d3 o U), K22 (d o U), P1 (g o U), P1^T (d3 o U)}
each raw and left-weighted by d, g, d3 (card K21/K22 off-diagonal, card K31 = t4 rho, P1 = 3 (k3_a/v_a)(K21_ac - C_ac k3_a/v_a);
d = pdf/sd = h2, d3 = -alpha pdf/v = h3), plus C U, U, and per-unit multipliers v, k3, k4, mu, g, d of U. For r35 at L5/9/12 also the
same R from the independent 8M draw (noise). Saves least-squares sufficient statistics per layer and k.
usage: Rp_data.py <rank> <threads>"""
import sys, json, time, math
from pathlib import Path
HERE0 = Path(__file__).resolve().parent; P2 = HERE0.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
HERE = HERE0
rank = int(sys.argv[1]); torch.set_num_threads(int(sys.argv[2]))
G19 = P2 / 'public_g19_attribution_v1'
PL = np.load(G19 / f'mps_planes_rank{rank:04d}.npz'); PN = np.load(G19 / 'mps_planes_rank0035_noise2.npz') if rank == 35 else None
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
p, a, c, e = model.torch_payload(torch, np, False)
card = model.Card(torch, np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy'), p, a, c, e); orig = card.env['_nonlinear_pair_batch']
def offz(X): return X - torch.diag(torch.diagonal(X))
NAMES = []
for z in ('K21', 'K21T', 'K31', 'K31T', 'K22', 'P1', 'P1T'):
    for wname in ('raw', 'd', 'g', 'd3'): NAMES.append(f'{z}_{wname}')
NAMES += ['CU', 'U', 'vU', 'k3U', 'k4U', 'muU', 'gU', 'dU']
stats = {}
def truth_cov(PLf, l, mu, C, K21, k4, k22, metric):
    C64 = C.double(); K = K21.double(); v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K); k4v = k4.double()
    K21t = offz(torch.as_tensor(PLf[f'K21_{l}'], dtype=torch.float64)) + torch.diag(k3)
    K22t = offz(torch.as_tensor(PLf[f'K22_{l}'], dtype=torch.float64)) + torch.diag(torch.diagonal(k22.double()))
    r31 = offz((torch.as_tensor(PLf[f'K31_{l}'], dtype=torch.float64) - k4v[:, None] * C64 / v[:, None]) / (v[:, None] * s[:, None] * s[None, :])).to(torch.float32)
    return orig(mu, C, K21t.to(K21.dtype), k4, K22t.to(k22.dtype), metric, remainder31=r31)[1].double()
def wpair(mu, C, K21, k4, k22_override=None, metric='identity', remainder31=None):
    out = orig(mu, C, K21, k4, k22_override, metric); l = card.call_layer
    if not (2 <= l <= 14): return out
    t0 = time.time()
    with torch.no_grad():
        mu64, C64, K = mu.double(), C.double(), K21.double(); k22 = k22_override.double(); k4v = k4.double()
        v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K); rad = float(k4v[0] / (v[0] * v[0]))
        al = mu64 / s; pdf = torch.exp(-al * al / 2) / math.sqrt(2 * math.pi); g = torch.erfc(-al / math.sqrt(2)) / 2; d = pdf / s; d3 = -al * pdf / v
        cov0 = out[1].double(); hh = out[0].double() / out[0].double().norm()
        def projP(X): return X - torch.outer(hh, hh @ X)
        def PDP(X): return projP(projP(X).T).T
        PCP = PDP(cov0); lam, V = torch.linalg.eigh((PCP + PCP.T) / 2); U = V[:, -30:].flip(1)
        R = PDP(truth_cov(PL, l, mu, C, K21, k4, k22_override, metric) - cov0) @ U
        K21o = offz(K); K22o = offz(k22); K31o = offz(rad * v[:, None] * C64); q = k3 / v
        P1 = offz(3 * q[:, None] * (K - C64 * q[:, None]))
        Z = {'K21': K21o @ (g[:, None] * U), 'K21T': K21o.T @ (d[:, None] * U), 'K31': K31o @ (g[:, None] * U), 'K31T': K31o.T @ (d3[:, None] * U),
             'K22': K22o @ (d[:, None] * U), 'P1': P1 @ (g[:, None] * U), 'P1T': P1.T @ (d3[:, None] * U)}
        W = {'raw': torch.ones_like(g), 'd': d, 'g': g, 'd3': d3}
        F = {f'{z}_{wn}': W[wn][:, None] * Z[z] for z in Z for wn in W}
        F.update({'CU': cov0 @ U, 'U': U, 'vU': v[:, None] * U, 'k3U': k3[:, None] * U, 'k4U': (rad * v * v)[:, None] * U,
                  'muU': mu64[:, None] * U, 'gU': g[:, None] * U, 'dU': d[:, None] * U})
        for k in (10, 30):
            X = torch.stack([projP(F[n_][:, :k]).reshape(-1) for n_ in NAMES], 1); y = R[:, :k].reshape(-1)
            stats[(l, k)] = dict(G=(X.T @ X).numpy(), b=(X.T @ y).numpy(), yy=float(y @ y))
        if PN is not None and f'K21_{l}' in PN.files:
            RB = PDP(truth_cov(PN, l, mu, C, K21, k4, k22_override, metric) - cov0) @ U
            for k in (10, 30):
                stats[(l, k)]['noise_share'] = float(((R[:, :k] - RB[:, :k]).norm() / math.sqrt(2) / R[:, :k].norm()) ** 2)
    print(json.dumps(dict(rank=rank, layer=l, R30_norm=round(float(R.norm()), 6), wall=round(time.time() - t0, 1))), flush=True)
    return out
card.env['_nonlinear_pair_batch'] = wpair
with torch.no_grad(): card.run(False)
out = dict(names=np.array(NAMES))
for (l, k), d_ in stats.items():
    for kk, vv in d_.items(): out[f'{kk}_{l}_{k}'] = np.asarray(vv)
np.savez(HERE / f'Rp_data_r{rank}.npz', **out); print('saved', flush=True)
