#!/usr/bin/env python3
"""TAG fable-analyst (AN-24): how much of the target dP (dP_data.py) is Monte Carlo noise of the 8M truth planes? r35 only: dP computed
with the main 8M planes (mps_planes_rank0035.npz) and with the independent 8M draw (mps_planes_rank0035_noise2.npz) at L2/5/9/12; the
noise of one draw's dP = ||dP_A - dP_B|| / sqrt(2). Reports the noise share of ||dP||^2 = the unpredictable floor. usage: dP_noise.py <threads>"""
import sys, json
from pathlib import Path
HERE0 = Path(__file__).resolve().parent; P2 = HERE0.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
torch.set_num_threads(int(sys.argv[1])); rank = 35
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
PA = np.load(P2 / 'public_g19_attribution_v1' / 'mps_planes_rank0035.npz'); PB = np.load(P2 / 'public_g19_attribution_v1' / 'mps_planes_rank0035_noise2.npz')
p, a, c, e = model.torch_payload(torch, np, False)
card = model.Card(torch, np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy'), p, a, c, e); orig = card.env['_nonlinear_pair_batch']
def offz(X): return X - torch.diag(torch.diagonal(X))
def wpair(mu, C, K21, k4, k22_override=None, metric='identity', remainder31=None):
    out = orig(mu, C, K21, k4, k22_override, metric)
    l = card.call_layer
    if l not in (2, 5, 9, 12): return out
    with torch.no_grad():
        C64 = C.double(); K = K21.double(); k22 = k22_override.double(); k4v = k4.double(); v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K)
        cov0 = out[1].double(); hh = out[0].double() / out[0].double().norm()
        def proj(D):
            Dh = D @ hh; hD = hh @ D; hDh = hh @ Dh
            return offz(D - torch.outer(hh, hD) - torch.outer(Dh, hh) + hDh * torch.outer(hh, hh))
        ys = []
        for PL in (PA, PB):
            K21t = offz(torch.as_tensor(PL[f'K21_{l}'], dtype=torch.float64)) + torch.diag(k3)
            K22t = offz(torch.as_tensor(PL[f'K22_{l}'], dtype=torch.float64)) + torch.diag(torch.diagonal(k22))
            r31 = offz((torch.as_tensor(PL[f'K31_{l}'], dtype=torch.float64) - k4v[:, None] * C64 / v[:, None]) / (v[:, None] * s[:, None] * s[None, :])).to(torch.float32)
            ys.append(proj(orig(mu, C, K21t.to(K21.dtype), k4, K22t.to(k22_override.dtype), metric, remainder31=r31)[1].double() - cov0))
        yA, yB = ys; nA = float(yA.norm()); nd = float((yA - yB).norm()) / np.sqrt(2)
        print(json.dumps(dict(layer=l, dP_norm=nA, noise_norm_one_draw=nd, noise_share_of_energy=round((nd / nA) ** 2, 4),
                              cos_AB=round(float((yA * yB).sum() / (yA.norm() * yB.norm())), 4))), flush=True)
    return out
card.env['_nonlinear_pair_batch'] = wpair
with torch.no_grad(): card.run(False)
