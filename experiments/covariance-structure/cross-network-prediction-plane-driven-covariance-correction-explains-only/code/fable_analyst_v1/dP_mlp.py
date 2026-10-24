#!/usr/bin/env python3
"""TAG fable-analyst (AN-26): NONLINEAR predictability ceiling of the plane-driven covariance correction. Per pair (a,c), a small MLP maps
dimensionless local features of the card's own state at layer l to the raw correction D[a,c] / (s_a s_c), D = cov_pair(truth planes) -
cov_pair(card planes) (dP_data.py construction, unprojected). Features: per unit a and c {alpha, skew k3/s^3, gate, pdf(alpha), q = mu k3/v^2},
pair {rho, T21_ac, T21_ca, T22_ac (card), T31_ac, T31_ca (card t4*rho)}, layer/14, radial. Train on one net (all layers 2-14, random pair
subsample), test on the other net on ALL pairs: R2 of D (raw) and of dP = offdiag(P D P) (the lead's target), both directions.
usage: dP_mlp.py <threads> [--epochs N]"""
import sys, json, time, math
from pathlib import Path
HERE0 = Path(__file__).resolve().parent; P2 = HERE0.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
HERE = HERE0
torch.set_num_threads(int(sys.argv[1])); EP = int([a_.split('=')[1] for a_ in sys.argv if a_.startswith('--epochs=')][0]) if any(a_.startswith('--epochs=') for a_ in sys.argv) else 6
STORE = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
man = json.loads((STORE / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
def offz(X): return X - torch.diag(torch.diagonal(X))
def collect(rank):
    planes = np.load(P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{rank:04d}.npz')
    p, a, c, e = model.torch_payload(torch, np, False)
    card = model.Card(torch, np.load(STORE / Path(man[rank]['path']).parent / 'weights.npy'), p, a, c, e); orig = card.env['_nonlinear_pair_batch']
    data = {}
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity', remainder31=None):
        out = orig(mu, C, K21, k4, k22_override, metric); l = card.call_layer
        if not (2 <= l <= 14): return out
        with torch.no_grad():
            mu64, C64, K = mu.double(), C.double(), K21.double(); k22 = k22_override.double(); k4v = k4.double()
            v = torch.diagonal(C64); s = v.sqrt(); k3 = torch.diagonal(K); rad = float(k4v[0] / (v[0] * v[0]))
            K21t = offz(torch.as_tensor(planes[f'K21_{l}'], dtype=torch.float64)) + torch.diag(k3)
            K22t = offz(torch.as_tensor(planes[f'K22_{l}'], dtype=torch.float64)) + torch.diag(torch.diagonal(k22))
            r31t = offz((torch.as_tensor(planes[f'K31_{l}'], dtype=torch.float64) - k4v[:, None] * C64 / v[:, None]) / (v[:, None] * s[:, None] * s[None, :])).to(torch.float32)
            Dm = orig(mu, C, K21t.to(K21.dtype), k4, K22t.to(k22_override.dtype), metric, remainder31=r31t)[1].double() - out[1].double()
            al = mu64 / s; g = torch.erfc(-al / math.sqrt(2)) / 2; pdf = torch.exp(-al * al / 2) / math.sqrt(2 * math.pi); sk = k3 / s ** 3; q = mu64 * k3 / (v * v)
            ss = torch.outer(s, s); rho = offz(C64 / ss); T21 = offz(K / (v[:, None] * s[None, :])); T22 = offz(k22 / torch.outer(v, v)); T31 = rad * rho
            unit = torch.stack([al, sk, g, pdf, q], 1)
            data[l] = dict(unit=unit.float(), rho=rho.float(), T21=T21.float(), T22=T22.float(), T31=T31.float(), rad=rad,
                           y=(offz(Dm) / ss).float(), ss=ss.float(), h=out[0].double())
        return out
    card.env['_nonlinear_pair_batch'] = wpair
    with torch.no_grad(): card.run(False)
    return data
def feats(d, l, idx_a, idx_c):
    u = d['unit']; ua, uc = u[idx_a], u[idx_c]
    pair = torch.stack([d['rho'][idx_a, idx_c], d['T21'][idx_a, idx_c], d['T21'][idx_c, idx_a], d['T22'][idx_a, idx_c], d['T31'][idx_a, idx_c], d['T31'][idx_c, idx_a]], 1)
    lay = torch.full((len(idx_a), 2), 0.0); lay[:, 0] = l / 14; lay[:, 1] = d['rad'] * 10
    return torch.cat([ua, uc, ua * uc, pair, lay], 1)
t0 = time.time(); DATA = {r: collect(r) for r in (35, 100)}; print(json.dumps(dict(collected=round(time.time() - t0, 1))), flush=True)
n = 1024; offm = ~torch.eye(n, dtype=torch.bool); IA, IC = torch.nonzero(offm, as_tuple=True)
def train_eval(tr, te):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    X, Y = [], []
    for l, d in DATA[tr].items():
        sel = torch.as_tensor(rng.choice(len(IA), 150_000, replace=False))
        X.append(feats(d, l, IA[sel], IC[sel])); Y.append(d['y'][IA[sel], IC[sel]])
    X = torch.cat(X); Y = torch.cat(Y); mx, sx = X.mean(0), X.std(0) + 1e-8; sy = Y.std()
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 96), torch.nn.SiLU(), torch.nn.Linear(96, 96), torch.nn.SiLU(), torch.nn.Linear(96, 1))
    opt = torch.optim.Adam(net.parameters(), lr=2e-3); Xn = (X - mx) / sx; Yn = Y / sy
    for ep in range(EP):
        perm = torch.randperm(len(Xn))
        for i in range(0, len(Xn), 4096):
            b = perm[i:i + 4096]; loss = ((net(Xn[b]).squeeze(1) - Yn[b]) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
        for gidx in opt.param_groups: gidx['lr'] *= 0.6
    res = {}
    with torch.no_grad():
        for l, d in DATA[te].items():
            pr = torch.zeros(n, n); Xf = (feats(d, l, IA, IC) - mx) / sx
            pr[IA, IC] = torch.cat([net(Xf[i:i + 262144]).squeeze(1) for i in range(0, len(Xf), 262144)]) * sy
            pr = (pr + pr.T) / 2
            Dp = (pr * d['ss']).double(); Dt = (d['y'] * d['ss']).double(); hh = d['h'] / d['h'].norm()
            def proj(M):
                Mh = M @ hh; hM = hh @ M; return offz(M - torch.outer(hh, hM) - torch.outer(Mh, hh) + (hh @ Mh) * torch.outer(hh, hh))
            yt, yp = proj(Dt)[offm], proj(Dp)[offm]; b = float(yp @ yt / (yp @ yp))
            res[l] = dict(R2_raw=round(1 - float(((Dt - Dp)[offm] ** 2).sum() / (Dt[offm] ** 2).sum()), 4),
                          R2_proj=round(1 - float(((yt - yp) ** 2).sum() / (yt @ yt)), 4), R2_proj_bestscale=round(1 - float(((yt - b * yp) ** 2).sum() / (yt @ yt)), 4))
    return res
for tr, te in ((35, 100), (100, 35)):
    t0 = time.time(); r = train_eval(tr, te)
    print(json.dumps(dict(train=tr, test=te, wall=round(time.time() - t0, 1), per_layer=r)), flush=True)
