#!/usr/bin/env python3
"""TAG fable-refit (RF-9): how much of the card's per-unit k3/k4/v ERROR (vs the 8M truth) is predictable from the causal GNN state h
that the cumulant heads read? Runs the research torch card on two nets, captures hstate and the pair-map inputs per layer, and fits
ridge (linear on h) and a small tanh-MLP (h -> 32 -> 1) for the relative errors; reports in-sample R2 on net A and transfer R2 on net B.
usage: head_info_probe.py <rankA> <rankB> <threads>"""
import sys, json, math
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
rA, rB, threads = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]); torch.set_num_threads(threads)
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
def capture(rank):
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    tr = np.load(P2 / "public_g19_attribution_v1" / f"mps_truth_rank{rank:04d}_P8M.npz")
    p, a, c, e = model.torch_payload(torch, np, False); card = model.Card(torch, w, p, a, c, e)
    rec = {}
    orig = card.env['_nonlinear_pair_batch']
    def pair(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = card.call_layer
        rec[l] = dict(h=card.cur_h.detach().numpy().copy(), k3=torch.diagonal(K21).detach().numpy().copy(), k4=k4.detach().numpy().copy(), v=torch.diagonal(C).detach().numpy().copy())
        return orig(mu, C, K21, k4, k22_override, metric)
    card.env['_nonlinear_pair_batch'] = pair
    orig_step = card.step
    def step(layer, state):
        card.cur_h = state[8]; return orig_step(layer, state)
    card.step = step; card.cur_h = None
    with torch.no_grad(): card.run(False)
    out = {}
    for l, d in rec.items():
        out[l] = dict(h=d['h'], e3=d['k3'] / tr['mc_pre_k3'][l] - 1, e4=d['k4'] / tr['mc_pre_k4'][l] - 1, ev=d['v'] / tr['mc_pre_v'][l] - 1)
    return out
A, B = capture(rA), capture(rB)
def r2(y, yhat): return 1 - float(((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def ridge(X, y, lam=1e-2):
    Xb = np.concatenate([X, np.ones((len(X), 1))], 1); return np.linalg.solve(Xb.T @ Xb + lam * np.eye(Xb.shape[1]), Xb.T @ y)
def mlp_fit(X, y, XB, yB, epochs=400):
    Xt, yt = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32); XBt = torch.tensor(XB, dtype=torch.float32)
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 1)); opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(epochs):
        opt.zero_grad(); loss = ((net(Xt).squeeze(1) - yt) ** 2).mean() + 1e-4 * sum((p ** 2).sum() for p in net.parameters()); loss.backward(); opt.step()
    with torch.no_grad(): return r2(y, net(Xt).squeeze(1).numpy()), r2(yB, net(XBt).squeeze(1).numpy())
for l in sorted(A):
    if l < 8: continue
    row = dict(layer=l)
    for key in ('e3', 'e4', 'ev'):
        X, y = A[l]['h'], A[l][key]; XB, yB = B[l]['h'], B[l][key]
        # clip outliers (units with tiny truth cumulants give huge relative errors)
        keep = np.abs(y) < np.quantile(np.abs(y), 0.98); keepB = np.abs(yB) < np.quantile(np.abs(yB), 0.98)
        X, y, XB, yB = X[keep], y[keep], XB[keepB], yB[keepB]
        beta = ridge(X, y); Xb = np.concatenate([X, np.ones((len(X), 1))], 1); XBb = np.concatenate([XB, np.ones((len(XB), 1))], 1)
        rl_in, rl_tr = r2(y, Xb @ beta), r2(yB, XBb @ beta)
        rm_in, rm_tr = mlp_fit(X, y, XB, yB)
        row[key] = dict(rms_err=float(np.sqrt((y ** 2).mean())), ridge_in=round(rl_in, 3), ridge_transfer=round(rl_tr, 3), mlp_in=round(rm_in, 3), mlp_transfer=round(rm_tr, 3))
    print(json.dumps(row), flush=True)
