#!/usr/bin/env python3
"""TAG fable-refit (RF-21): EXACT one-step Gaussian-born STAR term of the pre-activation (3,1) plane. The post-activation fourth cumulant of
ReLU(Gaussian x at layer l-1) has star trees centred at kink units i: d3_i * prod_{3 leaves} g_j C_ij with d3 = E[relu'''] = -alpha*phi/sigma^2.
Pushed to the pre-activation of layer l (B = W_l^T G C_{l-1} with zero self-edges, Wt = W_l^T):
K31_star[a,c] = 3 * ((Wt∘B∘B) diag(d3) B^T)[a,c] + ((B∘B∘B) diag(d3) Wt^T)[a,c]. Injected into the pair map's T31 (off-diagonal), modes:
'add' (on top of the card's isotropic t4*rho), 'addproj' (minus its projection on the card's isotropic form v_a C_ac), scaled by lam;
layers >= FROM. Torch card, learned parts ON, no refit. usage: star_k31_test.py <ranks> <modes> <lams> <from_layers> <threads>"""
import sys, json, ast, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; modes = sys.argv[2].split(','); lams = [float(x) for x in sys.argv[3].split(',')]; froms = [int(x) for x in sys.argv[4].split(',')]; torch.set_num_threads(int(sys.argv[5]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
text = model.SOURCE.read_text(); tree = ast.parse(text)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][-1]
src = ast.get_source_segment(text, node)
old = "    T31 = t4[:, None] * rho if k31 else None\n"
assert src.count(old) == 1
src = src.replace(old, "    T31 = t4[:, None] * rho if k31 else None\n    if _K31_EXTRA[0] is not None:\n        _X = _K31_EXTRA[0] / (v[:, None] * s[:, None] * s[None, :])\n        T31 = T31 + (_X - diag(diag(_X)))\n")
p, a, c, e = model.torch_payload(torch, np, False)
from scipy.stats import norm
def run(rank, mode, lam, FROM):
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    card.env['_K31_EXTRA'] = [None]; card.env['diag'] = torch.diag
    t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'pairstar', 'exec'), card.env)
    pair = card.env['_nonlinear_pair_batch']; prev = {}
    Wd = torch.tensor(w, dtype=torch.float64)
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = card.call_layer
        if lam and l >= FROM and 'C' in prev:
            Cp, gp, d3p = prev['C'], prev['g'], prev['d3']
            bp = gp[:, None] * Cp; bp = bp - torch.diag(torch.diagonal(bp))          # b_i = g o C[:, i], zero self edge (column i)
            Wl = Wd[l]; B = Wl.T @ bp                                               # B[a, i] = sum_j W_ja g_j C_ji
            Wt = Wl.T                                                               # Wt[a, i] = W_ia
            star = 3 * ((Wt * B * B) * d3p[None, :]) @ B.T + ((B * B * B) * d3p[None, :]) @ Wt.T
            if mode == 'addproj':
                v = torch.diagonal(C); iso = v[:, None] * C; off = ~torch.eye(1024, dtype=torch.bool)
                r = (star[off] * iso[off]).sum() / (iso[off] * iso[off]).sum(); star = star - r * iso
            card.env['_K31_EXTRA'][0] = lam * star
        out = pair(mu, C, K21, k4, k22_override, metric); card.env['_K31_EXTRA'][0] = None
        v = torch.diagonal(C); s = v.sqrt(); al = (mu / s).numpy()
        prev['C'] = C.detach().clone(); prev['g'] = torch.tensor(norm.cdf(al)); prev['d3'] = torch.tensor(-al * norm.pdf(al)) / v
        return out
    card.env['_nonlinear_pair_batch'] = wpair
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())
base = {r: run(r, 'add', 0.0, 99) for r in ranks}
for FROM in froms:
    for mode in modes:
        for lam in lams:
            rs = {r: run(r, mode, lam, FROM) / base[r] for r in ranks}
            print(json.dumps(dict(from_layer=FROM, mode=mode, lam=lam, mean_ratio=round(float(np.mean(list(rs.values()))), 4), per_net={k: round(v, 4) for k, v in rs.items()})), flush=True)
