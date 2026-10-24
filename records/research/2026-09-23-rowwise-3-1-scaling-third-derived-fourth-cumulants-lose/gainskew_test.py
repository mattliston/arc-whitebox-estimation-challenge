#!/usr/bin/env python3
"""TAG fable-refit (RF-20): gain-on-skewed-base fourth-order supply (CGF of h = G z to O(s2), radial = 12 s2):
K31_ac += f*(radial/4)*(3 mu_a K21_ac + mu_c k3_a); K22_ac += f*(radial/2)*(mu_a K21_ca + mu_c K21_ac); k4_a += f*radial*mu_a*k3_a.
All three are rebuilt each layer from radial and enter only the pair map (and k4 the final map). Variants: 'k31' (K31 only) and
'triple'. Torch research card, learned parts ON, no refit. usage: gainskew_test.py <ranks> <fs e.g. 0.3,0.5> <variants> <threads>"""
import sys, json, ast, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; fs = [float(x) for x in sys.argv[2].split(',')]; variants = sys.argv[3].split(','); torch.set_num_threads(int(sys.argv[4]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
text = model.SOURCE.read_text(); tree = ast.parse(text)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][-1]
src = ast.get_source_segment(text, node)
old = "    T31 = t4[:, None] * rho if k31 else None\n"
assert src.count(old) == 1
src = src.replace(old, "    T31 = t4[:, None] * rho if k31 else None\n    if _K31_EXTRA[0] is not None:\n        _X = _K31_EXTRA[0] / (v[:, None] * s[:, None] * s[None, :])\n        T31 = T31 + (_X - diag(diag(_X)))\n")
p, a, c, e = model.torch_payload(torch, np, False)
def run(rank, variant, f):
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    card.env['_K31_EXTRA'] = [None]; card.env['diag'] = torch.diag
    t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'pairgs', 'exec'), card.env)
    pair = card.env['_nonlinear_pair_batch']; scal = card.env['nonlinear_scalar']
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
        if f and card.call_layer >= 1:
            v = torch.diagonal(C); k3 = torch.diagonal(K21); radial = (k4 / (v * v)).mean()
            card.env['_K31_EXTRA'][0] = f * (radial / 4) * (3 * mu[:, None] * K21 + mu[None, :] * k3[:, None])
            if variant == 'triple':
                k4 = k4 + f * radial * mu * k3
                ex22 = f * (radial / 2) * (mu[:, None] * K21.T + mu[None, :] * K21)
                k22_override = torch.diagonal_scatter(k22_override + ex22, k4)
        out = pair(mu, C, K21, k4, k22_override, metric); card.env['_K31_EXTRA'][0] = None
        return out
    def wscal(mu, v, k3, k4):
        if f and variant == 'triple':
            radial = (k4 / (v * v)).mean(); k4 = k4 + f * radial * mu * k3
        return scal(mu, v, k3, k4)
    card.env['_nonlinear_pair_batch'] = wpair; card.env['nonlinear_scalar'] = wscal
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())
base = {r: run(r, 'k31', 0.0) for r in ranks}
for variant in variants:
    for f in fs:
        rs = {r: run(r, variant, f) / base[r] for r in ranks}
        print(json.dumps(dict(variant=variant, f=f, mean_ratio=round(float(np.mean(list(rs.values()))), 4), per_net={k: round(v, 4) for k, v in rs.items()})), flush=True)
