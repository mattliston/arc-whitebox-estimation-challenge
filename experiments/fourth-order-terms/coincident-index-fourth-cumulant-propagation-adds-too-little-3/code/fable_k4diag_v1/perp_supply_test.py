#!/usr/bin/env python3
"""TAG fable-k4diag (KD-2): PERP-PROJECTED covariance supplies. A physics supply changes the pair map's post covariance by
dC = C_mod - C_orig. Variant 'plain' uses C_mod (as shipped). Variant 'perp' uses C_orig + P dC P with P = I - h h^T/|h|^2 (h = the
pair map's post mean), i.e. keeps only the part of the change that kink units at l+1 can see and leaves the mean direction (where the
learned closure is co-adapted) untouched. Supplies: k31row (T31 row factor .85 + .5 mu k3/v^2, zero cost), k31truth (8M truth K31
planes, r35/r100 only). Torch card, learned ON, no refit; layers 2-14. usage: perp_supply_test.py <ranks> <configs> <threads>"""
import sys, json, ast, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; P2 = HERE.parent; sys.path.insert(0, str(P2 / 'fable_refit_v1'))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; configs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
text = model.SOURCE.read_text(); tree = ast.parse(text)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][-1]
src = ast.get_source_segment(text, node)
old = "    T31 = t4[:, None] * rho if k31 else None\n"
assert src.count(old) == 1
src = src.replace(old, "    T31 = ((t4 * _ROWF[0](mu, k3, v))[:, None] * rho if _K31T[0] is None else _K31T[0] / (v[:, None] * s[:, None] * s[None, :])) if k31 else None\n    T31 = T31 - diag(diag(T31)) if (k31 and _K31T[0] is not None) else T31\n")
def run(rank, cfg, planes):
    p, a, c, e = model.torch_payload(torch, np, False)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    card.env['_ROWF'] = [lambda mu, k3, v: 1.0]; card.env['_K31T'] = [None]; card.env['diag'] = torch.diag
    orig = card.env['_nonlinear_pair_batch']
    t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'pairmod', 'exec'), card.env)
    mod = card.env['_nonlinear_pair_batch']; card.env['_nonlinear_pair_batch'] = orig
    supply, mode = cfg.split(':') if cfg != 'base' else (None, None)
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = card.call_layer
        if supply is None or not (2 <= l <= 14):
            return orig(mu, C, K21, k4, k22_override, metric)
        if supply == 'k31row':
            card.env['_ROWF'][0] = lambda mu_, k3_, v_: 0.85 + 0.5 * mu_ * k3_ / (v_ * v_); card.env['_K31T'][0] = None
        elif supply == 'k31truth':
            card.env['_ROWF'][0] = lambda mu_, k3_, v_: 1.0; card.env['_K31T'][0] = torch.as_tensor(planes[f'K31_{l}'], dtype=C.dtype)
        out_mod = mod(mu, C, K21, k4, k22_override, metric)
        if mode == 'plain':
            return out_mod
        out_orig = orig(mu, C, K21, k4, k22_override, metric)
        h = out_orig[0]; hh = h / h.norm()
        D = out_mod[1] - out_orig[1]
        Dh = D @ hh; hD = hh @ D; hDh = hh @ Dh
        Dp = D - torch.outer(hh, hD) - torch.outer(Dh, hh) + hDh * torch.outer(hh, hh)
        return (out_mod[0], out_orig[1] + Dp) + tuple(out_mod[2:])
    card.env['_nonlinear_pair_batch'] = wpair
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())
for rk in ranks:
    planes = np.load(P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{rk:04d}.npz') if rk in (35, 100) else None
    base = None
    for cfg in ['base'] + configs:
        if cfg.startswith('k31truth') and planes is None: continue
        t0 = time.time(); mse = run(rk, cfg, planes)
        if base is None: base = mse
        print(json.dumps(dict(rank=rk, cfg=cfg, mse=mse, ratio=round(mse / base, 4), wall=round(time.time() - t0, 1))), flush=True)
