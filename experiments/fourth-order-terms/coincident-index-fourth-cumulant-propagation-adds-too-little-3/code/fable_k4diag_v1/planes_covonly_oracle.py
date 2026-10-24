#!/usr/bin/env python3
"""TAG fable-k4diag (KD-3): do the third/fourth-order PLANES explain the mean-orthogonal covariance lever? Feed 8M truth planes
(public_g19 mps_planes: K21, K22, K31 at L2-14) into the pair map, but let them change ONLY its covariance output; every other output
(post mean, post plane target, radial, aux) comes from the card's own inputs. Mode C = full change, perpC = change projected onto
P = I - hh^T/|h|^2. Compare with the exact-covariance oracle (Cperp@4+7+10+13 = .740 on r35 and r100).
usage: planes_covonly_oracle.py <ranks 35,100> <configs e.g. 21+22+31:perpC,31:perpC,21:perpC,21+22+31:C> <threads>"""
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
src = src.replace(old, "    T31 = (t4[:, None] * rho if _K31T[0] is None else _K31T[0] / (v[:, None] * s[:, None] * s[None, :])) if k31 else None\n    T31 = T31 - diag(diag(T31)) if (k31 and _K31T[0] is not None) else T31\n")
def run(rank, cfg, planes):
    p, a, c, e = model.torch_payload(torch, np, False)
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    card.env['_K31T'] = [None]; card.env['diag'] = torch.diag
    orig = card.env['_nonlinear_pair_batch']
    t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'pairmod', 'exec'), card.env)
    mod = card.env['_nonlinear_pair_batch']; card.env['_nonlinear_pair_batch'] = orig
    which, mode = cfg.split(':') if cfg != 'base' else ('', None)
    which = set(which.split('+')) if which else set()
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
        l = card.call_layer
        out_orig = orig(mu, C, K21, k4, k22_override, metric)
        if mode is None or not (2 <= l <= 14):
            return out_orig
        K21m, k22m = K21, k22_override
        if '21' in which:
            T21 = torch.as_tensor(planes[f'K21_{l}'], dtype=K21.dtype); K21m = T21 - torch.diag(torch.diagonal(T21)) + torch.diag(torch.diagonal(K21))
        if '22' in which and k22_override is not None:
            T22 = torch.as_tensor(planes[f'K22_{l}'], dtype=k22_override.dtype); k22m = T22 - torch.diag(torch.diagonal(T22)) + torch.diag(torch.diagonal(k22_override))
        card.env['_K31T'][0] = torch.as_tensor(planes[f'K31_{l}'], dtype=C.dtype) if '31' in which else None
        out_mod = mod(mu, C, K21m, k4, k22m, metric)
        card.env['_K31T'][0] = None
        D = out_mod[1] - out_orig[1]
        if mode == 'perpC':
            hh = out_orig[0] / out_orig[0].norm(); Dh = D @ hh; hD = hh @ D; hDh = hh @ Dh
            D = D - torch.outer(hh, hD) - torch.outer(Dh, hh) + hDh * torch.outer(hh, hh)
        return (out_orig[0], out_orig[1] + D) + tuple(out_orig[2:])
    card.env['_nonlinear_pair_batch'] = wpair
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())
for rk in ranks:
    planes = np.load(P2 / 'public_g19_attribution_v1' / f'mps_planes_rank{rk:04d}.npz')
    base = None
    for cfg in ['base'] + configs:
        t0 = time.time(); mse = run(rk, cfg, planes)
        if base is None: base = mse
        print(json.dumps(dict(rank=rk, cfg=cfg, mse=mse, ratio=round(mse / base, 4), wall=round(time.time() - t0, 1))), flush=True)
