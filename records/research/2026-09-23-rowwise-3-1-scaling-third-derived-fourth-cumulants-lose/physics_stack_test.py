#!/usr/bin/env python3
"""TAG fable-refit (RF-23): stack of ZERO-COST physics supplies that enter only the pair map / final map, measured with the learned parts
ON and OFF: (k4) per-unit k4 = radial v^2 + beta (k3^2/v - s v^2) (zero-sum) at layers >= 2 and the final map; (k31) per-row factor on the
isotropic T31: r_a = c0 + c1*mu_a*k3_a/v_a^2 (physicist: c0 .85, c1 .5). usage: physics_stack_test.py <ranks> <configs e.g. k31,k4,k31+k4> <threads>"""
import sys, json, ast
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; configs = sys.argv[2].split(','); torch.set_num_threads(int(sys.argv[3]))
BETA, C0, C1 = 0.5, 0.85, 0.5
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
text = model.SOURCE.read_text(); tree = ast.parse(text)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_nonlinear_pair_batch'][-1]
src = ast.get_source_segment(text, node)
old = "    T31 = t4[:, None] * rho if k31 else None\n"
assert src.count(old) == 1
src = src.replace(old, "    T31 = (t4 * (_ROWF[0](mu, k3, v) if _ROWF[0] is not None else 1.0))[:, None] * rho if k31 else None\n")
def run(rank, cfg, learned_off):
    p, a, c, e = model.torch_payload(torch, np, False)
    if learned_off:
        with torch.no_grad(): p['head'].zero_(); p['head_b'].zero_(); e['output'].zero_(); e['bias'].zero_()
    w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
    truth = np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    card = model.Card(torch, w, p, a, c, e)
    card.env['_ROWF'] = [(lambda mu, k3, v: C0 + C1 * mu * k3 / (v * v)) if 'k31' in cfg else None]
    t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'pairrow', 'exec'), card.env)
    pair = card.env['_nonlinear_pair_batch']; scal = card.env['nonlinear_scalar']
    def k4n(k3, v, k4):
        q = k3 * k3 / v; s = q.sum() / (v * v).sum(); kn = k4 + BETA * (q - s * v * v); return torch.maximum(kn, 0.1 * k4)
    def wpair(mu, C, K21, k4, k22_override=None, metric='identity'):
        if 'k4' in cfg and card.call_layer >= 2:
            k4 = k4n(torch.diagonal(K21), torch.diagonal(C), k4)
            if k22_override is not None: k22_override = torch.diagonal_scatter(k22_override, k4)
        return pair(mu, C, K21, k4, k22_override, metric)
    def wscal(mu, v, k3, k4):
        if 'k4' in cfg: k4 = k4n(k3, v, k4)
        return scal(mu, v, k3, k4)
    card.env['_nonlinear_pair_batch'] = wpair; card.env['nonlinear_scalar'] = wscal
    with torch.no_grad(): pred = card.run(False).numpy().astype(np.float64)
    return float(((pred[15] - truth[15]) ** 2).mean())
for learned_off in (False, True):
    base = {r: run(r, 'none', learned_off) for r in ranks}
    for cfg in configs:
        rs = {r: run(r, cfg, learned_off) / base[r] for r in ranks}
        print(json.dumps(dict(learned='off' if learned_off else 'on', cfg=cfg, mean_ratio=round(float(np.mean(list(rs.values()))), 4), per_net={k: round(v, 4) for k, v in rs.items()})), flush=True)
