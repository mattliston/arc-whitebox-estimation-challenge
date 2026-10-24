#!/usr/bin/env python3
"""TAG fable-refit (RF-11): second-order Edgeworth (k3^2 He6/72) terms in the marginal raw moments of the pair map and the final
scalar map, injected into the research torch card (no refit). E[f^(r)] = A[r]/sigma^(r-1) (card's derivative lists); relu^p
derivatives reduce to relu^(r-p+1) times p!; adds m += k3^2 A6/(72 s^5), r2 += k3^2 A5/(36 s^4), r3 += k3^2 A4/(12 s^3),
r4 += k3^2 A3/(3 s^2). usage: edgeworth2_test.py <ranks> <scale e.g. 1.0,0.5> <threads>"""
import sys, json, ast, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; scales = [float(x) for x in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
text = model.SOURCE.read_text(); tree = ast.parse(text)
def fsrc(name):
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]; return ast.get_source_segment(text, node)
pair_src, scalar_src = fsrc('_nonlinear_pair_batch'), fsrc('nonlinear_scalar')
def make(scale):
    p = pair_src
    old = ("    m = mg + k3 * e / 6 + k4 * f / 24\n    r2 = r2g + k3 * d / 3 + k4 * e / 12\n    r3 = r3g + k3 * g + k4 * d / 4\n    r4 = r4g + 4 * k3 * mg + k4 * g\n")
    assert p.count(old) == 1
    new = ("    _A3 = -a * pdf; _A4 = (a * a - 1) * pdf; _A5 = (3 * a - a ** 3) * pdf; _A6 = (a ** 4 - 6 * a * a + 3) * pdf; _q = %r * k3 * k3\n"
           "    m = mg + k3 * e / 6 + k4 * f / 24 + _q * _A6 / (72 * s ** 5)\n"
           "    r2 = r2g + k3 * d / 3 + k4 * e / 12 + _q * _A5 / (36 * s ** 4)\n"
           "    r3 = r3g + k3 * g + k4 * d / 4 + _q * _A4 / (12 * s ** 3)\n"
           "    r4 = r4g + 4 * k3 * mg + k4 * g + _q * _A3 / (3 * s ** 2)\n") % scale
    p = p.replace(old, new)
    sc = scalar_src
    old2 = ("    m = mg+k3*e/6+k4*f/24\n    r2 = r2g+k3*d/3+k4*e/12\n    r3 = r3g+k3*g+k4*d/4\n    r4 = r4g+4*k3*mg+k4*g\n")
    assert sc.count(old2) == 1
    new2 = ("    _A3 = -a*pdf; _A4 = (a*a-1)*pdf; _A5 = (3*a-a**3)*pdf; _A6 = (a**4-6*a*a+3)*pdf; _q = %r*k3*k3\n"
            "    m = mg+k3*e/6+k4*f/24 + _q*_A6/(72*s**5)\n    r2 = r2g+k3*d/3+k4*e/12 + _q*_A5/(36*s**4)\n    r3 = r3g+k3*g+k4*d/4 + _q*_A4/(12*s**3)\n    r4 = r4g+4*k3*mg+k4*g + _q*_A3/(3*s**2)\n") % scale
    sc = sc.replace(old2, new2)
    return p, sc
p, a, c, e = model.torch_payload(torch, np, False)
for scale in [0.0] + scales:
    psrc, ssrc = make(scale)
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
        truth = torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"][15].astype(np.float64))
        card = model.Card(torch, w, p, a, c, e)
        for src in (psrc, ssrc):
            t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'injected', 'exec'), card.env)
        t0 = time.time()
        with torch.no_grad(): pred = card.run(False)
        mse = float(((pred[15].to(torch.float64) - truth) ** 2).mean())
        print(json.dumps(dict(scale=scale, rank=rank, mse=mse, wall=round(time.time() - t0, 1))), flush=True)
