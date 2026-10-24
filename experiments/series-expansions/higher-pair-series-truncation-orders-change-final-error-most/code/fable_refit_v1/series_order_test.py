#!/usr/bin/env python3
"""TAG fable-refit (RF-10): truncation order of the pair map's Hermite series as an accuracy dial (no refit): Gaussian part to
order GO (bank: 6) and the third/fourth-order correction series to E rho-terms (bank: 3). Injects generalized copies of
_nonlinear_pair_batch / _pair_series_fused into the research torch card and reports the final MSE per (GO,E) on the given nets.
usage: series_order_test.py <ranks e.g. 35,100> <configs e.g. 6:3,8:3,6:4,8:4,10:5> <threads>"""
import sys, json, ast, time
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
from common import *
import model_bank_research as model
ranks = [int(x) for x in sys.argv[1].split(',')]; configs = [tuple(int(v) for v in c.split(':')) for c in sys.argv[2].split(',')]; torch.set_num_threads(int(sys.argv[3]))
P2 = HERE.parent; STORE = P2 / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
text = model.SOURCE.read_text(); tree = ast.parse(text)
def fsrc(name):
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]; return ast.get_source_segment(text, node)
pair_src, fused_src = fsrc('_nonlinear_pair_batch'), fsrc('_pair_series_fused')
def make(GO, E):
    p = pair_src.replace("gauss_order=6;edge_order=2;local=True;k31=True", f"gauss_order={GO};edge_order={E-1};local=True;k31=True")
    assert p != pair_src or (GO, E) == (6, 3)
    f = fused_src
    def R(a, b):
        nonlocal f
        assert f.count(a) == 1, a; f = f.replace(a, b)
    R("P = fnp.cumprod(fnp.broadcast_to(rho[None, :, :], (6, n, n)) / fnp.arange(1, 7, dtype=fnp.float32)[:, None, None], axis=0)",
                  f"P = fnp.cumprod(fnp.broadcast_to(rho[None, :, :], ({GO}, n, n)) / fnp.arange(1, {GO+1}, dtype=fnp.float32)[:, None, None], axis=0)")
    R("ans = fnp.sum(P6 * (Fr[1:7] * Gr[1:7]), axis=0)", f"ans = fnp.sum(P6 * (Fr[1:{GO+1}] * Gr[1:{GO+1}]), axis=0)")
    R("RR = fnp.stack((fnp.ones_like(rho), P[0], P[1]), axis=0)[:, None]", "RR = fnp.stack((fnp.ones_like(rho),) + tuple(P[i] for i in range(%d)), axis=0)[:, None]" % (E - 1))
    R("P12 = P6[:2]", f"P12 = P6[:{E-1}]")
    R("F1, F2, F3 = Fr[1:4], Fr[2:5], Fr[3:6]", f"F1, F2, F3 = Fr[1:{1+E}], Fr[2:{2+E}], Fr[3:{3+E}]")
    R("G1, G2, G3 = Gr[1:4], Gr[2:5], Gr[3:6]", f"G1, G2, G3 = Gr[1:{1+E}], Gr[2:{2+E}], Gr[3:{3+E}]")
    R("e3 = tri(t3Fr[4:6], Gr[1:3], P12) + tri(Fr[1:3], t3Gr[4:6], P12)", f"e3 = tri(t3Fr[4:{3+E}], Gr[1:{E}], P12) + tri(Fr[1:{E}], t3Gr[4:{3+E}], P12)")
    R("e4 = tri(t4Fr[5:7], Gr[1:3], P12) + tri(Fr[1:3], t4Gr[5:7], P12)", f"e4 = tri(t4Fr[5:{4+E}], Gr[1:{E}], P12) + tri(Fr[1:{E}], t4Gr[5:{4+E}], P12)")
    return p, f
p, a, c, e = model.torch_payload(torch, np, False)
for GO, E in configs:
    psrc, fsrc_ = make(GO, E)
    for rank in ranks:
        w = np.load(STORE / Path(man[rank]["path"]).parent / "weights.npy")
        truth = torch.tensor(np.load(P2 / "public_g22_operator_features_v1/truthpost" / f"truthpost_rank{rank:04d}.npz")["mc_post"][15].astype(np.float64))
        card = model.Card(torch, w, p, a, c, e)
        for src in (fsrc_, psrc):
            t_ = model.Translate().visit(ast.parse(src)); ast.fix_missing_locations(t_); exec(compile(t_, 'injected', 'exec'), card.env)
        t0 = time.time()
        with torch.no_grad(): pred = card.run(False)
        mse = float(((pred[15].to(torch.float64) - truth) ** 2).mean())
        print(json.dumps(dict(GO=GO, E=E, rank=rank, mse=mse, wall=round(time.time() - t0, 1))), flush=True)
