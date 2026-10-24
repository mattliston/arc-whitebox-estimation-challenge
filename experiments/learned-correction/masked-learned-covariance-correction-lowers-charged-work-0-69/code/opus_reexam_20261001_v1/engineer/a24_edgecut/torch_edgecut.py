"""TAG opus_engineer a24 (2026-10-02): how much of the learned covariance edge (layers 8-14) can be left uncomputed? Torch run M card,
forward only. The edge's raw output is T_ij pdf_i gate_j with T bounded (tanh layer times a head), then symmetrised. The native
lever Ecut evaluates it only on some rows x columns and writes 0 elsewhere; here the same mask is applied to the card's raw
output, so the cost of any choice of rows and columns can be read off without a native session:
  zr   rows kept: |alpha_i| <= zr            (the loop's response rows are |alpha| <= 2.33 and alive)
  fc   columns kept: gate_j >= fc            (the loop's alive columns are gate >= 0.01)
  off  1 = no edge at all (for scale)
The card's edge function is the native source function translated to torch (fable_refit_v1/model_bank_research.py); it is
re-translated here with one changed line (raw times the mask) and installed in the card's environment.
arms: name:zr=<z>,fc=<f>[,off=1][,family=shift ...]
usage: TRAINER_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 TRAINER_PL=1 TRAINER_CARD=M \\
       python -B torch_edgecut.py <rank> <out.npz (new)> arm [arm ...]
Output: pred_<arm> per arm; one JSON line per arm (per-layer MSE, share of the n x n grid kept per layer 8..14)."""
import ast
import json
import sys
import time
from pathlib import Path

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'trainer'))
import numpy as np
import otr_engine as E
from otr_common import data_paths
import model_bank_research as MB

torch = t = E.torch
rank, out, arms = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
assert not out.exists() and E.M is not None and E.PL is not None
weights_path, truth_path = data_paths(rank)
weights = np.load(weights_path, allow_pickle=False)
truth = np.load(truth_path, allow_pickle=False)['mc_post'].astype(np.float64)
OLD = "    raw=(fnp.tanh(h)@t['output'])*pf[:,None]*gf[None,:]\n"
NEW = "    raw=(fnp.tanh(h)@t['output'])*pf[:,None]*gf[None,:]*_EDGE_MASK(a,g)\n"
text = MB.SOURCE.read_text()
node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == '_message_edge_source')
segment = ast.get_source_segment(text, node) + '\n'
assert segment.count(OLD) == 1
PATCHED = compile(ast.fix_missing_locations(MB.Translate().visit(ast.parse(segment.replace(OLD, NEW)))), 'edge with a row and column mask', 'exec')


def install_mask(cd, o, log):
    def mask(a, g):
        rows = (a.abs() <= o['zr']).to(t.float32)
        cols = (g >= o['fc']).to(t.float32)
        log.append(round(float(rows.sum() * cols.sum()) / (1024 * 1024), 4))
        if o['off']:
            return t.zeros((1024, 1024), dtype=t.float32)
        return rows[:, None] * cols[None, :]
    cd.env['_EDGE_MASK'] = mask
    exec(PATCHED, cd.env)


saved = {}
for spec in arms:
    name, _unused, opts = spec.partition(':')
    o = dict(zr=99.0, fc=0.0, off=0.0)
    shifts = {}
    for item in filter(None, opts.split(',')):
        key, value = item.split('=')
        if key in o:
            o[key] = float(value)
        else:
            shifts[key] = [float(value)] * 16
    E.PL.set_values(shifts)
    E.LC.JT.DIALS_T = E.DL
    log = []
    t0 = time.monotonic()
    with torch.no_grad():
        cd = E.LC.card(torch, weights, E.p, E.T['a'], E.T['c'], E.e, fine_ckpt=False)
        cd.hooks['k4_15'] = lambda layer, value, ctx: value
        E.install_star(cd)
        E.PL.install(E, cd)
        install_mask(cd, o, log)      # always: the card's function environment may be shared between cards (a mask of ones is exact)
        E.M.install(cd)
        prediction = cd.run(False).detach().to(torch.float64).numpy().copy()
    del cd
    E.collect()
    assert prediction.shape == (16, 1024) and np.isfinite(prediction).all()
    saved['pred_' + name] = prediction
    mse = ((prediction - truth) ** 2).mean(1)
    print(json.dumps(dict(rank=rank, arm=name, opts={k: v for k, v in o.items() if v != dict(zr=99.0, fc=0.0, off=0.0)[k]},
                          final_x1e9=round(float(mse[15]) * 1e9, 4), grid_kept_layers_8_14=log, seconds=round(time.monotonic() - t0, 1))), flush=True)
np.savez(out, **saved)
