"""TAG opus_engineer a24 (2026-10-02): can the learned mean-correction net skip its messages for some units? Torch run M card, forward
only. gnn_step (layers 1-15) forms six message products W^p' @ src (n x n x 16 each, 1.7% of the bill together with the shared
edge messages); a unit's messages are the rows of those products. Here the messages of a set of TARGET units are replaced by 0
(what a native build would do by multiplying only the kept columns of W), optionally with their correction set to 0 as well:
  zlo   targets with alpha < -zlo get no messages      (dead side; 2.33 = not alive)
  zhi   targets with alpha > zhi get no messages       (saturated side; 2.33 = not a response row)
  dz    1 = the correction of those units is set to 0 too (default 1); 0 = the net's output without messages is kept
  l15   0 = the final layer keeps all messages (default 1: the mask applies there too)
The card's gnn_step is the native source function translated to torch; it is re-translated here with three changed lines.
arms: name:zlo=<z>,zhi=<z>[,dz=0][,l15=0][,family=shift ...]
usage: TRAINER_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 TRAINER_PL=1 TRAINER_CARD=M \\
       python -B torch_msgcut.py <rank> <out.npz (new)> arm [arm ...]
Output: pred_<arm> per arm; one JSON line per arm (final-layer MSE, share of the units that keep their messages per layer 1..15)."""
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
REPL = (('    x = fnp.clip((x.astype(fnp.float32)-P["centers"][l])/P["scales"][l],-8,8)\n',
         '    _mk = _MSG_MASK(l, x)\n    x = fnp.clip((x.astype(fnp.float32)-P["centers"][l])/P["scales"][l],-8,8)\n'),
        ('    msgs=fnp.concatenate([g*m1,g*m2,g*m3,fnp.log1p(fnp.maximum(e1,0)),fnp.log1p(fnp.maximum(e2,0)),fnp.log1p(fnp.maximum(e3,0))],axis=1)\n',
         '    msgs=fnp.concatenate([g*m1,g*m2,g*m3,fnp.log1p(fnp.maximum(e1,0)),fnp.log1p(fnp.maximum(e2,0)),fnp.log1p(fnp.maximum(e3,0))],axis=1)*_mk[0]\n'),
        ('    return delta,state\n', '    return delta*_mk[1],state\n'))
text = MB.SOURCE.read_text()
node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'gnn_step')
segment = ast.get_source_segment(text, node) + '\n'
for old, new in REPL:
    assert segment.count(old) == 1, old
    segment = segment.replace(old, new)
PATCHED = compile(ast.fix_missing_locations(MB.Translate().visit(ast.parse(segment))), 'gnn_step with a target mask on its messages', 'exec')


def install_mask(cd, o, log):
    def mask(l, x):
        alpha = x[:, 3].to(t.float64)               # the raw feature row: (2l/15 - 1, l/16, 1, alpha, ...)
        keep = (alpha >= -o['zlo']) & (alpha <= o['zhi'])
        if l == 15 and not o['l15']:
            keep = t.ones_like(keep)
        log.append(round(float(keep.sum()) / 1024, 4))
        k32 = keep.to(t.float32)
        return k32[:, None], (k32 if o['dz'] else t.ones_like(k32))
    cd.env['_MSG_MASK'] = mask
    exec(PATCHED, cd.env)


saved = {}
for spec in arms:
    name, _unused, opts = spec.partition(':')
    o = dict(zlo=99.0, zhi=99.0, dz=1.0, l15=1.0)
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
    print(json.dumps(dict(rank=rank, arm=name, opts={k: v for k, v in o.items() if v != dict(zlo=99.0, zhi=99.0, dz=1.0, l15=1.0)[k]},
                          final_x1e9=round(float(mse[15]) * 1e9, 4), units_with_messages_layers_1_15=log, seconds=round(time.monotonic() - t0, 1))), flush=True)
np.savez(out, **saved)
