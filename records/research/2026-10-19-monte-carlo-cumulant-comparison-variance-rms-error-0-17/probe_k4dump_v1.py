"""TAG fable-lead 2026-10-19: dump the card's PER-UNIT internals at every k4 band site for one fresh seed: mu, vv (= diag Cpre), radial
(scalar), k4_iso (radial vv^2, before the band scale), band scale, k4_total (after K4L). Pairs with mc_cumulants_v1 (the truth) for the
same seed. usage: probe_k4dump_v1.py <science candidate.py> <seed> <out npz>"""
import sys
import types
from pathlib import Path

import numpy as np

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
src = candidate.read_text()
BAND = "                k4 = k4 * (_S19_K4BAND[0] if l <= 7 else (_S19_K4BAND[1] if l <= 11 else _S19_K4BAND[2]))\n"
K4L = "                k4 = (_e2_append_inline((_e2_value_inline := k4)) or _e2_value_inline) + _hc_k4l\n"
assert src.count(BAND) == 4 and src.count(K4L) == 4
src = src.replace(BAND, "                _K4DUMP(l, 'pre', mu=mu, vv=vv, radial=radial, k4=k4, bpre=bpre)\n" + BAND + "                _K4DUMP(l, 'band', k4=k4)\n")
src = src.replace(K4L, K4L + "                _K4DUMP(l, 'total', k4=k4)\n")
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
REC = {}


def dump(l, stage, **kw):
    d = REC.setdefault(int(l), {})
    for k, v in kw.items():
        d[stage + '_' + k] = np.array(np.asarray(v, dtype=np.float64), copy=True)


mod._K4DUMP = dump
exec(compile(src, str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
import flopscope as flops


class MLP:
    def __init__(self, weights, seed):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, seed, 'fresh-%d' % seed


w = loader.weights(seed)
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(MLP([w[l] for l in range(16)], seed), 2**41), dtype=np.float64)
arrs = {'pred': y}
for l, d in REC.items():
    for k, v in d.items():
        arrs['L%d_%s' % (l, k)] = v
np.savez(out, **arrs)
print('seed %d: layers %s; shapes L8: %s' % (seed, sorted(REC), {k: v.shape for k, v in REC[8].items()}))
