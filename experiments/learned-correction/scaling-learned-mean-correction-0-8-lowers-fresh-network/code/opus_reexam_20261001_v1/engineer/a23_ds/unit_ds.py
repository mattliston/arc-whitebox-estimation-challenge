"""TAG opus_engineer a23 unit check of the mean-correction scale lever DS (light: gnn_step alone on synthetic inputs, one thread, no
network, no prediction, no setup). usage: unit_ds.py <candidate.py> <out.npz (new)>      then      unit_ds.py --compare <base.npz> <other.npz> <scale>
Calls the candidate's gnn_step(l, x, sigma, W, state, P) at layers 1, 7 and 15 with the candidate's own learned parameters and
random features / weights / hidden state, and stores the returned correction and hidden state."""
import sys
import types
from pathlib import Path

import numpy as np

if sys.argv[1] == '--compare':
    a, b, scale = np.load(sys.argv[2]), np.load(sys.argv[3]), float(sys.argv[4])
    for l in (1, 7, 15):
        da, db = a['delta_%d' % l].astype(np.float64), b['delta_%d' % l].astype(np.float64)
        print('layer %2d: correction rms %.3e; other / base - %g: max abs relative deviation %.2e; bytes equal %s; hidden state bytes equal %s' % (
            l, np.sqrt((da ** 2).mean()), scale, float(np.max(np.abs(db - scale * da)) / np.max(np.abs(da))), a['delta_%d' % l].tobytes() == b['delta_%d' % l].tobytes(),
            a['state_%d' % l].tobytes() == b['state_%d' % l].tobytes()))
    sys.exit(0)

import flopscope as flops

candidate, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_unit'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
fnp = mod.fnp
rng = np.random.default_rng(5)
n = 1024
saved = {}
with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
    P = mod._unpack_static_fp32(mod._GNN_LITERAL)
    hidden = int(P['head'].shape[1]); nx = int(P['centers'].shape[1])
    for l in (1, 7, 15):
        x = fnp.asarray((rng.standard_normal((n, nx)) * 1.5).astype(np.float32))
        sigma = fnp.asarray(rng.uniform(0.5, 1.5, n))
        W = fnp.asarray((rng.standard_normal((n, n)) / np.sqrt(n)).astype(np.float32))
        state = fnp.asarray((rng.standard_normal((n, hidden)) * 0.5).astype(np.float32))
        delta, new_state = mod.gnn_step(l, x, sigma, W, state, P)
        saved['delta_%d' % l] = np.asarray(delta); saved['state_%d' % l] = np.asarray(new_state)
        print('layer', l, 'delta dtype', saved['delta_%d' % l].dtype, 'rms %.3e' % float(np.sqrt((saved['delta_%d' % l].astype(np.float64) ** 2).mean())), flush=True)
np.savez(out, **saved)
