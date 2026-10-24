"""TAG fable-lead 2026-10-20: count client-side SYNCHRONISATION points per network (forum 18202: float()/bool()/int()/assert on a flopscope
array forces a materialising round-trip whose wait lands in the RESIDUAL bucket; 166 such points turned 5/50 failures into 50/50).
Wraps FlopscopeArray.__float__/__bool__/__int__/__index__/__array__/item/tolist/__iter__ and counts calls by caller site during one
in-process predict (local numpy-subclass mode: same code path as the evaluator, so the COUNTS transfer even though the cost does not).
usage: probe_syncpoints_v1.py <candidate.py> <store rank>"""
import collections
import json
import sys
import traceback
import types
from pathlib import Path

import numpy as np
import flopscope
import flopscope.numpy as fnp
from flopscope._ndarray import FlopscopeArray

O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
P2 = O.parent
candidate, rank = Path(sys.argv[1]).resolve(), int(sys.argv[2])
store = P2 / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = json.loads((store / 'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
COUNTS = collections.Counter()
ACTIVE = [False]


def wrap(name):
    orig = getattr(FlopscopeArray, name)

    def w(self, *a, **k):
        if ACTIVE[0]:
            site = '?'
            for fr in reversed(traceback.extract_stack(limit=12)[:-1]):
                if 'candidate' in fr.filename or fr.filename == str(candidate):
                    site = '%s:%d %s' % (Path(fr.filename).name, fr.lineno, (fr.line or '')[:70])
                    break
            COUNTS[(name, site)] += 1
        return orig(self, *a, **k)
    setattr(FlopscopeArray, name, w)


for n in ('__float__', '__bool__', '__int__', '__index__', '__array__', 'item', 'tolist', '__iter__', '__len__'):
    wrap(n)
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, 'release-%d' % rank


weights = np.load(store / Path(manifest[rank]['path']).parent / 'weights.npy')
ACTIVE[0] = True
with flopscope.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = est.predict(MLP([weights[l] for l in range(16)], rank), 2**41)
ACTIVE[0] = False
tot = collections.Counter()
for (name, site), c in COUNTS.items():
    tot[name] += c
print('rank %d: sync-point calls during predict by method: %s (total %d)' % (rank, dict(tot), sum(tot.values())))
print('top call sites:')
for (name, site), c in COUNTS.most_common(25):
    print('  %6d  %-10s %s' % (c, name, site))
