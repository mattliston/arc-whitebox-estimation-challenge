"""[fable-lead 2026-10-19] RTT read-row probe: run one predict in-process and print the candidate's _RTT_STATS['reads'] (per layer: m rows,
k young columns, c15 = response rows kept by the Phi(-1.5) gate floor, c2 = all response rows) plus _YNT_STATS and the deadcut rosters.
usage: probe_rtt_v1.py <candidate.py> <seed>"""
import json, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O / 'analyst/fresh')); import loader
candidate, seed = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9; mod._FAST_RESCUE_CAP_S = 1e9
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
print('total flops %.3f GF' % (total / 1e9))
reads = mod.__dict__.get('_RTT_STATS', {}).get('reads', [])
print('RTT reads (layer m k c15 c2 changed):')
for r in reads: print('  L%-2d m=%4d k=%5d c15=%4d c2=%4d changed=%s' % (r['layer'], r['m'], r['k'], r['c15'], r['c2'], r['changed']))
for name in ('_YNT_STATS', '_ASTRA_BCC_STATS', '_UVP_STATS', '_K4L_STATS', '_RTT_STATS'):
    v = mod.__dict__.get(name)
    if isinstance(v, dict): print(name, {k: (v[k] if not isinstance(v[k], list) else len(v[k])) for k in v})
