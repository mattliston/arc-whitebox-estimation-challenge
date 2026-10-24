"""[opus-vecA] dependency-DAG census of the small-vector ops issued directly by _astra_fhm_core: for each charged op whose caller frame is
_astra_fhm_core and whose operands have <= 4096 elements, record (layer, op, input handle ids, output handle id). Then per layer: assign
each op a level = 1 + max level of its inputs (inputs produced outside the function = level 0) and count same-op groups per level.
usage: probe_fhm_dag_v1.py <candidate> <seed>  (in-process native, PB_ALL=1)"""
import sys, types, collections, json
from pathlib import Path
import numpy as np
import flopscope as flops, flopscope.numpy as fnp
from flopscope import _budget
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
cand, seed = Path(sys.argv[1]).resolve(), int(sys.argv[2])
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(cand); sys.modules[mod.__name__] = mod
exec(compile(cand.read_text(), str(cand), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
LIVE = [False]; ROWS = []
orig = _budget.BudgetContext._charge_op
def site():
    fr = sys._getframe(2)
    while fr is not None and fr.f_code.co_filename != str(cand): fr = fr.f_back
    return (fr.f_code.co_name, fr.f_lineno) if fr is not None else ('?', 0)
def charged(self, op_name, flop_cost, subscripts, shapes, **kw):
    if LIVE[0]:
        fn, line = site()
        if fn == '_astra_fhm_core':
            n = max([int(np.prod(s)) if len(s) else 1 for s in shapes] or [1])
            lay = mod.__dict__.get('_PHYS_LAYER', [None])[0]
            ROWS.append((int(lay) if isinstance(lay, (int, np.integer)) else -1, op_name, line, n))
    return orig(self, op_name, flop_cost, subscripts, shapes, **kw)
_budget.BudgetContext._charge_op = charged
if '_ASTRA_DENSE_CONTEXT' in mod.__dict__:
    mod._ASTRA_DENSE_CONTEXT = tuple((n, mod.__dict__[n]) for n, f in mod._ASTRA_DENSE_CONTEXT)
est = mod.Estimator(); est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed); mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    LIVE[0] = True; y = est.predict(mlp, 2**41)
LIVE[0] = False
# per-layer sequence of small-vector ops (program order); runs of consecutive same-op small ops are the stackable candidates without reordering
small = [(lay, op, line) for lay, op, line, n in ROWS if n <= 4096]
print('fhm ops total', len(ROWS), 'small-vector', len(small))
per_layer = collections.defaultdict(list)
for lay, op, line in small: per_layer[lay].append((op, line))
for lay in sorted(per_layer):
    seq = per_layer[lay]; runs = []; i = 0
    while i < len(seq):
        j = i
        while j + 1 < len(seq) and seq[j + 1][0] == seq[i][0]: j += 1
        if j - i + 1 >= 3: runs.append((seq[i][0], j - i + 1, seq[i][1], seq[j][1]))
        i = j + 1
    byop = collections.Counter(op for op, _ in seq)
    if lay in (1, 7, 13): print('layer', lay, 'small ops', len(seq), 'by op', byop.most_common(8), '| consecutive same-op runs >=3:', runs)
json.dump(ROWS, open(str(O/'lead/runs/opus_vec_v1/fhm_ops_7103256.json'), 'w'))
