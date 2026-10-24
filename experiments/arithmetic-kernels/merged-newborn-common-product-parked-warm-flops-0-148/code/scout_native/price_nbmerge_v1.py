"""TAG scout-native price_nbmerge_v1 (2026-10-12): measurement only. Exact flopscope bill (and charged ops) of the young push and the
newborn-common push as two products vs ONE product on [p | base] (lever NB-MERGE), through the candidate's own _mm after setup, on
random float32 operands of the S50 twin's per-layer shapes (7100136, dims2_s50tw_136.json). Each product runs twice; the second
(warm: tree keys bound) is reported. The bill of a product depends only on its shapes.
usage: price_nbmerge_v1.py <candidate.py> <dims2.json> <out.json>"""
import json
import sys
import types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget

candidate, dims, out = Path(sys.argv[1]).resolve(), json.load(open(sys.argv[2])), Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_price'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
ops = [0]
ORIG = _budget.BudgetContext._charge_op


def counted(self, *a, **k):
    ops[0] += 1
    return ORIG(self, *a, **k)


_budget.BudgetContext._charge_op = counted
shapes = {}
for c in dims['calls']:
    if c['tag'] == 'mm' and c['parent'] in ('young_push', 'w5:newborn_common') and c['layer'] is not None:
        a, b = c['args']['a'], c['args']['b']
        shapes.setdefault(c['layer'], {})[c['parent']] = (a[0], a[1], b[1])
rng = np.random.default_rng(0)


def bill(m, k, n):
    A = fnp.asarray(rng.standard_normal((m, k)).astype(np.float32))
    B = fnp.asarray(rng.standard_normal((k, n)).astype(np.float32))
    res = None
    for _ in range(2):
        mod._BH_HOLD.clear() if hasattr(mod, '_BH_HOLD') else None
        ops[0] = 0
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            r = mod._mm(A, B)
        res = (int(ctx.summary_dict()['flops_used']), ops[0])
        del r
    return res


rows = []
for L in sorted(shapes):
    s = shapes[L]
    if 'young_push' not in s or 'w5:newborn_common' not in s:
        continue
    (m, k, q), (m2, k2, ns) = s['young_push'], s['w5:newborn_common']
    assert (m, k) == (m2, k2), (L, s)
    if q == 0:
        continue
    by, oy = bill(m, k, q)
    bn, on = bill(m, k, ns)
    bm, om = bill(m, k, q + ns)
    row = dict(layer=L, m=m, k=k, q=q, n_sel=ns, young=by, newborn=bn, merged=bm, ops_young=oy, ops_newborn=on, ops_merged=om,
               saving=by + bn - bm, concat=k * (q + ns))
    rows.append(row)
    print(json.dumps(row), flush=True)
_budget.BudgetContext._charge_op = ORIG
with out.open('x') as f:
    json.dump(dict(candidate=str(candidate), rows=rows), f)
tot = sum(r['saving'] - r['concat'] for r in rows)
print(json.dumps(dict(net_saving_flops=tot, ops_saving=sum(r['ops_young'] + r['ops_newborn'] - r['ops_merged'] for r in rows))), flush=True)
