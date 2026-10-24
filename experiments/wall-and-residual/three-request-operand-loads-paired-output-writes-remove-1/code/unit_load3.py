"""TAG opus_engineer a21 unit check of the three-request operand loads (light: a candidate's setup() and eleven tree products on
random float32 matrices (single and batched tree products, and products through the dispatcher), single thread, no network, no prediction).
usage: unit_load3.py <candidate.py> <out.json (new)>      then      unit_load3.py --compare <old.json> <new.json>
Per product call: sha256 of the output bytes, billed flops, charged ops, and the number of flopscope calls by name (a counting
proxy in front of the candidate's fnp: reshape / unstack / copyto / add / subtract / transpose / matmul / ...; index views are not
counted). Each case runs twice: the first call may bind its tree keys, the second is warm.
--compare prints, per case and call, whether the bytes are equal and the changes in flops, charged ops and calls."""
import collections
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np

if sys.argv[1] == '--compare':
    old, new = (json.load(open(x)) for x in sys.argv[2:4])
    same = True
    for a, b in zip(old['rows'], new['rows']):
        assert (a['case'], a['call']) == (b['case'], b['call'])
        equal = a['sha'] == b['sha']; same = same and equal
        calls = {k: b['calls'].get(k, 0) - a['calls'].get(k, 0) for k in sorted(set(a['calls']) | set(b['calls'])) if b['calls'].get(k, 0) != a['calls'].get(k, 0)}
        print('%-34s call %d: bytes %s  flops %+d (%+.4f%%)  charged ops %+d  counted calls %+d %s' % (
            a['case'], a['call'], 'EQUAL' if equal else 'DIFFER', b['flops'] - a['flops'], 100 * (b['flops'] / a['flops'] - 1),
            b['ops'] - a['ops'], sum(b['calls'].values()) - sum(a['calls'].values()), calls))
    print('ALL OUTPUTS BYTE-IDENTICAL' if same else 'OUTPUTS DIFFER')
    print('setup: flops %d -> %d, charged ops %d -> %d, views built %s' % (old['setup']['flops'], new['setup']['flops'], old['setup']['ops'], new['setup']['ops'], new.get('quads')))
    sys.exit(0)

import flopscope as flops

candidate, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_unit'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
real = mod.fnp
COUNT = collections.Counter()


class Counting:
    def __getattr__(self, name):
        value = getattr(real, name)
        if not callable(value) or isinstance(value, type):
            return value

        def counted(*args, **kwargs):
            COUNT[name] += 1
            return value(*args, **kwargs)
        counted.__name__ = name
        setattr(self, name, counted)
        return counted


est = mod.Estimator()
with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
    est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
    setup = dict(flops=int(ctx.summary_dict()['flops_used']), ops=len(ctx._op_log))
mod.fnp = Counting()
fnp = real
rng = np.random.default_rng(11)
f32 = lambda *shape: fnp.asarray(rng.standard_normal(shape).astype(np.float32))
CASES = [('single L6 1024x1024x1024', 6, 1, (1024, 1024, 1024)), ('single L6 1024x384x1024', 6, 1, (1024, 384, 1024)),
         ('single L6 704x960x640 (odd key)', 6, 1, (704, 960, 640)), ('single L5 1024x1024x480', 5, 1, (1024, 1024, 480)),
         ('single L7 1024x1024x1024', 7, 1, (1024, 1024, 1024)), ('batched L6 P=2 1024x1024x1024', 6, 2, (1024, 1024, 1024)),
         ('batched L6 P=3 512x1024x640', 6, 3, (512, 1024, 640)),
         # through the dispatcher _mm (L = 0 here): the k-split read with its own unwind, a padded k-split, the depth-7 read
         ('_mm 1024x4096x1024 (k-split)', 0, 1, (1024, 4096, 1024)), ('_mm 896x6144x960 (k-split)', 0, 1, (896, 6144, 960)),
         ('_mm 736x4096x1024 (k-split)', 0, 1, (736, 4096, 1024)), ('_mm 1024x2048x1024', 0, 1, (1024, 2048, 1024))]
rows = []
for case, L, P, (m, k, n) in CASES:
    for call in range(2):
        with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
            if call == 0:
                As = [f32(m, k) for _ in range(P)]; Bs = [f32(k, n) for _ in range(P)]
            outs = [fnp.empty((m, n), dtype=fnp.float32) for _ in range(P)]
            bill0, ops0 = int(ctx.summary_dict()['flops_used']), len(ctx._op_log)
            COUNT.clear()
            if L == 0:
                outs = [mod._mm(As[0], Bs[0])]
            elif P == 1:
                if not mod._ipt_fits(L, m, k, n):
                    raise SystemExit('case does not fit the tree regions: ' + case)
                mod._ipt_product_ab(As[0], Bs[0], outs[0], L)
            else:
                mod._bb_product(As, Bs, outs, L)
            bill, ops = int(ctx.summary_dict()['flops_used']) - bill0, len(ctx._op_log) - ops0
        got = [np.asarray(o) for o in outs]
        ref = [np.asarray(a, dtype=np.float64) @ np.asarray(b, dtype=np.float64) for a, b in zip(As, Bs)]
        rel = max(float(np.linalg.norm(g - r) / np.linalg.norm(r)) for g, r in zip(got, ref))
        rows.append(dict(case=case, call=call, sha=hashlib.sha256(b''.join(g.tobytes() for g in got)).hexdigest(), flops=bill, ops=ops,
                         calls=dict(COUNT), rel_err_vs_float64=rel))
        print(json.dumps({key: value for key, value in rows[-1].items() if key != 'calls'}), flush=True)
out.write_text(json.dumps(dict(candidate=str(candidate), setup=setup, rows=rows, quads=dict(getattr(mod, '_L3_STATS', {}), **getattr(mod, '_L3O_STATS', {})))))
