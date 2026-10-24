"""[opus-solo] matmul/einsum bill map: exec a candidate, patch fnp.matmul / fnp.einsum / the remote array's __matmul__, run setup + one
predict (PB_ALL) and record every call: caller function + line, operand shapes, billed flops delta, layer. Aggregates by site.
usage: probe_matmul_v1.py <candidate.py> <seed> <out.json>   env: PB_ALL=1, BLAS threads=1"""
import json, os, sys, types, collections
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
O = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1')
sys.path.insert(0, str(O/'analyst/fresh')); import loader
candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3]).resolve(); assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
CTX = [None]; ROWS = []
def shape(x):
    try: return list(int(v) for v in x.shape)
    except Exception: return None
def record(kind, args, f0):
    c = CTX[0]
    fr = sys._getframe(2)
    # walk up to the first frame inside the candidate module
    while fr is not None and fr.f_code.co_filename != str(candidate): fr = fr.f_back
    site = (fr.f_code.co_name + ':' + str(fr.f_lineno)) if fr is not None else '?'
    lay = mod.__dict__.get('_PHYS_LAYER', [None])[0]
    ROWS.append(dict(kind=kind, site=site, shapes=[shape(a) for a in args if shape(a) is not None][:3], sub=[a for a in args if isinstance(a, str)][:1],
                     flops=int(c.summary_dict()['flops_used'] or 0) - f0, layer=(int(lay) if isinstance(lay, (int, np.integer)) else None)))
_mm0 = fnp.matmul; _es0 = fnp.einsum
def mm(*a, **k):
    c = CTX[0]
    if c is None: return _mm0(*a, **k)
    f0 = int(c.summary_dict()['flops_used'] or 0); r = _mm0(*a, **k); record('matmul', a, f0); return r
def es(*a, **k):
    c = CTX[0]
    if c is None: return _es0(*a, **k)
    f0 = int(c.summary_dict()['flops_used'] or 0); r = _es0(*a, **k); record('einsum', a, f0); return r
fnp.matmul = mm; fnp.einsum = es
if mod.__dict__.get('fnp') is fnp: pass
probe = fnp.zeros((2, 2), dtype=fnp.float32); cls = type(probe)
_op0 = cls.__matmul__
def op(self, other):
    c = CTX[0]
    if c is None: return _op0(self, other)
    f0 = int(c.summary_dict()['flops_used'] or 0); r = _op0(self, other); record('@', (self, other), f0); return r
try: cls.__matmul__ = op; patched_op = True
except Exception as e: patched_op = str(e)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=seed, name=f'probe-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx; y = est.predict(mlp, 2**41); total = int(ctx.summary_dict()['flops_used'])
CTX[0] = None
agg = collections.defaultdict(lambda: [0, 0, set()])
for r in ROWS:
    a = agg[(r['kind'], r['site'])]; a[0] += 1; a[1] += r['flops']; a[2].add(tuple(tuple(s) for s in r['shapes']))
summary = sorted(((k[0], k[1], v[0], v[1], [list(map(list, s)) for s in list(v[2])[:3]]) for k, v in agg.items()), key=lambda t: -t[3])
json.dump(dict(candidate=str(candidate), seed=seed, total_flops=total, patched_op=patched_op, n_calls=len(ROWS), by_site=summary, rows=ROWS), out.open('x'), indent=0)
print(json.dumps(dict(seed=seed, total_flops=total, calls=len(ROWS), matmul_flops=sum(r['flops'] for r in ROWS), patched_op=patched_op)))
for k, s, n, f, sh in summary[:30]: print('%-7s %-36s calls %4d  %7.3f GF (%4.1f%%)  %s' % (k, s, n, f / 1e9, 100 * f / total, sh[:2]))
