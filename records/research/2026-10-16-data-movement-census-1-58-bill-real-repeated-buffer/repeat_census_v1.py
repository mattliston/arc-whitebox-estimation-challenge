"""[opus-exact] repeat census of the data-movement ops: for every billed reshape / ravel / flatten / astype / copy / asarray / stack /
concatenate / copyto / put / put_along_axis / take / fancy getitem, record (site, op, billed flops) and a key that says whether the
SAME result was already produced earlier in the process: views (reshape) by (data pointer, shape, strides, dtype, new shape); copies
by that plus a blake2b hash of every input's bytes (measurement only, outside the estimator). Setup = captured boundary setup.
usage (OpenBLAS env python, BLAS 1, under the guard): repeat_census_v1.py <candidate.py> <ranks> <outdir>"""
import collections, hashlib, json, os, sys, types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope._budget as fb
import flopscope._ndarray as fnd
import flopscope._symmetric as fsy
sys.path.insert(0, '/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline')
import captured_setup_replay_generic_v1 as replay
P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
candidate = Path(sys.argv[1]).resolve(); ranks = [int(x) for x in sys.argv[2].split(',')]; out = Path(sys.argv[3]).resolve(); out.mkdir()
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == os.environ['CANDIDATE_SHA256']
CF = str(candidate); fnp = flops.numpy
REC = []; STATE = dict(rank=-1, depth=0); SEEN = {}


def site():
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename != CF:
        f = f.f_back
    return (f.f_lineno, f.f_code.co_name) if f is not None else (0, '<outside>')


def desc(x, content):
    if isinstance(x, np.ndarray):
        a = np.asarray(x).view(np.ndarray)
        d = (a.__array_interface__['data'][0], a.shape, a.strides, str(a.dtype))
        if content and a.size and a.size <= 64_000_000:
            d += (hashlib.blake2b(np.ascontiguousarray(a).tobytes(), digest_size=12).hexdigest(),)
        return d
    if isinstance(x, (list, tuple)):
        return tuple(desc(y, content) for y in x)
    if isinstance(x, slice):
        return ('slice', x.start, x.stop, x.step)
    try:
        hash(x); return x
    except TypeError:
        return repr(x)[:60]


def wrap(op, fn, view):
    def w(*args, **kwargs):
        if STATE['depth']:
            return fn(*args, **kwargs)
        try:
            ctx = fb.get_active_budget()
        except Exception:
            ctx = None
        f0 = ctx.flops_used if ctx is not None else 0
        STATE['depth'] += 1
        try:
            r = fn(*args, **kwargs)
        finally:
            STATE['depth'] -= 1
        cost = (ctx.flops_used - f0) if ctx is not None else 0
        if cost > 0 or (ctx is None and view):
            key = (op, desc(args, not view), desc(tuple(sorted(kwargs.items())), not view))
            prev = SEEN.get(key)
            SEEN[key] = STATE['rank']
            if cost > 0:
                REC.append((STATE['rank'], site(), op, cost, prev))
        return r
    return w


for name in ('reshape', 'ravel', 'asarray', 'stack', 'concatenate', 'copyto', 'put', 'put_along_axis', 'take', 'copy'):
    setattr(fnp, name, wrap(name, getattr(fnp, name), name in ('reshape',)))
for cls in (fnd.FlopscopeArray, fsy.SymmetricTensor):
    for name in ('reshape', 'astype', 'copy', 'ravel', 'flatten', '__getitem__'):
        if name in cls.__dict__:
            setattr(cls, name, wrap('m.' + name, cls.__dict__[name], name == 'reshape'))
store = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {}); sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval'); mod.__file__ = CF; sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), CF, 'exec'), mod.__dict__)
expected_setup = replay.install(mod, os.environ['CAPTURED_SETUP_MANIFEST'])


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f'phase2-{rank}'


est = mod.Estimator()
STATE['rank'] = 0
with flops.BudgetContext(flop_budget=2**41, quiet=True):
    pass
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
assert replay.verify(mod, expected_setup)['all_setup_counts_exact']
used = {}
for rank in ranks:
    STATE['rank'] = rank
    weights = np.load(store/Path(manifest[rank]['path']).parent/'weights.npy')
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(MLP([weights[l] for l in range(16)], rank), 2**41), dtype=np.float64)
        used[rank] = int(ctx.flops_used)
    np.save(out/f'pred_{rank}.npy', y)
    print(json.dumps(dict(rank=rank, flops=used[rank])), flush=True)
agg = collections.defaultdict(lambda: [0, 0, 0, 0, collections.Counter()])   # calls, flops, repeat calls, repeat flops, prev ranks
for rank, (line, func), op, cost, prev in REC:
    a = agg[(rank, f'{line} {func} {op}')]; a[0] += 1; a[1] += cost
    if prev is not None:
        a[2] += 1; a[3] += cost; a[4][prev] += 1
rows = [dict(rank=k[0], site=k[1], calls=v[0], flops=v[1], repeat_calls=v[2], repeat_flops=v[3], prev=dict(v[4])) for k, v in agg.items()]
(out/'repeats.json').write_text(json.dumps(dict(candidate=CF, flops=used, rows=sorted(rows, key=lambda r: -r['repeat_flops']))) + '\n')
print('REPEATS_DONE', flush=True)
