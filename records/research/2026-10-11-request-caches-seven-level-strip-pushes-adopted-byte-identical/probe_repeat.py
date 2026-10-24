"""TAG opus_engineer a56 probe (2026-10-11, lead 03:5xZ request census): which no-arithmetic requests repeat on the SAME live object
with the SAME key (a view, reshape, transpose, unstack or float32 conversion that a cache could hand out again for no request), and
which allocations repeat with the same shape at the same site (a persistent buffer could take them). In process (local flopscope),
networks in one process (first = cold); counts for the LAST network. Weak references keep the object identity honest.
usage: probe_repeat.py <candidate.py> <comma ranks>"""
from pathlib import Path
import collections
import json
import sys
import types
import weakref
import numpy as np
import flopscope as flops
from flopscope import _ndarray as _nd
fnp = flops.numpy

P2 = Path(__file__).resolve().parents[3]
STORE = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
candidate, ranks = Path(sys.argv[1]).resolve(), [int(x) for x in sys.argv[2].split(',')]
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe_repeat'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
FILE = str(candidate)
calls, repeats, seen = collections.Counter(), collections.Counter(), {}
allocs = collections.Counter()


def site():
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename != FILE:
        f = f.f_back
    return (f.f_lineno, f.f_code.co_name) if f is not None else (0, '?')


def keyrepr(key):
    if isinstance(key, tuple):
        return tuple(keyrepr(k) for k in key)
    if isinstance(key, slice):
        return ('s', key.start, key.stop, key.step)
    if isinstance(key, (int, np.integer)) or key is None or key is Ellipsis:
        return key
    return ('obj', id(key))      # an index array: a gather, cached only if the same index object comes again


def note(kind, base, key):
    s = site() + (kind,)
    calls[s] += 1
    k = (s, id(base))
    entry = seen.get(k)
    if entry is not None and entry[0]() is base and key in entry[1]:
        repeats[s] += 1
    else:
        if entry is None or entry[0]() is not base:
            try:
                entry = seen[k] = (weakref.ref(base), set())
            except TypeError:
                return
        entry[1].add(key)


_getitem = _nd.FlopscopeArray.__getitem__
def getitem(self, key):
    note('__getitem__', self, keyrepr(key)); return _getitem(self, key)
_nd.FlopscopeArray.__getitem__ = getitem
for name in ('reshape', 'transpose', 'unstack'):
    orig = getattr(fnp, name)
    def wrapped(a, *args, _orig=orig, _name=name, **kw):
        if isinstance(a, _nd.FlopscopeArray):
            note(_name, a, (repr(args), repr(sorted(kw.items()))))
        return _orig(a, *args, **kw)
    setattr(fnp, name, wrapped)
_asarray = fnp.asarray
def asarray(a, *args, **kw):
    if isinstance(a, _nd.FlopscopeArray) and (args or 'dtype' in kw):
        note('asarray', a, repr((args, sorted(kw.items()))))
    return _asarray(a, *args, **kw)
fnp.asarray = asarray
for name in ('empty', 'zeros'):
    orig = getattr(fnp, name)
    def walloc(shape, *args, _orig=orig, _name=name, **kw):
        allocs[site() + (_name, repr(shape))] += 1
        return _orig(shape, *args, **kw)
    setattr(fnp, name, walloc)

est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = json.loads((STORE/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
for rank in ranks:
    calls.clear(); repeats.clear(); seen.clear(); allocs.clear()
    weights = np.load(STORE/Path(manifest[rank]['path']).parent/'weights.npy', allow_pickle=False)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[weights[l] for l in range(16)], seed=0, name=f'phase2-{rank}')
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        ops = len(ctx._op_log)
    print(json.dumps(dict(rank=rank, ops=ops, calls=sum(calls.values()), repeats=sum(repeats.values()))), flush=True)
top = sorted(calls, key=lambda s: -repeats[s])
print('site (line function kind): calls, repeats on the same live object with the same key')
for s in top[:45]:
    print(f'  {s[0]:6d} {s[1]:32s} {s[2]:12s} {calls[s]:6d} {repeats[s]:6d}')
by_alloc = collections.Counter()
for (ln, fn, kind, shape), n in allocs.items():
    by_alloc[(ln, fn, kind)] += n
print('allocations (line function kind): calls; distinct shapes')
for s, n in by_alloc.most_common(25):
    shapes = [k[3] for k in allocs if k[:3] == s]
    print(f'  {s[0]:6d} {s[1]:32s} {s[2]:6s} {n:6d}  shapes {len(shapes)}')
