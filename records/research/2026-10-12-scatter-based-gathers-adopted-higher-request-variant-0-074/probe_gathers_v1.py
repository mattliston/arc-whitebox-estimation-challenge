"""TAG opus_engineer a62 (2026-10-12 01:1xZ, lead: harvest small byte-identical bill cuts): MEASUREMENT of every fancy-index gather (getitem
with an integer or boolean array index) one prediction makes, by call site: source shape and dtype, output size, its bill (flopscope:
4 per output element in float32, 8 in float64), whether the same gather (same source object, equal index) was already made in this
layer, and the price of the same rows by scatter (put_along_axis: 1 per SOURCE element in float32, 2 in float64; rows only, unique
indices). Not a submission path: it wraps FlopscopeArray.__getitem__ in this process only.
usage: probe_gathers_v1.py <candidate.py> <seed> <out.json>"""
import collections
import hashlib
import json
import sys
import types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _ndarray

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader

candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
log = []
seen = {}
orig = _ndarray.FlopscopeArray.__getitem__


def fancy(key):
    keys = key if isinstance(key, tuple) else (key,)
    return [k for k in keys if isinstance(k, np.ndarray) and k.dtype.kind in 'iub' and k.ndim >= 1]


def probe(self, key):
    res = orig(self, key)
    try:
        arrs = fancy(key)
        if arrs and isinstance(res, np.ndarray):
            f = sys._getframe(1)
            site = '%s:%d' % (f.f_code.co_name, f.f_lineno)
            layer = mod._PHYS_LAYER[0] if hasattr(mod, '_PHYS_LAYER') else None
            plain = [np.asarray(a).view(np.ndarray) for a in arrs]   # local copies: no billed op in the probe's own arithmetic
            sig = (id(self), tuple(hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()[:12] for a in plain))
            repeat = seen.get(sig) == layer
            seen[sig] = layer
            per = 4 if self.dtype.itemsize <= 4 else 8
            rowgather = isinstance(key, np.ndarray) or (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], np.ndarray) and isinstance(key[1], slice) and key[1] == slice(None))
            idx = plain[0]
            uniq = idx.dtype.kind != 'b' and idx.ndim == 1 and len(set(idx.tolist())) == idx.shape[0]
            scatter = (self.shape[0] * (int(np.prod(self.shape[1:])) if self.ndim > 1 else 1)) * (1 if self.dtype.itemsize <= 4 else 2) if rowgather and uniq else None
            log.append(dict(site=site, layer=layer, src=list(self.shape), dtype=str(self.dtype), out=list(res.shape), bill=int(res.size * per),
                            repeat=bool(repeat), rowgather=bool(rowgather), unique=bool(uniq), scatter=scatter))
    except Exception as e:
        log.append(dict(error=repr(e)))
    return res


_ndarray.FlopscopeArray.__getitem__ = probe
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
_ndarray.FlopscopeArray.__getitem__ = orig
total = int(ctx.summary_dict()['flops_used'])
agg = collections.defaultdict(lambda: dict(n=0, bill=0, repeat_bill=0, scatter_save=0, shapes=set()))
for r in log:
    if 'error' in r:
        continue
    a = agg[r['site']]
    a['n'] += 1; a['bill'] += r['bill']
    if r['repeat']:
        a['repeat_bill'] += r['bill']
    if r['scatter'] is not None and r['scatter'] < r['bill']:
        a['scatter_save'] += r['bill'] - r['scatter']
    a['shapes'].add('%s->%s %s' % (tuple(r['src']), tuple(r['out']), r['dtype']))
rows = sorted(agg.items(), key=lambda kv: -kv[1]['bill'])
print('prediction bill %d; fancy gathers %d calls, %.3f%% of the bill; repeats %.3f%%; scatter-cheaper part saves %.3f%%' % (
    total, len([r for r in log if 'error' not in r]), 100 * sum(a['bill'] for a in agg.values()) / total,
    100 * sum(a['repeat_bill'] for a in agg.values()) / total, 100 * sum(a['scatter_save'] for a in agg.values()) / total))
for site, a in rows[:30]:
    print('%-44s n=%3d bill %.4f%% repeat %.4f%% scatter-save %.4f%%  %s' % (site, a['n'], 100 * a['bill'] / total, 100 * a['repeat_bill'] / total,
          100 * a['scatter_save'] / total, '; '.join(sorted(a['shapes']))[:150]))
out.write_text(json.dumps(dict(seed=seed, total=total, errors=[r for r in log if 'error' in r][:5],
                               sites={s: dict(a, shapes=sorted(a['shapes'])) for s, a in rows}), indent=0))
