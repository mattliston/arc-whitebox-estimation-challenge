"""TAG opus_engineer a62 (2026-10-12, lead 13:4xZ: a BILL CENSUS of the fired S50 twin for non-retraining cuts; measurement only).
One prediction of the candidate on network A, in-process, under a hook on flopscope's single charging point
(BudgetContext._charge_op: op name, subscripts, shapes, resolved billing dtype, adjusted cost), each charge attributed to the
innermost frame of the candidate file (function:line) and to the release / engineer fragment that line sits in.
Plus, through wrappers of the fnp functions, the FlopscopeArray operators and the file's own product helpers (_mm,
_shared_old384_mm), recorded per call site:
  - for 2-D square product results: the relative asymmetry |R - R'| / |R| (plain numpy views, not billed);
  - a fingerprint of every array argument (shape, dtype, three sampled sums), in call order.
Then a second prediction on network B with the fingerprints only: a call site whose argument fingerprints are equal call for call
on both networks computes the same numbers for every network (a constant inside predict).
Writes <out.json> (all aggregates) and prints the four lists the lead asked for: (1) the float64-rate share by fragment and site,
(2) square products with symmetric results and their bill, (3) network-independent bill by site, (4) the top 20 sites by bill.
usage: census_bill_v1.py <candidate.py> <seed A> <seed B> <out.json>"""
import collections
import json
import re
import sys
import types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _budget, _ndarray
from flopscope._dtype_billing import resolve_billing_dtype

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader

candidate, seed_a, seed_b, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
assert not out.exists()
text = candidate.read_text()
CFILE = str(candidate)

# line -> fragment (the BEGIN / END markers of the release and engineer fragments; the rest is 'base')
frag_of = {}
stack = []
for i, line in enumerate(text.split('\n'), 1):
    m = re.match(r'# BEGIN (OPUS-[A-Z-]+ (?:FRAGMENT |a62 |A62 FRAGMENT )?)(\S+)', line)
    if m:
        stack.append(m.group(2))
    if stack:
        frag_of[i] = stack[-1]
    if line.startswith('# END ') and stack:
        stack.pop()

stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_census'); mod.__file__ = CFILE; sys.modules[mod.__name__] = mod
exec(compile(text, CFILE, 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))

MODE = ['off']          # 'A' (full census), 'B' (fingerprints only), 'off'
charges = collections.defaultdict(lambda: dict(n=0, bill=0, raw=0, shapes=None, subscripts=None))
fps = {'A': collections.defaultdict(list), 'B': collections.defaultdict(list)}
sym = collections.defaultdict(lambda: dict(n=0, square=0, symmetric=0, bill=0, max_asym=0.0, shape=None))


def site_of(frame):
    f = frame
    while f is not None:
        if f.f_code.co_filename == CFILE:
            return '%s:%d' % (f.f_code.co_name, f.f_lineno), f.f_lineno
        f = f.f_back
    return 'outside', 0


ORIG_CHARGE = _budget.BudgetContext._charge_op


def charge(self, op_name, flop_cost, subscripts, shapes, **kw):
    if MODE[0] != 'A':
        return ORIG_CHARGE(self, op_name, flop_cost, subscripts, shapes, **kw)
    u0 = self._flops_used
    res = ORIG_CHARGE(self, op_name, flop_cost, subscripts, shapes, **kw)
    bill = self._flops_used - u0
    d = kw.get('dtypes')
    try:
        dt = resolve_billing_dtype(d).name if d else 'neutral'
    except Exception:
        dt = 'unresolved'
    site, line = site_of(sys._getframe(1))
    c = charges[(site, frag_of.get(line, 'base'), op_name, dt)]
    c['n'] += 1; c['bill'] += bill; c['raw'] += int(flop_cost)
    if c['shapes'] is None:
        c['shapes'] = str(shapes)[:160]; c['subscripts'] = subscripts
    return res


_budget.BudgetContext._charge_op = charge


def fingerprint(a):
    v = np.asarray(a).view(np.ndarray)
    if v.size == 0:
        return (v.shape, str(v.dtype))
    idx = np.unique(np.linspace(0, v.size - 1, 97).astype(np.int64))   # 97 fixed positions, any memory layout (flatiter: no full copy)
    with np.errstate(all='ignore'):
        s = v.flat[idx].astype(np.float64)
        return (v.shape, str(v.dtype), float(s.sum()), float((s * s).sum()), float(s[-1]))


def asym(r):
    v = np.asarray(r).view(np.ndarray)
    if v.ndim != 2 or v.shape[0] != v.shape[1] or v.shape[0] < 32 or v.dtype.kind != 'f':
        return None
    x = v.astype(np.float64)
    n = np.linalg.norm(x)
    return float(np.linalg.norm(x - x.T) / n) if n > 0 else 0.0


CTX = [None]


def wrap(name, fn, product=False):
    def inner(*args, **kwargs):
        mode = MODE[0]
        if mode == 'off':
            return fn(*args, **kwargs)
        site, line = site_of(sys._getframe(1))
        key = (name, site)
        fps[mode][key].append(tuple(fingerprint(a) for a in args if isinstance(a, np.ndarray)))
        if mode == 'A' and product:
            u0 = CTX[0].flops_used
            r = fn(*args, **kwargs)
            s = sym[key]
            s['n'] += 1; s['bill'] += CTX[0].flops_used - u0
            a = asym(r)
            if a is not None:
                s['square'] += 1; s['max_asym'] = max(s['max_asym'], a); s['shape'] = str(np.shape(r))
                if a < 1e-5:
                    s['symmetric'] += 1
            return r
        return fn(*args, **kwargs)
    return inner


NAMES = ('outer', 'multiply', 'add', 'subtract', 'divide', 'true_divide', 'sum', 'sqrt', 'exp', 'power', 'einsum', 'matmul', 'dot',
         'maximum', 'minimum', 'clip', 'abs', 'square', 'mean', 'diag', 'trace', 'where', 'tanh', 'log1p', 'log', 'erfc', 'cumprod', 'prod',
         'reshape', 'concatenate', 'stack', 'copyto', 'take', 'put_along_axis', 'transpose', 'fill_diagonal', 'negative', 'reciprocal')
PRODUCTS = {'einsum', 'matmul', 'dot', 'outer'}
for nm in NAMES:
    if hasattr(mod.fnp, nm):
        setattr(mod.fnp, nm, wrap('fnp.' + nm, getattr(fnp, nm), product=nm in PRODUCTS))
for helper in ('_mm', '_shared_old384_mm'):
    if helper in mod.__dict__:
        mod.__dict__[helper] = wrap(helper, mod.__dict__[helper], product=True)
OPS = {'__mul__': 'mul', '__rmul__': 'rmul', '__add__': 'add_op', '__radd__': 'radd', '__sub__': 'sub', '__rsub__': 'rsub', '__truediv__': 'div',
       '__rtruediv__': 'rdiv', '__matmul__': 'matmul_op', '__pow__': 'pow', '__neg__': 'neg'}
saved = {}
for dunder, short in OPS.items():
    if dunder in _ndarray.FlopscopeArray.__dict__:
        saved[dunder] = _ndarray.FlopscopeArray.__dict__[dunder]
        setattr(_ndarray.FlopscopeArray, dunder, wrap(short, saved[dunder], product=(dunder == '__matmul__')))


def predict(seed, mode):
    w = loader.weights(seed)
    mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        CTX[0] = ctx; MODE[0] = mode
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        MODE[0] = 'off'; CTX[0] = None
    return int(ctx.summary_dict()['flops_used']), y


total_a, ya = predict(seed_a, 'A')
total_b, yb = predict(seed_b, 'B')
for dunder, fn in saved.items():
    setattr(_ndarray.FlopscopeArray, dunder, fn)
_budget.BudgetContext._charge_op = ORIG_CHARGE

pct = lambda x: 100.0 * x / total_a
# (4) top sites by bill (innermost candidate frame), with their op forms
by_site = collections.defaultdict(lambda: dict(bill=0, n=0, forms=collections.Counter(), frag=None))
for (site, frag, op, dt), c in charges.items():
    s = by_site[site]; s['bill'] += c['bill']; s['n'] += c['n']; s['frag'] = frag
    s['forms'][(op, dt, c['subscripts'], c['shapes'])] += c['bill']
top = sorted(by_site.items(), key=lambda kv: -kv[1]['bill'])
# (1) float64 rate: by fragment and by site
f64_frag = collections.Counter(); f64_site = collections.Counter(); f32_like = collections.Counter()
for (site, frag, op, dt), c in charges.items():
    if dt == 'float64':
        f64_frag[frag] += c['bill']; f64_site[(site, frag, op)] += c['bill']
f64_total = sum(f64_frag.values())
# (3) network-independent sites: argument fingerprints equal call for call on A and B
const = []
for key, la in fps['A'].items():
    lb = fps['B'].get(key)
    if lb is not None and len(la) == len(lb) and la and all(x == y for x, y in zip(la, lb)) and any(len(x) for x in la):
        const.append(key)
const_sites = collections.Counter()
for (name, site) in const:
    const_sites[site] += 0
for (site, frag, op, dt), c in charges.items():
    if site in const_sites:
        const_sites[site] += c['bill']
partial = {}
for key, la in fps['A'].items():
    lb = fps['B'].get(key)
    if lb and la and key not in const:
        eq = sum(1 for x, y in zip(la, lb) if x == y and len(x))
        if eq:
            partial[key] = (eq, len(la), len(lb))
print('network A %d: bill %d; network B %d: bill %d' % (seed_a, total_a, seed_b, total_b))
print('\n(1) FLOAT64-RATE share of the bill: %.3f%%; by fragment:' % pct(f64_total))
for frag, b in f64_frag.most_common(12):
    print('   %-46s %.3f%%' % (frag, pct(b)))
print('   sites above 0.05%:')
for (site, frag, op), b in f64_site.most_common(40):
    if pct(b) < 0.05:
        break
    print('   %-44s %-30s %-14s %.3f%%' % (site, frag[:30], op, pct(b)))
print('\n(2) SQUARE PRODUCT RESULTS that are symmetric (|R - R\'| / |R| < 1e-5), by wrapped call site:')
for key, s in sorted(sym.items(), key=lambda kv: -kv[1]['bill']):
    if s['symmetric']:
        print('   %-12s %-44s calls %4d square %4d symmetric %4d  bill %.3f%%  max asym %.1e  %s' % (key[0], key[1], s['n'], s['square'], s['symmetric'],
              pct(s['bill']), s['max_asym'], s['shape']))
print('\n(3) NETWORK-INDEPENDENT sites (argument fingerprints equal on both networks, every call), bill on A:')
for site, b in const_sites.most_common(25):
    if b <= 0:
        continue
    print('   %-46s %.4f%%' % (site, pct(b)))
print('   total %.3f%% of the bill' % pct(sum(const_sites.values())))
print('\n(4) TOP 20 SITES by bill (innermost frame of the file):')
for site, s in top[:20]:
    (op, dt, sub, shp), b = s['forms'].most_common(1)[0]
    print('   %-44s %-28s %.3f%%  calls %5d  main form %s %s %s %s' % (site, (s['frag'] or 'base')[:28], pct(s['bill']), s['n'], op, dt, sub or '', shp[:70]))
out.write_text(json.dumps(dict(
    candidate=CFILE, seed_a=seed_a, seed_b=seed_b, total_a=total_a, total_b=total_b, f64_total=f64_total,
    f64_by_fragment=dict(f64_frag), f64_by_site=[dict(site=k[0], fragment=k[1], op=k[2], bill=v) for k, v in f64_site.most_common()],
    charges=[dict(site=k[0], fragment=k[1], op=k[2], dtype=k[3], **{x: v for x, v in c.items()}) for k, c in charges.items()],
    symmetric=[dict(op=k[0], site=k[1], **s) for k, s in sym.items() if s['square']],
    constant_sites=dict(const_sites), constant_keys=[list(k) for k in const], partial_constant=[dict(op=k[0], site=k[1], equal=v[0], calls_a=v[1], calls_b=v[2]) for k, v in partial.items()],
    top_sites=[dict(site=k, bill=v['bill'], n=v['n'], fragment=v['frag'], forms=[[list(f), b] for f, b in v['forms'].most_common(4)]) for k, v in top[:60]],
), indent=0))
print('written', out)
