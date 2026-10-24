"""TAG opus_engineer a62 (2026-10-12, lead 15:0xZ items 5 / 6; measurement only): the bill of one prediction by LAYER and by STEP.
The hook on flopscope's charging point (as census_bill_v1.py) reads, for every charge, the layer from the loop variable l of the
core _predict_binary frame on the stack, and the STEP = the candidate function that frame called (the first-level operation of the
layer: the push, the read, the pair map, the head hook, the F4c step, ...; a charge in the loop body itself is '_predict_binary:<line>').
Then, if asked, PERTURBATION TESTS for item 6: a named step function is wrapped so that, at the layers given, every float array it
returns is multiplied by (1 + 1e-6); one prediction per test, compared with the unperturbed prediction (final layer and all layers,
bit for bit). A step whose perturbation leaves the final layer's prediction bit-identical cannot reach the score there.
usage: census_layer_v1.py <candidate.py> <seed> <out.json> [<step>@<layers> ...]   (layers like 15 or 0-14)"""
import collections
import json
import sys
import types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O/'analyst/fresh'))
import loader

candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
tests = sys.argv[4:]
assert not out.exists()
CFILE = str(candidate)
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_layer'); mod.__file__ = CFILE; sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), CFILE, 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
ON = [False]
by_layer = collections.Counter(); by_step = collections.Counter(); by_site15 = collections.Counter()


def where(frame):
    inner = None; step = None; f = frame
    while f is not None:
        if f.f_code.co_filename == CFILE:
            if inner is None:
                inner = '%s:%d' % (f.f_code.co_name, f.f_lineno)
            if f.f_code.co_name == '_predict_binary' and 'l' in f.f_locals and 'means' in f.f_locals:
                return f.f_locals['l'], (step or '_predict_binary:%d' % f.f_lineno), inner
            step = '%s' % f.f_code.co_name
        f = f.f_back
    return None, step or 'outside', inner or 'outside'


ORIG = _budget.BudgetContext._charge_op


def charge(self, *args, **kw):
    if not ON[0]:
        return ORIG(self, *args, **kw)
    u0 = self._flops_used
    r = ORIG(self, *args, **kw)
    b = self._flops_used - u0
    l, step, inner = where(sys._getframe(1))
    by_layer[l] += b; by_step[(l, step)] += b
    if l == 15:
        by_site15[(step, inner, args[0])] += b
    return r


_budget.BudgetContext._charge_op = charge
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')


def predict(count):
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        ON[0] = count
        y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
        ON[0] = False
    return int(ctx.summary_dict()['flops_used']), y


total, y0 = predict(True)
pct = lambda x: 100.0 * x / total
print('network %d: bill %d' % (seed, total))
print('\n(5) BILL BY LAYER (layer of the core loop; None = outside the loop):')
for l in sorted(by_layer, key=lambda x: (-1 if x is None else x)):
    print('   L%-4s %6.3f%%' % (l, pct(by_layer[l])))
print('\n   L15 by step:')
for (l, step), b in sorted(by_step.items(), key=lambda kv: -kv[1]):
    if l == 15 and pct(b) >= 0.005:
        print('     %-40s %.4f%%' % (step, pct(b)))
print('   L15 top sites:')
for (step, inner, op), b in by_site15.most_common(25):
    print('     %-28s %-36s %-12s %.4f%%' % (step[:28], inner[:36], op, pct(b)))
print('\n   largest steps over all layers:')
steps_all = collections.Counter()
for (l, step), b in by_step.items():
    steps_all[step] += b
for step, b in steps_all.most_common(25):
    print('     %-40s %.3f%%' % (step, pct(b)))
results = []
for t in tests:
    name, layers = t.split('@')
    lo, hi = (int(x) for x in (layers.split('-') if '-' in layers else (layers, layers)))
    fn = mod.__dict__.get(name)
    if fn is None:
        results.append(dict(test=t, error='no such function')); print('TEST %s: no such function' % t); continue

    def bump(x):
        if isinstance(x, np.ndarray) and x.dtype.kind == 'f' and x.size:
            return x * (1 + 1e-6)
        if isinstance(x, tuple):
            return tuple(bump(e) for e in x)
        if isinstance(x, list):
            return [bump(e) for e in x]
        return x

    def wrapped(*a, _fn=fn, **k):
        r = _fn(*a, **k)
        l, _s, _i = where(sys._getframe(1))
        return bump(r) if (l is not None and lo <= l <= hi) else r

    mod.__dict__[name] = wrapped
    try:
        _t, y = predict(False)
        same_final = bool(np.array_equal(y[15], y0[15])); same_all = bool(np.array_equal(y, y0))
        d15 = float(np.max(np.abs(y[15] - y0[15])))
        layers_changed = [l for l in range(16) if not np.array_equal(y[l], y0[l])]
        results.append(dict(test=t, final_identical=same_final, all_identical=same_all, max_final_change=d15, layers_changed=layers_changed))
        print('TEST %-34s final layer %s (max change %.3e); layers changed %s' % (t, 'BIT-IDENTICAL' if same_final else 'CHANGED', d15, layers_changed))
    except Exception as e:
        results.append(dict(test=t, error=repr(e)[:300])); print('TEST %s: error %r' % (t, e))
    finally:
        mod.__dict__[name] = fn
_budget.BudgetContext._charge_op = ORIG
out.write_text(json.dumps(dict(candidate=CFILE, seed=seed, total=total, by_layer={str(k): v for k, v in by_layer.items()},
                               by_step=[dict(layer=k[0], step=k[1], bill=v) for k, v in by_step.items()],
                               l15_sites=[dict(step=k[0], site=k[1], op=k[2], bill=v) for k, v in by_site15.most_common()], tests=results), indent=0))
print('written', out)
