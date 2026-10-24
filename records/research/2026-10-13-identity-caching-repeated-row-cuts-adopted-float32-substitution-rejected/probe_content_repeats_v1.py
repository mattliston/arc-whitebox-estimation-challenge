"""TAG opus_engineer2 a62 probe_content_repeats_v1 (file written 2026-10-12T23:04:23Z; lead 22:5xZ: hunt byte-identical bill cuts, scout-native's list D):
MEASUREMENT ONLY. probe_repeats_v1 (the engineer, 10-12 01:3xZ) counted a repeat only when the same op met the same input OBJECTS in the
same layer (0.015% of S48b's bill). This counts a repeat by CONTENT: the same op with the same scalar arguments on inputs whose content is
identical (a fingerprint per input array: exact sum of its 32-bit words + sha1 of every 13th word, the probe's own unbilled numpy reads) recomputes an output already computed earlier in the prediction,
in the same layer or any earlier one. Each repeat is reported with its bill, its first and repeating call sites, the layers, and whether
every array input is the very same object as at the first call (cacheable by identity, the submission's rule) or only equal content
(cacheable only by caching upstream). Wraps the fnp functions by name, FlopscopeArray's arithmetic operators, __getitem__ (gathers) and the
layout methods reshape / astype / copy / ravel / flatten, in this process only. The prebind guard is lifted (PB_ALL as run_cert_v2) so the
prediction's ops are the steady state. usage: probe_content_repeats_v1.py <candidate.py> <seed> <out.json>"""
import collections
import hashlib
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from flopscope import _ndarray

O = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(O / 'analyst/fresh'))
import loader

candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
mod._PB_WALL_S = 1e9
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
CTX = [None]
DEPTH = [0]
first = {}                                   # content key -> (layer, site, input ids)
agg = collections.defaultdict(lambda: dict(n=0, bill=0, rep_n=0, rep_bill=0, rep_same_obj_bill=0, rep_same_layer_bill=0, firsts=collections.Counter(), shapes=None))
calls = [0]


def chash(a):   # a discovery fingerprint (every candidate is checked by reading before any cut): exact sum of all 32-bit words + sha1 of every 13th
    v = np.asarray(a).view(np.ndarray)       # plain numpy view: the probe's reads are not billed
    if v.size == 0:
        return (v.shape, str(v.dtype), '')
    b = np.ascontiguousarray(v).reshape(-1).view(np.uint8)
    if b.size % 4:
        return (v.shape, str(v.dtype), hashlib.sha1(b.data).hexdigest())
    u = b.view(np.uint32)
    return (v.shape, str(v.dtype), int(u.sum(dtype=np.uint64)), hashlib.sha1(np.ascontiguousarray(u[::13]).data).hexdigest())


def key_of(name, args, kwargs):
    parts, ids = [name], []
    for a in args:
        if isinstance(a, np.ndarray):
            parts.append(('arr',) + chash(a)); ids.append(id(a))
        elif isinstance(a, (int, float, str, bool, type(None), np.integer, np.floating)):
            parts.append(('val', repr(a)))
        elif isinstance(a, (tuple, list)) and all(isinstance(x, (int, float, str, bool, type(None), slice, np.integer)) for x in a):
            parts.append(('seq', repr(a)))
        elif isinstance(a, slice):
            parts.append(('slice', repr(a)))
        elif isinstance(a, (tuple, list)):
            sub = []
            for x in a:
                if isinstance(x, np.ndarray):
                    sub.append(chash(x)); ids.append(id(x))
                else:
                    sub.append(repr(x) if isinstance(x, (int, float, str, bool, type(None), slice, np.integer)) else type(x).__name__)
            parts.append(('seq', tuple(sub)))
        else:
            parts.append(('obj', type(a).__name__))
    for k in sorted(kwargs):
        v = kwargs[k]
        if k == 'out':
            continue                         # the written buffer is not an input (its prior content does not enter the result)
        if isinstance(v, np.ndarray):
            parts.append((k, chash(v))); ids.append(id(v))
        else:
            parts.append((k, repr(v) if isinstance(v, (int, float, str, bool, type(None), np.integer, np.floating, tuple)) else type(v).__name__))
    return tuple(parts), tuple(ids)


def wrap(name, fn):
    def inner(*args, **kwargs):
        ctx = CTX[0]
        if ctx is None or DEPTH[0]:
            return fn(*args, **kwargs)
        DEPTH[0] += 1
        try:
            u0 = ctx.flops_used
            res = fn(*args, **kwargs)
            bill = ctx.flops_used - u0
        finally:
            DEPTH[0] -= 1
        if bill < 1000:                      # 37k billed calls x < 1000 flops < 0.015% of the bill in all
            return res
        calls[0] += 1
        f = sys._getframe(1)
        site = '%s:%d' % (f.f_code.co_name, f.f_lineno)
        layer = mod._PHYS_LAYER[0]
        k, ids = key_of(name, args, kwargs)
        a = agg[(name, site)]
        a['n'] += 1; a['bill'] += bill
        if a['shapes'] is None:
            a['shapes'] = str([tuple(np.shape(x)) for x in args if isinstance(x, np.ndarray)])[:120]
        prev = first.get(k)
        if prev is None:
            first[k] = (layer, site, ids)
        else:
            a['rep_n'] += 1; a['rep_bill'] += bill
            if prev[2] == ids:
                a['rep_same_obj_bill'] += bill
            if prev[0] == layer:
                a['rep_same_layer_bill'] += bill
            a['firsts']['%s (L%s)' % (prev[1], prev[0])] += 1
        return res
    return inner


FN = ('outer', 'multiply', 'add', 'subtract', 'divide', 'true_divide', 'sum', 'sqrt', 'exp', 'power', 'einsum', 'matmul', 'dot', 'maximum',
      'minimum', 'clip', 'abs', 'square', 'mean', 'diag', 'trace', 'where', 'tanh', 'log1p', 'log', 'erfc', 'cumprod', 'prod', 'take', 'stack',
      'concatenate', 'copyto', 'put', 'put_along_axis', 'take_along_axis', 'reshape', 'transpose', 'tensordot', 'negative', 'reciprocal', 'cumsum',
      'unstack', 'split', 'fill_diagonal', 'diagonal', 'triu', 'tril', 'zeros_like', 'ones_like', 'full', 'arange', 'linspace', 'sort', 'argsort',
      'greater', 'less', 'equal', 'logical_and', 'logical_or', 'logical_not', 'isfinite', 'nan_to_num', 'sign', 'floor', 'ceil', 'round', 'cos',
      'sin', 'arctan2', 'hypot', 'expm1', 'erf')
for name in FN:
    if hasattr(mod.fnp, name):
        setattr(mod.fnp, name, wrap('fnp.' + name, getattr(fnp, name)))
OPS = {'__mul__': 'mul', '__rmul__': 'rmul', '__add__': 'add_op', '__radd__': 'radd', '__sub__': 'sub', '__rsub__': 'rsub', '__truediv__': 'div',
       '__rtruediv__': 'rdiv', '__matmul__': 'matmul_op', '__rmatmul__': 'rmatmul_op', '__pow__': 'pow', '__neg__': 'neg', '__getitem__': 'getitem',
       'reshape': 'm.reshape', 'astype': 'm.astype', 'copy': 'm.copy', 'ravel': 'm.ravel', 'flatten': 'm.flatten', 'sum': 'm.sum', 'mean': 'm.mean'}
saved = {}
for dunder, short in OPS.items():
    if dunder in _ndarray.FlopscopeArray.__dict__:
        saved[dunder] = _ndarray.FlopscopeArray.__dict__[dunder]
        setattr(_ndarray.FlopscopeArray, dunder, wrap(short, saved[dunder]))
w = loader.weights(seed)
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
t0 = time.monotonic()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    CTX[0] = ctx
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    CTX[0] = None
for dunder, fn in saved.items():
    setattr(_ndarray.FlopscopeArray, dunder, fn)
total = int(ctx.summary_dict()['flops_used'])
wrapped = sum(a['bill'] for a in agg.values())
rep = sum(a['rep_bill'] for a in agg.values())
same = sum(a['rep_same_obj_bill'] for a in agg.values())
lay = sum(a['rep_same_layer_bill'] for a in agg.values())
pred_sha = hashlib.sha256(y.astype(np.float32).tobytes()).hexdigest()[:16]
print('prediction bill %d (pred %s; %.0f s; %d billed wrapped calls covering %.2f%% of the bill)' % (total, pred_sha, time.monotonic() - t0, calls[0], 100 * wrapped / total))
print('content repeats (same op + args + identical input bytes as an earlier call): %.4f%% of the bill; of it same input OBJECTS %.4f%%, same layer %.4f%%' % (
    100 * rep / total, 100 * same / total, 100 * lay / total))
rows = sorted(agg.items(), key=lambda kv: -kv[1]['rep_bill'])
for (name, site), a in rows[:40]:
    if not a['rep_bill']:
        break
    print('%-12s %-40s calls %4d rep %4d  rep bill %.4f%% (same obj %.4f%%, same layer %.4f%%) shapes %s first: %s' % (
        name, site, a['n'], a['rep_n'], 100 * a['rep_bill'] / total, 100 * a['rep_same_obj_bill'] / total, 100 * a['rep_same_layer_bill'] / total,
        a['shapes'], ', '.join('%s x%d' % kv for kv in a['firsts'].most_common(3))))
out.write_text(json.dumps(dict(seed=seed, candidate=str(candidate), total=total, pred_sha=pred_sha, wrapped_bill=wrapped, repeat_bill=rep,
                               repeat_same_obj_bill=same, repeat_same_layer_bill=lay,
                               sites=[dict(op=k[0], site=k[1], n=a['n'], bill=a['bill'], rep_n=a['rep_n'], rep_bill=a['rep_bill'],
                                           rep_same_obj_bill=a['rep_same_obj_bill'], rep_same_layer_bill=a['rep_same_layer_bill'], shapes=a['shapes'],
                                           firsts=dict(a['firsts'])) for k, a in rows]), indent=0))
