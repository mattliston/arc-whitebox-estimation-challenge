"""TAG scout-native dims_probe_v2 (2026-10-12): measurement only. One prediction of a candidate on one fresh network, in-process,
with the bill (and charged-op count) of every flopscope charge attributed to the innermost of a set of wrapped candidate functions,
plus the argument shapes of those functions per layer (push: kept rows / inner / young columns / newborns / old columns / rank;
read: response rows / columns / live; reconstruction; covariance) and every _mm / _shared_old384_mm call (m, k, n, bill).
usage: dims_probe_v2.py <candidate.py> <seed> <out.json>"""
import collections
import json
import sys
import time
import types
from pathlib import Path
import numpy as np
import flopscope as flops
from flopscope import _budget

O = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(O / 'analyst/fresh'))
import loader

candidate, seed, out = Path(sys.argv[1]).resolve(), int(sys.argv[2]), Path(sys.argv[3])
assert not out.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_probe'); mod.__file__ = str(candidate); sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec', dont_inherit=True), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))

ON = [False]
LAYER = [None]
STACK = []
excl = collections.defaultdict(lambda: [0, 0])          # (layer, tag) -> [bill, ops], innermost tag
calls = []                                               # per wrapped call: dict(layer, tag, args, bill incl, ops incl)
ORIG = _budget.BudgetContext._charge_op
TOT = [0, 0]


def charge(self, op_name, flop_cost, subscripts, shapes, **kw):
    u0 = self._flops_used
    r = ORIG(self, op_name, flop_cost, subscripts, shapes, **kw)
    if ON[0]:
        b = self._flops_used - u0
        e = excl[(LAYER[0], STACK[-1] if STACK else 'other')]
        e[0] += b; e[1] += 1
        TOT[0] += b; TOT[1] += 1
    return r


_budget.BudgetContext._charge_op = charge


def shp(x):
    try:
        return [int(v) for v in x.shape]
    except Exception:
        return None


def describe(tag, args):
    a = args
    try:
        if tag == 'push':      # _deadcut_push(layer, w32, p, previous, newborn, lam, state, a)
            nb = a[4]
            return dict(p=shp(a[2]), previous=shp(a[3]), n_sel=(shp(nb[1]) if nb is not None else None), lam=shp(a[5]),
                        old_columns=int(a[6]['columns']), rank=a[6].get('rank'), n_dead_cut=int(np.sum(np.asarray(a[7]) < -mod._DEADCUT_Z)))
        if tag == 'young_push':    # _push_bank_shared_newborn(w, p, previous, newborn)
            nb = a[3]
            return dict(w=shp(a[0]), p=shp(a[1]), previous=shp(a[2]), n_sel=(shp(nb[1]) if nb is not None else None), q=(int(nb[0]) if nb is not None else None))
        if tag == 'old_recon':     # _slr_reconstruct(U, state)
            return dict(U=shp(a[0]), columns=int(a[1]['columns']))
        if tag == 'read':          # _cp_read_young_sq(left, young_right, lam, live_indices, state)
            return dict(left=shp(a[0]), right=shp(a[1]), lam=shp(a[2]), live=shp(a[3]), old_columns=int(a[4]['columns']))
        if tag in ('mm', 'old_mm'):
            return dict(a=shp(a[0]), b=shp(a[1]))
        if tag == 'cov_star':
            return dict(w=shp(a[0]))
        if tag == 'w5':
            return dict(key=a[3] if len(a) > 3 else None)
        if tag in ('fwdB', 'fwdA'):
            return dict(x=shp(a[0]), L=int(a[1]), d1=int(a[2]), d2=int(a[3]))
        if tag in ('fwdA_hzd', 'fwdB_hzd', 'unwind_ks', 'leaf_unwind', 'hz_leaf', 'ubc'):
            return dict(shapes=[shp(x) for x in a if hasattr(x, 'shape')][:3], ints=[x for x in a if isinstance(x, int)][:6])
    except Exception as ex:
        return dict(error=repr(ex))
    return {}


def wrap(name, tag):
    fn = mod.__dict__[name]

    def inner(*args, **kwargs):
        if not ON[0]:
            return fn(*args, **kwargs)
        t = tag
        if tag == 'w5' and len(args) > 3:
            t = 'w5:' + str(args[3])
        STACK.append(t)
        b0, o0 = TOT[0], TOT[1]
        try:
            return fn(*args, **kwargs)
        finally:
            STACK.pop()
            if tag not in ('mm', 'old_mm') or (len(STACK) == 0 or STACK[-1] not in ('mm', 'old_mm')):
                calls.append(dict(layer=LAYER[0], tag=t, parent=(STACK[-1] if STACK else None), path=[x for x in STACK if x not in ('mm', 'old_mm')], args=describe(tag, args),
                                  bill=TOT[0] - b0, ops=TOT[1] - o0))
    mod.__dict__[name] = inner


def begin(fn):
    def inner(ctx, mlp, l, *rest):
        LAYER[0] = int(l)
        return fn(ctx, mlp, l, *rest)
    return inner


mod.__dict__['_w5_port_begin'] = begin(mod.__dict__['_w5_port_begin'])
for name, tag in (('_deadcut_push', 'push'), ('_shared_old384_push', 'old384_push'), ('_push_bank_shared_newborn', 'young_push'),
                  ('_slr_reconstruct', 'old_recon'), ('_w5_port_call', 'w5'), ('_cp_read_young_sq', 'read'),
                  ('_cp_slr_inner_sq', 'read_old_inner'), ('_np_left', 'np_left'), ('_dc_cov_star', 'cov_star'),
                  ('_l15_rowcut_k3', 'l15'), ('_dc_cov_diagonal', 'l15_diag'), ('_mm', 'mm'), ('_shared_old384_mm', 'old_mm'),
                  ('_k4l_local', 'k4l_local'), ('_b6_old', 'b6_old'), ('omitted_radial_source', 'omitted_radial'),
                  ('_f4c_step', 'f4c_step'), ('_bd_update', 'bd_update'), ('_np_update', 'np_update'), ('_np_forms', 'np_forms'),
                  ('_ott_trace', 'ott_trace'), ('_ynt_trace', 'ynt_trace'), ('nonlinear', 'pair_map'), ('_gn_dcov', 'gn_dcov'),
                  ('_edge_source', 'edge'), ('_shared_old384_commit', 'commit'), ('_ipt_forward_b_ab', 'fwdB'), ('_ipt_forward_a_ab', 'fwdA'), ('_hzd_forward_a', 'fwdA_hzd'), ('_hzd_forward_b', 'fwdB_hzd'), ('_ks_unwind', 'unwind_ks'), ('_ipt_leaf_unwind_ab', 'leaf_unwind'), ('_hz_leaf', 'hz_leaf'), ('_ubc_product', 'ubc')):
    if name in mod.__dict__:
        wrap(name, tag)
    else:
        print('missing', name, flush=True)

w = loader.weights(seed)
truth = loader.truth(seed, check_weights=False)['mc_post']
mlp = types.SimpleNamespace(width=1024, depth=16, weights=[w[l] for l in range(16)], seed=0, name=f'fresh-{seed}')
t0 = time.monotonic()
with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
    ON[0] = True
    y = np.asarray(est.predict(mlp, 2**41), dtype=np.float64)
    ON[0] = False
total = int(ctx.summary_dict()['flops_used'])
mse = float(((y[15] - truth[15]) ** 2).mean())
res = dict(candidate=str(candidate), seed=seed, total=total, counted=TOT[0], ops=TOT[1], mse_final=mse, wall_s=round(time.monotonic() - t0, 1),
           exclusive=[dict(layer=k[0], tag=k[1], bill=v[0], ops=v[1]) for k, v in sorted(excl.items(), key=lambda kv: (-1 if kv[0][0] is None else kv[0][0], kv[0][1]))],
           calls=calls)
with out.open('x') as f:
    json.dump(res, f)
print(json.dumps(dict(total=total, counted=TOT[0], ops=TOT[1], mse=mse, wall=res['wall_s'], n_calls=len(calls))), flush=True)
