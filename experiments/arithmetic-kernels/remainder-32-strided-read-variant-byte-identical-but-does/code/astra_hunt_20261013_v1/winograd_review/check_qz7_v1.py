"""Small QZ7 structural/bill check. Written for root to run; not run by author.

Usage: python -B check_qz7_v1.py NEW_OUTPUT.json
No estimator setup, prebind, network generation or process creation. The only
arena is 7**7*(4+32+32) float32 elements, about 224 MB. Run under the root's guard.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import collections
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import flopscope as flops

HERE = Path(__file__).resolve().parent
P2 = HERE.parent.parent
BASE = P2 / 'opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py'
PATCH = HERE / 'fragment_qz7_v1.py'
OUT = Path(sys.argv[1]).resolve()
assert not OUT.exists(), OUT
assert hashlib.sha256(BASE.read_bytes()).hexdigest().startswith('b1d647e9')
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('qz7_check_candidate')
mod.__file__ = str(BASE)
sys.modules[mod.__name__] = mod
exec(compile(BASE.read_text(), str(BASE), 'exec', dont_inherit=True), mod.__dict__)

# Allocate only enough room for the tiny forced-depth-7 shapes. All cases have
# padded m,k <=256 and packet width 2048. No other depth is entered.
region = tuple(7**7 * x for x in (4, 32, 32))
mod._IPT_REGION[7] = region
mod._PB_SINGLE = ()
mod._LVF_EAGER[0] = False
mod._LVF_KEEP[0] = False
with flops.BudgetContext(flop_budget=2**42, quiet=True):
    mod._SCRATCH = [mod.fnp.empty((sum(region),), dtype=mod.fnp.float32)] + [None] * 6

real_fnp = mod.fnp
calls = collections.Counter()


class Counting:
    def __getattr__(self, name):
        value = getattr(real_fnp, name)
        if not callable(value) or isinstance(value, type):
            return value

        def counted(*args, **kwargs):
            calls[name] += 1
            return value(*args, **kwargs)
        setattr(self, name, counted)
        return counted


mod.fnp = Counting()
leaf_parent = mod._hz_leaf
audit_state = {'shape': None, 'done': False, 'checks': []}


def zero_assert(x, label):
    assert np.count_nonzero(x) == 0, label
    audit_state['checks'].append(label)


def audited_leaf(views):
    # Native views only: audit the actual encoded arrays, without adding any
    # flopscope operations. Only the cold parent call is audited, never warm.
    if not audit_state['done']:
        m, k, _n = audit_state['shape']
        M, K = (m + 127) // 128 * 128, (k + 127) // 128 * 128
        A = np.asarray(mod._IPT_A[(7, M, K)][0])
        B = np.asarray(mod._IPT_B[(7, K, 2048)][0])
        full = slice(None)
        if m % 128 == 32:
            for c in (4, 6):
                zero_assert(A[(full,) * 6 + (c, M // 128 - 1, full)],
                            'top row-zero A chunk %d' % c)
                zero_assert(A[(c,) + (full,) * 6 + (M // 128 - 1, full)],
                            'deep row-zero A chunk %d' % c)
        if k % 128 == 32:
            for c in (2, 6):
                zero_assert(A[(full,) * 6 + (c, full, K // 128 - 1)],
                            'top inner-zero A chunk %d' % c)
            for c in (2, 3):
                zero_assert(B[(full,) * 6 + (c, K // 128 - 1, full)],
                            'top inner-zero B chunk %d' % c)
        audit_state['done'] = True
    return leaf_parent(views)


mod._hz_leaf = audited_leaf
CASES = [(160, 256, 2048), (256, 160, 2048), (160, 160, 2048),
         (192, 160, 2048), (160, 192, 2048), (192, 192, 2048),
         (160, 160, 4096)]
results = {}
parent_outputs = {}


def evaluate(shape, arm):
    m, k, n = shape
    rng = np.random.default_rng(7100136 + m + 3*k + 5*n)
    an = rng.standard_normal((m, k)).astype(np.float32)
    bn = rng.standard_normal((k, n)).astype(np.float32)
    ref = an.astype(np.float64) @ bn.astype(np.float64)
    with flops.BudgetContext(flop_budget=2**42, quiet=True):
        a, b = real_fnp.asarray(an), real_fnp.asarray(bn)
    audit_state.update(shape=shape, done=(arm != 'parent'), checks=[])
    measured = []
    for repeat in range(2):
        # No transform-reuse comparison artifact between the two arms.
        mod._TR_LAST[0] = None
        calls.clear()
        with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
            start = time.monotonic()
            y = mod._hz7_product(a, b, m, k, n)
            wall = time.monotonic() - start
            bill = int(ctx.summary_dict()['flops_used'])
            ops = len(ctx._op_log)
        got = np.asarray(y).copy()
        err = float(np.linalg.norm(got.astype(np.float64) - ref) / np.linalg.norm(ref))
        assert np.isfinite(got).all(), (shape, arm, 'nonfinite')
        assert err < 0.01, (shape, arm, 'gross multiplication error', err)
        row = dict(arm=arm, shape=list(shape), repeat=repeat, flops=bill,
                   charged_ops=ops, wall_s=wall, calls=dict(calls),
                   sha256=hashlib.sha256(got.tobytes()).hexdigest(), relerr_fp64=err)
        if arm == 'patch':
            old = parent_outputs[shape]
            row['byte_equal_parent'] = got.tobytes() == old.tobytes()
            row['relative_delta_parent'] = float(np.linalg.norm(got.astype(np.float64) - old) / np.linalg.norm(ref))
            parent_err = results[shape]['parent'][1]['relerr_fp64']
            assert err <= max(5*parent_err, 2e-5), (shape, 'patch error materially worse', parent_err, err)
        measured.append(row)
        print(json.dumps(row), flush=True)
    if arm == 'parent':
        parent_outputs[shape] = got
    else:
        assert mod._QZ7_CONTEXT == [], 'recursive product context was not restored'
    return measured, list(audit_state['checks'])


for shape in CASES:
    rows, checks = evaluate(shape, 'parent')
    results[shape] = dict(parent=rows, structural_checks=checks)

exec(compile(PATCH.read_text(), str(PATCH), 'exec', dont_inherit=True), mod.__dict__)
for shape in CASES:
    rows, _checks = evaluate(shape, 'patch')
    results[shape]['patch'] = rows
    before, after = results[shape]['parent'][1], rows[1]
    gain = before['flops'] - after['flops']
    if shape[0] % 128 == 32 or shape[1] % 128 == 32:
        assert gain > 0, (shape, 'no warm bill saving', gain)
    else:
        assert gain == 0 and after['byte_equal_parent'], (shape, 'control changed')

payload = dict(base=str(BASE), base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),
               patch=str(PATCH), patch_sha256=hashlib.sha256(PATCH.read_bytes()).hexdigest(),
               arena_bytes=4*sum(region), stats=mod._QZ7_STATS,
               cases=[dict(shape=list(shape), **results[shape]) for shape in CASES])
with OUT.open('x') as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(dict(done=True, output=str(OUT), stats=mod._QZ7_STATS)), flush=True)
