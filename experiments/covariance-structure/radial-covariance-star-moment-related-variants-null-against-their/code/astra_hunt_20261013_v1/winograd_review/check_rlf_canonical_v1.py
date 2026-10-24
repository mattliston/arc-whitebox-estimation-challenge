"""Root-only toy certificate; authored source-only, never run by this agent.

Usage: .venv/bin/python -B check_rlf_canonical_v1.py NEW_OUTPUT.json
Both fused and unfused depth6 contexts run, with exact integer tensor
references and actual RLF old-product
callbacks with HZ row/column strips. Uses a38MiB arena, no estimator setup.
"""
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'VECLIB_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'
import collections
import hashlib
import json
import sys
import types
from pathlib import Path
import numpy as np
import flopscope as flops

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BASE = ROOT.parent / 'opus_reexam_20261001_v1/engineer/a62_modules/s49/s51k2tw_x/candidate.py'
RLF = ROOT / 'bank_algebra/fragment_read_leaf_fuse_v1.py'
PATCH = HERE / 'fragment_rlf_canonical_v1.py'
OUT = Path(sys.argv[1]).resolve()
assert OUT.is_relative_to(ROOT) and not OUT.exists(), OUT
assert hashlib.sha256(BASE.read_bytes()).hexdigest().startswith('b1d647e9')
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub


def reference_canonical(initial, integer=False):
    """Independent tensor-linear oracle; no candidate helpers or op order.

    D maps seven leaf components to canonical C slots TL,BL,TR,BR.
    Apply D separately to all six tensor axes, then turn each C digit into
    (column bit,row bit) and reverse levels to physical row/column order.
    Integer leaves prove exact algebra/axes: all magnitudes stay <=4**6,
    so every float32 intermediate is exactly representable.
    """
    dtype = np.int64 if integer else np.float64
    D = np.array([[1,0,1,0,0,0,0],
                  [1,1,0,0,0,1,-1],
                  [1,0,0,1,1,1,0],
                  [1,1,0,0,1,1,0]], dtype=dtype)
    T = initial.astype(dtype)
    L, r, c = 6, T.shape[-2], T.shape[-1]
    for axis in range(L):
        T = np.moveaxis(np.tensordot(D, T, axes=(1, axis)), 0, axis)
    T = T.reshape((2,2)*L+(r,c))
    order = ([2*(L-level)+1 for level in range(1,L+1)] + [2*L]
             + [2*(L-level) for level in range(1,L+1)] + [2*L+1])
    return T.transpose(order).reshape(64*r,64*c)


def run_context(fused):
    name = 'rlfd_toy_fused' if fused else 'rlfd_toy_unfused'
    mod = types.ModuleType(name)
    mod.__file__ = str(BASE)
    sys.modules[name] = mod
    for source in (BASE, RLF, PATCH):
        exec(compile(source.read_text(), str(source), 'exec', dont_inherit=True), mod.__dict__)
    region = tuple(7**6*x for x in (16, 32, 32))
    mod._IPT_REGION[6] = region
    mod._PB_SINGLE = ()
    mod._LVF_EAGER[0] = fused
    mod._LVF_KEEP[0] = False
    real = mod.fnp
    with flops.BudgetContext(flop_budget=2**42, quiet=True):
        mod._SCRATCH = [real.empty((sum(region),), dtype=real.float32)] + [None]*6
    calls, double_calls = collections.Counter(), []

    class Counting:
        def __getattr__(self, attr):
            value = getattr(real, attr)
            if not callable(value) or isinstance(value, type):
                return value
            def counted(*args, **kwargs):
                calls[attr] += 1
                if kwargs.get('dtype') is real.float64:
                    double_calls.append(dict(op=attr, elements=int(np.prod(args[0].shape)),
                        out_dtype=str(kwargs['out'].dtype) if kwargs.get('out') is not None else None))
                return value(*args, **kwargs)
            setattr(self, attr, counted)
            return counted

    mod.fnp = Counting()
    # The tap only copies the post-add leaf into an ordinary NumPy diagnostic
    # snapshot; it does not alter candidate arrays or add counted operations.
    snapshot = {}
    chosen_parent = [mod._ASTRA_RLFD_PARENT]
    def tap(TC, Z1, ops, basis, out, L, m, n):
        snapshot['leaf'] = np.asarray(TC).copy()
        return chosen_parent[0](TC, Z1, ops, basis, out, L, m, n)
    mod._ASTRA_RLF_PARENT_UNWIND = tap
    rows = []
    cases = [(128, 128, False, False), (192, 256, True, False),
             (256, 192, False, True), (192, 192, True, True)]
    cases = [(M,N,hr,hc,integer) for M,N,hr,hc in cases for integer in (True,False)]
    for M, N, hzr, hzc, integer in cases:
        shape = (7,)*6+(M//64, N//64)
        rank = 64
        mr, nc = M-32*hzr, N-32*hzc
        rng = np.random.default_rng(7310000+13*M+N+int(hzr)+3*int(hzc))
        young_leaf = rng.standard_normal(shape).astype(np.float32)
        an = rng.standard_normal((mr, rank)).astype(np.float32)
        bn = rng.standard_normal((rank, nc)).astype(np.float32)
        if integer:
            young_leaf = rng.integers(-1,2,size=shape,dtype=np.int32).astype(np.float32)
            # Keep the genuine old-product callback and strip machinery,
            # but zero its operands to isolate an exact integer decoder test.
            an.fill(0); bn.fill(0)
        with flops.BudgetContext(flop_budget=2**42, quiet=True):
            inner, right = real.asarray(an), real.asarray(bn)
            TC, Z1, slot, ops, basis = mod._ipt_side(2, 6, M, N)
            output = real.empty((M, N), dtype=real.float32)
        measured, parent_bytes = {}, None
        for arm in ('parent', 'off', 'on'):
            chosen_parent[0] = mod._ASTRA_RLFD_PARENT if arm == 'parent' else mod._astra_rlfd_unwind
            mod._ASTRA_RLFD_ON = arm == 'on'
            for repeat in range(2):
                with flops.BudgetContext(flop_budget=2**42, quiet=True):
                    real.copyto(mod._SCRATCH[0], np.float32(np.nan))
                    real.copyto(TC, young_leaf)
                mod._TR_LAST[0] = None
                mod._ASTRA_RLF_CONTEXT[0] = dict(inner=inner, right=right, M=M, N=N,
                    hzr=hzr, hzc=hzc, done=False)
                calls.clear(); double_calls.clear()
                with flops.BudgetContext(flop_budget=2**42, quiet=True) as ctx:
                    result = mod._ks_unwind(TC, Z1, ops, basis, output, 6, M, N)
                    bill = int(ctx.summary_dict()['flops_used'])
                    charged = len(ctx._op_log)
                assert mod._ASTRA_RLF_CONTEXT[0]['done']
                mod._ASTRA_RLF_CONTEXT[0] = None
                got = np.asarray(result).copy()
                assert np.isfinite(got).all(), (fused, M, N, arm)
                row = dict(fused=fused, M=M, N=N, hzr=hzr, hzc=hzc,
                    arm=arm, repeat=repeat, integer=integer, bill=bill, calls=dict(calls),
                    requests=sum(calls.values()), charged_ops=charged,
                    sha256=hashlib.sha256(got.tobytes()).hexdigest(),
                    double_calls=list(double_calls))
                if arm == 'parent':
                    parent_bytes = got.tobytes()
                    parent_array = got.copy()
                    if integer:
                        expected = reference_canonical(snapshot['leaf'], integer=True)
                        row['integer_exact_parent'] = bool(np.array_equal(got,expected))
                        assert row['integer_exact_parent'], row
                elif arm == 'off':
                    row['byte_equal_parent'] = got.tobytes() == parent_bytes
                    assert row['byte_equal_parent'], row
                else:
                    expected = reference_canonical(snapshot['leaf'], integer=integer)
                    error = got.astype(np.float64)-expected
                    row['max_abs_error_vs_tensor_oracle'] = float(np.max(np.abs(error)))
                    row['rms_error_vs_tensor_oracle'] = float(np.sqrt(np.mean(error*error)))
                    row['integer_exact'] = bool(integer and np.array_equal(got,expected))
                    assert not double_calls, row
                    if integer:
                        assert row['integer_exact'], row
                    else:
                        # A conservative a priori bound, not an accuracy
                        # target fitted to a seed or to the observed output.
                        eps = np.finfo(np.float32).eps/2
                        bound = 100*eps*4**6*float(np.max(np.abs(snapshot['leaf'])))
                        row['conservative_roundoff_bound'] = float(bound)
                        assert row['max_abs_error_vs_tensor_oracle'] <= bound, row
                rows.append(row)
                print(json.dumps(row), flush=True)
                if repeat == 1:
                    measured[arm] = row
                mod._BH_HOLD.clear()
        assert measured['off']['bill'] == measured['parent']['bill'], measured
        assert measured['off']['requests'] == measured['parent']['requests'], measured
        assert measured['off']['charged_ops'] == measured['parent']['charged_ops'], measured
        assert measured['on']['bill']-measured['parent']['bill'] == 27867*(M//64)*(N//64), measured
        assert measured['on']['requests'] == measured['parent']['requests'], measured
        assert measured['on']['charged_ops'] == measured['parent']['charged_ops'], measured
    return dict(fused=fused, arena_bytes=4*sum(region), rows=rows,
                stats=mod._ASTRA_RLFD_STATS)


# A fixed exact binary adversarial example demonstrates the cancelled term:
# u=1, z4=2**24, z1=z6=0. Both algorithms use the same five float32 inputs.
z0,z5,z4,z1,z6 = [np.float32(x) for x in (1,0,2**24,0,0)]
original = ((z0+z5)+z4+z1)-(z4+z6)
balanced = (z0+z5)+(z1-z6)
assert original == 0 and balanced == 1
payload = dict(adversarial=dict(original=float(original), balanced=float(balanced), exact=1), sources={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (BASE, RLF, PATCH, Path(__file__))},
               contexts=[run_context(True), run_context(False)])
with OUT.open('x') as handle:
    json.dump(payload, handle, indent=2)
print(json.dumps(dict(done=True, output=str(OUT))), flush=True)
