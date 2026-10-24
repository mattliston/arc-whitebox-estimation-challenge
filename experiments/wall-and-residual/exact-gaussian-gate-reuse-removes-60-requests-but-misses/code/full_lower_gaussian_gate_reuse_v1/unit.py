"""Actual FNP parity, no-mutation and duplicate-work removal fixtures."""
import importlib.util
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE, PARENT, read, write, pin, verify, seal_check

seal_check()
build = read(HERE/'build.json')
for key in ['parent', 'candidate', 'disabled']:
    verify(build[key])
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        spec.loader.exec_module(m)
    return m

parent = load(PARENT, 'gaussian_parent')
candidate = load(HERE/'candidate.py', 'gaussian_candidate')

def snapshot(value):
    if isinstance(value, dict):
        return {k:snapshot(v) for k,v in value.items()}
    if isinstance(value, (tuple, list)):
        return [snapshot(v) for v in value]
    if value is None:
        return None
    return np.asarray(value).copy()

def exact(a, b):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            exact(a[key], b[key])
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x,y in zip(a,b):
            exact(x,y)
    elif a is None:
        assert b is None
    else:
        assert a.dtype == b.dtype and a.shape == b.shape
        assert a.tobytes() == b.tobytes()
        assert np.isfinite(a).all()

def prepared(mu, v):
    clipped = fnp.maximum(v, 1e-10)
    sigma = fnp.sqrt(clipped)
    alpha = mu/sigma
    gate = flops.stats.norm.cdf(alpha)
    return clipped, sigma, alpha, gate

def raw_call(module, mu_np, v_np, reuse):
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        mu = fnp.asarray(mu_np)
        v = fnp.asarray(v_np)
    before = snapshot([mu, v])
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        if reuse:
            packet = prepared(mu, v)
            gate = packet[3]
            result = module.gaussian_raw(mu, v, packet)
        else:
            gate = flops.stats.norm.cdf(mu/fnp.sqrt(fnp.maximum(v, 1e-10)))
            result = module.gaussian_raw(mu, v)
    exact(before, snapshot([mu, v]))
    return snapshot([gate, result]), ctx.summary_dict()['flops_used']

raw_rows = []
removed1024 = None
for dtype, n in [(np.float32,7), (np.float32,13), (np.float64,7), (np.float64,13), (np.float64,1024)]:
    mu = np.linspace(-25,25,n).astype(dtype)
    variance = np.resize(np.array([-1,0,1e-25,1e-10,.001,.3,2],dtype=dtype),n)
    old, old_bill = raw_call(parent, mu, variance, False)
    default, default_bill = raw_call(candidate, mu, variance, False)
    reused, reused_bill = raw_call(candidate, mu, variance, True)
    again, again_bill = raw_call(candidate, mu, variance, True)
    exact(old, default); exact(old, reused); exact(reused, again)
    assert old_bill == default_bill and reused_bill == again_bill and 0 < reused_bill < old_bill
    raw_rows.append(dict(dtype=str(np.dtype(dtype)), width=n, exact=True, no_input_mutation=True,
        default_flops=old_bill, reused_flops=reused_bill, removed_flops=old_bill-reused_bill))
    if n == 1024:
        removed1024 = old_bill-reused_bill

def pair_call(module, name, reuse):
    n = 7
    rng = np.random.default_rng(2026092067)
    u = rng.normal(0,.07,(n,n))
    arrays = [np.linspace(-.5,.5,n), u@u.T + .2*np.eye(n), rng.normal(0,.0001,(n,n)),
              np.full(n,.00005), np.eye(n)*.00005]
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        args = [fnp.asarray(x) for x in arrays]
    before = snapshot(args)
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        vv = fnp.diag(args[1])
        if reuse:
            packet = prepared(args[0], vv)
            result = getattr(module, name)(*args, 'covariance', _gaussian_prepared=packet)
        else:
            unused_gate = flops.stats.norm.cdf(args[0]/fnp.sqrt(fnp.maximum(vv, 1e-10)))
            result = getattr(module, name)(*args, 'covariance')
    exact(before, snapshot(args))
    return snapshot(result), ctx.summary_dict()['flops_used']

pair_rows = []
for name in ['_nonlinear_pair_batch', '_nonlinear_first_gaussian', '_nonlinear_terminal_live']:
    old, old_bill = pair_call(parent, name, False)
    default, default_bill = pair_call(candidate, name, False)
    reused, reused_bill = pair_call(candidate, name, True)
    again, again_bill = pair_call(candidate, name, True)
    exact(old, default); exact(old, reused); exact(reused, again)
    assert old_bill == default_bill and reused_bill == again_bill and reused_bill < old_bill
    assert old_bill-reused_bill == next(x['removed_flops'] for x in raw_rows if x['dtype']=='float64' and x['width']==7)
    pair_rows.append(dict(function=name, exact=True, no_input_mutation=True,
        default_flops=old_bill, reused_flops=reused_bill, removed_flops=old_bill-reused_bill))
assert removed1024 > 0
write(HERE/'unit.json', dict(passed=True, raw=raw_rows, pairs=pair_rows,
    duplicate_removed_per_1024_vector=removed1024, expected_full_call_removed_flops=15*removed1024,
    final_scalar_default_path_unchanged=True, global_numerical_gate_cache=False,
    targets_accessed=False, candidate=build['candidate'], source=pin(__file__)))
print(dict(passed=True, duplicate_removed_per_1024_vector=removed1024,
           expected_full_call_removed_flops=15*removed1024), flush=True)
