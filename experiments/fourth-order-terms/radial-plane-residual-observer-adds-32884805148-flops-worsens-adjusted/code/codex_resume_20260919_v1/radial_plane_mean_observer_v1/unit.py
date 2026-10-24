import ast
from collections import Counter
from itertools import product
import math
import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import *

seal_check()
ns = dict(fnp=fnp)
exec(compile(ast.parse((HERE / 'helper.py').read_text()), str(HERE / 'helper.py'), 'exec'), ns)
rng = np.random.default_rng(2026092078)


def relative(actual, reference):
    error = float(np.linalg.norm(actual-reference))
    norm = float(np.linalg.norm(reference))
    return error/norm if norm else error


def dense_tensor(d):
    n = len(d)
    t = np.zeros((n, n, n, n))
    for index in product(range(n), repeat=4):
        counts = Counter(index)
        if len(counts) == 1:
            i = index[0]; t[index] = d[i, i]
        elif sorted(counts.values()) == [2, 2]:
            i, j = counts; t[index] = d[i, j]
    return t


records = []
for case in range(6):
    n = 4 + case % 3
    a = rng.normal(size=(n, n)); c = a@a.T + np.eye(n)
    q = rng.normal(size=(n, n)); q = (q+q.T)/2
    if case == 1:
        c = c + .01*(a-a.T); q = q + .02*(a-a.T)
    if case == 2:
        q[:] = 0
    if case == 3:
        scale = np.geomspace(.2, 3, n); c = c*np.outer(scale, scale)
    if case == 4:
        q = np.diag(np.linspace(-1, 2, n))
    if case == 5:
        q = (c*c)*.03 - np.outer(np.diag(c), np.diag(c))*.07
    cs = (c+c.T)/2; qs = (q+q.T)/2
    denom = math.fsum(float(cs[i,i]) for i in range(n))**2
    denom += 2*math.fsum(float(x)**2 for x in cs.flat)
    radial = 3*math.fsum(float(x) for x in qs.flat)/denom
    ref = np.array([[qs[i,j] - radial*(cs[i,i]*cs[j,j]+2*cs[i,j]**2)/3
                     for j in range(n)] for i in range(n)])
    trace_error = abs(math.fsum(float(x) for x in ref.flat))
    trace_scale = float(np.linalg.norm(qs))
    assert trace_error <= 2e-11*trace_scale if trace_scale else trace_error == 0
    original_bytes = [x.tobytes() for x in [c,q]]
    w = rng.normal(size=(n,n)).astype(np.float32)/math.sqrt(n)
    outputs, bills = [], []
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=2**35, quiet=True) as ctx:
            cf, qf, wf = fnp.asarray(c), fnp.asarray(q), fnp.asarray(w)
            d = ns['_r22_source'](cf,qf)
            ray = ns['_r22_ray'](d,wf)
        outputs.append((np.asarray(d).copy(),np.asarray(ray).copy()))
        bills.append(int(ctx.summary_dict()['flops_used']))
    assert bills[0] == bills[1]
    assert all(x.tobytes()==y.tobytes() for x,y in zip(*outputs))
    dr, ray = outputs[0]
    assert relative(dr.astype(float), ref) <= 2e-6
    assert [x.tobytes() for x in [c,q]] == original_bytes
    tensor = dense_tensor(dr.astype(float))
    wf64 = w.astype(float)
    expected = np.einsum('ijkl,ia,ja,ka,la->a',tensor,wf64,wf64,wf64,wf64,optimize=False)
    ray_error = relative(ray,expected); assert ray_error <= 3e-5
    direct = 3*np.einsum('ia,ij,ja->a',wf64**2,dr.astype(float),wf64**2)
    direct -= 2*np.einsum('i,ia->a',np.diag(dr).astype(float),wf64**4)
    assert relative(direct,expected) <= 2e-11
    perm = rng.permutation(n)
    assert relative(dense_tensor(dr[np.ix_(perm,perm)]),tensor[np.ix_(perm,perm,perm,perm)]) <= 2e-11
    scale = np.geomspace(.5,2,n)
    scaled_d = dr.astype(float)*np.outer(scale**2,scale**2)
    sw = wf64/scale[:,None]
    scaled_ray = np.einsum('ijkl,ia,ja,ka,la->a',dense_tensor(scaled_d),sw,sw,sw,sw,optimize=False)
    assert relative(scaled_ray,expected) <= 2e-11
    state = dict(plane=None,mean=None); mean_ref = None; recurrence_errors=[]
    for layer in range(3):
        wl = rng.normal(size=(n,n)).astype(np.float32)/math.sqrt(n)
        alpha = np.linspace(-1.7,1.4,n); sigma = np.linspace(.7,1.3,n)
        pdf = np.exp(-alpha**2/2)/math.sqrt(2*math.pi)
        gate = np.array([.5*(1+math.erf(x/math.sqrt(2))) for x in alpha])
        with flops.BudgetContext(flop_budget=2**35,quiet=True):
            state['plane'] = fnp.asarray(dr)
            aux = {k:fnp.asarray(v) for k,v in dict(alpha=alpha,sigma=sigma,pdf=pdf,gate=gate).items()}
            ns['_r22_advance'](state,fnp.asarray(wl),fnp.asarray(wl.astype(float)),aux)
            actual = np.asarray(state['mean']).copy()
        wld = wl.astype(float)
        rr = np.einsum('ijkl,ia,ja,ka,la->a',tensor,wld,wld,wld,wld,optimize=False)
        source = (alpha**2-1)*pdf*rr/(24*sigma**3)
        if mean_ref is not None:
            source += gate*(wld.T@mean_ref)
        mean_ref = source
        recurrence_errors.append(relative(actual,mean_ref))
    assert max(recurrence_errors) <= 3e-5
    records.append(dict(case=case,width=n,source_relative=relative(dr.astype(float),ref),
        ray_relative=ray_error,recurrence_relative=max(recurrence_errors),double_trace_error=trace_error,
        repeated_bytes_and_bill=True,unit_bill=bills[0]))
write(HERE / 'unit.json',dict(passed=True,cases=records,
    dense_tensor_certified=True,coordinate_scaling_and_permutation=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,cases=len(records),max_ray=max(r['ray_relative'] for r in records),
    max_recurrence=max(r['recurrence_relative'] for r in records)),flush=True)

