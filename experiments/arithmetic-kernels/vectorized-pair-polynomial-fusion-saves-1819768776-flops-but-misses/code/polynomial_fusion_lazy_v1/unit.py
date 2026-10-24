"""Independent parent polynomial comparison, exact-real and billed-FP32 cases."""
import ast
import hashlib
import json
from pathlib import Path
import resource
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

HERE = Path(__file__).resolve().parent
sha = lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
build = json.loads((HERE/'build.json').read_text())
assert sha(HERE/'candidate.py')==build['candidate_sha256']
assert sha(HERE/'fused_helper.py')==build['fragment_sha256']
assert all(t['num_threads']==1 for t in threadpool_info())
parent = HERE.parent/'lazy_rotation_objective_v1/candidate.py'
assert sha(parent)==build['parent_sha256']
source = parent.read_text(); lines = source.splitlines()
node = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_pair_series_fused'][0]
ns = {'fnp':fnp}
exec(compile('\n'.join(lines[node.lineno-1:node.end_lineno]),str(parent),'exec'),ns)
old = ns['_pair_series_fused']
ns2 = {'fnp':fnp}
exec(compile((HERE/'fused_helper.py').read_text(),str(HERE/'fused_helper.py'),'exec'),ns2)
new = ns2['_pair_series_fused']
rng = np.random.default_rng(2026092007)

def fixture(n,c,dtype,asymmetric):
    F = rng.normal(0,.25,size=(c,7,n)).astype(dtype)
    G = rng.normal(0,.25,size=(c,7,n)).astype(dtype)
    z = rng.normal(size=(n,n))
    rho = ((z+z.T)*(.035/np.sqrt(2))).astype(dtype)
    if asymmetric:
        rho += rng.normal(0,2e-6,size=(n,n)).astype(dtype)
    np.fill_diagonal(rho,0)
    t3 = rng.normal(0,.03,n).astype(dtype)
    t4 = rng.normal(0,.03,n).astype(dtype)
    T21 = rng.normal(0,.002,size=(n,n)).astype(dtype)
    q = rng.normal(0,.002,size=(n,n))
    T22 = ((q+q.T)/2).astype(dtype)
    T31 = (t4.astype(np.float64)[:,None]*rho.astype(np.float64)).astype(dtype)
    return (F,G,rho,T21,T22,T31,t3,t4,n)

cases = [('identity_c1_f64',fixture(32,1,np.float64,False)),
         ('identity_c3_f64',fixture(32,3,np.float64,False)),
         ('symmetric_c3_f32',fixture(64,3,np.float32,False)),
         ('asymmetric_c3_f32',fixture(64,3,np.float32,True)),
         ('full_width_c3_f32',fixture(1024,3,np.float32,True))]
rows = []
for name,args in cases:
    arrays = args[:-1]
    immutable = [x.tobytes() for x in arrays]
    history = []
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=10**10,quiet=True) as ctx:
            fargs = [fnp.asarray(x.copy()) for x in arrays]+[args[-1]]
            before = ctx.summary_dict()['flops_used']
            a = old(*fargs)
            old_bill = ctx.summary_dict()['flops_used']-before
            fargs[5] = None
            before = ctx.summary_dict()['flops_used']
            b = new(*fargs)
            new_bill = ctx.summary_dict()['flops_used']-before
            aa,bb = np.asarray(a).copy(),np.asarray(b).copy()
            ingress_unchanged = all(np.asarray(x).tobytes()==original
                for i,(x,original) in enumerate(zip(fargs[:-1],immutable)) if i!=5)
        assert ingress_unchanged and immutable==[x.tobytes() for x in arrays]
        assert aa.dtype==bb.dtype==arrays[0].dtype and np.isfinite(bb).all()
        delta = aa.astype(np.float64)-bb.astype(np.float64)
        maxabs,rms = float(np.max(np.abs(delta))),float(np.sqrt(np.mean(delta*delta)))
        if arrays[0].dtype==np.float64:
            assert maxabs<=2e-12,(name,maxabs)
        else:
            assert maxabs<=2e-6 and rms<=2e-7,(name,maxabs,rms)
        history.append((aa,bb,old_bill,new_bill,ctx.summary_dict()['flops_used']))
    a,b = history
    assert a[0].tobytes()==b[0].tobytes() and a[1].tobytes()==b[1].tobytes() and a[2:]==b[2:]
    if name=='full_width_c3_f32':assert new_bill<old_bill
    rows.append(dict(case=name,maxabs=maxabs,rms=rms,parent_helper_flops=old_bill,
                     fused_helper_flops=new_bill,saved_helper_flops=old_bill-new_bill,
                     complete_context_flops=history[0][-1],repeat_bytes_and_bills_exact=True,
                     inputs_unchanged=True))
out = HERE/'unit_v1';out.mkdir(exist_ok=False)
record = dict(passed=True,rows=rows,seed=2026092007,source_sha256=sha(__file__),
              candidate_sha256=build['candidate_sha256'],target_arrays_opened=False,
              full_estimator_called=False,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record),flush=True)
