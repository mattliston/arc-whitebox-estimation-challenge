from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,statistics,sys,time,types
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_info

D=Path(__file__).resolve().parent
parent=D.parent/'candidate_old_both_tail_trace_bounded_v6.py'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
seal=json.loads((D/'seal_v2.json').read_text())
for p,h in seal['sources'].items():assert sha(Path(p))==h
assert sha(parent)==seal['parent_sha256']
assert all(r['num_threads']==1 for r in threadpool_info())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('covariance_stripe_bench');mod.__file__=str(parent);sys.modules[mod.__name__]=mod
exec(compile(parent.read_text(),str(parent),'exec'),mod.__dict__)
exec(compile((D/'fragment.py').read_text(),str(D/'fragment.py'),'exec'),mod.__dict__)
mod.Estimator().setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
assert mod._W5_PORT_SCOPE is None
rng=np.random.default_rng(2026092737)

# Exact signed orientation fixture, independently write each scalar output.
a0=rng.integers(-2,3,(12,7)).astype(np.float32)
b0=rng.integers(-2,3,(7,12)).astype(np.float32)
ref0=np.empty((12,12),dtype=np.float64)
for i in range(12):
    for j in range(12):
        ii,jj=(i,j) if i//4<=j//4 else (j,i)
        ref0[i,j]=sum(float(a0[ii,t])*float(b0[t,jj]) for t in range(7))
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    a,b=mod.fnp.asarray(a0),mod.fnp.asarray(b0)
    q=mod._cov_upper_stripes(a,b,block=4)
    assert np.array_equal(np.asarray(q),ref0)
    assert np.array_equal(np.asarray(a),a0) and np.array_equal(np.asarray(b),b0)
del a,b,q
n=1024
r=rng.standard_normal((n,n)).astype(np.float32)/32
c=(r@r.T+np.eye(n,dtype=np.float32)*.4).astype(np.float32)
w=(rng.standard_normal((n,n))/32).astype(np.float32)
a0=w.T.copy()
with flops.BudgetContext(flop_budget=2**41,quiet=True):
    cc,ww=mod.fnp.asarray(c),mod.fnp.asarray(w)
    first=mod._mm(cc,ww)
    b0=np.asarray(first).copy()
    a=mod.fnp.asarray(a0);b=first
reference=a0.astype(np.float64)@b0.astype(np.float64)
norm=float(np.linalg.norm(reference))
methods={'parent':lambda:mod._sym_upper(a,b),'stripes128':lambda:mod._cov_upper_stripes(a,b)}
rows={k:[] for k in methods};values={}
for rep in range(-1,3):
    for name in (('parent','stripes128') if rep%2==0 else ('stripes128','parent')):
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            t=time.monotonic();cpu=time.process_time();out=methods[name]()
            wall=time.monotonic()-t;cpu=time.process_time()-cpu
        arr=np.asarray(out).copy();summary=ctx.summary_dict()
        err=float(np.linalg.norm(arr.astype(np.float64)-reference)/norm)
        assert np.isfinite(arr).all() and err<1e-4,(name,err)
        if name in values:assert arr.tobytes()==values[name].tobytes()
        else:values[name]=arr.copy()
        if rep>=0:rows[name].append(dict(flops=int(summary['flops_used']),wall_s=wall,cpu_s=cpu,
            relative_error=err,operations=sum(int(v.get('calls',v.get('count',0))) for v in summary['operations'].values())))
        del out,arr
assert np.asarray(a).tobytes()==a0.tobytes() and np.asarray(b).tobytes()==b0.tobytes()
for rr in rows.values():assert len({r['flops'] for r in rr})==1
aggregate={name:{k:statistics.median(r[k] for r in rr) for k in rr[0]} for name,rr in rows.items()}
ratio={k:aggregate['stripes128'][k]/aggregate['parent'][k] for k in ('flops','cpu_s','wall_s','operations')}
diff=float(np.linalg.norm(values['stripes128'].astype(np.float64)-values['parent'].astype(np.float64))/norm)
assert diff<1e-4
gates=dict(bill=ratio['flops']<=.8,cpu=ratio['cpu_s']<=1.25,numerical=True,input_ownership=True)
rec=dict(asof_utc=datetime.now(timezone.utc).isoformat(),source_seal_sha256=sha(D/'seal_v2.json'),
         small_signed_orientation_exact=True,repeated_warm_bytes_and_bills_exact=True,
         rows=rows,aggregate=aggregate,ratios=ratio,candidate_parent_relative_difference=diff,
         gates=gates,passed=all(gates.values()),scope='Isolated second covariance product, synthetic inputs; not a complete prediction or score',
         network_data_accessed=False,new_samples=False,submitted=False)
with (D/'result_v2.json').open('x') as f:json.dump(rec,f,indent=2,allow_nan=False)
print(json.dumps(rec),flush=True)
