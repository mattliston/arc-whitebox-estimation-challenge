"""Actual FNP helpers against independently formed dense tensors, no labels."""
import json,time,resource
from common import HERE,sha,read,write,seal_check
seal_check();started=time.monotonic()
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
assert all(r['num_threads']==1 for r in threadpool_info())
env=dict(fnp=fnp,_mm=lambda a,b:a@b);exec(compile((HERE/'helpers.py').read_text(),str(HERE/'helpers.py'),'exec'),env)
rng=np.random.Generator(np.random.PCG64DXSM(2026092010));records=[]
for n,m in [(3,2),(5,7),(8,5),(4,4)]:
    for signed in (False,True):
        w=(rng.normal(size=(n,m))*.3).astype(np.float32);a=rng.normal(size=(n,n)).astype(np.float32)
        z=rng.normal(size=(n,n));c=(z@z.T+.3*np.eye(n)) if not signed else z
        tensor=np.zeros((n,n,n),np.float64)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    tensor[i,j,k]=((a[k,i] if i==j else 0.)+(a[j,i] if i==k else 0.)+(a[i,j] if j==k else 0.))/3
        wd=w.astype(np.float64);td=np.einsum('ai,bj,ck,abc->ijk',wd,wd,wd,tensor,optimize=True)
        expected_star=np.stack([td[i,i,:] for i in range(m)])
        expected_cov=wd.T@c.astype(np.float32).astype(np.float64)@wd
        first=None;bills=[];maximum=0.
        for repeat in range(2):
            with flops.BudgetContext(flop_budget=10_000_000,quiet=True) as ctx:
                fw,fa,fc=fnp.asarray(w),fnp.asarray(a),fnp.asarray(c)
                cv=env['_flms_covariance'](fw,fc);v,v2=env['_flms_covariance_diagonal'](fw,fc)
                st=env['_flms_star'](fw,fa);sd=env['_flms_star_diagonal'](fw,fa)
            arrays=[np.asarray(x).copy() for x in (cv,v,v2,st,sd)]
            expectations=[expected_cov,np.diag(expected_cov),np.diag(expected_cov),expected_star,np.diag(expected_star)]
            errors=[float(np.max(np.abs(x-y))/max(1.,float(np.max(np.abs(y))))) for x,y in zip(arrays,expectations)]
            assert max(errors)<=3e-6,errors
            assert all(x.dtype==np.float64 and np.isfinite(x).all() for x in arrays)
            if first is None:first=[x.tobytes() for x in arrays]
            else:assert [x.tobytes() for x in arrays]==first
            bills.append(ctx.summary_dict()['flops_used']);maximum=max(maximum,max(errors))
        assert bills[0]==bills[1]
        # Full covariance-trace identity, for the symmetric formal tensor.
        b=rng.normal(size=n);cs=(c+c.T)/2
        trace_tensor=b[:,None,None]*cs[None,:,:]+b[None,:,None]*cs[:,None,:]+b[None,None,:]*cs[:,:,None]
        pushed=np.einsum('ai,bj,ck,abc->ijk',wd,wd,wd,trace_tensor,optimize=True)
        bp=wd.T@b;cp=wd.T@cs@wd;plane=np.diag(cp)[:,None]*bp[None,:]+2*cp*bp[:,None]
        trace_error=float(np.max(np.abs(plane-np.stack([pushed[i,i,:] for i in range(m)])))/max(1.,float(np.max(np.abs(pushed)))))
        assert trace_error<=2e-12
        records.append(dict(n=n,m=m,signed_nonsymmetric_covariance=signed,max_helper_scaled_error=maximum,trace_identity_scaled_error=trace_error,repeat_exact=True,bills=bills))
write(HERE/'unit_result.json',dict(completed=True,passed=True,records=records,truth_accessed=False,new_model_calls=0,
    source_seal_sha256=sha(HERE/'sources_sealed.json'),elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(json.dumps(records))
