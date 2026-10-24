import json,time,resource
from common import HERE,verify,write,pin
from model import inputs,forward
def main():
    import numpy as np
    import flopscope as flops
    import flopscope.numpy as fnp
    from threadpoolctl import threadpool_info
    assert all(x['num_threads']==1 for x in threadpool_info())
    verify('learning_sources_sealed.json');started=time.monotonic()
    rng=np.random.Generator(np.random.PCG64DXSM(2026092057));n=17
    f=rng.normal(size=(n,30)).astype(np.float32);h=rng.normal(size=(n,64)).astype(np.float32)
    mu=(rng.normal(size=94)*.2).astype(np.float32);sd=np.exp(rng.normal(size=94)*.2).astype(np.float32)
    p=dict(W1=(rng.normal(size=(102,32))*.015).astype(np.float32),b1=np.where(np.arange(32)%2==0,1.,-1.).astype(np.float32),
        W2=(rng.normal(size=32)*.1).astype(np.float32),b2=np.asarray(.03,np.float32))
    rows=[];fixture=dict(features=f,hidden=h,mu=mu,sd=sd,**p)
    for layer in [8,11,15]:
        x=inputs(f,h,layer,mu,sd,np);expected=forward(x,p,np)
        reference=(np.maximum(x.astype(np.float64)@p['W1'].astype(np.float64)+p['b1'],0)@p['W2'].astype(np.float64)+float(p['b2']))
        assert np.max(np.abs(expected-reference))/(1+np.max(np.abs(reference)))<=2e-5
        bill=[];out=[]
        for repeat in range(2):
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                pf={k:fnp.asarray(v) for k,v in p.items()}
                xf=inputs(fnp.asarray(f),fnp.asarray(h),layer,fnp.asarray(mu),fnp.asarray(sd),fnp)
                answer=forward(xf,pf,fnp)
            a=np.asarray(answer);bill.append(ctx.summary_dict()['flops_used']);out.append(a.copy())
        assert bill[0]==bill[1] and out[0].tobytes()==out[1].tobytes()
        err=float(np.max(np.abs(out[0]-expected))/(1+np.max(np.abs(expected))));assert err<=2e-5
        assert out[0].shape==(n,) and out[0].dtype==np.float32 and np.isfinite(out[0]).all()
        rows.append(dict(layer=layer,scaled_error=err,bills=bill,repeat_exact=True))
    pzero={**p,'W2':np.zeros_like(p['W2']),'b2':np.asarray(0,np.float32)}
    assert not forward(x,pzero,np).any()
    path=HERE/'unit_fixture.npz';np.savez(path,**fixture)
    result=dict(passed=True,rows=rows,zero_head_exact=True,fixture=pin(path),truth_accessed=False,
        wall_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(HERE/'unit_fnp.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
