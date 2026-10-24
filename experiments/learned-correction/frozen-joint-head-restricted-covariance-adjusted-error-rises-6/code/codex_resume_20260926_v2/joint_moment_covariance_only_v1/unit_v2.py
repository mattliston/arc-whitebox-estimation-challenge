"""Independent FP64 covariance formula and complete virtual native map."""
from datetime import datetime,timezone
import hashlib,json,math,sys,types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

H=Path(__file__).resolve().parent;OLD=H.parents[1]/'codex_resume_20260919_v1/joint_incoming_moment_head_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
build=json.loads((H/'build.json').read_text());assert sha(H/'candidate.py')==build['candidate_sha256']
assert build['exact_source_inverse'] and build['helpers_ast_inverse'] and build['all_parameter_bytes_exact']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
m=types.ModuleType('covonly_test');m.__file__=str(H/'candidate.py');sys.modules[m.__name__]=m
exec(compile((H/'candidate.py').read_text(),m.__file__,'exec'),m.__dict__)
ref={};exec(compile((OLD/'head.py').read_text(),str(OLD/'head.py'),'exec'),ref)
t=dict(np.load(build['model']['path'],allow_pickle=False))
cdf=np.vectorize(lambda x:.5*math.erfc(-float(x)/math.sqrt(2)),otypes=[float])


def reference(layer,mu,C,a,b,c):
    s=np.sqrt(np.maximum(np.diag(C),1e-10));alpha=mu/s
    p=np.exp(-.5*alpha*alpha)/np.sqrt(2*np.pi);g=cdf(alpha)
    A=[g,p,-alpha*p,(alpha*alpha-1)*p,-(alpha**3-3*alpha)*p]
    R=C/np.outer(s,s);np.fill_diagonal(R,0.)
    dc=np.zeros_like(C)
    for r in range(3):
        rr=R**r/math.factorial(r)
        dc+=rr*(a*np.outer(A[r+1],A[r])+a.T*np.outer(A[r],A[r+1]))/2
        dc+=rr*c*np.outer(A[r+1],A[r+1])/4
        dc+=rr*(b*np.outer(A[r+2],A[r])+b.T*np.outer(A[r],A[r+2]))/6
    if layer in m._GN_COV_LAYERS:
        dn=a*(m._GN_KNOBS['k21n']*(np.abs(alpha)>=1.5))[:,None]
        cf=sum(R**r/math.factorial(r)*np.outer(A[r+1],A[r]) for r in range(3))
        x=dn*cf/2;dc+=x+x.T
    dc*=np.outer(s,s);np.fill_diagonal(dc,0.)
    return dc


rng=np.random.default_rng(359734);n=9;rows=[]
for index,layer in enumerate((8,11,14)):
    A=rng.normal(size=(n,n));C=A@A.T/n+np.eye(n)
    C[0,2]+=1e-4;C[3,1]-=2e-4
    if index==2:
        C[0,:]=0.;C[:,0]=0.;C[0,0]=1e-12
    s=np.sqrt(np.maximum(np.diag(C),1e-10));v=np.diag(C)
    mu=s*np.linspace(-2.4,2.4,n)
    K=rng.normal(size=(n,n))*.003*(v[:,None]*s[None,:])
    k4=.012*v*v;k22=.012*(np.outer(v,v)+2*C*C)/3
    W=(rng.normal(size=(n,n))/np.sqrt(n)).astype(np.float32)
    state=(rng.normal(size=(n,64))*.2).astype(np.float32)
    a,b,c=(rng.normal(size=(n,n)).astype(np.float32)*.006 for _ in range(3));c=(c+c.T)/2
    for x in (a,b,c):np.fill_diagonal(x,0.)
    F,R,S,_=ref['features'](np,cdf,layer,mu,C,K,.012,W,state)
    ha,hb,hc=ref['predict_planes'](np,F,R,S,t)
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        fm,fC,fK,f4,f22,fW,fH=(fnp.asarray(x) for x in (mu,C,K,k4,k22,W,state))
        fa,fb,fc=(fnp.asarray(x) for x in (a,b,c))
        m._SCRATCH=[fnp.empty(72*n*n,dtype=fnp.float32)]
        m._E2_PAIR_ARENA.clear();m._E2_LAYER.clear();m._PHYS_LAYER[0]=layer
        base=m._nonlinear_pair_batch(fm,fC,fK,f4,f22,'covariance')
        aux=base[4]
        st=m._e2_pair_state(n);arena=m._e2_pair_arena(n)
        buffers=[st['FT'],st['P16'],arena['T']]
        before=[np.asarray(x).copy() for x in buffers]
        inputs=[fm,fC,fK,fa,fb,fc,fW,fH]
        original=[np.asarray(x).copy() for x in inputs]
        head=np.asarray(m._jm_covariance_delta(layer,fm,fC,fK,.012,fW,fH,aux)).copy()
        actual=np.asarray(m._jm_covariance_from_planes(layer,fa,fb,fc,aux)).copy()
        z=fnp.zeros((n,n),dtype=fnp.float32)
        null=np.asarray(m._jm_covariance_from_planes(layer,z,z,z,aux)).copy()
        assert np.array_equal(null,np.zeros_like(null))
        assert all(np.array_equal(np.asarray(x),b) for x,b in zip(buffers,before))
        assert all(np.array_equal(np.asarray(x),b) for x,b in zip(inputs,original))
        gn=m._gn_dcov(layer,fm,fC,fK,f4,aux)
        base_cov=np.asarray(base[1]+gn).copy()
        newK=fK+fa.astype(fnp.float64)*fnp.asarray(v[:,None]*s[None,:])
        new22=f22+fc.astype(fnp.float64)*fnp.asarray(np.outer(v,v))
        extra=fb.astype(fnp.float64)*fnp.asarray((v*s)[:,None]*s[None,:])
        alt=m._nonlinear_pair_batch(fm,fC,newK,f4,new22,'covariance',extra31=extra)
        gn_alt=m._gn_dcov(layer,fm,fC,newK,f4,alt[4])
        alt_cov=np.asarray(alt[1]+gn_alt).copy()
        full_delta=alt_cov-base_cov
        flops_used=int(ctx.summary_dict()['flops_used'])
    expected=reference(layer,mu,C,a.astype(float),b.astype(float),c.astype(float))
    expected_head=reference(layer,mu,C,ha.astype(float),hb.astype(float),hc.astype(float))
    errors=[]
    for result,want in ((actual,expected),(head,expected_head)):
        err=float(np.max(np.abs(result-want)));tol=1e-13+2e-6*float(np.max(np.abs(want)))
        assert np.isfinite(result).all() and err<=tol,(layer,err,tol)
        assert np.array_equal(np.diag(result),np.zeros(n))
        errors.append(dict(max_abs=err,tolerance=tol))
    direct_error=float(np.max(np.abs(actual-full_delta)))
    rounding=16*np.finfo(np.float32).eps*max(1.,float(np.max(np.abs(base_cov))),float(np.max(np.abs(alt_cov))))
    assert direct_error<=rounding,(layer,direct_error,rounding)
    rows.append(dict(layer=layer,floor_fixture=index==2,formula_errors=errors,direct_map_error=direct_error,
                     fp32_rounding_allowance=rounding,input_and_coefficient_bytes_unchanged=True,zero_exact=True,flops=flops_used))
out=dict(passed=True,rows=rows,candidate_sha256=sha(H/'candidate.py'),model_sha256=sha(Path(build['model']['path'])),
         utc_finished=datetime.now(timezone.utc).isoformat(),truth_accessed=False)
with (H/'unit.json').open('x') as f:json.dump(out,f,indent=2);f.write('\n')
print(json.dumps(out,indent=2),flush=True)
