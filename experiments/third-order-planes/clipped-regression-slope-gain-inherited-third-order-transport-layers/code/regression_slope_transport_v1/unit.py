"""Independent scalar integrals and dense ownership checks; no network/targets."""
import hashlib
import json
from math import erf,sqrt,pi
from pathlib import Path
import resource
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

HERE=Path(__file__).resolve().parent
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
seal=read(HERE/'sources_sealed.json')
for item in seal['sources']+seal['external_pins']:
    assert sha(item['path'])==item['sha256'],item['path']
assert all(t['num_threads']==1 for t in threadpool_info())
started=time.monotonic()
source=(HERE/'kernel.py').read_text()
ns={'fnp':np};exec(compile(source,'RGT numpy helper','exec'),ns)
native={'fnp':fnp};exec(compile(source,'RGT native helper','exec'),native)

def moments(alpha,sigma,skew,kurt):
    variance=sigma*sigma;mu=alpha*sigma
    pdf=np.exp(-alpha*alpha/2)/sqrt(2*pi)
    gate=np.array([.5*(1+erf(float(a)/sqrt(2))) for a in alpha])
    mg=sigma*pdf+mu*gate
    k3=skew*sigma**3;k4=kurt*variance**2
    host=mg-k3*alpha*pdf/variance/6+k4*(alpha*alpha-1)*pdf/(variance*sigma)/24
    raw2=(mu*mu+variance)*gate+mu*sigma*pdf+k3*pdf/sigma/3-k4*alpha*pdf/variance/12
    return mu,host,dict(gate=gate,pre_v=variance,raw2=raw2)

alpha=np.array([-8.,-6.,-4.,-2.,-.5,0.,.5,2.,4.,6.,8.])
sigma=np.linspace(.3,1.7,len(alpha));nodes,weights=np.polynomial.legendre.leggauss(192)
scalar=[]
for skew,kurt in ((0.,0.),(.02,.03),(-.02,.03),(.1,-.05)):
    mu,host,aux=moments(alpha,sigma,skew,kurt)
    raw=(aux['raw2']-mu*host)/aux['pre_v']
    integrated=[]
    for a in alpha:
        lo=-a;hi=16.;z=(hi-lo)*nodes/2+(hi+lo)/2
        density=np.exp(-z*z/2)/sqrt(2*pi)
        H3=z**3-3*z;H4=z**4-6*z*z+3
        # Cov(X, ReLU(X))/Var(X) under the specified formal centered law.
        integrated.append((hi-lo)/2*np.dot(weights,z*(z+a)*density*(1+skew*H3/6+kurt*H4/24)))
    independent=np.array(integrated)
    analytic=aux['gate']+np.exp(-alpha*alpha/2)/sqrt(2*pi)*(skew*(alpha*alpha+2)/6-kurt*alpha*(alpha*alpha+1)/24)
    errors=dict(quadrature=float(np.max(np.abs(raw-independent))),
                simplified_unfloored=float(np.max(np.abs(raw-analytic))))
    if skew==kurt==0:errors['gaussian_identity']=float(np.max(np.abs(raw-aux['gate'])))
    assert max(errors.values())<=2e-11,errors
    q,_=ns['_rgt_rebase_history'](7,mu,host,aux,np.eye(len(alpha)),np.eye(len(alpha)))
    assert np.array_equal(q,np.clip(raw,0,1))
    scalar.append(dict(skew=skew,kurt=kurt,errors=errors,
                       raw_below_zero=int((raw<0).sum()),raw_above_one=int((raw>1).sum())))

# The private Gaussian floor is not silently substituted for original pre_v.
floor_aux=dict(gate=np.array([.5]),pre_v=np.array([1e-12]),raw2=np.array([.5e-10]))
qfloor,_=ns['_rgt_rebase_history'](1,np.zeros(1),np.array([1e-5/sqrt(2*pi)]),floor_aux,np.ones((1,1)),np.ones((1,1)))
assert qfloor[0]==1.

# This counterexample is deliberately RETAINED, not treated as a physics pass.
phi0=1/sqrt(2*pi)
formal_mixed_coefficient=-phi0/6
regression_mixed_coefficient=phi0/3
assert formal_mixed_coefficient<0<regression_mixed_coefficient
assert abs(regression_mixed_coefficient+2*formal_mixed_coefficient)<1e-15

def cubic(p,lam):return np.einsum('ir,jr,kr,r->ijk',p,p,p,lam)
def repeated(t):return np.einsum('iij->ij',t)
def star(plane):
    n=len(plane);out=np.zeros((n,n,n))
    for i in range(n):
        out[i,i,i]=plane[i,i]
        for j in range(n):
            if i!=j:out[i,i,j]=out[i,j,i]=out[j,i,i]=plane[i,j]
    return out

rng=np.random.Generator(np.random.PCG64(2026092040))
n=7;p=rng.normal(size=(n,8))/3;lam=rng.normal(size=4);lam=np.r_[lam,-lam]
t=cubic(p,lam);cp=repeated(t)
mu,host,aux=moments(np.linspace(-2,2,n),np.linspace(.7,1.3,n),.05,.03)
g=aux['gate'];inherited=g[:,None]**2*g[None,:]*cp
q,plane=ns['_rgt_rebase_history'](7,mu,host,aux,cp,inherited)
carried=cubic(q[:,None]*p,lam)
direct=t*q[:,None,None]*q[None,:,None]*q[None,None,:]
newborn=cubic(rng.normal(size=(n,6))/4,rng.normal(size=6))
c=rng.normal(size=(n,n));c=c@c.T/20;b=rng.normal(size=n)/20
trace=b[:,None,None]*c[None,:,:]+b[None,:,None]*c[:,None,:]+b[None,None,:]*c[:,:,None]
target=rng.normal(size=(n,n));rebuilt=carried+newborn+trace+star(target-plane-repeated(newborn)-repeated(trace))
Q=rng.normal(size=(n,3));F=rng.normal(size=(3,8))
dense_errors=dict(three_leg=float(np.max(np.abs(carried-direct))),
                  inherited_plane=float(np.max(np.abs(plane-repeated(carried)))),
                  star_restoration=float(np.max(np.abs(repeated(rebuilt)-target))),
                  shared_basis=float(np.max(np.abs((q[:,None]*Q)@F-q[:,None]*(Q@F)))))
assert max(dense_errors.values())<=2e-11
rounded=cubic((q.astype(np.float32)[:,None]*p.astype(np.float32)).astype(np.float64),lam)
rounding_error=float(np.linalg.norm(rounded-carried)/max(1.,np.linalg.norm(carried)))
assert rounding_error<=2e-6
original_innovation=np.diag(target)-np.diag(inherited)-np.diag(repeated(trace))
early_innovation=np.diag(target)-np.diag(plane)-np.diag(repeated(trace))
early_difference=float(np.max(np.abs(early_innovation-original_innovation)))
assert early_difference>1e-7

history=[]
for repeat in range(2):
    with flops.BudgetContext(flop_budget=10**6,quiet=True) as ctx:
        fmu=fnp.asarray(mu.copy());fhost=fnp.asarray(host.copy())
        fa={k:fnp.asarray(v.copy()) for k,v in aux.items()}
        fp=fnp.asarray(cp.copy());fi=fnp.asarray(inherited.copy())
        operands=[fmu,fhost,fp,fi]+[fa[k] for k in sorted(fa)]
        before=[np.asarray(z).tobytes() for z in operands]
        g0,p0=native['_rgt_rebase_history'](0,fmu,fhost,fa,fp,fi)
        assert g0 is fa['gate'] and p0 is fi
        native['_RGT_ENABLED']=False
        gn,pn=native['_rgt_rebase_history'](7,fmu,fhost,fa,fp,fi)
        assert gn is fa['gate'] and pn is fi
        native['_RGT_ENABLED']=True
        qn,rn=native['_rgt_rebase_history'](7,fmu,fhost,fa,fp,fi)
        qn=np.asarray(qn).copy();rn=np.asarray(rn).copy()
        assert before==[np.asarray(z).tobytes() for z in operands]
    assert np.max(np.abs(qn-q))<=2e-11 and np.max(np.abs(rn-plane))<=2e-11
    history.append(dict(gain=qn.tobytes(),plane=rn.tobytes(),bill=ctx.summary_dict()['flops_used']))
assert history[0]==history[1]

invalid=[]
for field,value in (('pre_v',0.),('pre_v',float('nan')),('raw2',float('nan')),('mu',float('inf')),('host',float('inf'))):
    bad={k:v.copy() for k,v in aux.items()};bm=mu.copy();bh=host.copy()
    if field=='mu':bm[0]=value
    elif field=='host':bh[0]=value
    else:bad[field][0]=value
    try:ns['_rgt_rebase_history'](7,bm,bh,bad,cp,inherited)
    except RuntimeError:invalid.append(field)
    else:raise AssertionError('invalid input accepted')

out=dict(passed=True,scalar_cases=scalar,uses_original_variance_in_floor_case=True,
         derivative_correctness_claim=False,mixed_coefficient_counterexample_retained=True,
         formal_mixed_coefficient=formal_mixed_coefficient,regression_mixed_coefficient=regression_mixed_coefficient,
         dense_errors=dense_errors,rounded_factor_scaled_error=rounding_error,
         original_innovation_preserved_by_source=True,early_insertion_innovation_difference=early_difference,
         native_null_returns_original_objects=True,native_inputs_unchanged=True,
         native_repeat_outputs_and_bills_exact=True,native_complete_context_flops=history[0]['bill'],
         invalid_cases_rejected=len(invalid),truth_accessed=False,model_calls=0,
         candidate_sha256=read(HERE/'build.json')['candidate_sha256'],
         elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (HERE/'unit.json').open('x') as stream:json.dump(out,stream,indent=2)
print(json.dumps(out),flush=True)
