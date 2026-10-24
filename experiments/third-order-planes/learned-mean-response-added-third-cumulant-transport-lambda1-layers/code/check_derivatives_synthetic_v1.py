"""Bounded synthetic math fixtures only. No card/model, weights, truth, or network import."""
from pathlib import Path
import math,json,hashlib,itertools,os
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent
OUT=D/'SYNTHETIC_DERIVATIVE_RESULT_v1.json'
assert not OUT.exists()
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','GOTO_NUM_THREADS'):
    assert os.environ.get(key)=='1',(key,os.environ.get(key))
ACTIVE=(3,5,6,7,8,9,11,20,21)
COLS=ACTIVE+tuple(range(30,78))

def marginal(a,s,t3,t4):
    g=.5*(1+math.erf(a/math.sqrt(2)));p=math.exp(-a*a/2)/math.sqrt(2*math.pi)
    mu=a*s;v=s*s;k3=t3*s**3;k4=t4*s**4
    mg=mu*g+s*p
    r2g=(mu*mu+v)*g+mu*s*p
    r3g=mu*(mu*mu+3*v)*g+(mu*mu+2*v)*s*p
    r4g=(mu**4+6*mu*mu*v+3*v*v)*g+mu*(mu*mu+5*v)*s*p
    den=p/s;e=-a*p/v;f=(a*a-1)*p/(v*s)
    m=mg+k3*e/6+k4*f/24
    r2=r2g+k3*den/3+k4*e/12
    r3=r3g+k3*g+k4*den/4
    r4=r4g+4*k3*mg+k4*g
    pv=r2-m*m;pk3=r3-3*m*r2+2*m**3
    pk4=r4-4*m*r3+6*m*m*r2-3*m**4-3*pv*pv
    q=g+t3*(a*a-1)*p/6-t4*(a*a*a-3*a)*p/24
    u=m/s;vv=pv/v;tt=pk3/s**3
    x=np.array([a,g,p,mg/s,u,(m-mg)/s,vv,tt,pk4/s**4],dtype=np.float64)
    dx=np.array([1,p,-a*p,g,q,q-g,2*u*(1-q),3*(1-q)*(vv-u*u),4*(1-q)*(tt+u**3-3*u*vv)])
    return x,dx

marg=[]
for a,s,t3,t4 in itertools.product((-4.,-2.,-.3,0.,.7,2.,4.),(.1,1.,3.),(-.04,0.,.03),(-.05,0.,.08)):
    x,d=marginal(a,s,t3,t4);eps=1e-5
    fd=(marginal(a+eps,s,t3,t4)[0]-marginal(a-eps,s,t3,t4)[0])/(2*eps)
    error=float(np.max(np.abs(fd-d)))
    assert error < 2e-7,(a,s,t3,t4,error,d,fd)
    marg.append(error)

rng=np.random.default_rng(390173)

def silu(x):return x/(1+np.exp(-x))
def build(n,dtype):
    x=rng.normal(size=(n,30)).astype(dtype)
    dx=np.zeros((n,30),dtype=dtype)
    for i in range(n):
        y,dy=marginal(float(-2+4*i/max(n-1,1)),1.,.03,-.05)
        x[i,list(ACTIVE)]=y;dx[i,list(ACTIVE)]=dy
    scales=np.linspace(.25,1.25,30,dtype=dtype)
    centers=np.zeros(30,dtype=dtype)
    # Keep explicit clipped local/gate cases away from nondifferentiable boundary.
    x[0,3]=dtype(10)*scales[3];x[1,5]=dtype(-10)*scales[5]
    M=rng.normal(0,1,(n,48)).astype(dtype);M[2,:]=dtype(100)
    energies=rng.normal(size=(n,48)).astype(dtype)
    W0=rng.normal(0,.04,(126,128)).astype(dtype)
    W2=rng.normal(0,.04,(128,128)).astype(dtype)
    W4=rng.normal(0,.04,(128,64)).astype(dtype)
    head=rng.normal(0,.1,64).astype(dtype)
    b0=rng.normal(0,.05,(n,128)).astype(dtype)
    b2=rng.normal(0,.05,128).astype(dtype)
    return x,dx,scales,centers,M,energies,W0,W2,W4,head,b0,b2

def forward(data,eps=0):
    x,dx,scales,centers,M,E,W0,W2,W4,head,b0,b2=data
    raw=(x+eps*dx-centers)/scales
    Hx=np.clip(raw,-8,8);signed0=Hx[:,5:6]*M;signed=np.clip(signed0,-8,8)
    H=np.concatenate((Hx,signed,E),axis=1)
    z0=H@W0+b0;h0=silu(z0);z1=h0@W2+b2;h1=silu(z1)
    state=h1@W4;out=state@head
    return out,state,(raw,signed0,z0,z1)

def tangent_np(data,cache):
    x,dx,scales,centers,M,E,W0,W2,W4,head,b0,b2=data
    raw,sm,z0,z1=cache
    dhx=(dx/scales)*((raw>-8)&(raw<8))
    dsm=dhx[:,5:6]*M*((sm>-8)&(sm<8))
    dh=np.concatenate((dhx[:,list(ACTIVE)],dsm),axis=1)
    dz0=dh@W0[list(COLS)]
    s0=1/(1+np.exp(-z0));dh0=dz0*(s0*(1+z0*(1-s0)))
    dz1=dh0@W2;s1=1/(1+np.exp(-z1));dh1=dz1*(s1*(1+z1*(1-s1)))
    return dh1@(W4@head)

head_records=[]
for n,dtype in itertools.product((8,64),(np.float64,np.float32)):
    data=build(n,dtype);out,state,cache=forward(data)
    bytes_before=(out.tobytes(),state.tobytes())
    tangent=tangent_np(data,cache)
    assert bytes_before==(out.tobytes(),state.tobytes())
    eps=1e-5 if dtype==np.float64 else 2e-3
    fd=(forward(data,eps)[0]-forward(data,-eps)[0])/(2*eps)
    abs_err=float(np.max(np.abs(fd-tangent)));rel=float(np.linalg.norm(fd-tangent)/np.linalg.norm(tangent))
    assert rel<(2e-6 if dtype==np.float64 else .003),(n,dtype,abs_err,rel)
    head_records.append(dict(n=n,dtype=str(np.dtype(dtype)),max_abs_error=abs_err,relative_l2_error=rel,forward_cached_outputs_unchanged=True))

# Count the synthetic additional JVP at real width; synthetic matrices only.
# Conversion is outside context because all arguments are forward/cache inputs.
data=build(1024,np.float32);out,state,cache=forward(data)
x,dx,sc,ce,M,E,W0,W2,W4,head,b0,b2=data
raw,sm,z0,z1=cache
with flops.BudgetContext(flop_budget=2**33,quiet=True):
    F={k:fnp.asarray(v) for k,v in dict(dx=dx,sc=sc,M=M,raw=raw,sm=sm,z0=z0,z1=z1,W0=W0,W2=W2,W4=W4,head=head).items()}
with flops.BudgetContext(flop_budget=2**33,quiet=True) as budget:
    dhx=(F['dx']/F['sc'])*((F['raw']>-8)&(F['raw']<8))
    dsm=dhx[:,5:6]*F['M']*((F['sm']>-8)&(F['sm']<8))
    dh=fnp.concatenate((dhx[:,list(ACTIVE)],dsm),axis=1)
    dz0=dh@F['W0'][list(COLS)]
    s0=1/(1+fnp.exp(-F['z0']));dh0=dz0*(s0*(1+F['z0']*(1-s0)))
    dz1=dh0@F['W2'];s1=1/(1+fnp.exp(-F['z1']));dh1=dz1*(s1*(1+F['z1']*(1-s1)))
    tangent=dh1@(F['W4']@F['head'])
    result=tangent*fnp.float32(.0001)*fnp.float32(1.01)
summary=budget.summary_dict()
ref=tangent_np(data,cache)*np.float32(.0001)*np.float32(1.01)
assert np.array_equal(np.asarray(result),ref)
# A deliberately generous per-element allowance for the 9-feature tangent builder,
# assuming it consumes already-computed a/g/pdf/q/m/s/post_v/post_k3.
# <=256 charged float32-equivalent scalar ops per unit incl casts/masks/copies.
per_layer_allowance=1024*256
helper_per_layer=int(summary['flops_used'])
upper14=14*(helper_per_layer+per_layer_allowance)
assert upper14<800_000_000
live=sum(v.nbytes for v in data+cache+(out,state,ref))
assert live<25_000_000
record=dict(status='PASS_SYNTHETIC_DERIVATIVES_AND_HELPER_COST_ONLY',
 checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 proposal_sha256=hashlib.sha256((D/'SOURCE_MATH_PROPOSAL_v1.json').read_bytes()).hexdigest(),
 marginal_cases=len(marg),marginal_max_abs_error=max(marg),head_cases=head_records,
 nondifferentiable_clip_boundary='Derivative defined as zero; exact boundary has no mathematical central derivative and is not claimed to pass central differences.',
 full_width_synthetic_native_helper=dict(n=1024,hidden=128,active_columns=57,flops=helper_per_layer,ops=summary.get('ops'),numpy_formula_byteexact=True),
 added_14_layer_intrinsic_allowance=upper14,extra_local_feature_allowance_per_layer=per_layer_allowance,
 bound_conditions='Only this specified helper plus <=256 charged float32-equivalent ops per unit marginal builder; final source must be checked against envelope. No global network bill bound because downstream active rows/births can change.',
 max_counted_numpy_live_bytes=live,no_card_model_network_or_learned_weights_imported=True,
 no_accuracy_or_native_delivery_claim=True)
with OUT.open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({'path':str(OUT),'sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'marginal_cases':len(marg),'helper_flops':helper_per_layer,'intrinsic_allowance_14':upper14,'max_numpy_bytes':live}))
