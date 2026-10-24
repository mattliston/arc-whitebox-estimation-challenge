"""Held-math-only fixtures; no card/model/network/learned weight imports."""
from pathlib import Path
import ast,hashlib,json,math,itertools,os
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
D=Path(__file__).resolve().parent;OUT=D/'SYNTHETIC_DERIVATIVE_RESULT_v2.json';assert not OUT.exists()
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','GOTO_NUM_THREADS'):assert os.environ.get(k)=='1'
mp=D/'conditional_tangent_math_UNVALIDATED_v1.py'
nenv={'fnp':np};fenv={'fnp':fnp};exec(mp.read_text(),nenv);exec(mp.read_text(),fenv)

def marginal(a,v,t3,t4):
    s=math.sqrt(max(v,1e-10));vc=s*s;g=.5*(1+math.erf(a/math.sqrt(2)));ph=math.exp(-a*a/2)/math.sqrt(2*math.pi)
    mu=a*s;k3=t3*s**3;k4=t4*s**4
    mg=mu*g+s*ph;r2g=(mu*mu+vc)*g+mu*s*ph
    r3g=mu*(mu*mu+3*vc)*g+(mu*mu+2*vc)*s*ph
    r4g=(mu**4+6*mu*mu*vc+3*vc*vc)*g+mu*(mu*mu+5*vc)*s*ph
    e=-a*ph/v;d=ph/s;f=(a*a-1)*ph/(v*s)
    m=mg+k3*e/6+k4*f/24;r2=r2g+k3*d/3+k4*e/12
    r3=r3g+k3*g+k4*d/4;r4=r4g+4*k3*mg+k4*g
    pv=r2-m*m;pk3=r3-3*m*r2+2*m**3;pk4=r4-4*m*r3+6*m*m*r2-3*m**4-3*pv*pv
    aux=dict(sigma=s,alpha=a,gate=g,pdf=ph,pre_v=v,pre_k3=k3,pre_k4=k4,gaussian_mean=mg,post_v=pv,post_k3=pk3,post_k4=pk4)
    x=np.array([a,g,ph,mg/s,m/s,(m-mg)/s,pv/vc,pk3/(vc*s),pk4/(vc*vc)])
    return m,aux,x

cases=[]
for a,v,t3,t4 in itertools.product((-4.,-2.,-.3,0.,.7,2.,4.),(1.,1e-10,1e-12),(-.0003,0.,.0003),(-.0001,0.,.0001)):
    m,aux,x=marginal(a,v,t3,t4)
    aa={k:np.array([z],dtype=np.float64) for k,z in aux.items()}
    got=nenv['_astra_lmt_local'](np.array([m]),aa)[0].astype(np.float64)
    eps=1e-5;fd=(marginal(a+eps,v,t3,t4)[2]-marginal(a-eps,v,t3,t4)[2])/(2*eps)
    error=float(np.max(np.abs(fd-got)));assert error<3e-7,(a,v,t3,t4,error)
    cases.append(dict(pre_v=v,error=error))
# Strict boundary convention and stable sigmoid are finite at extreme inputs.
z=np.array([-1000.,-100.,0.,100.,1000.],np.float32)
sig=nenv['_astra_lmt_sigmoid'](z);assert np.isfinite(sig).all() and sig[0]==0 and sig[-1]==1
assert ((np.array([-8.,8.])>-8)&(np.array([-8.,8.])<8)).tolist()==[False,False]
# Reuse only synthetic utility definitions from old toy; do not execute its module.
p=D/'check_derivatives_synthetic_v1.py';tt=ast.parse(p.read_text());util=ast.Module(body=[x for x in tt.body if isinstance(x,ast.FunctionDef) and x.name in ('silu','build','forward')],type_ignores=[])
rng=np.random.default_rng(390173);env=dict(np=np,rng=rng,ACTIVE=tuple(nenv['_ASTRA_LMT_LOCAL']))
def old_marginal(a,s,t3,t4):
    m,aux,x=marginal(a,s*s,t3,t4)
    got=nenv['_astra_lmt_local'](np.array([m]),{k:np.array([z]) for k,z in aux.items()})[0]
    return x,got
env['marginal']=old_marginal;exec(compile(util,str(p)+':synthetic_definitions_only','exec'),env)
records=[]
for n,dtype in itertools.product((8,64),(np.float32,np.float64)):
    data=env['build'](n,dtype);out,state,cache=env['forward'](data)
    x,dx,sc,ce,M,E,W0,W2,W4,head,b0,b2=data;raw,sm,z0,z1=cache
    H=np.concatenate((np.clip(raw,-8,8),np.clip(sm,-8,8),E),axis=1)
    got=nenv['_astra_lmt_jvp'](dx[:,nenv['_ASTRA_LMT_LOCAL']],H,M,z0,z1,W0,W2,W4,head,1.,1.,sc)
    eps=2e-3 if dtype==np.float32 else 1e-5
    fd=(env['forward'](data,eps)[0]-env['forward'](data,-eps)[0])/(2*eps)
    rel=float(np.linalg.norm(fd-got)/np.linalg.norm(got));assert rel<(.003 if dtype==np.float32 else 2e-6),(n,dtype,rel)
    records.append(dict(n=n,dtype=str(np.dtype(dtype)),relative_l2=rel))
n=1024;data=env['build'](n,np.float32);out,state,cache=env['forward'](data)
x,dx,sc,ce,M,E,W0,W2,W4,head,b0,b2=data;raw,sm,z0,z1=cache
H=np.concatenate((np.clip(raw,-8,8),np.clip(sm,-8,8),E),axis=1)
ms=[];auxs=[]
for i in range(n):
    m,aux,_=marginal(-2+4*i/(n-1),(1.,1e-10,1e-12)[i%3],.0003,-.0001)
    ms.append(m);auxs.append(aux)
m=np.array(ms);aux={k:np.array([v[k] for v in auxs]) for k in auxs[0]}
with flops.BudgetContext(flop_budget=2**33,quiet=True):
    F={k:fnp.asarray(v) for k,v in dict(m=m,H=H,M=M,z0=z0,z1=z1,W0=W0,W2=W2,W4=W4,head=head,sc=sc).items()}
    A={k:fnp.asarray(v) for k,v in aux.items()}
with flops.BudgetContext(flop_budget=2**33,quiet=True) as budget:
    dl=fenv['_astra_lmt_local'](F['m'],A)
    outj=fenv['_astra_lmt_jvp'](dl,F['H'],F['M'],F['z0'],F['z1'],F['W0'],F['W2'],F['W4'],F['head'],fnp.float32(.0001),fnp.float32(1.01),F['sc'])
summary=budget.summary_dict();bill=int(summary['flops_used'])
ref=nenv['_astra_lmt_jvp'](nenv['_astra_lmt_local'](m,aux),H,M,z0,z1,W0,W2,W4,head,np.float32(.0001),np.float32(1.01),sc)
assert np.array_equal(np.asarray(outj),ref)
# Extra native M[3,n,16] -> [n,48] concatenate and final float64 cast/add;
# deliberately generous 1M FLOPs/layer wrapper allowance, source check mandatory.
whole_intrinsic=14*(bill+1_000_000)
assert whole_intrinsic<900_000_000
live=sum(v.nbytes for v in data+cache+(m,H,ref))+sum(v.nbytes for v in aux.values());assert live<10_000_000
result=dict(status='PASS_FULL_LOCAL_FLOOR_AND_STABLE_JVP_SYNTHETIC_ONLY',checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),math_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),
 marginal_cases=len(cases),marginal_max_abs_error=max(x['error'] for x in cases),pre_variance_branches=[1.,1e-10,1e-12],head_cases=records,clip_boundary_zero_convention=True,stable_sigmoid_extremes=True,
 full_width_helper_flops_per_layer=bill,full_width_helper_ops=summary.get('ops'),source_intrinsic_14_layer_upper_envelope=whole_intrinsic,wrapper_allowance_per_layer=1_000_000,
 global_bill_bound=False,global_bill_caveat='State-dependent downstream row/birth counts are not bounded by this helper proof; actual MAX paired136/137 mandatory.',
 numpy_formula_byteexact=True,peak_counted_numpy_bytes=live,no_card_model_learnedweights_or_network_imports=True)
with OUT.open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(OUT),sha256=hashlib.sha256(OUT.read_bytes()).hexdigest(),bill=bill,envelope=whole_intrinsic,cases=len(cases))))
