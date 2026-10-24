import importlib.util,math,sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,read,write,pin,verify,seal_check,guarded,relative
from reference import reference
seal_check();guarded('build');b=read(HERE/'build.json');verify(b['candidate']);verify(b['parameters'])
with np.load(b['parameters']['path'],allow_pickle=False) as z:p={k:z[k] for k in z.files}
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('edge_candidate',b['candidate']['path']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    spec.loader.exec_module(mod)
    mod._EDGE_PARAMETERS={k:fnp.asarray(v,dtype=fnp.float32) for k,v in mod._EDGE_LITERAL.items()}
setup=ctx.summary_dict()['flops_used']
rng=np.random.Generator(np.random.PCG64DXSM(2026092064));n=17
a=np.linspace(-2,2,n);s=np.exp(rng.normal(size=n)*.1);g=np.asarray([.5*(1+math.erf(v/math.sqrt(2))) for v in a]);q=np.exp(-a*a/2)/math.sqrt(2*math.pi)
m=s*(q+a*g);v=rng.normal(size=(n,4));C=(v@v.T)/20;np.fill_diagonal(C,1);C=C*np.outer(s,s)
k3=rng.normal(size=n)*.01*s**3;k4=rng.normal(size=n)*.01*s**4
K=rng.normal(size=(n,n))*.01*(s[:,None]**2*s[None,:]);np.fill_diagonal(K,k3)
aux=dict(alpha=a,sigma=s,gate=g,pdf=q,pre_k3=k3,pre_k4=k4,pre_cov=C)
rows=[]
for layer in [8,11,14]:
    expected=reference(m,K,aux,p);ref64=reference(m,K,aux,p,wide=True);answers=[];bills=[]
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            answer=mod._edge_source(layer,fnp.asarray(m),fnp.asarray(K),{k:fnp.asarray(v) for k,v in aux.items()})
        answers.append(np.asarray(answer).copy());bills.append(ctx.summary_dict()['flops_used'])
    value=answers[0];err=relative(value,expected);rerr=relative(value,ref64)
    assert value.shape==(n,n) and value.dtype==np.float64 and np.isfinite(value).all()
    assert value.tobytes()==answers[1].tobytes() and bills[0]==bills[1]
    assert np.array_equal(value,value.T) and max(err,rerr)<=3e-5,(err,rerr)
    rows.append(dict(layer=layer,numpy_relative=err,fp64_relative=rerr,bills=bills,repeat_and_symmetry_exact=True))
perm=rng.permutation(n);ix=np.ix_(perm,perm)
ax={k:(v[ix] if k=='pre_cov' else v[perm]) for k,v in aux.items()}
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    permuted=mod._edge_source(14,fnp.asarray(m[perm]),fnp.asarray(K[ix]),{k:fnp.asarray(v) for k,v in ax.items()})
perr=relative(np.asarray(permuted),value[ix]);assert perr<=3e-5
permutation_bill=ctx.summary_dict()['flops_used']
mod._EDGE_PARAMETERS=None
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    cold=mod._edge_source(14,fnp.asarray(m),fnp.asarray(K),{k:fnp.asarray(v) for k,v in aux.items()})
cold_bill=ctx.summary_dict()['flops_used'];assert cold_bill==rows[-1]['bills'][0]+setup
assert np.asarray(cold).tobytes()==value.tobytes()
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    mod._EDGE_PARAMETERS['output']=fnp.zeros(16,dtype=fnp.float32)
    zero=mod._edge_source(14,fnp.asarray(m),fnp.asarray(K),{k:fnp.asarray(v) for k,v in aux.items()})
assert not np.asarray(zero).any()
write(HERE/'unit.json',dict(passed=True,rows=rows,setup_flops=setup,cold_flops=cold_bill,
    cold_output_and_bill_exact=True,permutation_relative=perr,permutation_bill=permutation_bill,
    zero_bill=ctx.summary_dict()['flops_used'],zero_head_exact=True,truth_accessed=False))
print(read(HERE/'unit.json'),flush=True)
