import ast,importlib.util,math,sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,D,PARENT,read,write,pin,verify,seal_check
sys.path.insert(0,str(D/'full_lower_covariance_edge_native_v1'))
from reference import reference
seal_check();build=read(HERE/'build.json')
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
rng=np.random.Generator(np.random.PCG64DXSM(2026092065));n=17
a=np.linspace(-2,2,n);s=np.exp(rng.normal(size=n)*.1);g=np.array([.5*(1+math.erf(v/math.sqrt(2))) for v in a]);p=np.exp(-a*a/2)/math.sqrt(2*math.pi)
m=s*(p+a*g);z=rng.normal(size=(n,4));C=z@z.T/20;np.fill_diagonal(C,1);C*=np.outer(s,s)
k3=rng.normal(size=n)*.01*s**3;k4=rng.normal(size=n)*.01*s**4;K=rng.normal(size=(n,n))*.01*(s[:,None]**2*s[None,:]);np.fill_diagonal(K,k3)
aux=dict(alpha=a,sigma=s,gate=g,pdf=p,pre_k3=k3,pre_k4=k4,pre_cov=C)
sources=[dict(name='base',source=pin(PARENT))]+[dict(name=f"head{x['coordinate']:02d}_{x['label']}",source=x['source']) for x in build['variants'] if x['coordinate'] in [0,8]]
rows=[];values={}
for index,entry in enumerate(sources):
    verify(entry['source']);spec=importlib.util.spec_from_file_location(f'head_unit_{index}',entry['source']['path']);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:spec.loader.exec_module(mod)
    assert ctx.summary_dict()['flops_used']==0
    parameters={k:np.asarray(v,np.float32) for k,v in mod._EDGE_LITERAL.items()}
    expected=reference(m,K,aux,parameters);wide=reference(m,K,aux,parameters,wide=True)
    for layer in [8,11,14]:
        answers=[];bills=[]
        for repeat in range(2):
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                result=mod._edge_source(layer,fnp.asarray(m),fnp.asarray(K),{k:fnp.asarray(v) for k,v in aux.items()})
            answers.append(np.asarray(result).copy());bills.append(ctx.summary_dict()['flops_used'])
        value=answers[0];scale=max(np.linalg.norm(wide),1e-30)
        assert value.dtype==np.float64 and value.shape==(n,n) and np.isfinite(value).all() and np.array_equal(value,value.T)
        assert value.tobytes()==answers[1].tobytes()
        assert bills[0]-bills[1]==(1090 if layer==8 else 0)
        err=float(np.linalg.norm(value-expected)/scale);err64=float(np.linalg.norm(value-wide)/scale)
        assert max(err,err64)<=3e-5
        rows.append(dict(source=entry['name'],layer=layer,numpy_relative=err,fp64_relative=err64,bills=bills,repeat_exact=True))
        values[(entry['name'],layer)]=value
linear=[]
for j in [0,8]:
    for layer in [8,11,14]:
        y=values[('base',layer)];plus=values[(f'head{j:02d}_plus',layer)];minus=values[(f'head{j:02d}_minus',layer)];double=values[(f'head{j:02d}_double',layer)]
        scale=max(np.linalg.norm(plus-minus),1e-30)
        symmetric=float(np.linalg.norm(plus+minus-2*y)/scale);twice=float(np.linalg.norm(double-2*plus+y)/scale)
        assert max(symmetric,twice)<=3e-5,(j,layer,symmetric,twice)
        linear.append(dict(coordinate=j,layer=layer,symmetric_relative=symmetric,double_relative=twice))
write(HERE/'unit.json',dict(passed=True,rows=rows,local_head_linearity=linear,setup_flops=1090,all_repeats_exact=True,new_targets_accessed=False))
print(dict(passed=True,small_source_calls=sum(len(r['bills']) for r in rows),max_fp64_relative=max(r['fp64_relative'] for r in rows),max_linearity_relative=max(max(r['symmetric_relative'],r['double_relative']) for r in linear)),flush=True)
