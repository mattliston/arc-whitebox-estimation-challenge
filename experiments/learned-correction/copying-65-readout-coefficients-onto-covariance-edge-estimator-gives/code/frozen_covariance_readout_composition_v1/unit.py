import ast,importlib.util,resource,sys,types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from common import *
seal_check();done('build');b=read(HERE/'build.json');verify(b['candidate'])
assert all(t['num_threads']==1 for t in threadpool_info())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('readout_composition_unit',b['candidate']['path']);mod=importlib.util.module_from_spec(spec)
with flops.BudgetContext(flop_budget=2**35,quiet=True) as ctx:
    spec.loader.exec_module(mod);P=mod._unpack_static_fp32(mod._GNN_LITERAL)
load_bill=int(ctx.summary_dict()['flops_used'])
a={k:np.asarray(v).copy() for k,v in P.items()}
with np.load(D/'full_lower_readout_fit_v1/all_train_coefficients.npz',allow_pickle=False) as z:head=z['head'].copy()
assert np.array_equal(np.r_[a['head'][15],a['head_b'][15]],head)
source=(HERE/'candidate.py').read_text();step=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='gnn_step')
node=[n for n in step.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='delta' for t in n.targets)][0]
delta_source='def actual_delta(state,sigma,P):\n    l=15\n    '+ast.get_source_segment(source,node)+'\n    return delta\n'
env=dict(fnp=fnp);exec(compile(delta_source,'extracted_actual_delta','exec'),env);actual=env['actual_delta']
rng=np.random.Generator(np.random.PCG64DXSM(2026092184));rows=[]
for case in range(3):
    H=(rng.standard_normal((17,64))*.4).astype(np.float32) if case else np.zeros((17,64),np.float32)
    sigma=np.linspace(.0001,1.7,17).astype(np.float64)
    host=np.linspace(-.001,.003,17).astype(np.float64)
    outputs=[];bills=[]
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=2**30,quiet=True) as c:
            y=actual(fnp.asarray(H),fnp.asarray(sigma),P)
            pred=fnp.maximum(fnp.asarray(host)+y.astype(fnp.float64),0).astype(fnp.float32)
        outputs.append((np.asarray(y).copy(),np.asarray(pred).copy()));bills.append(c.summary_dict()['flops_used'])
    assert bills[0]==bills[1] and all(x.tobytes()==y.tobytes() for x,y in zip(outputs[0],outputs[1]))
    first=sigma.astype(np.float32)*a['rms'][15]
    middle=H@a['head'][15]+a['head_b'][15]
    last=np.float32(1)+np.float32(.25)*np.tanh(a['gain'][15:16])
    reference=(first*middle)*last
    fp64=sigma.astype(np.float32).astype(float)*float(a['rms'][15])*(H.astype(float)@a['head'][15].astype(float)+float(a['head_b'][15]))*(1+.25*np.tanh(float(a['gain'][15])))
    assert np.max(np.abs(outputs[0][0]-reference))<=2e-7 and relative(outputs[0][0],fp64)<3e-5
    expected=np.maximum(host+outputs[0][0].astype(float),0).astype(np.float32)
    assert np.array_equal(outputs[0][1],expected) and np.isfinite(expected).all()
    rows.append(dict(case=case,flops=bills[0],fp32_max_abs=float(np.max(np.abs(outputs[0][0]-reference))),fp64_relative=relative(outputs[0][0],fp64),repeat_exact=True,clipping_exact=True))
write(HERE/'unit.json',dict(passed=True,candidate=b['candidate'],fixed_all_train_coefficients_exact=True,
    delta_assignment=delta_source,parameter_load_flops=load_bill,rows=rows,targets_decoded=False,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(passed=True,rows=rows),flush=True)
