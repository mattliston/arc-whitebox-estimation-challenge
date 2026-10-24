"""Separate module setup from repeated actual-terminal helper budgets."""
import importlib.util
import sys
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from common import HERE,OLD,MODEL,read,write,pin,verify,seal_check,guarded,relative

seal_check();guarded('build');verify(read(HERE/'build.json')['candidate'])
for p in read(HERE/'repair_sources_sealed.json')['pins']:verify(p)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
spec=importlib.util.spec_from_file_location('native_mean_candidate',HERE/'candidate.py')
mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
oldspec=importlib.util.spec_from_file_location('frozen_model',OLD/'model.py')
old=importlib.util.module_from_spec(oldspec);oldspec.loader.exec_module(old)
with np.load(MODEL,allow_pickle=False) as z:p={k:z[k] for k in z.files}
rng=np.random.Generator(np.random.PCG64DXSM(2026092063));n=17
raw=(rng.normal(size=(n,94)).astype(np.float32)*p['sd']+p['mu']).astype(np.float32)
f=raw[:,:30];h=raw[:,30:];sigma=np.exp(rng.normal(size=n)*.1).astype(np.float64)
with flops.BudgetContext(flop_budget=2**41,quiet=True) as setup_context:
    spec.loader.exec_module(mod)
setup_bill=setup_context.summary_dict()['flops_used']
rows=[]
for layer in [8,11,15]:
    x=old.inputs(f,h,layer,p['mu'],p['sd'],np)
    expected=old.forward(x,p,np)*p['label_rms']*sigma
    x64=np.concatenate([(raw.astype(np.float64)-p['mu'])/p['sd'],np.eye(8)[np.full(n,layer-8)]],axis=1)
    reference=(np.maximum(x64@p['W1'].astype(np.float64)+p['b1'],0)@p['W2'].astype(np.float64)+float(p['b2']))*float(p['label_rms'])*sigma
    out=[];bills=[]
    for repeat in range(2):
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            answer=mod._lmio_source(fnp.asarray(f),fnp.asarray(h),layer,fnp.asarray(sigma))
        out.append(np.asarray(answer).copy());bills.append(ctx.summary_dict()['flops_used'])
    a=out[0];err=relative(a,expected);rerr=relative(a,reference)
    assert a.shape==(n,) and a.dtype==np.float64 and np.isfinite(a).all()
    assert max(err,rerr)<=3e-5,(err,rerr)
    assert a.tobytes()==out[1].tobytes() and bills[0]==bills[1],dict(bills=bills,outputs_exact=a.tobytes()==out[1].tobytes())
    rows.append(dict(layer=layer,numpy_relative=err,fp64_relative=rerr,repeat_exact=True,bills=bills))
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    mod._LMIO_PARAMETERS['W2']=fnp.zeros(32,dtype=fnp.float32)
    mod._LMIO_PARAMETERS['b2']=fnp.asarray(0,dtype=fnp.float32)
    zero=mod._lmio_source(fnp.asarray(f),fnp.asarray(h),15,fnp.asarray(sigma))
assert not np.asarray(zero).any()
result=dict(passed=True,rows=rows,module_setup_flops=setup_bill,zero_output_head_exact=True,
    zero_bill=ctx.summary_dict()['flops_used'],candidate=pin(HERE/'candidate.py'),truth_accessed=False,new_fit_steps=0,
    original_failed_unit=pin(HERE/'unit.log'),original_two_call_flops_inferred=setup_bill+sum(rows[0]['bills']))
write(HERE/'unit_v2.json',result);print(result,flush=True)
