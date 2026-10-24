from pathlib import Path
import sys,types,time,gc,ctypes,json,hashlib
import numpy as np
import flopscope as flops
HERE=Path(__file__).resolve().parent;H=HERE.parent
PARENT=H/'classical_covariance_response_v1';PROBE=H/'native_innovation_readout_response_v1'
BASE=H/'recurrent_covariance_observer_v1'
read=lambda p:json.loads(p.read_text())
def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        while b:=f.read(1<<20):h.update(b)
    return dict(path=str(p.resolve()),sha256=h.hexdigest(),bytes=p.stat().st_size)
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
for e in read(HERE/'seal.json')['sources']:assert pin(e['path'])==e
g=read(HERE/'prepare_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert read(HERE/'preparation.json')['passed']
name=sys.argv[1];disabled=name=='disabled';prior_name='zero' if disabled else name
assert name in ('disabled','zero','d0_plus','d0_minus','d0_halfplus','d0_halfminus','d7_plus','d7_minus','d7_halfplus','d7_halfminus')
if not disabled:
    g=read(HERE/'disabled_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    assert read(HERE/'disabled/complete.json')['all16_and_bills_exact']
fast={r['mini']:r for r in read(PROBE/prior_name/'complete.json')['rows']}
control={r['mini']:r for r in read(PARENT/'zero/complete.json')['rows']}
e=fast[57]['packet'];assert pin(e['path'])==e
with np.load(e['path']) as z:readout=z['readout'].copy()
with np.load(PROBE/'basis.npz') as z:directions=z['directions'].copy()
if prior_name=='zero':expected=np.zeros((1,32),dtype='float32')
else:
    d=int(prior_name[1]);amp={'plus':.125,'minus':-.125,'halfplus':.0625,'halfminus':-.0625}[prior_name.split('_')[1]]
    expected=(amp*directions[:,d]).astype('float32')[None,:]
assert readout.tobytes()==expected.tobytes()
out=HERE/name;out.mkdir();candidate=HERE/'candidate.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('stable_training_reference_'+name);mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec'),mod.__dict__)
before={k:np.asarray(v,dtype='float32') for k,v in mod._CF_PAYLOAD.items()}
mod._SR_ON=not disabled;mod._CF_PAYLOAD['output.weight']=readout.tolist();mod._CF_PAYLOAD['output.bias']=[0.]
for k,v in mod._CF_PAYLOAD.items():
    if k not in ('output.weight','output.bias'):assert np.asarray(v,dtype='float32').tobytes()==before[k].tobytes()
saved={};old_dispatch=mod._cr_dispatch
def dispatch(l,w,C,A):
    result=old_dispatch(l,w,C,A)
    if l==9:
        assert not saved;saved['Cpre9_diag']=np.diag(np.asarray(result[0])).copy()
    return result
mod._cr_dispatch=dispatch
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
np.testing.assert_array_equal(np.asarray(mod._CF_PARAMS['output.weight']),readout)
assert mod._CR_ON and mod._WINOGRAD and np.asarray(mod._CF_PARAMS['output.bias']).item()==0
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t;rows=[]
for mini in (57,68):
    b=base[mini];assert pin(b['weights']['path'])==b['weights'];w=np.load(b['weights']['path'])
    e=b['packet'];assert pin(e['path'])==e
    with np.load(e['path']) as z:parent=z['prediction'].copy()
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:pred=np.asarray(est.predict(mlp,2**41),dtype='float64')
    wall=time.monotonic()-start;bill=int(ctx.summary_dict()['flops_used']);calls=len(rows)+1
    assert pred.shape==(16,1024) and np.isfinite(pred).all() and np.min(pred)>=0
    assert pred[:9].tobytes()==parent[:9].tobytes() and mod._WINOGRAD
    assert mod._CF_STATS['calls']==7*calls and mod._CR_COUNTS==dict(full=6*calls,diagonal=calls)
    assert mod._SR_STATS==dict(late_dispatch=0 if disabled else 6*calls,wide_sketch=0 if disabled else 2*calls)
    if disabled:
        e=control[mini]['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:assert pred.tobytes()==z['prediction'].tobytes()
        assert bill==control[mini]['flops']
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=pred,readout=readout,**saved)
    row=dict(mini=mini,packet=pin(p),flops=bill,fast_parent_flops=fast[mini]['flops'],fast_bill_ratio=bill/fast[mini]['flops'],
        covariance_baseline_flops=control[mini]['flops'],covariance_baseline_bill_ratio=bill/control[mini]['flops'],
        wall_s=wall,first9_exact=True,switch_restored=True,final_targets_read=False)
    rows.append(row);print(json.dumps(dict(event='stable_reference_probe',name=name,**row)),flush=True)
    saved.clear();del w,mlp,pred,parent;gc.collect();relief(None,0)
once(out/'complete.json',dict(name=name,disabled=disabled,rows=rows,all16_and_bills_exact=disabled,
    first9_exact=True,training_reference_only=True,final_targets_read=False,coefficients_fitted=False,
    covariance_counts=mod._CR_COUNTS,reference_counts=mod._SR_STATS))
