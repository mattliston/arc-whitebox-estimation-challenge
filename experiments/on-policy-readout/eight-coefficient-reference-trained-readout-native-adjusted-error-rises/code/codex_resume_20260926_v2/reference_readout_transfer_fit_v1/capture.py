import sys,types,time,gc,ctypes
import flopscope as flops
from common import *
verify();name=sys.argv[1];assert name in NAMES
with np.load(BASIS/'basis.npz') as z:directions=z['directions'].copy()
if name=='zero':readout=np.zeros((1,32),dtype='float32')
else:
    d=int(name[1]);amplitude=.125 if name.endswith('plus') else -.125
    readout=(amplitude*directions[:,d]).astype('float32')[None,:]
old={r['mini']:r for r in read(REFERENCE/name/'complete.json')['rows']} if name=='zero' or name.startswith(('d0_','d7_')) else None
out=HERE/name;out.mkdir();candidate=REFERENCE/'candidate.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('reference_transfer_capture_'+name);mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate)+':TRAIN6','exec'),mod.__dict__)
mod._CF_PAYLOAD['output.weight']=readout.tolist();mod._CF_PAYLOAD['output.bias']=[0.]
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
assert mod._SR_ON and mod._CR_ON and mod._WINOGRAD
np.testing.assert_array_equal(np.asarray(mod._CF_PARAMS['output.weight']),readout)
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t;rows=[]
for mini in TRAIN:
    b=base[mini];assert pin(b['weights']['path'])==b['weights'];w=np.load(b['weights']['path'])
    parent=packet(b['packet'],['prediction'])['prediction']
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:pred=np.asarray(est.predict(mlp,2**41),dtype='float64')
    bill=int(ctx.summary_dict()['flops_used']);wall=time.monotonic()-start;count=len(rows)+1
    assert pred.shape==(16,1024) and np.isfinite(pred).all() and np.min(pred)>=0
    assert pred[:9].tobytes()==parent[:9].tobytes() and mod._WINOGRAD
    assert mod._CF_STATS['calls']==7*count and mod._CR_COUNTS==dict(full=6*count,diagonal=count)
    assert mod._SR_STATS==dict(late_dispatch=6*count,wide_sketch=2*count)
    repeated=bool(old is not None and mini in old)
    if repeated:
        a=packet(old[mini]['packet'],['prediction','readout'])
        assert pred.tobytes()==a['prediction'].tobytes() and readout.tobytes()==a['readout'].tobytes() and bill==old[mini]['flops']
        del a
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=pred,readout=readout)
    row=dict(mini=mini,packet=pin(p),flops=bill,wall_s=wall,first9_exact=True,repeated_prior_case=repeated,
        repeated_output_and_bill_exact=True if repeated else None,switch_restored=True,final_targets_read=False)
    rows.append(row);print(json.dumps(dict(event='TRAIN_reference_capture',name=name,**row)),flush=True)
    del w,mlp,pred,parent;gc.collect();relief(None,0)
once(out/'complete.json',dict(name=name,rows=rows,final_targets_read=False,coefficients_fitted=False,reference=pin(candidate)))
