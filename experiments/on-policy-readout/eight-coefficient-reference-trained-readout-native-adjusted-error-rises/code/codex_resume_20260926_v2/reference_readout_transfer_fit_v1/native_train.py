import sys,types,time,gc,ctypes
import flopscope as flops
from common import *
verify();guard('fit')
for e in read(HERE/'native_seal.json')['sources']:assert pin(e['path'])==e
mode=sys.argv[1];assert mode in ('parent','fit')
if mode=='fit':guard('native_parent')
source=H/'candidate_old_both_tail_trace_bounded_v6.py' if mode=='parent' else HERE/'candidate.py'
if mode=='fit':assert pin(source)==read(HERE/'fit_result.json')['candidate']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('actual_fast_train_'+mode);mod.__file__=str(source);sys.modules[mod.__name__]=mod
exec(compile(source.read_text(),str(source),'exec'),mod.__dict__)
assert '_SR_ON' not in mod.__dict__ and '_CR_ON' not in mod.__dict__
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
assert mod._WINOGRAD
if mode=='fit':
    model=packet(read(HERE/'fit_result.json')['model'])
    assert np.asarray(mod._CF_PARAMS['output.weight']).tobytes()==model['readout'].tobytes()
    assert np.asarray(mod._CF_PARAMS['output.bias']).item()==0
    with np.load(FIT/'model.npz') as z:
        for key in z.files:
            if key not in ('output.weight','output.bias'):assert np.asarray(mod._CF_PARAMS[key]).tobytes()==z[key].astype('float32').tobytes()
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
target=packet(read(HERE/'targets.json')['packet']);assert tuple(target['train'])==TRAIN
out=HERE/('native_'+mode);out.mkdir();rows=[]
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
for i,mini in enumerate(TRAIN):
    b=base[mini];assert pin(b['weights']['path'])==b['weights'];w=np.load(b['weights']['path'])
    expected=packet(b['packet'],['prediction'])['prediction']
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:y=np.asarray(est.predict(mlp,2**41),dtype='float64')
    bill=int(ctx.summary_dict()['flops_used']);wall=time.monotonic()-start
    assert y.shape==(16,1024) and np.isfinite(y).all() and np.min(y)>=0 and mod._WINOGRAD
    assert y[:9].tobytes()==expected[:9].tobytes()
    if mode=='parent':assert y.tobytes()==expected.tobytes() and bill==b['flops']
    else:assert mod._CF_STATS['calls']==7*(i+1)
    mse=math.fsum(float(v)**2 for v in y[-1]-target['targets'][i])/1024
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=y)
    row=dict(mini=mini,packet=pin(p),flops=bill,wall_s=wall,mse=mse,adjusted=mse*bill/(2**41),
        first9_exact=True,all16_parent_exact=mode=='parent',check_targets_read=False)
    rows.append(row);print(json.dumps(dict(event='actual_fast_TRAIN',mode=mode,**row)),flush=True)
    del w,mlp,y,expected;gc.collect();relief(None,0)
once(out/'complete.json',dict(mode=mode,rows=rows,candidate=pin(source),actual_fast_path=True,training_reference_deployed=False,
    train=list(TRAIN),check_targets_read=False))
