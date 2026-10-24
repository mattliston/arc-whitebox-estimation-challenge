import sys,types,time,zipfile,gc,ctypes
import flopscope as flops
from common import *

assert read(HERE/'native_train_result.json')['passed']
for e in read(HERE/'pairs_seal.json')['sources']: assert pin(e['path'])==e
mode=sys.argv[1]; assert mode in ('parent','fit')
source=H/'candidate_old_both_tail_trace_bounded_v6.py' if mode=='parent' else HERE/'candidate.py'
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('check_pair_'+mode);mod.__file__=str(source);sys.modules[mod.__name__]=mod
exec(compile(source.read_text(),str(source),'exec'),mod.__dict__)
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
base={r['mini']:r for r in read(BASE/'capture/complete.json')['rows']}
out=HERE/('check_'+mode);out.mkdir();rows=[]
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
for i,mini in enumerate(CHECK):
    b=base[mini];assert pin(b['weights']['path'])==b['weights'];w=np.load(b['weights']['path'])
    expected=packet(b['packet'],['prediction'])['prediction']
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:y=np.asarray(est.predict(mlp,2**41),dtype='float64')
    bill=int(ctx.summary_dict()['flops_used']);wall=time.monotonic()-start
    assert y.shape==(16,1024) and np.isfinite(y).all() and np.min(y)>=0
    assert y[:9].tobytes()==expected[:9].tobytes()
    if mode=='parent':assert y.tobytes()==expected.tobytes()
    receipt=read(OLD/f'download_mini{mini:04d}.json');assert receipt['passed'] and receipt['ordinal']==mini
    archive=receipt['payload'];assert pin(archive['path'])==archive
    with np.load(archive['path']) as z:
        assert int(z['n_samples'].item())==100_000_000 and int(z['global_index'].item())==mini
        name=z['mlp_name'].item();name=name.decode() if isinstance(name,bytes) else name
        assert name==b['identity']['mlp_name']
        target=z['gt_mean'][15].astype('float64')
    mse=math.fsum(float(v)**2 for v in y[-1]-target)/1024
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=y,target=target)
    row=dict(mini=mini,packet=pin(p),flops=bill,wall_s=wall,mse=mse,adjusted=mse*max(.1,bill/(2**41)),first9_exact=True,archive=archive)
    rows.append(row);print(json.dumps(dict(event='actual_fast_CHECK',mode=mode,**row)),flush=True)
    del w,mlp,y,expected,target;gc.collect();relief(None,0)
once(out/'complete.json',dict(mode=mode,rows=rows,candidate=pin(source),check=list(CHECK),reused_research_networks=True))
