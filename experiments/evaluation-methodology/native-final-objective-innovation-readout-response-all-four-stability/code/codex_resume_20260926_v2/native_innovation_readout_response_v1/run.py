import sys,types,time,gc,ctypes
import flopscope as flops
from common import *
verify();guard('prepare');frozen=read(HERE/'basis_frozen.json');a=packet(frozen['packet'])
if sys.argv[1]=='zero':name='zero';direction=None;amplitude=0.;readout=np.zeros((1,32),dtype='float32')
else:
    direction=int(sys.argv[1]);amplitude=float(sys.argv[2]);assert direction in (0,7) and amplitude in (.125,-.125,.0625,-.0625)
    name=mode_name(direction,amplitude);readout=(amplitude*a['directions'][:,direction]).astype('float32')[None,:]
    guard('zero');assert read(HERE/'zero/complete.json')['all16_exact']
out=HERE/name;out.mkdir();candidate=FIT/'candidate.py'
assert pin(candidate)==read(FIT/'result.json')['candidate']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('native_final_readout_'+name);mod.__file__=str(candidate);sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate)+':readout_probe','exec'),mod.__dict__)
assert mod._CF_PARAMS is None
before={k:np.asarray(v,dtype='float32') for k,v in mod._CF_PAYLOAD.items()}
mod._CF_PAYLOAD['output.weight']=readout.tolist();mod._CF_PAYLOAD['output.bias']=[0.]
for k,v in mod._CF_PAYLOAD.items():
    if k not in ('output.weight','output.bias'):assert np.asarray(v,dtype='float32').tobytes()==before[k].tobytes()
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
np.testing.assert_array_equal(np.asarray(mod._CF_PARAMS['output.weight']),readout)
assert np.asarray(mod._CF_PARAMS['output.bias']).item()==0
base={x['mini']:x for x in read(BASE/'capture/complete.json')['rows']}
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t;rows=[]
for mini in MINIS:
    b=base[mini];assert pin(b['weights']['path'])==b['weights'];w=np.load(b['weights']['path'])
    parent=packet(b['packet'],['prediction'])['prediction']
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    bill=int(ctx.summary_dict()['flops_used']);assert y.shape==(16,1024) and np.isfinite(y).all() and np.min(y)>=0
    assert y[:9].tobytes()==parent[:9].tobytes() and mod._CF_STATS['calls']==7*(len(rows)+1)
    if name=='zero':assert y.tobytes()==parent.tobytes()
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,prediction=y,readout=readout)
    row=dict(mini=mini,packet=pin(p),flops=bill,wall_s=time.monotonic()-start,first9_exact=True,final_targets_read=False)
    rows.append(row);print(json.dumps(dict(event='probe',name=name,**row)),flush=True)
    del w,mlp,parent,y;gc.collect();relief(None,0)
once(out/'complete.json',dict(name=name,direction=direction,amplitude=amplitude,rows=rows,all16_exact=name=='zero',
    final_targets_read=False,only_output_changed=True,output_bias_zero=True,model=pin(FIT/'model.npz'),basis=pin(HERE/'basis.npz')))
