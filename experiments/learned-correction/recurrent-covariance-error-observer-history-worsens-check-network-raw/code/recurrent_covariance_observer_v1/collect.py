import sys,types,time,gc,ctypes
import numpy as np
import flopscope as flops
from common import *
verify()
for name in ('baseline','capture'):
    g=read(PREV/f'{name}_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
base={r['mini']:r for r in read(PREV/'baseline/complete.json')['rows']}
out=HERE/'capture';out.mkdir()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('recurrent_covariance_observer_capture');mod.__file__=str(PARENT);sys.modules[mod.__name__]=mod
original=PARENT.read_text()
line='            means.append(m)\n            if l == depth - 1:'
tap='            _capture_post(l,m,C)\n'
assert original.count(line)==1
source=original.replace(line,'            means.append(m)\n'+tap+'            if l == depth - 1:')
assert source.replace(tap,'')==original
exec(compile(source,str(PARENT)+':recurrent_capture','exec'),mod.__dict__)
current={}
def post(l,m,C):
    if l in LAYERS:current[f'C_{l}']=np.array(C);current[f'm_{l}']=np.array(m)
mod._capture_post=post
oldfeatures,oldstep=mod.features,mod.gnn_step
def features(l,m,C,K21,aux,w):
    x=oldfeatures(l,m,C,K21,aux,w)
    if l in LAYERS:
        current[f'K_{l}']=np.array(K21)
        for key in ('sigma','gate','pdf'):current[f'{key}_{l}']=np.array(aux[key])
    return x
def step(l,x,sigma,w,state,P):
    y=oldstep(l,x,sigma,w,state,P)
    if l in LAYERS:current[f'F_{l}']=np.concatenate((np.asarray(x),np.asarray(y[1])),axis=1).astype(np.float32)
    return y
mod.features,mod.gnn_step=features,step
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief;relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
with np.load(OLDPAIR/'parent_parameters.npz') as z:expected={k:z[k] for k in z.files}
records=[];start=time.monotonic()
for mini in ALL:
    assert not current
    wp,identity=weights(mini);w=np.load(wp)
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=identity['mlp_name'])
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    bill=int(ctx.summary_dict()['flops_used']);assert y.shape==(16,1024) and np.isfinite(y).all()
    for k,x in expected.items():assert np.array_equal(x,np.asarray(mod._GNN_PARAMS[k])),k
    e=base[mini]['packet'];assert pin(e['path'])==e
    with np.load(e['path']) as z:assert y.tobytes()==z['prediction'].tobytes(),mini
    assert bill==base[mini]['flops'],(mini,bill,base[mini]['flops'])
    assert len(current)==49,(len(current),sorted(current))
    current['prediction']=y;p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**current)
    row=dict(mini=mini,identity=identity,weights=pin(wp),packet=pin(p),flops=bill,wall_s=time.monotonic()-started,all16_bytes_exact=True,bill_exact=True,targets_accessed=False)
    records.append(row)
    with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(dict(event='capture',mini=mini,bytes=p.stat().st_size,wall_s=row['wall_s'],flops=bill,all16_and_bill_exact=True)),flush=True)
    current.clear();del w,mlp,y;gc.collect();relief(None,0)
once(out/'complete.json',dict(rows=records,parent_sha256=PARENT_SHA,post_tap_inverse_exact=True,elapsed_s=time.monotonic()-start,targets_accessed=False))
