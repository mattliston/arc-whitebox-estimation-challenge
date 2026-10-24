"""Privileged diagnostic; no deployable estimator is created."""
import sys,types,time,gc,ctypes
import flopscope as flops
from common import *

verify();mode=sys.argv[1]
assert mode=='branch'
target=int(sys.argv[2]) if mode=='branch' else None
if target is not None:assert target in range(9,15)
name=f'branch{target}' if mode=='branch' else mode
guard_ok('prepare')
assert read(PILOT/'result.json')['verification_passed'] and read(PILOT/'result.json')['arms'][1]['passed']
if mode=='fixed':
    guard_ok('assemble')
    labels=read(HERE/'labels/complete.json')
    sources={x['mini']:x['packet'] for x in labels['rows']}
base,refs=tables();out=HERE/name;out.mkdir()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('exact_native_innovation_'+name);mod.__file__=str(PARENT)
sys.modules[mod.__name__]=mod
original=PARENT.read_text();needle='            means.append(m)\n            if l == depth - 1:'
tap='            C = _research_innovation(l,m,C)\n'
assert original.count(needle)==1
source=original.replace(needle,tap+needle);assert source.replace(tap,'')==original
exec(compile(source,str(PARENT)+':native_innovation','exec'),mod.__dict__)
state={};saved={};cells=[]
def hook(l,m,C):
    if l not in LAYERS:return C
    a=np.asarray(C);mean=np.asarray(m)
    if mode=='null' or (mode=='branch' and l==target-1):
        assert a.tobytes()==state['base'][f'C_{l}'].tobytes(),('C prefix',l)
        assert mean.tobytes()==state['base'][f'm_{l}'].tobytes(),('m prefix',l)
    if mode=='branch' and l==target:
        saved.update(C=a.copy(),m=mean.copy())
    if mode=='null' or (mode=='branch' and l!=target-1):return C
    if mode=='branch':delta=project(state['truth'][f'C_{l}']-a,mean)
    else:delta=project(state['sources'][f'J_{l}'],mean)
    changed=a+delta;assert np.isfinite(changed).all()
    h=mean/np.linalg.norm(mean)
    cells.append(dict(layer=l,delta_energy=float(np.sum(delta**2)),
        annihilation_max=float(np.max(abs(delta@h))),
        min_variance=float(np.min(np.diag(changed)))))
    return mod.fnp.asarray(changed,dtype=mod.fnp.float64)
mod._research_innovation=hook
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,
    flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
rows=[]
for mini in MINIS:
    b=base[mini];assert pin(b['weights']['path'])==b['weights']
    w=np.load(b['weights']['path']);parent=checked_npz(b['packet'],['prediction'])['prediction']
    keys=[f'{k}_{l}' for l in LAYERS for k in ('C','m')]
    state['base']=checked_npz(b['packet'],keys)
    if mode=='branch':state['truth']=checked_npz(refs[mini]['packet'],[f'C_{target-1}'])
    if mode=='fixed':
        z=checked_npz(sources[mini]);state['sources']={f'J_{l}':unpack(z[f'J_{l}_upper']) for l in LAYERS};del z
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=b['identity']['mlp_name'])
    start=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        pred=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    bill=int(ctx.summary_dict()['flops_used'])
    assert pred.shape==(16,1024) and np.isfinite(pred).all() and np.min(pred)>=0
    prefix=16 if mode=='null' else target if mode=='branch' else 9
    assert pred[:prefix].tobytes()==parent[:prefix].tobytes()
    if mode=='null':assert bill==b['flops'] and not cells
    elif mode=='branch':assert [x['layer'] for x in cells]==[target-1] and set(saved)=={'C','m'}
    else:assert [x['layer'] for x in cells]==list(LAYERS)
    packet=dict(prediction=pred,**saved)
    row=dict(mini=mini,mode=mode,target_layer=target,prefix_bytes_exact=True,
        diagnostic_flops=bill,wall_s=time.monotonic()-start,cells=list(cells),deployable=False)
    if mode in ('fixed','null'):
        final=checked_npz(refs[mini]['packet'],['final'])['final']
        a=float(np.mean((pred[-1]-final)**2));b_mse=float(np.mean((parent[-1]-final)**2))
        row.update(raw_mse=a,parent_mse=b_mse,raw_ratio=a/b_mse)
        packet.update(final=final,parent=parent)
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**packet)
    row['packet']=pin(p);rows.append(row)
    with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(dict(event='native',mini=mini,mode=mode,target=target,
        raw_ratio=row.get('raw_ratio'),wall_s=row['wall_s'],flops=bill)),flush=True)
    state.clear();saved.clear();cells.clear();del w,mlp,pred,parent,packet;gc.collect();relief(None,0)
result=dict(mode=mode,target_layer=target,rows=rows,source_inverse_exact=True,
    all16_and_bill_exact=mode=='null',parent_sha256=PARENT_SHA,privileged=True,deployable=False)
if mode in ('fixed','null'):
    ratio=sum(x['raw_mse'] for x in rows)/sum(x['parent_mse'] for x in rows)
    result.update(pooled_raw_ratio=ratio,passed=mode=='fixed' and ratio<=.90 and all(x['raw_ratio']<=.95 for x in rows))
once(out/'complete.json',result)
print(json.dumps(dict(event='complete',mode=mode,target=target,pooled_raw_ratio=result.get('pooled_raw_ratio'),passed=result.get('passed'))),flush=True)
