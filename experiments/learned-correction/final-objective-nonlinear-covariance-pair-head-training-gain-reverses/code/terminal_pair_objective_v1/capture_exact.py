import sys,types,time,gc,ctypes
import numpy as np
import flopscope as flops
from common import *
verify();group=int(sys.argv[1]);assert group==0
with np.load(HERE/'qualification_directions_v3.npz') as z:directions={k:z[k] for k in z.files}
if group:
    assert read(HERE/'qualification.json')['passed']
    g=read(HERE/'qualification_guard.json');assert g['completed'] and g['max_swap_growth_bytes']==0 and g['whole_group_retired']
ranks=TRAIN[:2] if group==0 else TRAIN[2:]+CHECK
out=HERE/f'capture_exact_{group:02d}';out.mkdir()
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=read(store/'manifest.json')['roles']['observer_fit']['row_manifests']
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('terminal_pair_capture');mod.__file__=str(PARENT);sys.modules[mod.__name__]=mod
s=PARENT.read_text();old='                m, aux = nonlinear_scalar(mu, v, k3, _e2_H(_e2_H(radial * v) * v))'
assert s.count(old)==1
s=s.replace(old,'                _capture_scalar(mu,v,k3,bpre,radial)\n'+old)
exec(compile(s,str(PARENT)+':passive_scalar_capture','exec'),mod.__dict__)
current={}
def scalar(mu,v,k3,bpre,radial):
    current.update({k:np.array(x,dtype=np.float64) for k,x in locals().copy().items()})
mod._capture_scalar=scalar
oldfeatures,oldstep,oldcov,oldcut,oldfinal=mod.features,mod.gnn_step,mod._flms_covariance_diagonal,mod._l15_rowcut_k3,mod.final_scalar_features
def features(l,m,C,K21,aux,w):
    x=oldfeatures(l,m,C,K21,aux,w)
    if l==14:
        current.update(K=np.array(K21),s14=np.array(aux['sigma']))
    return x
def step(l,x,sigma,w,state,P):
    y=oldstep(l,x,sigma,w,state,P)
    if l==14:current['F']=np.concatenate((np.asarray(x),np.asarray(y[1])),axis=1).astype(np.float32)
    if l==15:
        k=mod._e2_gnn_consts(P,len(sigma))
        current.update(M=np.asarray(mod._E2_MSG['v'][2]).transpose(1,0,2).reshape(len(sigma),48).copy(),
                       elog=np.array(k['HM36']),native_delta=np.array(y[0]))
    return y
def covariance(w,C):
    assert 'C' not in current
    current.update(C=np.array(C),W=np.array(w))
    return oldcov(w,C)
def cut(*args):
    y=oldcut(*args);current['cut']=np.array(y)
    original_limit=mod._L15_ROWCUT_Z
    try:
        mod._L15_ROWCUT_Z=1e20
        full=oldcut(*args)
    finally:mod._L15_ROWCUT_Z=original_limit
    full=np.array(full);mu=np.asarray(args[1]);v=np.asarray(args[2]);z=np.abs(mu/np.sqrt(np.maximum(v,1e-10)))
    count=min(len(z),((np.count_nonzero(z<=3.5)+31)//32)*32)
    rows=np.argsort(z,kind='stable')[:count];full[rows]=current['cut'][rows]
    current['fullcut']=full
    for j in range(2):
        D=directions[f'{rank}_{j}_D']
        for amp in (1.,.5):
            for sign,label in ((1,'plus'),(-1,'minus')):
                vp,_=oldcov(args[3],mod.fnp.asarray(current['C']+sign*amp*D))
                newargs=list(args);newargs[2]=vp
                kp=oldcut(*newargs)
                current[f'cut_{j}_{amp}_{label}']=np.array(kp)
                current[f'v_{j}_{amp}_{label}']=np.array(vp)
    return y
def final(l,m,aux,w):
    x=oldfinal(l,m,aux,w);current.update(native_x=np.array(x),native_m=np.array(m));return x
mod.features,mod.gnn_step,mod._flms_covariance_diagonal,mod._l15_rowcut_k3,mod.final_scalar_features=features,step,covariance,cut,final
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief
relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
records=[];start=time.monotonic()
for rank in ranks:
    assert not current
    wp=store/Path(manifest[rank]['path']).parent/'weights.npy';w=np.load(wp)
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=f'phase2-{rank}')
    t=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    bill=int(ctx.summary_dict()['flops_used']);assert y.shape==(16,1024) and np.isfinite(y).all()
    oldpacket=H2/'covariance_hidden_messages_v1'/('capture_00' if rank in TRAIN[:2] else 'capture_01')/f'row_{rank:04d}.npz'
    with np.load(oldpacket) as z:assert y.tobytes()==z['prediction'].tobytes()
    if group==0:
        base=[readrow for line in (H2/'both_tail_train2_v6/rows.jsonl').read_text().splitlines() if (readrow:=json.loads(line))['rank']==rank][0]
        assert bill>base['flops']  # metered research-only full CP/star probes
    current['prediction']=y
    with np.load(HERE/'capture_00'/f'row_{rank:04d}.npz') as old:
        for key in old.files:assert current[key].tobytes()==old[key].tobytes(),(rank,key)
    packet=out/f'row_{rank:04d}.npz'
    with packet.open('xb') as f:np.savez(f,**current)
    if not (HERE/'parent_parameters.npz').exists():
        with (HERE/'parent_parameters.npz').open('xb') as f:np.savez(f,**{k:np.array(v) for k,v in mod._GNN_PARAMS.items()})
    rec=dict(rank=rank,packet=pin(packet),flops=bill,wall_s=time.monotonic()-t,weights=pin(wp),all16_bytes_exact=True,matched_bill_exact=False,research_probes_metered=True,targets_accessed=False)
    records.append(rec)
    with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
    print(json.dumps(dict(event='capture',rank=rank,wall_s=rec['wall_s'],bytes=packet.stat().st_size)),flush=True)
    current.clear();del w,mlp,y;gc.collect();relief(None,0)
once(out/'complete.json',dict(rows=records,elapsed_s=time.monotonic()-start,targets_accessed=False,parent_sha256=PARENT_SHA,group=group))
