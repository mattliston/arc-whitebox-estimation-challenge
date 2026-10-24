import sys,types,time,gc,ctypes
import numpy as np
import flopscope as flops
from common import *
verify();mode=sys.argv[1];assert mode in ('baseline','capture')
u=read(HERE/'unit_result.json');g=read(HERE/'unit_guard.json')
assert u['passed'] and u['seal_sha256']==sha(HERE/'seal.json')
assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
if mode=='capture':
    g=read(HERE/'baseline_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
    base={r['mini']:r for r in read(HERE/'baseline/complete.json')['rows']}
out=HERE/mode;out.mkdir()
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('current_covariance_supervision_'+mode);mod.__file__=str(PARENT);sys.modules[mod.__name__]=mod
original=PARENT.read_text();source=original
tap='                _capture_scalar(mu,v,k3,bpre,radial)\n'
line='                m, aux = nonlinear_scalar(mu, v, k3, _e2_H(_e2_H(radial * v) * v))'
if mode=='capture':
    assert source.count(line)==1;source=source.replace(line,tap+line);assert source.replace(tap,'')==original
exec(compile(source,str(PARENT)+':'+mode,'exec'),mod.__dict__)
current={}
if mode=='capture':
    def scalar(mu,v,k3,bpre,radial):current.update({k:np.array(x,dtype=np.float64) for k,x in locals().copy().items()})
    mod._capture_scalar=scalar
    oldfeatures,oldstep,oldcov,oldcut,oldfinal=mod.features,mod.gnn_step,mod._flms_covariance_diagonal,mod._l15_rowcut_k3,mod.final_scalar_features
    def features(l,m,C,K21,aux,w):
        x=oldfeatures(l,m,C,K21,aux,w)
        if l==14:current.update(K=np.array(K21),s14=np.array(aux['sigma']))
        return x
    def step(l,x,sigma,w,state,P):
        y=oldstep(l,x,sigma,w,state,P)
        if l==14:current['F']=np.concatenate((np.asarray(x),np.asarray(y[1])),axis=1).astype(np.float32)
        if l==15:
            k=mod._e2_gnn_consts(P,len(sigma))
            current.update(M=np.asarray(mod._E2_MSG['v'][2]).transpose(1,0,2).reshape(len(sigma),48).copy(),elog=np.array(k['HM36']),native_delta=np.array(y[0]))
        return y
    def covariance(w,C):
        assert 'C' not in current;current.update(C=np.array(C),W=np.array(w));return oldcov(w,C)
    def cut(*args):
        y=oldcut(*args);current['cut']=np.array(y);limit=mod._L15_ROWCUT_Z
        try:
            mod._L15_ROWCUT_Z=1e20;full=oldcut(*args)
        finally:mod._L15_ROWCUT_Z=limit
        full=np.array(full);mu=np.asarray(args[1]);v=np.asarray(args[2]);z=np.abs(mu/np.sqrt(np.maximum(v,1e-10)))
        count=min(len(z),((np.count_nonzero(z<=3.5)+31)//32)*32);rows=np.argsort(z,kind='stable')[:count]
        full[rows]=current['cut'][rows];current['fullcut']=full
        return y
    def final(l,m,aux,w):
        x=oldfinal(l,m,aux,w);current.update(native_x=np.array(x),native_m=np.array(m));return x
    mod.features,mod.gnn_step,mod._flms_covariance_diagonal,mod._l15_rowcut_k3,mod.final_scalar_features=features,step,covariance,cut,final
est=mod.Estimator();est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
relief=ctypes.CDLL('/usr/lib/libSystem.B.dylib').malloc_zone_pressure_relief;relief.argtypes=[ctypes.c_void_p,ctypes.c_size_t];relief.restype=ctypes.c_size_t
with np.load(OLDPAIR/'parent_parameters.npz') as z:expected_params={k:z[k] for k in z.files}
records=[];start=time.monotonic()
for mini in ALL:
    assert not current
    wp,identity=weights(mini);w=np.load(wp)
    mlp=types.SimpleNamespace(width=1024,depth=16,weights=[w[l] for l in range(16)],seed=0,name=identity['mlp_name'])
    started=time.monotonic()
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:y=np.asarray(est.predict(mlp,2**41),dtype=np.float64)
    bill=int(ctx.summary_dict()['flops_used']);assert y.shape==(16,1024) and np.isfinite(y).all()
    for k,x in expected_params.items():assert np.array_equal(x,np.asarray(mod._GNN_PARAMS[k])),k
    if mode=='capture':
        e=base[mini]['packet'];assert pin(e['path'])==e
        with np.load(e['path']) as z:assert y.tobytes()==z['prediction'].tobytes(),mini
        assert bill>base[mini]['flops'],(mini,bill,base[mini]['flops'])
    current['prediction']=y
    p=out/f'mini{mini:04d}.npz'
    with p.open('xb') as f:np.savez(f,**current)
    row=dict(mini=mini,identity=identity,weights=pin(wp),packet=pin(p),flops=bill,wall_s=time.monotonic()-started,targets_accessed=False)
    if mode=='capture':row.update(all16_bytes_exact=True,research_fullcut_probe_metered=True,extra_bill=bill-base[mini]['flops'])
    records.append(row)
    with (out/'rows.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(dict(event=mode,mini=mini,wall_s=row['wall_s'],flops=bill,all16_bytes_exact=mode=='capture')),flush=True)
    current.clear();del w,mlp,y;gc.collect();relief(None,0)
once(out/'complete.json',dict(rows=records,parent_sha256=PARENT_SHA,scalar_tap_inverse_exact=mode=='capture',mode=mode,elapsed_s=time.monotonic()-start,targets_accessed=False))
