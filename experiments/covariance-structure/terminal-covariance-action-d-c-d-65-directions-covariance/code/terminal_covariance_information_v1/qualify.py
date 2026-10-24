"""Target-free local tangent and finite native final-block replay."""
import hashlib,importlib.util,json,resource,sys,time,types
from pathlib import Path
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info
from tangent import covariance_jacobian,terminal
HERE=Path(__file__).resolve().parent;D=HERE.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert all(t['num_threads']==1 for t in threadpool_info())
seal=read(HERE/'sources_sealed.json')
for x in seal['sources']+seal['external_pins']:assert sha(x['path'])==x['sha256'],x['path']
assert read(HERE/'unit.json')['passed']
guard=read(HERE/'capture_watch.json');assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
capt=read(HERE/'capture/result.json');assert capt['completed'] and capt['repeat_all_captured_arrays_exact']
out=HERE/'qualification';out.mkdir(exist_ok=False)
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
source=D/'lazy_rotation_objective_v1/candidate.py';spec=importlib.util.spec_from_file_location('tci_replay_parent',source);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
with flops.BudgetContext(flop_budget=2**41,quiet=True) as setup:
    spec.loader.exec_module(mod)
    arena=fnp.empty((mod._PACKED_ELEMENTS,),dtype=fnp.float32)
    mod._SCRATCH=[arena,mod._cp3_bind_arena_views(arena,mod._PACKED_PUSH),mod._cp3_bind_arena_views(arena,mod._PACKED_READ),None,None,None,None]
    mod._bind_push6(arena)

def native(a,q):
    with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
        C=fnp.asarray(a['C']);W=fnp.asarray(a['W']);H=fnp.asarray(a['H14'].copy())
        P={k[2:]:fnp.asarray(v.copy()) for k,v in a.items() if k.startswith('P_')}
        Q=fnp.asarray(q,dtype=fnp.float32)
        scale=fnp.exp(.005*fnp.tanh(H@Q[:64]+Q[64]))
        C=(C*scale[:,None])*scale[None,:]
        v,av=mod._cov_compact_diagonal(W,C,fnp.asarray(a['indices']),fnp.asarray(a['alive']))
        k3=fnp.asarray(a['k3base'])+3*av*fnp.asarray(a['bpre'])
        m,aux=mod.nonlinear_scalar(fnp.asarray(a['mu']),v,k3,fnp.asarray(a['radial'])*v*v)
        x=mod.final_scalar_features(15,m,aux,W)
        delta,_=mod.gnn_step(15,x,aux['sigma'],W,H,P)
        answer=fnp.asarray(fnp.maximum(m+fnp.asarray(delta,dtype=fnp.float64),0),dtype=fnp.float32)
    return np.asarray(answer).astype(np.float64).copy(),ctx.summary_dict()['flops_used']

started=time.monotonic();records=[];all_pass=True
for rec in capt['records'][:2]:
    assert sha(rec['capture'])==rec['capture_sha256']
    with np.load(rec['capture'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    X=np.column_stack([a['H14'].astype(np.float64),np.ones(1024)])
    dv,da=covariance_jacobian(a['C'],a['W'],a['indices'],a['alive'],X)
    J=terminal(a,dv,da,native_anchors=True).d
    assert J.shape==(1024,65) and np.isfinite(J).all()
    direction=np.sin(.37*np.arange(1,66,dtype=np.float64));direction/=np.sqrt(np.mean((X@direction)**2))
    prediction=a['prediction'][-1].astype(np.float64);zero,zero_F=native(a,np.zeros(65))
    delta0=zero-prediction;zero_max=float(np.max(np.abs(delta0)));zero_rms=float(np.sqrt(np.mean(delta0**2)))
    secants=[];bills=[zero_F];replays={'zero':zero,'direction':direction}
    for scale in [.05,.025]:
        plus,pF=native(a,scale*direction);minus,mF=native(a,-scale*direction)
        secants.append((plus-minus)/(2*scale));bills.extend([pF,mF]);replays['plus_'+str(scale)]=plus;replays['minus_'+str(scale)]=minus
    t=J@direction;s=secants[1];norm=float(np.linalg.norm(s))
    assert norm>0
    secant_rms=norm/np.sqrt(1024)
    amplitude_error=float(np.linalg.norm(secants[0]-s)/norm)
    tangent_error=float(np.linalg.norm(t-s)/norm)
    passed=zero_max<=1e-6 and zero_rms<=1e-7 and secant_rms>1e-7 and amplitude_error<=.05 and tangent_error<=.05
    feature_path=out/f"rank{rec['rank']:04d}.npz"
    B=a['sigma'].astype(np.float64)[:,None]*np.column_stack([a['H15'].astype(np.float64),np.ones(1024)])
    np.savez(feature_path,J=J,B=B,prediction=a['prediction'],tangent_direction=t,secant_coarse=secants[0],secant_fine=s,**replays)
    row=dict(rank=rec['rank'],passed=bool(passed),zero_maxabs=zero_max,zero_rms=zero_rms,secant_rms=secant_rms,secant_amplitude_relative_error=amplitude_error,tangent_to_native_secant_relative_error=tangent_error,path=str(feature_path),sha256=sha(feature_path),capture_sha256=rec['capture_sha256'],local_replay_flops=bills,parent_flops=rec['parent_flops'])
    records.append(row);all_pass=all_pass and passed;print(json.dumps(row),flush=True)
    del a,X,J,B,dv,da
result=dict(completed=True,passed=bool(all_pass),records=records,truth_accessed=False,full_model_calls=0,local_replay_calls=10,all_native_directions_certified=False,source_seal_sha256=sha(HERE/'sources_sealed.json'),capture_receipt_sha256=sha(HERE/'capture/result.json'),elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with (out/'result.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps({'completed':True,'passed':bool(all_pass)}),flush=True)
