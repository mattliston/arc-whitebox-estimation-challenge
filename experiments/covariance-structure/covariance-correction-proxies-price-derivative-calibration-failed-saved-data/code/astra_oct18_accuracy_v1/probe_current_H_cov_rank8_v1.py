"""Fixed rank8 current-H covariance readout, historical saved-data screen only."""
import json, math, re, resource, subprocess, time
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
C=D.parent/'astra_current_variance_capture_v1'
S=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
RANKS=(16,259,277,287);RIDGE=.1;DIM=8;START=time.monotonic()
def guard():
    raw=subprocess.check_output(['memory_pressure'],text=True)
    free=int(re.search(r'System-wide memory free percentage: (\d+)%',raw).group(1))
    if free<50:raise RuntimeError(f'memory free {free}<50')
    return free
def read(rank):
    guard()
    p=C/f'run_capture_train{rank}_v1/native_{rank}_capture_capture.npz'
    with np.load(p)as z:
        keys=['L13_state64','L13_sigma','L13_gate','L14_pre_v','L14_pre_k3','L14_pre_k4','L14_alpha','L14_sigma','L14_pdf','L15_gate','predictions_float64']
        c={k:z[k].astype('d') for k in keys}
    w=np.load(S/f'rows/observer_fit/{rank:04d}/weights.npy',mmap_mode='r')
    c.update(rank=rank,W14=w[14].astype('d'),W15=w[15].astype('d'))
    for half in ('A','B'):
        with np.load(O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_{half}32M.npz')as z:
            c['tv'+half]=z['mc_pre_v'][14].astype('d');c['t'+half]=z['mc_post'][15].astype('d')
    a=c['L14_alpha'];v=c['L14_pre_v'];s=c['L14_sigma'];p=c['L14_pdf']
    c['sensitivity']=p/(2*s)+c['L14_pre_k3']*(3*a-a**3)*p/(12*v*v)+c['L14_pre_k4']*(a**4-6*a*a+3)*p/(48*v*v*s)
    c['target']=c['sensitivity']*((c['tvA']+c['tvB'])/2-v)
    return c
PAIRS=[(a,b) for a in range(DIM) for b in range(a,DIM)]
def features(c,center,rotation):
    B=(c['L13_state64']-center)@rotation
    B*= (c['L13_sigma']*c['L13_gate'])[:,None]
    V=c['W14'].T@B
    local=np.stack([B[:,a]*B[:,b]*(1 if a==b else 2) for a,b in PAIRS],axis=1)
    pushed=np.stack([V[:,a]*V[:,b]*(1 if a==b else 2) for a,b in PAIRS],axis=1)
    F=pushed-(c['W14']*c['W14']).T@local
    return F,F*c['sensitivity'][:,None]
def fit(train):
    center=np.mean([c['L13_state64'].mean(axis=0) for c in train],axis=0)
    gram=sum((c['L13_state64']-center).T@(c['L13_state64']-center) for c in train)/(1024*len(train))
    ev,V=np.linalg.eigh(gram);assert ev[-DIM]>0
    rotation=V[:,-DIM:]/np.sqrt(ev[-DIM:])
    XX=[features(c,center,rotation)[1]for c in train]
    scale=np.sqrt(np.mean([np.mean(x*x,axis=0)for x in XX],axis=0));assert scale.min()>0
    gram=np.eye(len(PAIRS))*RIDGE;rhs=np.zeros(len(PAIRS))
    for c,x in zip(train,XX):
        x=x/scale;gram+=x.T@x/(1024*len(train));rhs+=x.T@c['target']/(1024*len(train))
    theta=np.linalg.solve(gram,rhs)/scale
    return center,rotation,theta
def score(c,model):
    center,rotation,theta=model
    F,X=features(c,center,rotation);dv=F@theta;dm=X@theta
    shift=c['L15_gate']*(c['W15'].T@dm)
    base=c['predictions_float64'][15]
    new=(base+shift).astype('f').astype('d')
    U0=float(np.mean((base-c['tA'])*(base-c['tB'])));U=float(np.mean((new-c['tA'])*(new-c['tB'])))
    s=c['sensitivity'];v=c['L14_pre_v']
    V0=float(np.mean(s*s*(v-c['tvA'])*(v-c['tvB'])));VV=float(np.mean(s*s*(v+dv-c['tvA'])*(v+dv-c['tvB'])))
    return dict(rank=c['rank'],base_U=U0,candidate_U=U,proxy_adjusted_pct=100*(U/U0-1),variance_effect_U_change_pct=100*(VV/V0-1),shift_rms=float(np.sqrt(np.mean(shift*shift))),relative_variance_delta_rms=float(np.sqrt(np.mean((dv/v)**2))))
data=[read(r)for r in RANKS];rows=[]
for held in data:
    model=fit([c for c in data if c is not held]);rows.append(score(held,model))
    print(rows[-1],flush=True)
mean=float(np.mean([r['proxy_adjusted_pct']for r in rows]));pooled=100*(sum(r['candidate_U']for r in rows)/sum(r['base_U']for r in rows)-1)
passed=mean<=-.5 and sum(r['proxy_adjusted_pct']<0 for r in rows)>=3
res=dict(status='HISTORICAL_PROXY_PASS_NEEDS_CURRENT_NATIVE'if passed else'CLOSED_FIXED_HISTORICAL_PROXY',rank=DIM,ridge=RIDGE,ranks=RANKS,rows=rows,mean_proxy_adjusted_pct=mean,pooled_proxy_adjusted_pct=pooled,
         optimistic_current_score_projection=1.451748e-9*(1+mean/100),projection_warning='Unbanked historical YR7 proxy only; actual current-lineage score cannot be inferred reliably',
         approximate_deploy_flops=2*1024*64*8+2*1024*8*8+2*1024*8*1024+3*1024*1024,
         elapsed_s=time.monotonic()-START,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,memory_free_end_pct=guard(),no_new_model_calls=True)
if passed:
    center,rotation,theta=fit(data);np.savez(D/'H_COV8_ALLFIT_SCREEN_ONLY_v1.npz',center=center,rotation=rotation,theta=theta,pairs=PAIRS)
p=D/'H_COV8_PROXY_v1.json';assert not p.exists();p.write_text(json.dumps(res,indent=2));print(json.dumps({k:res[k]for k in ['status','mean_proxy_adjusted_pct','pooled_proxy_adjusted_pct','optimistic_current_score_projection','elapsed_s','maxrss_bytes','memory_free_end_pct']},indent=2))
