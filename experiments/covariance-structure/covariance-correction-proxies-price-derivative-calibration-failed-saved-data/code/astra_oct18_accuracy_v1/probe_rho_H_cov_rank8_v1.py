"""ONE fixed current-v61j rank16 rho-gated H8 covariance fit; no grid.
Four neuron blocks are within-network CV, never independent-network evidence.
"""
import json,re,resource,subprocess,time
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
DIM=8;RIDGE=.1;BILL_PCT=.03;BOARD=1.4515052706908603e-9
PAIRS=[(a,b)for a in range(DIM)for b in range(a,DIM)]
def guard():
    t=subprocess.check_output(['/usr/bin/memory_pressure'],text=True)
    f=int(re.search(r'System-wide memory free percentage: (\d+)%',t).group(1));assert f>=50;return f
def mdv(s,a,p,k3,k4):
    v=s*s;return p/(2*s)+k3*(3*a-a**3)*p/(12*v*v)+k4*(a**4-6*a*a+3)*p/(48*v*v*s)
guard();t0=time.monotonic()
z=np.load(D/'PRICE_CAPTURE_RANK16_v1.npz')
historical=np.load(D.parent/'astra_current_variance_capture_v1/run_capture_train16_v1/native_16_capture_capture.npz')
def v(l,k):return z[f'L{l}_{k}'].astype(float)
ww=np.load(O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0016/weights.npy',mmap_mode='r')
W14=np.asarray(ww[14],dtype=float);W15=np.asarray(ww[15],dtype=float)
truth=[np.load(O/f'analyst/fresh/pu/mps_truth_rank0016_{ab}32M.npz')for ab in('A','B')]
H=v(13,'state');center=H.mean(axis=0);H-=center
ev,Q=np.linalg.eigh(H.T@H/1024);rotation=Q[:,-DIM:]/np.sqrt(ev[-DIM:])
B=(H@rotation)*v(13,'gate')[:,None]
C=v(13,'pre_cov');C=(C+C.T)/2
cw=[C@(W14*B[:,b,None])for b in range(DIM)]
W2=W14*W14;diagC=np.diag(C)
F=[]
for a,b in PAIRS:
    f=np.sum((W14*B[:,a,None])*cw[b],axis=0)-W2.T@(diagC*B[:,a]*B[:,b])
    F.append(f*(1 if a==b else 2))
F=np.stack(F,axis=1);del cw,W2,H
s=v(14,'sigma');a=v(14,'alpha');p=v(14,'pdf');g=v(14,'gate');k3=v(14,'pre_k3');k4=v(14,'pre_k4');var=s*s
d=mdv(s,a,p,k3,k4);X=F*d[:,None]
target=d*((truth[0]['mc_pre_v'][14]+truth[1]['mc_pre_v'][14])/2-var)
def fit(rows):
    xx=X[rows];scale=np.maximum(np.sqrt(np.mean(xx*xx,axis=0)),1e-30);xx=xx/scale
    return np.linalg.solve(xx.T@xx/len(xx)+RIDGE*np.eye(len(PAIRS)),xx.T@target[rows]/len(xx))/scale
cv=np.zeros(1024);folds=[];inds=np.arange(1024)
for j in range(4):
    held=(inds//256)==j;theta=fit(~held);cv[held]=X[held]@theta
    folds.append(dict(block=j,target_mean_square=float(np.mean(target[held]**2)),
        corrected_target_mean_square=float(np.mean((target[held]-cv[held])**2))))
base=z['predictions'][15].astype(float);ta=truth[0]['mc_post'][15];tb=truth[1]['mc_post'][15]
u0=float(np.mean((base-ta)*(base-tb)))
gate15=historical['L15_gate'];s15=historical['L15_sigma'];a15=historical['L15_alpha'];p15=historical['L15_pdf']
d15=mdv(s15,a15,p15,historical['L15_pre_k3'],historical['L15_pre_k4'])
def score(shift):
    u=float(np.mean((base+shift-ta)*(base+shift-tb)));raw=100*(u/u0-1)
    adj=100*((u/u0)*(1+BILL_PCT/100)-1)
    return dict(U=u,proxy_accuracy_pct=raw,estimated_bill_pct=BILL_PCT,proxy_adjusted_pct=adj,
        optimistic_projected_score=BOARD*(1+adj/100),shift_rms=float(np.sqrt(np.mean(shift*shift))))
cv_score=score(gate15*(W15.T@cv))
theta=fit(np.ones(1024,dtype=bool));A=np.zeros((DIM,DIM))
for coeff,(aa,bb)in zip(theta,PAIRS):A[aa,bb]=A[bb,aa]=coeff
deltaC=C*((B@A)@B.T);np.fill_diagonal(deltaC,0)
pre14=W14.T@deltaC@W14;dv14=np.diag(pre14);dm14=d*dv14
post14=pre14*(g[:,None]*g[None,:])
r2dv=g+k3*(a*a-1)*p/(6*var*s)+k4*(3*a-a**3)*p/(24*var*var)
np.fill_diagonal(post14,(r2dv-2*v(14,'host')*d)*dv14)
dv15=np.einsum('ij,ij->j',W15,post14@W15)
allfit_score=score(gate15*(W15.T@dm14)+d15*dv15)
passed=cv_score['proxy_adjusted_pct']<=-.5 and allfit_score['proxy_adjusted_pct']<=-.5
result=dict(status='ELIGIBLE_NATIVE_OFF_NETWORK_TEST'if passed else'CLOSED_FIXED_RHO_H8_PROXY',
    source='current v61j ac93b10f',rank=16,dimension=DIM,ridge=RIDGE,no_grid=True,
    crossvalidation='four contiguous256-neuron blocks on ONE network; not independent-network confirmation',
    cv_endpoint_mean_path_only=cv_score,allfit_endpoint_with_covariance_propagation=allfit_score,folds=folds,
    covariance_relative_rms=float(np.linalg.norm(deltaC)/np.linalg.norm(v(13,'cov_after_edge'))),
    linearization_caveat='CV endpoint includes inheritedmean only. Allfit endpoint additionally propagates first-chaos covariance with exact diagonal scalarresponse; terminaljacobian is historical YR7. Native required.',
    not_banked=True,elapsed_s=time.monotonic()-t0,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    memory_free_end_pct=guard())
with(D/'RHO_H_COV8_PROXY_v1.json').open('x')as f:json.dump(result,f,indent=2)
if passed:np.savez(D/'RHO_H_COV8_RANK16FIT_HELD_v1.npz',center=center,rotation=rotation,A=A,theta=theta,pairs=PAIRS)
print(json.dumps(result,indent=2))
