"""Fixed full-negative-part endpoint proxy; one saved rank, not deployable."""
import json,resource,time
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;A=D.parent;O=A.parents[2]
start=time.monotonic()
z=np.load(A/'PRICE_CAPTURE_RANK16_v1.npz')
f=np.load(D/'PSD_NEGATIVE_FACTORS_v1.npz')
h=np.load(A.parent/'astra_current_variance_capture_v1/run_capture_train16_v1/native_16_capture_capture.npz')
W=np.load(O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0016/weights.npy',mmap_mode='r')
def val(l,k):return z[f'L{l}_{k}'].astype(float)
def mdv(s,a,p,k3,k4):
 v=s*s
 return p/(2*s)+k3*(3*a-a**3)*p/(12*v*v)+k4*(a**4-6*a*a+3)*p/(48*v*v*s)
W14=np.asarray(W[14],dtype=float);W15=np.asarray(W[15],dtype=float)
q13=f['L13_vectors']*np.sqrt(f['L13_magnitudes'])
q14=W14.T@q13
dv14=np.sum(q14*q14,axis=1)
s=val(14,'sigma');a=val(14,'alpha');p=val(14,'pdf');g=val(14,'gate');v=s*s
md=mdv(s,a,p,val(14,'pre_k3'),val(14,'pre_k4'))
dm14=md*dv14
r2dv=g+val(14,'pre_k3')*(a*a-1)*p/(6*v*s)+val(14,'pre_k4')*(3*a-a**3)*p/(24*v*v)
postvdv=r2dv-2*val(14,'host')*md
q15=W15.T@(g[:,None]*q14)
local15=W15.T@(f['L14_vectors']*np.sqrt(f['L14_magnitudes']))
dv15=np.sum(q15*q15,axis=1)+(W15*W15).T@((postvdv-g*g)*dv14)+np.sum(local15*local15,axis=1)
shift=h['L15_gate']*(W15.T@dm14)+mdv(h['L15_sigma'],h['L15_alpha'],h['L15_pdf'],h['L15_pre_k3'],h['L15_pre_k4'])*dv15
ys=[np.load(O/f'analyst/fresh/pu/mps_truth_rank0016_{ab}32M.npz')['mc_post'][15].astype(float)for ab in ('A','B')]
pred=z['predictions'][15].astype(float)
u0=float(np.mean((pred-ys[0])*(pred-ys[1])))
u1=float(np.mean((pred+shift-ys[0])*(pred+shift-ys[1])))
pct=100*(u1/u0-1)
relative_norm=float(np.sqrt(np.mean(shift*shift)/u0))
result=dict(status='CLOSED_NEGLIGIBLE_FULL_PSD_ORACLE',rank=16,
 fixed_full_negative_part=True,no_fit=True,no_sdk=True,no_model=True,
 base_crosshalf_U=u0,proxy_accuracy_pct=pct,
 optimistic_zero_cost_projected_adjusted_score=1.4468565569762618e-9*(1+pct/100),
 full_oracle_shift_rms=float(np.sqrt(np.mean(shift*shift))),
 absolute_most_favorable_alignment_proxy_pct=100*(2*relative_norm-relative_norm*relative_norm),
 caveat='Linearized Gaussian covariance propagation, analytic marginal response, historical terminal Jacobian. Not a global bound or independent confirmation; scale is orders below .5%.',
 elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with(D/'PSD_ENDPOINT_PROXY_v1.json').open('x')as out:json.dump(result,out,indent=2)
print(json.dumps(result,indent=2))
