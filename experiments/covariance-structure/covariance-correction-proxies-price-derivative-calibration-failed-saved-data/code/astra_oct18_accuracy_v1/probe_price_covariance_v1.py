"""Fixed unit-strength L13/L14 Price proxy, ONE current-v61j rank16 capture.
No fit and no alternative strength. Nonlocal/HC message context is fixed.
"""
import importlib.util,json,re,resource,subprocess,time
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[2]
def free():
    t=subprocess.check_output(['/usr/bin/memory_pressure'],text=True)
    return int(re.search(r'System-wide memory free percentage: (\d+)%',t).group(1))
assert free()>=50
start=time.monotonic()
spec=importlib.util.spec_from_file_location('pj',D.parent/'astra_oct18_bill_v1/partial_mean_jvp_v1.py')
pj=importlib.util.module_from_spec(spec);spec.loader.exec_module(pj)
z=np.load(D/'PRICE_CAPTURE_RANK16_v1.npz')
P={k[2:]:z[k].astype(float)for k in z.files if k.startswith('P_')}
W=np.load(O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3/rows/observer_fit/0016/weights.npy',mmap_mode='r')
historical=np.load(D.parent/'astra_current_variance_capture_v1/run_capture_train16_v1/native_16_capture_capture.npz')
def val(l,k):return z[f'L{l}_{k}'].astype(float)
def mean_dv(s,a,p,k3,k4):
    v=s*s
    return p/(2*s)+k3*(3*a-a**3)*p/(12*v*v)+k4*(a**4-6*a*a+3)*p/(48*v*v*s)
def directions(l):
    x=val(l,'x');a=val(l,'alpha');p=val(l,'pdf');g=val(l,'gate');s=val(l,'sigma')
    t3=val(l,'pre_k3')/s**3;t4=val(l,'pre_k4')/s**4
    q=g+t3*(a*a-1)*p/6+t4*(-a**3+3*a)*p/24
    m=val(l,'host')/s;v=val(l,'post_v')/s**2;k3=val(l,'post_k3')/s**3
    dx=np.zeros_like(x)
    dx[:,3]=1;dx[:,5]=p;dx[:,6]=-a*p;dx[:,7]=g;dx[:,8]=q;dx[:,9]=q-g
    dx[:,11]=2*m*(1-q);dx[:,20]=3*(1-q)*(v-m*m);dx[:,21]=4*(1-q)*(k3+m**3-3*m*v)
    dx[:,26]=1/1024;dx[:,29]=q/1024
    slope=pj.partial_mean_jvp(np,l,x,dx,val(l,'M'),val(l,'z0'),val(l,'z1'),P,
        mean=val(l,'host'),delta=val(l,'delta'),base_mean_dmu=q)
    return slope,q
correction={};detail={}
for l in (13,14):
    slope,q=directions(l)
    correction[l]=pj.first_price_covariance(np,val(l,'pre_cov'),val(l,'gate'),slope)
    detail[l]=dict(derivative_rms=float(np.sqrt(np.mean(slope*slope))),derivative_mean=float(slope.mean()),
        correction_rms=float(np.sqrt(np.mean(correction[l]**2))),
        relative_covariance_rms=float(np.linalg.norm(correction[l])/np.linalg.norm(val(l,'cov_after_edge'))),
        clipped_count=int((val(l,'delta')<=-val(l,'host')).sum()))
W14=np.asarray(W[14],dtype=float);W15=np.asarray(W[15],dtype=float)
dpre14=W14.T@correction[13]@W14;dv14=np.diag(dpre14)
s14=val(14,'sigma');a14=val(14,'alpha');p14=val(14,'pdf');g14=val(14,'gate')
k314=val(14,'pre_k3');k414=val(14,'pre_k4');v14=s14*s14
mdv14=mean_dv(s14,a14,p14,k314,k414)
dm14=mdv14*dv14
# Leading first-chaos propagation, with exact analytic marginal variance response.
dpost14=(g14[:,None]*g14[None,:])*dpre14
r2dv14=g14+k314*(a14*a14-1)*p14/(6*v14*s14)+k414*(3*a14-a14**3)*p14/(24*v14*v14)
postvdv14=r2dv14-2*val(14,'host')*mdv14
np.fill_diagonal(dpost14,postvdv14*dv14)
combined14=dpost14+correction[14]
dv15=np.einsum('ij,ij->j',W15,combined14@W15)
s15=historical['L15_sigma'];a15=historical['L15_alpha'];p15=historical['L15_pdf'];g15=historical['L15_gate']
mdv15=mean_dv(s15,a15,p15,historical['L15_pre_k3'],historical['L15_pre_k4'])
shift=g15*(W15.T@dm14)+mdv15*dv15
ys=[np.load(O/f'analyst/fresh/pu/mps_truth_rank0016_{ab}32M.npz')['mc_post'][15].astype(float)for ab in ('A','B')]
pred=z['predictions'][15].astype(float)
base=float(np.mean((pred-ys[0])*(pred-ys[1])))
arm=float(np.mean((pred+shift-ys[0])*(pred+shift-ys[1])))
acc=100*(arm/base-1);bill=.052
adj=100*((arm/base)*(1+bill/100)-1)
result=dict(status='ELIGIBLE_FOR_NATIVE_ONLY'if adj<=-.5 else 'CLOSED_FIXED_UNIT_PROXY',rank=16,
    current_parent='v61j ac93b10f',fixed_unit_strength=True,no_fit=True,partial_local_derivative=True,
    base_crosshalf_U=base,proxy_crosshalf_U=arm,proxy_accuracy_pct=acc,estimated_bill_pct=bill,
    proxy_adjusted_pct=adj,optimistic_projected_score=1.451748e-9*(1+adj/100),
    detail=detail,shift_rms=float(np.sqrt(np.mean(shift*shift))),
    variance14_shift_rms_relative=float(np.sqrt(np.mean((dv14/v14)**2))),
    terminal_jacobian='historical YR7 rank16 L15; v61j endpoint and L13/L14 state exact',
    omitted_responses='HC/message/context and learnedhead variance response; higher-chaos offdiagonal propagation; gate/moment feedback',
    not_banked=True,elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    memory_free_end_pct=free())
with(D/'PRICE_COV_PROXY_v1.json').open('x')as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
