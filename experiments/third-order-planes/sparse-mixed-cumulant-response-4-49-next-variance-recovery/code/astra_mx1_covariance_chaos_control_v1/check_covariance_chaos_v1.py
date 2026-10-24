"""One fixed covariance-chaos all-cohort MX1 control variate, existing saved data only."""
from pathlib import Path
import os,json,time,resource,hashlib,math
import numpy as np
assert os.getpriority(os.PRIO_PROCESS,0)>=10
assert all(os.environ[k]=='1'for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
D=Path(__file__).resolve().parent;O=D.parents[2];start=time.monotonic();rank=1
store=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
entry=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][35]
weights=np.load((store/entry['path']).parent/'weights.npy',mmap_mode='r')
with np.load(O/'physicist/t2cap_rank0035.npz')as z:
 birth=z['P_11'].astype(float);half=birth.shape[1]//2;delta=birth[:,:half]-birth[:,half:];hubs=np.argmax(np.abs(delta),axis=0)
 tp=z['plus_11'][hubs];tm=z['minus_11'][hubs];d=z['d_11'][hubs]
 delta[hubs,np.arange(half)]=0;assert np.max(np.abs(delta))<1e-12
 pushed=weights[12].astype(float).T@birth;plus,minus=pushed[:,:half],pushed[:,half:]
 E=(plus-minus)/(tp-tm)[None,:];Y=(tp[None,:]*minus-tm[None,:]*plus)/(tp-tm)[None,:];A=E*Y*d[None,:]
 mu=z['mu_12'].astype(float);v=np.diag(z['Cpre_12']).copy();s=np.sqrt(v)
 Cprev=z['C_11'].astype(float)
 B=(tp[None,:]*birth[:,half:]-tm[None,:]*birth[:,:half])/(tp-tm)[None,:]
 varw=2.0/len(mu)
 meanA=varw*d*B[hubs,np.arange(len(hubs))]
 coeff=d*np.sum(B*Cprev[:,hubs],axis=0)/np.sum(Cprev*Cprev)
 q=v-varw*np.trace(Cprev)
 vcheck=np.sum(weights[12].astype(float)*(Cprev@weights[12].astype(float)),axis=0)
 Ahat=q[:,None]*coeff[None,:]+meanA[None,:]

with np.load(O/'e15cols_rank0035_L12.npz')as z:
 stops=np.r_[0,z['stops']];k0=int(stops[-2]);Pold=z['pp'][:,:k0].astype(float);lam=z['lam'][:k0].astype(float);response=z['resp'].astype(bool)
with np.load(O/'physicist/t4cols_rank0035_L11.npz')as z:
 previous=z['pp'];assert previous.shape[1]==k0
 Pprev=previous[hubs].astype(float);del previous
coefficient=coeff@Pprev;mean_coefficient=meanA@Pprev;del Pprev
hat=6*(q[:,None]*coefficient[None,:]+mean_coefficient[None,:]);del coefficient,mean_coefficient
approx=(hat*Pold*lam[None,:])@Pold.T
del hat,Pold
with np.load(O/'k31diag_rank0035_diag.npz')as z:
 teacher=z['MX1_12'].astype(float);teacher_mu=z['mu_12'].astype(float);teacher_v=z['v_12'].astype(float)
np.fill_diagonal(approx,0);np.fill_diagonal(teacher,0)
a=mu/s;pdf=np.exp(-a*a/2)/np.sqrt(2*np.pi);gate=np.array([.5*math.erfc(-x/np.sqrt(2))for x in a]);e=-a*pdf/v
cov=e[:,None]*teacher*gate[None,:];cov=(cov+cov.T)/6
covhat=e[:,None]*approx*gate[None,:];covhat=(covhat+covhat.T)/6
wn=weights[13].astype(float);dv=np.sum(wn*(cov@wn),axis=0);dvh=np.sum(wn*(covhat@wn),axis=0)
def shares(t,p):
 den=np.sum(t*t);pd=np.sum(p*p);cross=np.sum(t*p)
 return dict(relative_energy=float(pd/den),cross_share=float(cross/den),explained=float((2*cross-pd)/den),cosine=float(cross/np.sqrt(den*pd)))
n=len(mu);h=len(hubs);m=int(response.sum());R=k0
# Runtime uses response rows for theta and the existing read for its final plane.
# Include all previous-column gather, newborn extraction, storage copy, casts,
# factor formation, coefficient/reconstruction, fold and scatter allowances.
items=dict(covariance_mode_coefficients=4*n*h+2*n*n,allcohort_coefficients=4*h*R,reconstruct_response_theta=4*m*R,fold_into_existing_read=5*m*R,previous_copy_gather_cast=2*n*R+h*R,diagonal_handling=5*n)
out=dict(status='FIXED_COVARIANCE_CHAOS_CONTROL_ALL_COHORTS_SAVED_PROXY',rank=rank,
 dimensions=dict(n=n,hubs=h,all_older_columns=R,response_rows=m,cohorts=len(stops)-2),
 A_projection=shares(A,Ahat),quadratic_mode_v_relative_mismatch=float(np.linalg.norm(v-vcheck)/np.linalg.norm(v)),A_mean_RMS=float(np.sqrt(np.mean(meanA*meanA))),signed_MX1=shares(teacher,approx),covariance_response=shares(cov,covhat),next_variance_response=shares(dv,dvh),
 teacher_mu_relative_mismatch=float(np.linalg.norm(mu-teacher_mu)/np.linalg.norm(mu)),teacher_variance_relative_mismatch=float(np.linalg.norm(v-teacher_v)/np.linalg.norm(v)),
 shape_arithmetic_and_conservative_linear_allowances=items,layer_shape_cost=sum(items.values()),layer_bill_pct=100*sum(items.values())/238299111296,
 seven_late_layers_same_geometry_bill_pct=700*sum(items.values())/238299111296,
 runtime_storage_previous_pp_bytes=4*n*R,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 models=0,captures=0,truth_reads=0,seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 limits=['Historical network35 layer12; no native final gain inferred.','Exact Gaussian inner-product coefficient onto C_previous quadratic mode; no fitting or rank/coefficient grid.','Cost is explicit shape arithmetic/linear allowance, not native full-model billing.','No cost credit for the unchanged read; retaining/gathering previous all-column pp is additional.'])
assert out['rss_bytes']<2_000_000_000
(D/'SAVED_COVARIANCE_CHAOS_RESULT_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
