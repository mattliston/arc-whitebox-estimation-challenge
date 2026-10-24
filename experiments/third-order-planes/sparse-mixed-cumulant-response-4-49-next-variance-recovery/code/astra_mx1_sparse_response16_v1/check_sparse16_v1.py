"""One fixed sparse exact MX1 row rule, saved L12 only, zero unsampled rows."""
from pathlib import Path
import os,json,time,resource,hashlib
import numpy as np
from scipy.special import ndtr
assert os.getpriority(os.PRIO_PROCESS,0)>=10
THREADS=('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS')
assert all(os.environ[x]=='1'for x in THREADS)
D=Path(__file__).resolve().parent;O=D.parents[2];t0=time.monotonic();n=1024;keep=16
store=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
entry=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][35]
weights=np.load((store/entry['path']).parent/'weights.npy',mmap_mode='r');wn=weights[13].astype(float);wn2=wn*wn
with np.load(O/'physicist/t2cap_rank0035.npz')as z:
 mu=z['mu_12'].astype(float);v=np.diag(z['Cpre_12']).copy();sigma=np.sqrt(v)
 m=z['m_12'].astype(float);postdiag=np.diag(z['C_12']).copy()
 birth=z['P_11'];hubs=birth.shape[1]//2;del birth
proxy_mu=wn.T@m;proxy_v=np.maximum(wn2.T@np.maximum(postdiag,1e-10),1e-10);proxy_s=np.sqrt(proxy_v)
proxy_pdf=np.exp(-.5*(proxy_mu/proxy_s)**2)/np.sqrt(2*np.pi);h=proxy_pdf/(2*proxy_s)
alpha=mu/sigma;pdf=np.exp(-.5*alpha*alpha)/np.sqrt(2*np.pi);gate=ndtr(alpha);e=-alpha*pdf/v
score=(alpha*pdf*sigma)**2*(wn2@(h*h))
selected=np.argsort(-score,kind='stable')[:keep];mask=np.zeros(n,dtype=bool);mask[selected]=True
with np.load(O/'e15cols_rank0035_L12.npz')as z:R=int(z['stops'][-2])
with np.load(O/'k31diag_rank0035_diag.npz')as z:teacher=z['MX1_12'].astype(float)
np.fill_diagonal(teacher,0);sparse=teacher*mask[:,None]
cov=e[:,None]*teacher*gate[None,:];cov=(cov+cov.T)/6
covhat=e[:,None]*sparse*gate[None,:];covhat=(covhat+covhat.T)/6
full_dv=np.sum(wn*(cov@wn),axis=0);kept_dv=np.sum(wn*(covhat@wn),axis=0)
def metric(t,p):
 den=np.sum(t*t);pe=np.sum(p*p);cross=np.sum(t*p)
 return dict(relative_energy=float(pe/den),cross_share=float(cross/den),explained=float((2*cross-pe)/den),cosine=float(cross/np.sqrt(den*pe)))
items=dict(selected_A_from_paid_newborn=10*keep*hubs,exact_allcohort_theta=2*keep*hubs*R,fold_selected_left=5*keep*R,selector_weight_square= n*n,selector_three_matvecs=6*n*n,selector_vector_allowance=20*n)
out=dict(status='FIXED_TOP16_SPARSE_EXACT_MX1_SAVED_L12',selection='descending (alpha*pdf*sqrt(v))^2 times Wnext^2 @ hproxy^2; hproxy from current m and diagonal-only propagated current postvariance; stable ties',keep=keep,selected_rows=selected.tolist(),teacher_used_for_selection=False,unsampled_rule='exact zero',signed_MX1=metric(teacher,sparse),covariance_response=metric(cov,covhat),next_variance_response=metric(full_dv,kept_dv),next_mean_proxy_response=metric(h*full_dv,h*kept_dv),omitted_next_variance_RMS=float(np.sqrt(np.mean((full_dv-kept_dv)**2))),full_next_variance_RMS=float(np.sqrt(np.mean(full_dv**2))),selected_score_share=float(np.sum(score[selected])/np.sum(score)),dimensions=dict(n=n,hubs=hubs,all_older_columns=R,selected_rows=keep),minimum_direct_shape_cost=items,minimum_layer_flops=sum(items.values()),minimum_layer_bill_pct=100*sum(items.values())/238299111296,seven_layers_same_geometry_minimum_bill_pct=700*sum(items.values())/238299111296,limits=['One historical network and layer; no final accuracy inference.','Cost is optimistic direct shape arithmetic; omits carrying/gathering previous pp, no exact native tariff.','Existing read reuse assumed for folded selected left; a separate selective read adds 2*keep*R*n FLOPs.','Proxy future marginals derive only from current paid moments and known weights.'],models=0,captures=0,truth_reads=0,nice=os.getpriority(os.PRIO_PROCESS,0),seconds=time.monotonic()-t0,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
assert out['rss_bytes']<2_000_000_000
(D/'SAVED_SPARSE16_RESULT_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
