"""One fixed L13 320-Gram+64 actual next-weight cubic contractions; saved only."""
from pathlib import Path
import hashlib,json,os,resource,time
import numpy as np
D=Path(__file__).resolve().parent;O=D.parents[3]
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):assert os.environ[key]=='1'
archive=O/'engineer/a35_join/state/s39_rank0651.npz'
auxfile=O/'engineer/a36_carrier/state/s38_rank0651.npz'
store=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
entry=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][651]
wp=store/Path(entry['path']).parent/'weights.npy'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prereg=dict(rank=384,gram_columns=320,cubic_columns=64,layer=13,actual_next_weight_columns=list(range(0,1024,16)),evaluation_seed=20261017,evaluation_columns=64,
 sketch='Z=J@[lambda*(J.T@W14[:,0:1024:16])^2], exact uncentered next-weight contractions of new signed columns; fixed64 uniformly spaced output neurons; no target selection',
 basis='Single QR([G[:,top320], Z columns scaled to equal Euclidean norm]); retain same384 factors/reconstruction; no future core',
 archives={str(p):sha(p) for p in (archive,auxfile,wp)},source_sha256=sha(Path(__file__)))
(D/'PREREG_v1.json').open('x').write(json.dumps(prereg,indent=2)+'\n')
t0=time.monotonic();z=np.load(archive)
J=z['J_13'].astype(float);lam=z['lam_13'].astype(float);Qp=z['Qp_13'].astype(float);core=z['core_13'].astype(float);prev=z['prev_13']
old_rows=z['rows_12'];saved_old=z['old_12'].astype(float)
full_old=np.zeros((1024,saved_old.shape[1]));full_old[old_rows]=saved_old
observed_old=(full_old*z['gate_12'][:,None])[prev];valid=np.abs(observed_old).sum(axis=1)>0
old_factors=np.linalg.lstsq(Qp[valid],observed_old[valid],rcond=None)[0]
old_fit_error=float(np.linalg.norm(Qp[valid]@old_factors-observed_old[valid])/np.linalg.norm(observed_old[valid]))
old=Qp@old_factors;del full_old,observed_old,saved_old
lo=z['lamold_12'].astype(float);F=np.concatenate((old,J),axis=1);co=np.r_[lo,lam];del old
W=np.load(wp,mmap_mode='r')[13,prev,:].astype(float);m,k=J.shape
G=(J*(lam*lam)[None,:])@J.T+Qp@core@Qp.T;G=(G+G.T)*.5
order=np.argsort(-np.diag(G),kind='stable')
omega=W[:,::16].copy();assert omega.shape==(m,64)
ha=J.T@omega
Z=J@(lam[:,None]*(ha*ha))
Z/=np.linalg.norm(Z,axis=0)[None,:]
Qbase=np.linalg.qr(G[:,np.sort(order[:384])],mode='reduced')[0]
Qnew=np.linalg.qr(np.concatenate((G[:,np.sort(order[:320])],Z),axis=1),mode='reduced')[0]
del ha,omega,Z,G
A=W.T@F;cp=(A*A*co[None,:])@A.T;diag=np.diag(cp).copy()
aux=np.load(auxfile)['aux_14'];alpha=aux[3];sigma=aux[2]
sens=-alpha*np.exp(-.5*alpha*alpha)/np.sqrt(2*np.pi)/(6*sigma*sigma)
den_col=np.sum((A*co[None,:])**2);den_cp=np.linalg.norm(cp)
erng=np.random.default_rng(20261017);ea=erng.choice((-1.,1.),size=(m,64));eb=erng.choice((-1.,1.),size=(m,64))
def probes(X):return X@(co[:,None]*(X.T@ea)*(X.T@eb))
truthprobe=probes(F);probe_den=float(np.sum(truthprobe*truthprobe))
results=[]
for name,Q in [('static384',Qbase),('actualnext320plus64',Qnew)]:
 factors=Q.T@F;B=(W.T@Q)@factors;residual=A-B
 dcp=(B*B*co[None,:])@B.T-cp;dd=np.diag(dcp)
 # Full signed cubic tensor projected in all three legs, evaluated on an
 # independent fixed batch; probes used for basis creation never scored here.
 projected=Q@factors;ep=probes(projected)-truthprobe
 row=dict(name=name,rank=384,weighted_pushed_factor_relative_squared_error=float(np.sum((residual*co[None,:])**2)/den_col),
 signed_K21_relative_frobenius_error=float(np.linalg.norm(dcp)/den_cp),signed_K3_relative_error=float(np.linalg.norm(dd)/np.linalg.norm(diag)),
 signed_K3_mean_sensitivity_error_norm=float(np.linalg.norm(dd*sens)),
 heldout_cubic_contraction_relative_squared_error=float(np.sum(ep*ep)/probe_den),
 heldout_probe_error2_per_column=np.sum(ep*ep,axis=0).tolist(),
 max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
 assert row['max_rss_bytes']<2_000_000_000
 results.append(row);print(json.dumps({k:v for k,v in row.items() if not isinstance(v,list)}),flush=True)
 del factors,B,residual,dcp,dd,projected,ep
metrics=[k for k in results[0] if k not in ('name','rank','max_rss_bytes','heldout_probe_error2_per_column')]
ratios={k:results[1][k]/results[0][k] for k in metrics}
doc=dict(status='FIXED_L13_ACTUALNEXT64_SAVED_AUDIT',prereg=prereg,dimensions=dict(m=m,young_columns=k,old_columns=len(lo),old_rank=Qp.shape[1]),rows=results,ratio_to_static384=ratios,old_saved_row_fit_relative_error=old_fit_error,seconds=time.monotonic()-t0,max_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,new_truths=0,limits=['Historical S39 network651 factors; auxiliary sensitivity from same-network S38, not current S2tr native accuracy.','The64 basis directions use actual W14 uniformly spaced columns, no random probes or response selection. Old tier remains represented in320 original Gram columns.','Independent64 Rademacher contractions give a fixed stochastic full-cubic Frobenius proxy, not an exact norm or downstream terminal loss.','Raw pushed K21/K3 do not include NP correction or later feedback. Mean sensitivity corrected to saved aux14 to match W14; comparison values are regenerated on both bases.','Same rank384 and final-join-only avoid the mixed-core invalidity; no next join core required.','No architecture/rank/seed/cutoff sweep, fit, truth or model call.'])
(D/'SAVED_RESULT_v1.json').open('x').write(json.dumps(doc,indent=2)+'\n')
print('RATIOS',json.dumps(ratios),flush=True)
