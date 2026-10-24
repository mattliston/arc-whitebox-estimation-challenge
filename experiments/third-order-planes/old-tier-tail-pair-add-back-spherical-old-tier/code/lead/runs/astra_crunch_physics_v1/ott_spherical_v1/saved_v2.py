"""One fixed saved L13 test of spherical instead of Gaussian OTT trace; no model/truth/fit."""
from pathlib import Path
import ast,hashlib,json,os,resource,time
import numpy as np
D=Path(__file__).resolve().parent;R=D.parents[1];O=D.parents[3]
assert os.getpriority(os.PRIO_PROCESS,0)>=10
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS','GOTO_NUM_THREADS'):assert os.environ[key]=='1'
source=R/'astra_crunch_runtime_v1/s2tr_peel_v1/candidate_s2tr_peel_release_v17.py'
P=O/'engineer/a35_join/state/s39_rank0651.npz';S=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3';auxpath=O/'engineer/a36_carrier/state/s38_rank0651.npz'
entry=json.loads((S/'manifest.json').read_text())['roles']['observer_fit']['row_manifests'][651];wp=S/Path(entry['path']).parent/'weights.npy'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prereg=dict(layer=13,network=651,join_rank=384,common_rank=256,coefficient='3/(r+2), uniquely spherical harmonic trace projection',rule='Replace global3mean(U^2) Ut@t by3||Ut_i||^2/(r+2) Ut@t separately for old-map and new-join blocks.',data='historical factors/weights/aux only; no truth or terminal calibration',source_sha256=sha(source),script_sha256=sha(Path(__file__)),archives={str(p):sha(p)for p in(P,wp,auxpath)})
prereg['fixed_additional_comparator']='Diagonal C_U=mean(U²,axis=0);3 U F[lambda ((F²)^T diag(C_U))], Gaussian first chaos, no fitted coefficient.'
(D/'PREREG_v2.json').open('x').write(json.dumps(prereg,indent=2)+'\n')
t0=time.monotonic();z=np.load(P);J=z['J_13'].astype(float);lam=z['lam_13'].astype(float);Qp=z['Qp_13'].astype(float);core=z['core_13'].astype(float);prev=z['prev_13'];lo=z['lamold_12'].astype(float)
saved=z['old_12'].astype(float);full=np.zeros((1024,saved.shape[1]));full[z['rows_12']]=saved;observed=(full*z['gate_12'][:,None])[prev];valid=np.abs(observed).sum(axis=1)>0
oldF=np.linalg.lstsq(Qp[valid],observed[valid],rcond=None)[0];old_fit_error=float(np.linalg.norm(Qp[valid]@oldF-observed[valid])/np.linalg.norm(observed[valid]));del saved,full,observed
G=(J*(lam*lam)[None,:])@J.T+Qp@core@Qp.T;G=(G+G.T)*.5;ix=np.sort(np.argsort(-np.diag(G),kind='stable')[:384]);Q=np.linalg.qr(G[:,ix],mode='reduced')[0];del G
Fj=Q.T@J;overlap=Q.T@Qp;W=np.load(wp,mmap_mode='r')[13,prev,:].astype(float);U=W.T@Q;Um=U@overlap;del J,W,Q,Qp,core
base=np.zeros(1024);new=np.zeros(1024);diag_cov=np.zeros(1024);truth_raw=np.zeros(1024);truth_effective=np.zeros(1024);rows=[]
for name,X,F,coeff in [('old_map',Um,oldF,lo),('new_join',U,Fj,lam)]:
 r=X.shape[1];n2=np.sum(F*F,axis=0);t=F@(coeff*n2);v=X@t;rho=np.sum(X*X)/X.size
 base+=3*rho*v;new+=3*np.sum(X*X,axis=1)/(r+2)*v
 cdiag=np.mean(X*X,axis=0);nd=(F*F).T@cdiag;diag_cov+=3*(X@(F@(coeff*nd)))
 A=X@F;truth_raw+=np.sum((A*A*A)*coeff,axis=1);del A
 cuts=np.r_[0,np.flatnonzero(np.sign(coeff[1:])!=np.sign(coeff[:-1]))+1,len(coeff)]
 eff=np.empty_like(F)
 for j in range(0,len(cuts)-1,2):
  a,b,c=map(int,cuts[j:j+3]);assert b-a==c-b and np.array_equal(coeff[a:b],-coeff[b:c])
  common=.5*(F[:,a:b]+F[:,b:c]);diff=.5*(F[:,a:b]-F[:,b:c]);common[256:]=0
  eff[:,a:b]=common+diff;eff[:,b:c]=common-diff
 A=X@eff;truth_effective+=np.sum((A*A*A)*coeff,axis=1);del A,eff,common,diff
 rows.append(dict(block=name,rank=r,columns=F.shape[1],global_rho=float(rho),radius_rms_to_mean=float(np.std(np.sum(X*X,axis=1))/np.mean(np.sum(X*X,axis=1))),incremental_flop_upper=2*X.size+6*X.shape[0],diag_cov_extra_flop_upper=2*X.size+3*F.size+5*X.shape[0],column_variance_relative_sd=float(np.std(cdiag)/np.mean(cdiag))))
aux=np.load(auxpath)['aux_13'];alpha=aux[3];sigma=aux[2];pdf=aux[4];gate=aux[0];sens=-alpha*pdf/(6*sigma*sigma)
# Match current64-rounded tail selection; archived aux provides the row scores, all1024 rows diagnostic.
score=2.5-alpha;negative=(alpha < -2.5)&(gate<.01);score=np.where(negative,alpha+2.5,score);kept=min(1024,((int(np.count_nonzero(score>=0))+63)//64)*64);order=np.argsort(-score,kind='stable');tail=order[kept:]
band=np.flatnonzero((np.abs(alpha)>2.5)&(np.abs(alpha)<=3.0))
def metric(target,idx,weighted=False,prediction=None):
 a=base[idx]-target[idx];b=(new if prediction is None else prediction)[idx]-target[idx]
 if weighted:a=a*sens[idx];b=b*sens[idx]
 aa=float(a@a);bb=float(b@b);delta=b-a
 return dict(n=len(idx),baseline_error2=aa,new_error2=bb,error2_recovery_pct=100*(1-bb/aa)if aa else None,correction_cosine=float(-a@delta/(np.linalg.norm(a)*np.linalg.norm(delta)))if np.linalg.norm(a)*np.linalg.norm(delta)else None)
metrics=[]
for name,target in [('raw_fullrank_trace_target',truth_raw),('current_style_common256_target',truth_effective)]:
 for label,idx in [('all_rows',np.arange(1024)),('rounded_ott_tail',tail),('fixed_2p5_3_band',band)]:
  metrics.append(dict(target=name,rows=label,raw=metric(target,idx),mean_weighted=metric(target,idx,True),diag_cov_raw=metric(target,idx,prediction=diag_cov),diag_cov_mean_weighted=metric(target,idx,True,diag_cov)))
# Tiny identity: spherical trace tensor Sym(t,I)/(r+2) has exact trace t.
r=7;rng=np.random.default_rng(20261018);f=rng.normal(size=(r,19));c=rng.normal(size=19);t=f@(c*np.sum(f*f,axis=0));I=np.eye(r);T=(np.einsum('i,jk->ijk',t,I)+np.einsum('j,ik->ijk',t,I)+np.einsum('k,ij->ijk',t,I))/(r+2);tiny=float(np.max(np.abs(np.einsum('ijj->i',T)-t)))
assert tiny<1e-12
prereg['fixed_additional_comparator']='Diagonal C_U=mean(U²,axis=0);3 U F[lambda ((F²)^T diag(C_U))], Gaussian first chaos, no fitted coefficient.'
out=dict(status='SAVED_PROXY_ONLY_DIAGONAL_COV_COMPARISON',prereg=prereg,old_saved_row_fit_relative_error=old_fit_error,tail_count=len(tail),rows=rows,metrics=metrics,tiny_trace_identity_error=tiny,incremental_L13_flop_upper=sum(x['incremental_flop_upper']for x in rows),incremental_L13_wholebill_pct=100*sum(x['incremental_flop_upper']for x in rows)/244596740000,seconds=time.monotonic()-t0,maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,models=0,limits=['HistoricalS39factors andS38aux, notcurrentSTOREterminalerror.','CurrentOTTactive atL13 confirmed fromv17source; lazyold/newblocks evaluated separately.','No best-scale, threshold, layer, rank or seed selection.','Roundedtailuses1024archiveaux; currentdeadcutcompactrows maydiffer.'])
assert out['maxrss']<2_000_000_000
(D/'SAVED_RESULT_v2.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
