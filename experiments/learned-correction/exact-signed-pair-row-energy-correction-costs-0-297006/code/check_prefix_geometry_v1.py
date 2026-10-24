from pathlib import Path
import json,time,resource,os
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent;C=R/'astra_native_terminal_linear_refit_v1/run_capture_s0_v1/train16_read_observer_v1'
assert os.getpriority(os.PRIO_PROCESS,0)==10
assert all(os.environ[k]=='1'for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'))
start=time.monotonic();records=[]
def score(pred,target):
 return dict(cosine=float(pred@target/max(np.linalg.norm(pred)*np.linalg.norm(target),1e-30)),relative_error=float(np.linalg.norm(pred-target)/max(np.linalg.norm(target),1e-30)),total_ratio=float(pred.sum()/max(target.sum(),1e-30)))
for l in range(2,15):
 with np.load(C/f'L{l:02d}.npz')as z:
  A=z['left_young'].astype(float);right=z['right_young'].astype(float);lam=z['lambda_young'].astype(float);rows=z['rows'];cols=z['cols'];Q=z['Q'].astype(float);prefix=z['prefix'].astype(float);cut=int(z['c15']);var=z['vc'].astype(float)
 v=var[rows];sig=np.sqrt(var[cols]);weighted=right*lam[None,:]/sig[:,None]
 # Both prefix alternatives share the exact current Q, not a fitted projector.
 youngprefix=A@weighted[:cut].T
 fullprefix=prefix/sig[None,:cut]
 residualfull=(fullprefix-Q@(Q.T@fullprefix))/v[:,None]
 residualyoung=(youngprefix-Q@(Q.T@youngprefix))/v[:,None]
 residualtail=((A-Q@(Q.T@A))@weighted[cut:].T)/v[:,None]
 _,ir,ic=np.intersect1d(rows,cols[:cut],return_indices=True);residualfull[ir,ic]=0;residualyoung[ir,ic]=0
 _,ir,ic=np.intersect1d(rows,cols[cut:],return_indices=True);residualtail[ir,ic]=0
 ratio=np.sum(1/var[cols[cut:]])/np.sum(1/var[cols[:cut]])
 efull=np.sum(residualfull**2,axis=1)*ratio;eyoung=np.sum(residualyoung**2,axis=1)*ratio;target=np.sum(residualtail**2,axis=1)
 records.append(dict(layer=l,m=len(rows),k=len(lam),prefix=cut,tail=len(cols)-cut,variance_weight_ratio=float(ratio),full_prefix=score(efull,target),young_prefix=score(eyoung,target),old_contamination=score(efull,eyoung),old_residual_prefix_energy_fraction=float(np.sum((residualfull-residualyoung)**2)/max(np.sum(residualfull**2),1e-30)),exact_tail_energy_sum=float(target.sum())))
 del A,right,weighted,prefix,youngprefix,fullprefix,residualfull,residualyoung,residualtail,Q
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss<2_000_000_000 and time.monotonic()-start<180
out=dict(scope='One fixed paid-prefix residual-energy estimate, inverse-variance sum ratio, same current Q, global diagonal excluded. Old-contamination assessed by separately computed offline young prefix; no deployment of that expensive product.',rows=records,models=0,captures=0,fit=False,seconds=time.monotonic()-start,rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'PREFIX_RESIDUAL_GEOMETRY_v1.json').open('x').write(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
