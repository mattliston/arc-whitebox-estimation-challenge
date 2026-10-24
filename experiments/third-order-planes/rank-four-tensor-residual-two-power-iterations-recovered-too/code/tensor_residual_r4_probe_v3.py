"""Saved-only fixed first-four-QR-direction tensor residual contraction.
No source/model/SDK change. r8 is a cost diagnostic, not a tunable native arm.
"""
import gc,json,re,resource,subprocess,time
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;R=D.parent.parent

def free():
 t=subprocess.check_output(['/usr/bin/memory_pressure'],text=True)
 p=int(re.search(r'System-wide memory free percentage: (\d+)%',t).group(1));assert p>=50,p
 return p

start=time.monotonic();rows=[]
for rank in(35,200):
 before=free();z=np.load(R/f'opus_fix336687_v1/joinsketch/join{rank}_L13.npz')
 J=z['J'].astype('f');lam=z['lam'].astype('f');wt=lam*lam
 Qp=z['Qp'].astype('f');Fprev=z['F'].astype('f');core=z['core'].astype('f')
 W=z['w32'][z['previous']].T.astype('f');old=z['lam_old'].astype('f')
 m,k=J.shape;diag=(J*J)@wt+np.sum((Qp@core)*Qp,axis=1)
 ri=np.sort(np.argsort(-diag.astype('d'),kind='stable')[:384])
 ci=np.argsort(-(np.sum(J*J,axis=0)*wt).astype('d'),kind='stable')[:1536]
 Y=J[:,ci]@(wt[ci,None]*J[ri][:,ci].T)+(Qp@core)@Qp[ri].T
 Q=np.linalg.qr(Y)[0].astype('f');F=Q.T@J
 sample=np.arange(0,1024,16);Ws=W[sample];WQ=Ws@Q
 X=np.concatenate((Qp@Fprev,J),axis=1);weights=np.concatenate((old,lam))
 exact=Ws@X;projected=WQ@(Q.T@X)
 truth=(exact*exact*weights)@exact.T;baseline=(projected*projected*weights)@projected.T
 truth_diag=np.diag(truth).copy().astype('d');base_diag=np.diag(baseline).copy().astype('d')
 diagerr=float(np.sum((base_diag-truth_diag)**2))
 np.fill_diagonal(truth,0);np.fill_diagonal(baseline,0)
 error=float(np.sum((baseline.astype('d')-truth)**2))
 q=WQ@F;e=Ws@J-q
 full_first=(q*q*lam)@e.T+2*(q*e*lam)@q.T
 full_first_diag=np.diag(full_first).copy().astype('d')
 np.fill_diagonal(full_first,0)
 results={}
 for r in(4,8):
  a=F[:r];u=WQ[:,:r];pairs=[(i,j)for i in range(r)for j in range(i,r)]
  B=np.stack([lam*a[i]*a[j]for i,j in pairs],axis=1)
  # Saved probe uses sampled residual directly; native factorization is
  # V=J@B-Q@(F@B), then W@V. No whole-bank residual materialization needed.
  ideal_v=e@B
  nativeV=J@B-Q@(F@B)
  v=Ws@nativeV
  left=[];right=[]
  for d,(i,j)in enumerate(pairs):
   if i==j:
    left.extend((u[:,i]*u[:,i],2*u[:,i]*v[:,d]));right.extend((v[:,d],u[:,i]))
   else:
    left.extend((2*u[:,i]*u[:,j],2*u[:,i]*v[:,d],2*u[:,j]*v[:,d]));right.extend((v[:,d],u[:,j],u[:,i]))
  correction=np.stack(left,axis=1)@np.stack(right,axis=0)
  correction_diag=np.diag(correction).copy().astype('d')
  corrected_diagerr=float(np.sum((base_diag+correction_diag-truth_diag)**2))
  np.fill_diagonal(correction,0)
  after=float(np.sum((baseline.astype('d')+correction-truth)**2))
  qsmall=u@a
  direct=(qsmall*qsmall*lam)@e.T+2*(qsmall*e*lam)@qsmall.T
  np.fill_diagonal(direct,0)
  d=len(pairs);n=1024;cpcols=len(left)
  # Conservative ordinary dense tariff floor/upper algebra estimate; no
  # old-tensor residual, setup, SDK routing or request effects priced here.
  ops=dict(J_times_B=2*m*k*d,F_times_B=2*384*k*d,Q_times_FB=2*m*384*d,
      W_times_V=2*n*m*d,CP_plane_full=2*n*n*cpcols,
      coefficient_products=2*k*d,vector_subtract=m*d)
  results[str(r)]=dict(pair_vectors=d,CP_rank_columns=cpcols,CP_error_squared=after,
      CP_error_ratio=after/error,CP_error_recovered_pct=100*(1-after/error),
      K3_diagonal_error_squared=corrected_diagerr,K3_diagonal_error_ratio=corrected_diagerr/diagerr,
      K3_error_recovered_pct=100*(1-corrected_diagerr/diagerr),
      K3_correction_RMS=float(np.sqrt(np.mean(correction_diag**2))),
      fixed_four_primary=r==4,approx_common_energy_fraction=float(np.sum(qsmall.astype('d')**2)/np.sum(q.astype('d')**2)),
      direct_formula_maxabs_difference=float(np.max(np.abs(correction-direct))),
      residual_contraction_relative_error=float(np.linalg.norm(v-ideal_v)/max(np.linalg.norm(ideal_v),1e-30)),
      native_residual_contraction_used=True,
      ordinary_dense_flops=ops,ordinary_dense_total=sum(ops.values()),
      bill_pct_at240GF=100*sum(ops.values())/240e9,
      optimistic_no_accuracy_penalty_score=1.4468565569762618e-9*(1+sum(ops.values())/240e9))
 fullerr=float(np.sum((baseline.astype('d')+full_first-truth)**2))
 row=dict(rank=rank,m=m,new_columns=k,baseline_CP_error_squared=error,
      baseline_K3_diagonal_error_squared=diagerr,
      full_rank_new_residual_firstorder_K3_error_ratio=float(np.sum((base_diag+full_first_diag-truth_diag)**2))/diagerr,
      full_rank_new_residual_firstorder_error_ratio=fullerr/error,
      primary='4',arms=results,free_start=before,free_end=free())
 rows.append(row);print(json.dumps(row),flush=True)
 del z,J,lam,wt,Qp,Fprev,core,W,old,Y,Q,F,Ws,WQ,X,weights,exact,projected,truth,baseline,q,e,full_first
 gc.collect()
receipt=dict(status='SAVED_ONLY_NO_ENDPOINT',primary='first4_QR_columns',rows=rows,
      full_tensor_symmetric_first_order=True,new_J_only_not_inherited_old_residual=True,
      diagonal_measured_separately=True,NP_and_future_carrier_unmodeled=True,no_SDK=True,no_truths_read=True,
      gate='Primary r4 signed CP-error recovery>=20% on each saved rank; no native authorization implied',
      eligible=all(row['arms']['4']['CP_error_ratio']<=.8 for row in rows),
      elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with(D/'TENSOR_RESIDUAL_R4_RESULT_v3.json').open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps({k:v for k,v in receipt.items()if k!='rows'},indent=2))
