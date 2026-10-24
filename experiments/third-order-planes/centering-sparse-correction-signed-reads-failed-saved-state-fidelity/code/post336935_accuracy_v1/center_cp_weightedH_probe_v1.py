"""One fixed CP-content weighted H selection versus prior H-energy policy.
No truths/endpoints. Prior screening cutoff is not an adjusted-score bound.
Full signed factors are projected consistently on both sides of sampled reads.
"""
import gc,json,re,resource,subprocess,time
from pathlib import Path
import numpy as np
D=Path(__file__).resolve().parent;A=D.parent;R=A.parent
def free():
 t=subprocess.check_output(['/usr/bin/memory_pressure'],text=True)
 p=int(re.search(r'System-wide memory free percentage: (\d+)%',t).group(1))
 assert p>=50,p
 return p
start=time.monotonic();results=[]
for rank in (35,200):
 pressure=free()
 z=np.load(R/f'opus_fix336687_v1/joinsketch/join{rank}_L13.npz')
 J=z['J'].astype('f');lam=z['lam'].astype('f');wt=lam*lam
 Qp=z['Qp'].astype('f');F=z['F'].astype('f');core=z['core'].astype('f')
 wp=z['w32'][z['previous']].T.astype('f');old=z['lam_old'].astype('f')
 splits=np.flatnonzero(np.sign(lam[1:])!=np.sign(lam[:-1]))+1
 widths=np.diff(np.r_[0,splits,len(lam)]);assert len(widths)==6
 assert np.all(widths[::2]==widths[1::2]);starts=np.r_[0,np.cumsum(widths)]
 ip=np.concatenate([np.arange(starts[c],starts[c+1])for c in(0,2,4)])
 im=np.concatenate([np.arange(starts[c],starts[c+1])for c in(1,3,5)])
 assert np.array_equal(wt[ip],wt[im])
 M=(J[:,ip]+J[:,im])*.5;H=(J[:,ip]-J[:,im])*.5
 diag=(J*J)@wt+np.sum((Qp@core)*Qp,axis=1)
 ri=np.sort(np.argsort(-diag.astype('d'),kind='stable')[:384])
 ci=np.argsort(-(np.sum(J*J,axis=0)*wt).astype('d'),kind='stable')[:1536]
 score=np.sum(J[:,ip]**2+J[:,im]**2,axis=0)*wt[ip]
 pairs=np.argsort(-score.astype('d'),kind='stable')[:768]
 oldY=(Qp@core)@Qp[ri].T
 Y0=J[:,ci]@(wt[ci,None]*J[ri][:,ci].T)+oldY
 Yc=M[:,pairs]@((2*wt[ip[pairs]])[:,None]*M[ri][:,pairs].T)+oldY
 # Preserve H correlation; select H columns by their own energy across allpairs.
 hscore=np.sum(H*H,axis=0)*wt[ip]
 hi=np.argsort(-hscore.astype('d'),kind='stable')
 target_trace=float(np.sum(hscore[pairs]))
 repair={};details={}
 cp_score=hscore.astype('d')*np.sum(M.astype('d')**2,axis=0)**2
 cp_hi=np.argsort(-cp_score,kind='stable')
 for name,order in [('center_full_Hhalf_energy',hi),('center_full_Hhalf_CPcontent',cp_hi)]:
  hs=order[:384]
  trace_scale=target_trace/float(np.sum(hscore[hs]))
  repair[name]=Yc+H[:,hs]@((2*trace_scale*wt[ip[hs]])[:,None]*H[ri][:,hs].T)
  details[name]=dict(center_columns=768,H_columns=384,H_trace_scale=trace_scale,
      H_columns_outside_parent_pairs=int(np.count_nonzero(~np.isin(hs,pairs))),
      overlap_with_H_energy_top384=int(np.count_nonzero(np.isin(hs,hi[:384]))))
 gh=2*(H[ri][:,pairs]*wt[ip[pairs]])@H[ri][:,pairs].T
 ghF=float(np.sum(gh.astype('d')**2))
 X=np.concatenate((Qp@F,J),axis=1);L=np.concatenate((old,lam))
 # Fixed label-free 64x64 output block; all signed columns and old factors.
 out=np.arange(0,1024,16);Wsample=wp[out]
 exact=Wsample@X;truth=(exact*exact*L)@exact.T
 np.fill_diagonal(truth,0)
 weightH=H*np.sqrt(wt[ip])[None,:]
 hnorm=float(np.sum(weightH.astype('d')**2))
 y_noise=np.random.default_rng(20261018+rank).standard_normal(Y0.shape).astype('f')
 Ynoise=Y0+np.float32(4e-6*np.sqrt(np.mean(Y0.astype('d')**2)))*y_noise
 arms={}
 for name,Y in [('baseline',Y0),('baseline_noise4e6',Ynoise),('center_only',Yc)]+list(repair.items()):
  Q=np.linalg.qr(Y)[0].astype('f')
  projected=(Wsample@Q)@(Q.T@X)
  plane=(projected*projected*L)@projected.T
  np.fill_diagonal(plane,0)
  plane_error=float(np.sum((plane.astype('d')-truth)**2))
  hproj=Q.T@weightH
  herr=max(hnorm-float(np.sum(hproj.astype('d')**2)),0)
  if name=='baseline':Qbase=Q.copy()
  overlap=Qbase.T@Q
  gap=max(1-float(np.sum(overlap.astype('d')**2))/Q.shape[1],0)
  arms[name]=dict(signed_CP_error_squared=plane_error,weighted_H_projection_error_squared=herr,
      subspace_chord_squared_per_rank=gap)
  del Q,projected,plane,hproj,overlap
 for a in arms.values():
  a['CP_error_ratio_vs_baseline']=a['signed_CP_error_squared']/arms['baseline']['signed_CP_error_squared']
  a['H_error_ratio_vs_baseline']=a['weighted_H_projection_error_squared']/arms['baseline']['weighted_H_projection_error_squared']
 rows=dict(rank=rank,m=int(J.shape[0]),new_columns=int(J.shape[1]),selected_pairs=len(pairs),
     all_pairs=len(ip),H_gram_offdiagonal_energy_fraction=1-float(np.sum(np.diag(gh).astype('d')**2))/ghF,
     arms=arms,policy_details=details,memory_free_start=pressure,memory_free_end=free())
 results.append(rows);print(json.dumps(rows),flush=True)
 del z,J,lam,wt,Qp,F,core,wp,old,M,H,diag,oldY,Y0,Yc,repair,gh,X,L,exact,truth,weightH,Ynoise,Qbase
 gc.collect()
candidates=('center_full_Hhalf_energy','center_full_Hhalf_CPcontent')
scores={a:float(np.mean([r['arms'][a]['CP_error_ratio_vs_baseline']for r in results]))for a in candidates}
choice=min(scores,key=scores.get)
eligible=all(r['arms'][choice]['CP_error_ratio_vs_baseline']<=1.05 and r['arms'][choice]['H_error_ratio_vs_baseline']<=1.05 for r in results)
receipt=dict(status='SAVED_CP_CONTENT_WEIGHTING_DIAGNOSTIC',
     candidate_chosen_without_labels=choice,mean_CP_error_ratio=scores,
     rule='One fixed alternative H score = lambda^2 * ||M||^4 * ||H||^2; no endpoint selection or eligibility inference',
     rows=results,no_truths_read=True,no_SDK=True,no_endpoint_oracle=True,
     caveat='Historical L13 joins, exact CPU float32 algebra. NP/coadaptation/current native transfer not tested.',
     elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with(D/'CENTER_CP_WEIGHTED_H_RESULT_v1.json').open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps({k:v for k,v in receipt.items()if k!='rows'},indent=2))
