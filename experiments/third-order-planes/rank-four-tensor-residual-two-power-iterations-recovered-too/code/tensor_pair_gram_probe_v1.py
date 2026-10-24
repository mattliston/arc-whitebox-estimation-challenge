"""Fixed exact per-pair third-tensor mode Gram, saved-only falsifier.
Same rank/column count; parent selected complete-pair set; no endpoint labels.
Require >=20% signed CP-error reduction on both saved joins, including noise.
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
 # Exact per-pair cubic T=2*lambda*(sym(M,M,H)+H^3), mode1 Gram.
 # This block-diagonal pair approximation omits cross-PAIR tensor Gram terms.
 Mp=M[:,pairs].astype('d');Hp=H[:,pairs].astype('d')
 aa=np.sum(Mp*Mp,axis=0);bb=np.sum(Hp*Hp,axis=0);cc=np.sum(Mp*Hp,axis=0)
 ww=wt[ip[pairs]].astype('d')
 A=4*ww*(2*aa*bb+2*cc*cc)
 B=4*ww*(2*cc*(aa+bb))
 C=4*ww*(aa*aa+bb*bb+2*cc*cc)
 root=np.sqrt(np.maximum(A,1e-300));cross=B/root
 schur=C-cross*cross
 U=(Mp*root+Hp*cross).astype('f')
 V=(Hp*np.sqrt(np.maximum(schur,0))).astype('f')
 parent_trace=float(np.sum(J[:,ci].astype('d')**2*wt[ci]))
 tensor_trace=float(np.sum(U.astype('d')**2)+np.sum(V.astype('d')**2))
 scale=parent_trace/tensor_trace
 UV=np.concatenate((U,V),axis=1)
 Ytensor=(UV@UV[ri].T)*np.float32(scale)+oldY
 details=dict(selected_complete_pairs=len(pairs),parent_columns=len(ci),tensor_columns=UV.shape[1],
     trace_normalization=scale,parent_new_gram_trace=parent_trace,unscaled_tensor_trace=tensor_trace,
     minimum_schur=float(np.min(schur)),negative_schur_count=int(np.count_nonzero(schur<0)),
     full_H_cubed_included=True,cross_pair_terms_in_metric_omitted=True)
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
 Tnoise=Ytensor+np.float32(4e-6*np.sqrt(np.mean(Ytensor.astype('d')**2)))*y_noise
 arms={}
 for name,Y in [('baseline',Y0),('baseline_noise4e6',Ynoise),('tensor_pair_gram',Ytensor),('tensor_pair_gram_noise4e6',Tnoise)]:
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
 del z,J,lam,wt,Qp,F,core,wp,old,M,H,diag,oldY,Y0,Ytensor,Tnoise,gh,X,L,exact,truth,weightH,Ynoise,Qbase,Mp,Hp,U,V,UV
 gc.collect()
candidates=('tensor_pair_gram','tensor_pair_gram_noise4e6')
scores={a:float(np.mean([r['arms'][a]['CP_error_ratio_vs_baseline']for r in results]))for a in candidates}
choice='tensor_pair_gram'
eligible=all(r['arms'][a]['CP_error_ratio_vs_baseline']<=0.8 for r in results for a in candidates)
receipt=dict(status='ELIGIBLE_SAVED_PROXY_ONLY'if eligible else 'CLOSED_TENSOR_MODE_PROXY_GATE',
     candidate_chosen_without_labels=choice,mean_CP_error_ratio=scores,
     rule='Fixed exact per-pair cubic modeGram, traces matched; CP squared-error ratio<=.8 on EACH saved rank, with and without4e-6 noise; H diagnostic only',
     rows=results,no_truths_read=True,no_SDK=True,no_endpoint_oracle=True,
     caveat='Historical L13 joins, exact CPU float32 algebra. NP/coadaptation/current native transfer not tested.',
     elapsed_s=time.monotonic()-start,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
with(D/'TENSOR_PAIR_GRAM_RESULT_v1.json').open('x')as f:json.dump(receipt,f,indent=2)
print(json.dumps({k:v for k,v in receipt.items()if k!='rows'},indent=2))
