"""Predeclared fixed signed-h residual64, both CP sides; saved proxy only."""
import json, math, resource, time, gc, re as regex, subprocess
def guard():
    s=subprocess.check_output(["memory_pressure"],text=True)
    free=int(regex.search(r"System-wide memory free percentage: (\d+)%",s).group(1))
    if free<50: raise RuntimeError(f"memory guard free {free}<50")
    return free
from pathlib import Path
import numpy as np
from scipy.special import ndtr
O=Path(__file__).resolve().parents[3]; D=O/'lead/runs/opus_fix336687_v1'
S=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rows=[]; t0=time.monotonic()
for rank in (35,200):
    free_start=guard()
    d=np.load(D/f'joinsketch/join{rank}_L13.npz')
    J=d['J'].astype('f'); lam=d['lam'].astype('f'); Qp=d['Qp'].astype('f'); F=d['F'].astype('f'); core=d['core'].astype('f')
    wp=d['w32'][d['previous']].T.astype('f'); wt=lam*lam
    X=np.concatenate((Qp@F,J),axis=1); L=np.concatenate((d['lam_old'],lam)).astype('f')
    diag=(J*J)@wt+np.sum((Qp@core)*Qp,axis=1)
    ri=np.sort(np.argsort(-diag.astype('d'),kind='stable')[:384])
    ci=np.argsort(-(np.sum(J*J,axis=0)*wt).astype('d'),kind='stable')[:1536]
    Y=J[:,ci]@(wt[ci,None]*J[ri][:,ci].T)+(Qp@core)@Qp[ri].T
    Q=np.linalg.qr(Y)[0].astype('f'); R=Q.T@X
    p=wp@X; q=(wp@Q)@R
    true_plane=(p*p*L)@p.T; old_plane=(q*q*L)@q.T
    a=np.load(O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_A32M.npz'); b=np.load(O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_B32M.npz')
    mu=.5*(a['mc_pre_m']+b['mc_pre_m']); v=.5*(a['mc_pre_v']+b['mc_pre_v'])
    z=mu/np.sqrt(v); gate=ndtr(z); pdf=np.exp(-z*z/2)/math.sqrt(2*math.pi)
    h=pdf[13]/np.sqrt(v[13]); g=gate[13]
    def cov(plane):
        t=.5*h[:,None]*g[None,:]*plane
        c=t+t.T;np.fill_diagonal(c,0);return c
    residual=cov(old_plane-true_plane)
    weights=np.load(S/f'rows/observer_fit/{rank:04d}/weights.npy',mmap_mode='r')
    W14=weights[14].astype('d'); W15=weights[15].astype('d')
    def forward(c):
        dv=np.einsum('ij,ij->j',W14,c@W14)
        return gate[15]*(W15.T@(.5*pdf[14]/np.sqrt(v[14])*dv))
    e=forward(residual); n0=float(e@e)
    # Apply the card's fixed .01 gate and 32-row rounding to truth geometry.
    score=np.minimum(g,1-g)
    count=min(1024,((int(np.count_nonzero(score>=.01))+31)//32)*32)
    keep=np.argsort(-score,kind='stable')[:count]; ko=F.shape[1]; m,K=J.shape
    pj=p[keep,ko:]; qj=q[keep,ko:]; right=q[:,ko:]
    def measure(name, fixed, raw):
        delta=((fixed*fixed-qj*qj)*lam)@right.T
        repair=np.zeros_like(old_plane);repair[keep]=delta
        re=cov(repair); ne=e+forward(re)
        share=1-float(ne@ne)/n0
        bill=100*.40*raw/(1.452e-9/1.315e-8*2.2e12)
        return dict(name=name,propagated_mean_error_energy_recovered_pct=100*share,optimistic_adjusted_accuracy_pct=-1.2*share,extra_dense_flops=raw,optimistic_bill_pct=bill,optimistic_net_pct=bill-1.2*share)
    Fj=R[:,ko:]
    cuts=np.flatnonzero(np.sign(lam[1:])!=np.sign(lam[:-1]))+1
    sizes=np.diff(np.r_[0,cuts,len(lam)]); assert len(sizes)==6 and np.all(sizes[::2]==sizes[1::2])
    starts=np.r_[0,np.cumsum(sizes)]
    ip=np.concatenate([np.arange(starts[c],starts[c+1]) for c in (0,2,4)])
    im=np.concatenate([np.arange(starts[c],starts[c+1]) for c in (1,3,5)])
    H=(J[:,ip]-J[:,im])*.5; Fh=(Fj[:,ip]-Fj[:,im])*.5
    en=np.maximum(np.sum(H*H,axis=0)-np.sum(Fh*Fh,axis=0),0)
    cols=np.argsort(-(en*wt[ip]).astype('d'),kind='stable')[:64]
    Z=H[:,cols]-Q@Fh[:,cols];Z=Z-Q@(Q.T@Z)
    Z=np.linalg.qr(Z)[0].astype('f'); coeff=Z.T@H
    pushed=wp@Z; dh=pushed@coeff
    joined=q[:,ko:].copy();joined[:,ip]+=dh;joined[:,im]-=dh
    repair=(joined*joined*lam)@joined.T-old_plane+((q[:,:ko]*q[:,:ko]*L[:ko])@q[:,:ko].T)
    re=cov(repair);ne=e+forward(re);share=1-float(ne@ne)/n0
    # Conservative plain-matmul prices: no assumed Winograd discount.
    P=K//2;nr=1024;raw=3*2*m*384*64+2*64*m*P+2*nr*m*64+2*nr*64*P+2*count*P*64+2*count*64*1024+4*m*64*64
    bill=100*raw/(1.452e-9/1.315e-8*2.2e12)
    cand=[dict(name='signed_h64_both_sides',propagated_mean_error_energy_recovered_pct=100*share,optimistic_adjusted_accuracy_pct=-1.2*share,plain_dense_qr_flops=raw,plain_bill_pct=bill,optimistic_net_with_plain_bill_pct=bill-1.2*share,response_rows=count)]
    # Fixed spectral audit only; compare low-rank energy on selected paired sketch.
    M=(J[:,ip]+J[:,im])*.5
    score=np.sum(J[:,ip]**2+J[:,im]**2,axis=0)*wt[ip]
    ks=np.argsort(-score.astype('d'),kind='stable')[:768]
    gm=(M[ri][:,ks]*wt[ip[ks]])@M[ri][:,ks].T
    gh=(H[ri][:,ks]*wt[ip[ks]])@H[ri][:,ks].T
    em=np.linalg.eigvalsh(gm.astype('d'));eh=np.linalg.eigvalsh(gh.astype('d'))
    spectral=dict(common_total=float(np.sum(em)),difference_total=float(np.sum(eh)),common_top64_pct=float(100*np.sum(em[-64:])/np.sum(em)),common_top128_pct=float(100*np.sum(em[-128:])/np.sum(em)),difference_top64_pct=float(100*np.sum(eh[-64:])/np.sum(eh)),difference_top128_pct=float(100*np.sum(eh[-128:])/np.sum(eh)))
    cand[0]['spectral_selected384rows']=spectral
    cand[0]['memory_free_end_pct']=guard()
    rows.append(dict(rank=rank,candidate=cand));print(rows[-1],flush=True)
    del d,J,lam,Qp,F,core,wp,wt,X,L,diag,ri,ci,Y,Q,R,p,q,true_plane,old_plane,a,b,mu,v,z,gate,pdf,h,g,residual,weights,W14,W15,e,keep,pj,qj,right,H,Fh,Z,coeff,pushed,dh,joined,repair,re,M,gm,gh
    gc.collect()
res=dict(status='Saved proxy only; oracle geometry and historical bank, full old right; no NP or scalar K3 response modeled, no deadcut/tail mask. Complete CP plane only, not final estimator MSE',rows=rows,wall_s=time.monotonic()-t0,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
out=Path(__file__).resolve().parent/'L13_SIGNED_H64_PROXY_v1.json';assert not out.exists();out.write_text(json.dumps(res,indent=2));print('done',res['wall_s'],res['maxrss_bytes'],flush=True)
