"""Fixed saved-join ceiling: exact only selected left rows at the L13 read.

Historical join captures, rebuilt JSK-lite basis; not a current-native candidate.
64/128 rows fixed in advance, ranked by Gaussian covariance sensitivity.
All truth use is explicit oracle geometry; no trained coefficient or model run.
"""
import json, math, resource, time
from pathlib import Path
import numpy as np
from scipy.special import ndtr

O=Path(__file__).resolve().parents[3]; D=O/'lead/runs/opus_fix336687_v1'
S=O.parent/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rows=[]; t0=time.monotonic()
for rank in (35,200):
    d=np.load(D/f'joinsketch/join{rank}_L13.npz')
    J=d['J'].astype('f'); lam=d['lam'].astype('f'); Qp=d['Qp'].astype('f'); F=d['F'].astype('f'); core=d['core'].astype('f')
    wp=d['w32'][d['previous']].T.astype('f'); wt=lam*lam
    old=Qp@F; X=np.concatenate((old,J),axis=1); del old
    L=np.concatenate((d['lam_old'],lam)).astype('f')
    diag=(J*J)@wt+np.sum((Qp@core)*Qp,axis=1)
    ri=np.sort(np.argsort(-diag.astype('d'),kind='stable')[:384])
    ci=np.argsort(-(np.sum(J*J,axis=0)*wt).astype('d'),kind='stable')[:1536]
    Y=J[:,ci]@(wt[ci,None]*J[ri][:,ci].T)+(Qp@core)@Qp[ri].T
    Q=np.linalg.qr(Y)[0].astype('f')
    R=Q.T@X; p=wp@X; q=(wp@Q)@R
    true_plane=(p*p*L)@p.T; old_plane=(q*q*L)@q.T
    a=np.load(O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_A32M.npz')
    b=np.load(O/f'analyst/fresh/pu/mps_truth_rank{rank:04d}_B32M.npz')
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
        dm=.5*pdf[14]/np.sqrt(v[14])*dv
        return gate[15]*(W15.T@dm)
    e=forward(residual); n0=float(e@e)
    cand=[]
    for r in (64,128):
        keep=np.argsort(-h,kind='stable')[:r]
        delta=((p[keep]*p[keep]-q[keep]*q[keep])*L)@q.T
        repair=np.zeros_like(old_plane);repair[keep]=delta
        # Covariance-only: marginal third moment and future compressed bank stay as parent.
        re=cov(repair); ne=e+forward(re)
        m,K=J.shape; ko=F.shape[1]; rb=384;n=1024
        raw_push=2*r*(m*K+m*rb+rb*ko)
        raw_read=2*r*(rb*(ko+K)+rb*n)
        raw=raw_push+raw_read
        cand.append(dict(rows=r,covariance_error_energy_recovered_pct=100*(1-float(np.sum((residual+re)**2))/float(np.sum(residual**2))),propagated_mean_error_energy_recovered_pct=100*(1-float(ne@ne)/n0),optimistic_adjusted_accuracy_pct=-1.2*(1-float(ne@ne)/n0),extra_dense_flops=raw,optimistic_bill_pct=100*.40*raw/(1.452e-9/1.315e-8*2.2e12),less_optimistic_bill_pct=100*.60*raw/(1.452e-9/1.315e-8*2.2e12)))
    rows.append(dict(rank=rank,candidate=cand));print(rows[-1],flush=True)
res=dict(status='Saved proxy only; full old factor right, no NP corrections; oracle kink geometry; no release claim',rows=rows,wall_s=time.monotonic()-t0,maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
out=Path(__file__).resolve().parent/'L13_ROW_REPAIR_PROXY_v1.json';assert not out.exists();out.write_text(json.dumps(res,indent=2));print('done',res['wall_s'],res['maxrss_bytes'],flush=True)
