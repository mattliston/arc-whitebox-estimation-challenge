"""Read-only Schur moment consistency on existing target-free parent tapes."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigvalsh
from threadpoolctl import threadpool_limits

here=Path(__file__).resolve().parent
out=here/'current_moment_schur_audit_v1.json'
assert not out.exists()
start=time.time();rows=[]
with threadpool_limits(limits=4):
    for rank in (35,100):
        folder=here/'rows_current_transport_shadow_v2'/f'row_{rank:04d}'
        cp=np.load(folder/'pre_cov.npy',mmap_mode='r')
        tp=np.load(folder/'k21.npy',mmap_mode='r')
        kp=np.load(folder/'k4.npy',mmap_mode='r')
        alive=np.load(folder/'alive.npy')
        for l in (4,8,12,15):
            idx=np.flatnonzero(alive[l])
            C=cp[l][np.ix_(idx,idx)].astype(np.float64)
            C=(C+C.T)/2
            v=np.diag(C).copy();s=np.sqrt(v)
            R=C/np.outer(s,s)
            T=tp[l][np.ix_(idx,idx)].astype(np.float64)/(v[:,None]*s[None,:])
            alpha=float(np.mean(kp[l,idx].astype(np.float64)/v**2))
            Q=alpha*(1+2*R*R)/3+2*R*R
            ce=eigvalsh(R,check_finite=False)
            record=dict(rank=rank,layer=l,active=len(idx),cov_min=float(ce[0]),cov_max=float(ce[-1]),
                        covariance_condition=float(ce[-1]/ce[0]) if ce[0]>0 else None,radial_kurtosis=alpha)
            if ce[0] <= 1e-10*ce[-1]:
                record['status']='UNRESOLVED_COVARIANCE_CONDITIONING_NO_REPAIR'
            else:
                factor=cho_factor(R,lower=True,check_finite=False)
                solved=cho_solve(factor,T.T,check_finite=False)
                projected=T@solved
                S=Q-projected;S=(S+S.T)/2
                se=eigvalsh(S,check_finite=False)
                scalar=np.diag(projected)/np.diag(Q)
                negative=np.minimum(se,0)
                record.update(status='SOLVED',schur_min=float(se[0]),schur_max=float(se[-1]),
                              negative_count=int(np.count_nonzero(se < -1e-8)),
                              negative_trace_fraction=float(-negative.sum()/np.trace(Q)),
                              negative_energy_fraction=float(np.sum(negative**2)/np.sum(se**2)),
                              cauchy_max=float(scalar.max()),cauchy_mean=float(scalar.mean()),
                              cauchy_violations=int(np.count_nonzero(scalar>1+1e-8)))
            rows.append(record);print(json.dumps(record),flush=True)
receipt=dict(rows=rows,wall_s=time.time()-start,target_access=False,
             qualification='Necessary moment-cone audit of approximate current states; no identified correction or accuracy gain.')
out.write_text(json.dumps(receipt,indent=2)+'\n')
