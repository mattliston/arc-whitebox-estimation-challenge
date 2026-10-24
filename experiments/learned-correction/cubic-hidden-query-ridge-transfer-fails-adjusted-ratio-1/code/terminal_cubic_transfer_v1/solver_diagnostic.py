import resource
import numpy as np
from common import *
from solver import fit
seal_check();rng=np.random.default_rng(2026092179);rows=[]
for case in range(4):
    n=31+case;d=7;X=rng.normal(size=(n,d));y=rng.normal(size=n);w=np.exp(rng.normal(size=n))
    X[:,0]=1;X[:,1]=0;X[:,2]=X[:,3]
    if case==3:y[:]=0
    beta,cert=fit(np,X,y,w);wn=w/w.sum();xn=X/cert['scale']
    aug=np.vstack([np.sqrt(wn)[:,None]*xn,np.sqrt(1e-3)*np.eye(d)])
    yy=np.concatenate([np.sqrt(wn)*y,np.zeros(d)])
    scaled_ref=np.linalg.lstsq(aug,yy,rcond=None)[0];alt=scaled_ref/cert['scale']
    row=dict(case=case,unscaled_relative=float(np.linalg.norm(beta-alt)/max(np.linalg.norm(beta),1e-30)),
        scaled_relative=float(np.linalg.norm(cert['scaled_beta']-scaled_ref)/max(np.linalg.norm(cert['scaled_beta']),1e-30)),
        prediction_relative=float(np.linalg.norm(X@beta-X@alt)/max(np.linalg.norm(X@beta),1e-30)),
        zero_column_scale=float(cert['scale'][1]),zero_column_normal_coefficient=float(cert['scaled_beta'][1]),
        zero_column_augmented_coefficient=float(scaled_ref[1]))
    rows.append(row)
write(HERE/'solver_diagnostic.json',dict(rows=rows,no_native_calls=True,targets_decoded=False,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(rows,flush=True)
