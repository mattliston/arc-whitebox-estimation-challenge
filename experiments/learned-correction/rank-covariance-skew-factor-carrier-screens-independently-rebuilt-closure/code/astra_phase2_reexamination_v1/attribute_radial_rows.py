"""Offline state attribution using frozen honest rows and independent32M labels.

This is an oracle diagnostic, never a deployment candidate or feature source.
Mean targets remain the saved 1e9 post means. Approximate pre means remain
fixed across the per-unit cumulant substitutions.
"""
import json
from pathlib import Path
import numpy as np
from k3_radial_numpy import gaussian_raw
HERE=Path(__file__).resolve().parent
G19=HERE.parent/"public_g19_attribution_v1"
G22=HERE.parent/"public_g22_operator_features_v1"
outfile=HERE/"radial_state_attribution_v1.json"
if outfile.exists():raise FileExistsError(outfile)

def mean(mu,v,k3,k4,second=False):
    s,a,pdf,g,m,*_=gaussian_raw(mu,v)
    out=m-k3*a*pdf/(6*v)+k4*(a*a-1)*pdf/(24*v*s)
    if second:out+=k3*k3*(a**4-6*a*a+3)*pdf/(72*s**5)
    return out

results=[]
for rank in (35,100):
    row=HERE/"rows_radial_covmetric_v1"/f"row_{rank:04d}"
    x=np.load(row/"raw20.npy").astype(np.float64);s=np.load(row/"sigma.npy").astype(np.float64)
    host=np.load(row/"host.npy").astype(np.float64)
    mu=x[:,:,3]*s;v=s*s;k3=x[:,:,10]*s**3;k4=x[:,:,12]*s**4
    truth=np.load(G22/"truthpost"/f"truthpost_rank{rank:04d}.npz")["mc_post"].astype(np.float64)
    labels=[np.load(G19/f"mps_truth_rank{rank:04d}_{tag}.npz") for tag in ("P32M","indep32M")]
    true=[sum(z[f"mc_pre_{key}"] for z in labels)/2 for key in ("v","k3","k4")]
    tv,t3,t4=true
    variants={"baseline":host,"true_v":mean(mu,tv,k3,k4),"true_k3":mean(mu,v,t3,k4),
              "true_k4":mean(mu,v,k3,t4),"true_all":mean(mu,tv,t3,t4),
              "true_k3_k4":mean(mu,v,t3,t4),"k3_squared":mean(mu,v,k3,k4,True),
              "true_all_k3_squared":mean(mu,tv,t3,t4,True)}
    res=dict(rank=rank,truth="1e9 post mean; cumulant supplies averaged independent32M and P32M, oracle only",
             layer_mse={k:np.mean((p-truth)**2,axis=1).tolist() for k,p in variants.items()},
             late_mse={k:float(np.mean((p[10:]-truth[10:])**2)) for k,p in variants.items()})
    results.append(res);print(json.dumps(res),flush=True)
outfile.write_text(json.dumps(results,indent=2)+"\n")
