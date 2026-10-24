import json
import numpy as np
import sys
sys.path.insert(0,".")
from run_tangent_v2 import c1_std_factors
rank=35
d=np.load(f"states_rank{rank:04d}_lite.npz") if __import__("pathlib").Path(f"states_rank{rank:04d}_lite.npz").exists() else np.load(f"states_rank{rank:04d}.npz")
z=np.load(f"host_cov_rank{rank:04d}_base.npz")
tp=np.load(f"tensor_planes_rank{rank:04d}.npz")
kt=np.load(f"tensor_planes_rank{rank:04d}_kt.npz")
s=np.load(f"sidecar_states_rank{rank:04d}.npz")
tr=np.load(f"mps_truth_rank{rank:04d}_P8M.npz")
dep=np.load(f"deployed_rank{rank:04d}.npy")
raw20=s["raw20"]
P2=__import__("pathlib").Path(".").resolve().parent
STORE=(P2/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3")
manifest=json.loads((STORE/"manifest.json").read_text())
entry=manifest["roles"]["observer_fit"]["row_manifests"][rank]
W=np.load((STORE/entry["path"]).parent/"weights.npy",mmap_mode="r")
Wd=[np.asarray(W[k],dtype=np.float64) for k in range(16)]
resid=(tr["mc_post"][15]-dep[15]).astype(np.float64)
e1={};a2={};Sv={}
for l in range(1,16):
    i=l-1
    m_=d["host_m"][i].astype(np.float64); v_=np.maximum(d["host_v"][i].astype(np.float64),1e-30)
    c1,e,sd=c1_std_factors(m_,v_)
    e1[l]=c1[1]/sd; a2[l]=c1[2]/sd**2
    al=m_/sd
    Sv[l]=np.exp(-0.5*al*al)/np.sqrt(2*np.pi)/(2*sd)
g={}
acc=resid.copy()
for L in range(15,0,-1):
    g[L]=acc.copy()
    acc=Wd[L]@(raw20[L,:,5].astype(np.float64)*acc)
Mcore=None
v_ok=v_kt=v_oab=0.0
per=[]
for l in range(14,0,-1):
    A=e1[l][:,None]*Wd[l+1]
    core=(Mcore if Mcore is not None else np.zeros((1024,1024)))
    core=core+np.diag(Sv[l+1]*g[l+1])
    Mcore=A@core@A.T
    M=(a2[l][:,None]*Mcore)*e1[l][None,:]
    if f"pre_{l}" in tp.files:
        h=z["k21"][l-1].astype(np.float64)
        dfull=tp[f"pre_{l}"].astype(np.float64)-h
        dkt=kt[f"pre_{l}"].astype(np.float64)-h
        doab=dfull-dkt
        vf=float(np.sum(dfull*M)); vk=float(np.sum(dkt*M)); vo=float(np.sum(doab*M))
        v_ok+=vf; v_kt+=vk; v_oab+=vo
        per.append((l,round(vf*1e7,2),round(vk*1e7,2),round(vo*1e7,2)))
print("per-layer value x1e7 (l, full, mapgap_kt, out_and_back):")
for row in per[::-1]: print(" ",row)
print(f"TOTALS x1e7: full {v_ok*1e7:.2f}  mapgap {v_kt*1e7:.2f}  oab {v_oab*1e7:.2f}")
print(f"shares of full value: mapgap {v_kt/v_ok:.2f}  oab {v_oab/v_ok:.2f}")
