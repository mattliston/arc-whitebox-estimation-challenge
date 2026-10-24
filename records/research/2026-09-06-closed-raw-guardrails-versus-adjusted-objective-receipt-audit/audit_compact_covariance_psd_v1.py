"""Read-only covariance-validity audit on the frozen learned full-pair loop.

Check active post-covariance before/after current mean correction, without
changing the trajectory or using truth in features. This is a diagnostic,
not an estimator or a permission to deploy an expensive PSD projection.
"""
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_feed_fullpair_v1 import predict
from radial_binary_compact import select
from radial_features import features
from radial_gnn_numpy import gnn_step

HERE=Path(__file__).resolve().parent
STORE=HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
out=HERE/"compact_covariance_psd_r35_v1.json"
if out.exists():
    raise FileExistsError(out)
man=json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W=np.load(STORE/Path(man[35]["path"]).parent/"weights.npy",mmap_mode="r")
P=dict(np.load(HERE/"payload_compact_feed_fullpair_v1.npz"))
state=np.zeros((1024,P["head"].shape[1]),np.float32)
records=[]


def spectrum(C):
    scale=max(float(np.trace(C)),1e-30)
    eigen=np.linalg.eigvalsh((C+C.T)/2)
    return dict(min_eigenvalue=float(eigen[0]),max_eigenvalue=float(eigen[-1]),
                negative_count=int(np.count_nonzero(eigen < -1e-10*scale)),
                negative_trace_fraction=float(-np.minimum(eigen,0).sum()/scale),
                negative_frobenius_fraction=float(np.linalg.norm(np.minimum(eigen,0))/np.linalg.norm(eigen)),
                asymmetry_relative=float(np.linalg.norm(C-C.T)/max(np.linalg.norm(C),1e-30)))


def hook(l,m,C,K21,aux,w):
    global state
    x=features(l,m,C,K21,aux,w)
    d,state=gnn_step(l,x,aux["sigma"],w,state,P)
    if l in (8,12,14):
        indices,_=select(aux["gate"],.01)
        before=C[np.ix_(indices,indices)]
        mi=m[indices];di=d[indices].astype(np.float64)
        after=before-np.outer(mi,di)-np.outer(di,mi)-np.outer(di,di)
        rec=dict(layer=l,active=len(indices),before=spectrum(before),after=spectrum(after),
                 mean_delta_rms=float(np.sqrt(np.mean(d*d))))
        records.append(rec);print(json.dumps(rec),flush=True)
    return d


started=time.time()
with threadpool_limits(limits=4):
    mean,_=predict(W,gate_floor=.01,hook=hook,collect_states=False)
mean=np.maximum(mean,0)
reference=np.load(HERE/"compact_feed_fullpair_closedloop_r35_v1.npz")["mean"]
deviation=float(np.max(np.abs(mean-reference)))
assert deviation<1e-12,deviation
receipt=dict(rank=35,records=records,trajectory_max_abs=deviation,wall_s=time.time()-started,
             interpretation="Read-only active-covariance diagnosis; no model change and no current-layer target feedback.")
out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(trajectory_max_abs=deviation,wall_s=time.time()-started)),flush=True)
