"""Closed-loop NumPy development screen of an exported mean-head payload."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from k3_radial_numpy import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
HERE=Path(__file__).resolve().parent
G22=HERE.parent/"public_g22_operator_features_v1"
STORE=HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap=argparse.ArgumentParser()
ap.add_argument("--rank",type=int,default=35)
ap.add_argument("--payload",required=True)
ap.add_argument("--tag",required=True)
args=ap.parse_args()
outpath=HERE/f"{args.tag}.json"
if outpath.exists() or outpath.with_suffix(".npz").exists(): raise FileExistsError(outpath)
P=dict(np.load(HERE/args.payload))
assert set(P)=={"centers","scales","rms","msg_w","msg_b","enc0_w","enc0_b","enc2_w","enc2_b","enc4_w","enc4_b","head","head_b","gain","first"}
assert P["first"][0]==1
man=json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W=np.load(STORE/Path(man[args.rank]["path"]).parent/"weights.npy",mmap_mode="r")
truth=np.load(G22/"truthpost"/f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
state=np.zeros((1024,P["head"].shape[1]),np.float32)
xs,ds=[],[]
def hook(l,m,C,K21,aux,w):
    global state
    x=features(l,m,C,K21,aux,w)
    delta,state=gnn_step(l,x,aux["sigma"],w,state,P)
    xs.append(x.copy()); ds.append(delta.copy())
    return delta
t0=time.time()
with threadpool_limits(limits=4):
    mean,_=predict(W,hook=hook,cov32=True,metric="covariance",collect_states=False)
mean=np.maximum(mean,0)
r=dict(args=vars(args),wall_s=time.time()-t0,layer_mse=np.mean((mean-truth)**2,axis=1).tolist(),final_mse=float(np.mean((mean[-1]-truth[-1])**2)))
np.savez(outpath.with_suffix(".npz"),mean=mean,x=np.stack(xs),delta=np.stack(ds))
outpath.write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r),flush=True)
