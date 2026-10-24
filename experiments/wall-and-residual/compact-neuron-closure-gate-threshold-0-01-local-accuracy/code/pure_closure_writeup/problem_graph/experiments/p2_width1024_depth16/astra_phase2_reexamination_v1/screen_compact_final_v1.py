"""Compare scalar final execution against frozen on-policy training inputs."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from compact_final_reference_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input,final_delta

HERE = Path(__file__).resolve().parent
G22 = HERE.parent / "public_g22_operator_features_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap = argparse.ArgumentParser()
ap.add_argument("--rank",type=int,required=True)
ap.add_argument("--tag",required=True)
args = ap.parse_args()
out = HERE/f"{args.tag}.json"
if out.exists() or out.with_suffix(".npz").exists():
    raise FileExistsError(out)
P = dict(np.load(HERE/"payload_binary_compact_g01_v1.npz"))
Q = dict(np.load(HERE/"payload_compact_final_head_v1.npz"))
man = json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE/Path(man[args.rank]["path"]).parent/"weights.npy",mmap_mode="r")
truth = np.load(G22/"truthpost"/f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
state = np.zeros((1024,P["head"].shape[1]),np.float32)
xs,ds = [],[]
saved = {}


def hook(l,m,C,K21,aux,w):
    global state
    x = features(l,m,C,K21,aux,w)
    d,state = gnn_step(l,x,aux["sigma"],w,state,P)
    xs.append(x.copy()); ds.append(d.copy())
    return d


def final_hook(l,m,aux,w):
    x = final_scalar_features(l,m,aux,w)
    h = final_input(x,w,state,P)
    d = final_delta(h,aux["sigma"],Q)
    saved.update(final_h=h,final_host=m.copy(),final_delta=d.copy())
    return d


started = time.time()
with threadpool_limits(limits=4):
    mean,_ = predict(W,hook=hook,final_hook=final_hook,collect_states=False)
mean = np.maximum(mean,0)
frozen = dict(np.load(HERE/"rows_compact_final_head_v1"/f"row_{args.rank:04d}"/"data.npz"))
with threadpool_limits(limits=4):
    frozen_prediction = frozen["host"]+final_delta(frozen["h"],frozen["sigma"],Q)
receipt = dict(rank=args.rank,wall_s=time.time()-started,final_mse=float(np.mean((mean[-1]-truth[-1])**2)),
               frozen_input_prediction_mse=float(np.mean((frozen_prediction-truth[-1])**2)),
               scalar_vs_frozen_host_max=float(np.max(np.abs(saved["final_host"]-frozen["host"]))),
               scalar_vs_frozen_input_max=float(np.max(np.abs(saved["final_h"]-frozen["h"]))),
               layer_mse=np.mean((mean-truth)**2,axis=1).tolist())
np.savez(out.with_suffix(".npz"),mean=mean,x=np.stack(xs),delta=np.stack(ds),**saved)
out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
