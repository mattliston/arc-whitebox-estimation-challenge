"""Only 24 fresh trajectories; freeze/verify parent state before labels."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_feed_fullpair_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from joint_plane_features_v1 import joint_features,FEATURE_NAMES

HERE=Path(__file__).resolve().parent
STORE=HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
G22=HERE.parent/"public_g22_operator_features_v1"
ap=argparse.ArgumentParser();ap.add_argument("--rank",type=int,required=True);args=ap.parse_args()
assert 35<=args.rank<59
out=HERE/"rows_joint_plane_probe_v1"/f"row_{args.rank:04d}"
assert not out.exists();out.mkdir(parents=True)
man=json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W=np.load(STORE/Path(man[args.rank]["path"]).parent/"weights.npy",mmap_mode="r")
truth=np.load(G22/"truthpost"/f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
P=dict(np.load(HERE/"payload_compact_feed_fullpair_v1.npz"))
state=np.zeros((1024,P["head"].shape[1]),np.float32)
xs=[];js=[];hosts=[];sigmas=[];deltas=[]
def hook(l,m,C,K21,aux,w):
    global state
    x=features(l,m,C,K21,aux,w)
    j=joint_features(m,K21,aux)
    delta,state=gnn_step(l,x,aux["sigma"],w,state,P)
    xs.append(x.copy());js.append(j.copy());hosts.append(m.copy())
    sigmas.append(aux["sigma"].copy());deltas.append(delta.copy())
    return truth[l]-m if l>=1 else np.zeros_like(m)
t0=time.time()
with threadpool_limits(limits=4):
    predict(W,gate_floor=.01,hook=hook,cov32=True,metric="covariance",collect_states=False)
x=np.stack(xs);j=np.stack(js);host=np.stack(hosts);sigma=np.stack(sigmas);delta=np.stack(deltas)
parent=HERE/"rows_compact_feed_fullpair_v1"/f"row_{args.rank:04d}"
assert np.array_equal(host,np.load(parent/"host.npy")),"host drift"
assert np.array_equal(sigma,np.load(parent/"sigma.npy")),"sigma drift"
assert np.array_equal(x[:,:,:20],np.load(parent/"raw20.npy")),"old feature drift"
assert np.array_equal(x[:,:,20:],np.load(parent/"mech.npy")),"old mech drift"
assert j.shape==(16,1024,48) and np.isfinite(j).all()
np.savez(out/"data.npz",x=x,joint=j,sigma=sigma,host=host,delta=delta,target=truth-host)
receipt=dict(rank=args.rank,parent_byte_exact=True,feature_names=FEATURE_NAMES,
    wall_s=time.time()-t0,one_step_mse=np.mean((truth-host)**2,axis=1).tolist(),
    frozen_parent_head_residual_mse=np.mean((truth-host-delta)**2,axis=1).tolist())
(out/"result.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(dict(rank=args.rank,wall_s=receipt["wall_s"],parent_byte_exact=True)),flush=True)
