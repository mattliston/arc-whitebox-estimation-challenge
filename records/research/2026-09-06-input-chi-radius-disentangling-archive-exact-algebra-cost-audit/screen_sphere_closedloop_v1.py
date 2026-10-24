"""Matched angular GNN, evaluated in the standard-Gaussian target frame."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from radial_compact_sphere_v1 import predict, gaussian_radius_mean
from radial_features import features
from radial_gnn_numpy import gnn_step

HERE = Path(__file__).resolve().parent
G22 = HERE.parent / "public_g22_operator_features_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
ap.add_argument("--tag", required=True)
args = ap.parse_args()
out = HERE / f"{args.tag}.json"
if out.exists() or out.with_suffix(".npz").exists():
    raise FileExistsError(out)
P = dict(np.load(HERE / "payload_compact_sphere_v1.npz"))
assert len(P)==16 and P["centers"].shape==(16,30)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[args.rank]["path"]).parent / "weights.npy",mmap_mode="r")
truth = np.load(G22 / "truthpost" / f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
baseline = np.load(HERE / f"compact_feed_fullpair_closedloop_r{args.rank}_v1.npz")["mean"]
state = np.zeros((1024,P["head"].shape[1]),np.float32)
xs, deltas = [], []


def hook(l,m,C,K21,aux,w):
    global state
    x = features(l,m,C,K21,aux,w)
    delta,state = gnn_step(l,x,aux["sigma"],w,state,P)
    xs.append(x.copy()); deltas.append(delta.copy())
    return delta


start = time.time()
with threadpool_limits(limits=4):
    mean, _ = predict(W,gate_floor=.01,hook=hook,collect_states=False)
mean = np.maximum(mean,0)
assert np.isfinite(mean).all()
mse = float(np.mean((mean[-1]-truth[-1])**2))
baseline_mse = float(np.mean((baseline[-1]-truth[-1])**2))
receipt = dict(rank=args.rank, final_mse=mse, baseline_final_mse=baseline_mse, relative_mse=mse/baseline_mse,
               layer_mse=np.mean((mean-truth)**2,axis=1).tolist(),wall_s=time.time()-start,
               gaussian_radius_mean=gaussian_radius_mean(1024),
               frame="Matched 200-net/fixed160-40/C64/40-epoch angular GNN; final outputs in standard-Gaussian units; full final feature map.")
np.savez(out.with_suffix(".npz"),mean=mean,x=np.stack(xs),delta=np.stack(deltas))
out.write_text(json.dumps(receipt,indent=2)+"\n")
print(json.dumps(receipt),flush=True)
