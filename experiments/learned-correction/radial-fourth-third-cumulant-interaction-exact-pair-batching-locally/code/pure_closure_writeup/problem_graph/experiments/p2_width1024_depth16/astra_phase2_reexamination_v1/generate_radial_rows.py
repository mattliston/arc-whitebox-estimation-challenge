"""Honest one-step radial K3 rows; features captured before any truth feedback."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from k3_radial_numpy import predict
from radial_features import features,FEATURE_NAMES

HERE = Path(__file__).resolve().parent
G22 = HERE.parent/"public_g22_operator_features_v1"
STORE = HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap = argparse.ArgumentParser()
ap.add_argument("--rank",type=int,required=True)
ap.add_argument("--rows",required=True)
ap.add_argument("--cov32",action="store_true")
ap.add_argument("--metric",choices=("identity","covariance"),default="covariance")
args = ap.parse_args()
out = HERE/args.rows/f"row_{args.rank:04d}"
if (out/"result.json").exists():
    print(f"existing completed row {args.rank}",flush=True)
    raise SystemExit(0)
if out.exists():
    raise FileExistsError(f"Incomplete row is preserved: {out}")
out.mkdir(parents=True)
man = json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE/Path(man[args.rank]["path"]).parent/"weights.npy",mmap_mode="r")
truth = np.load(G22/"truthpost"/f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
assert W.shape==(16,1024,1024)
xs,hosts,sigmas = [],[],[]
def hook(l,m,C,K21,aux,w):
    xs.append(features(l,m,C,K21,aux,w))
    hosts.append(m.copy()); sigmas.append(aux["sigma"].copy())
    return truth[l]-m if l>=1 else np.zeros_like(m)
t0 = time.time()
with threadpool_limits(limits=4):
    pred,_ = predict(W,hook=hook,cov32=args.cov32,metric=args.metric,collect_states=False)
x = np.stack(xs); host = np.stack(hosts); sigma = np.stack(sigmas)
assert np.all(np.isfinite(x)) and np.all(np.isfinite(host))
np.save(out/"raw20.npy",x[:,:,:20]); np.save(out/"mech.npy",x[:,:,20:])
np.save(out/"host.npy",host); np.save(out/"sigma.npy",sigma)
result = dict(rank=args.rank,args=vars(args),wall_s=time.time()-t0,feature_names=FEATURE_NAMES,
              one_step_mse=np.mean((host-truth)**2,axis=1).tolist(),
              correction_order="features and host frozen before truth feedback; covariance updated afterward")
(out/"result.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
