"""Predeclared compare: same CP subset, with/without future-query fitting."""
import argparse,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from radial_lookahead import make_predict
HERE=Path(__file__).resolve().parent;G22=HERE.parent/"public_g22_operator_features_v1"
STORE=HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap=argparse.ArgumentParser();ap.add_argument("--rank",type=int,required=True);args=ap.parse_args()
outfile=HERE/f"radial_lookahead_r{args.rank}_v1.json"
if outfile.exists():raise FileExistsError(outfile)
man=json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W=np.load(STORE/Path(man[args.rank]["path"]).parent/"weights.npy",mmap_mode="r")
truth=np.load(G22/"truthpost"/f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
records=[];t0=time.time()
with threadpool_limits(limits=4):
    for fit in (False,True):
        predict,log=make_predict(fit=fit)
        out,_=predict(W,cov32=True,metric="covariance",collect_states=False)
        r=dict(rank=args.rank,fit=fit,final_mse=float(np.mean((out[-1]-truth[-1])**2)),layer_mse=np.mean((out-truth)**2,axis=1).tolist(),compression=log)
        records.append(r);print(json.dumps(r),flush=True)
        np.savez(HERE/f"radial_lookahead_fit{int(fit)}_r{args.rank}_v1.npz",mean=out)
outfile.write_text(json.dumps(dict(records=records,wall_s=time.time()-t0),indent=2)+"\n")
