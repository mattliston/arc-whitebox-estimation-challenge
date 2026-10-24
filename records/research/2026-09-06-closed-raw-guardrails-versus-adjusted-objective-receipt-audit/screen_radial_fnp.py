"""Meter and save the new FlopScope candidate on an explicitly named Phase2 net."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time
import numpy as np
import flopscope as flops
from threadpoolctl import threadpool_limits
from whestbench.domain import MLP
from whestbench.seeds import derive_estimator_seed
HERE = Path(__file__).resolve().parent
G22 = HERE.parent/"public_g22_operator_features_v1"
STORE = HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap = argparse.ArgumentParser()
ap.add_argument("--candidate",required=True)
ap.add_argument("--rank",type=int,default=35)
ap.add_argument("--plain",action="store_true")
ap.add_argument("--tag",required=True)
args = ap.parse_args()
outpath = HERE/f"{args.tag}.json"
if outpath.exists() or outpath.with_suffix(".npz").exists():
    raise FileExistsError(outpath)
spec = importlib.util.spec_from_file_location("astra_radial_candidate",HERE/args.candidate)
mod = importlib.util.module_from_spec(spec); sys.modules[spec.name]=mod; spec.loader.exec_module(mod)
mod._WINOGRAD = not args.plain
man = json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
rowpath = STORE/man[args.rank]["path"]; row=json.loads(rowpath.read_text())
W = np.load(rowpath.parent/"weights.npy",mmap_mode="r")
mlp=MLP(width=1024,depth=16,weights=list(W),seed=int(derive_estimator_seed(int(row["mlp_seed"]))),name=f"radial-{args.rank}")
truth = np.load(G22/"truthpost"/f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
t0=time.time()
with threadpool_limits(limits=4):
    with flops.BudgetContext(flop_budget=2**41,wall_time_limit_s=900,quiet=True) as ctx:
        out = mod.Estimator().predict(mlp,2**41)
out=np.asarray(out,dtype=np.float64)
r=dict(args=vars(args),util=int(ctx.flops_used)/2**41,flops=int(ctx.flops_used),wall_s=time.time()-t0,
       layer_mse=np.mean((out-truth)**2,axis=1).tolist(),final_mse=float(np.mean((out[-1]-truth[-1])**2)))
np.savez(outpath.with_suffix(".npz"),mean=out)
outpath.write_text(json.dumps(r,indent=2)+"\n")
print(json.dumps(r),flush=True)
