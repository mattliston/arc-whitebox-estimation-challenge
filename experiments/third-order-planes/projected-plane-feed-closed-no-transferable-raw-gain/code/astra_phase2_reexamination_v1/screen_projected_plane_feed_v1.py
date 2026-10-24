"""One natural-coefficient carried-source screen, no retuning or old heads."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from projected_plane_feed_v1 import predict

HERE=Path(__file__).resolve().parent
STORE=HERE.parent/"learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
G22=HERE.parent/"public_g22_operator_features_v1"
ap=argparse.ArgumentParser();ap.add_argument("--rank",type=int,choices=(35,100),required=True);args=ap.parse_args()
out=HERE/f"projected_plane_feed_r{args.rank}_v1.json"
assert not out.exists() and not out.with_suffix(".npz").exists()
assert (HERE/"projected_plane_feed_certificate_v1.json").is_file()
man=json.loads((STORE/"manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W=np.load(STORE/Path(man[args.rank]["path"]).parent/"weights.npy",mmap_mode="r")
truth=np.load(G22/"truthpost"/f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
parent=np.load(HERE/f"compact_activeonly_r{args.rank}_v1.npz")["mean"][0]
t0=time.time()
with threadpool_limits(limits=4):
    mean,states=predict(W,gate_floor=.01,collect_states=True)
assert np.isfinite(mean).all()
mse=float(np.mean((np.maximum(mean[-1],0)-truth[-1])**2))
base=float(np.mean((np.maximum(parent[-1],0)-truth[-1])**2))
record=dict(rank=args.rank,final_mse=mse,parent_final_mse=base,relative_mse=mse/base,
    layer_mse=np.mean((np.maximum(mean,0)-truth)**2,axis=1).tolist(),states=states,
    wall_s=time.time()-t0,frame="Unlearned same-source current full-pair; stored1e9 means, no metering or panel claim.")
np.savez(out.with_suffix(".npz"),mean=mean)
out.write_text(json.dumps(record,indent=2)+"\n")
print(json.dumps(record),flush=True)
