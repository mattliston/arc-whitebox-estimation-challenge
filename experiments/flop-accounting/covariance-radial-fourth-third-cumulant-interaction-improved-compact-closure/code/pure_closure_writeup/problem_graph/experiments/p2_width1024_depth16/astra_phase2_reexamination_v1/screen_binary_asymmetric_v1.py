"""Fixed two-net causal test of stronger cold pruning at fixed birth floor."""
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from radial_binary_compact_asymmetric_v1 import predict, predict_shadow

HERE = Path(__file__).resolve().parent
G22 = HERE.parent / "public_g22_operator_features_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
outfile = HERE / "binary_asymmetric_v1.json"
if outfile.exists():
    raise FileExistsError(outfile)
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
records = []
started = time.time()
with threadpool_limits(limits=4):
    for net in (35, 100):
        w = np.load(STORE / Path(man[net]["path"]).parent / "weights.npy", mmap_mode="r")
        truth = np.load(G22 / "truthpost" / f"truthpost_rank{net:04d}.npz")["mc_post"].astype(np.float64)
        for floor in (.03, .05):
            for shadow, method in ((False, predict), (True, predict_shadow)):
                mean, states = method(w, gate_floor=floor, birth_floor=.01, collect_states=True)
                record = dict(net=net, gate_floor=floor, birth_floor=.01, shadow=shadow,
                              final_mse=float(np.mean((mean[-1]-truth[-1])**2)),
                              layer_mse=np.mean((mean-truth)**2, axis=1).tolist(), states=states)
                records.append(record)
                print(json.dumps(record), flush=True)
                path = HERE / f"binary_asymmetric_r{net}_g{floor}_shadow{int(shadow)}_v1.npz"
                if path.exists():
                    raise FileExistsError(path)
                np.savez(path, mean=mean)
outfile.write_text(json.dumps(dict(records=records, wall_s=time.time()-started), indent=2)+"\n")
