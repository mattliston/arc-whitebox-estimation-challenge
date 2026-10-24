"""On-policy final-only rows, frozen learned prefix; no truth state writes."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from radial_binary_compact import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_final_head_v1 import final_input

HERE = Path(__file__).resolve().parent
G22 = HERE.parent / "public_g22_operator_features_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
ap = argparse.ArgumentParser()
ap.add_argument("--rank", type=int, required=True)
args = ap.parse_args()
out = HERE / "rows_compact_final_head_v1" / f"row_{args.rank:04d}"
if (out / "result.json").exists():
    print("existing completed row", args.rank, flush=True)
    raise SystemExit(0)
if out.exists():
    raise FileExistsError(out)
out.mkdir(parents=True)
payload = HERE / "payload_binary_compact_g01_v1.npz"
P = dict(np.load(payload))
man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
W = np.load(STORE / Path(man[args.rank]["path"]).parent / "weights.npy", mmap_mode="r")
state = np.zeros((1024, P["head"].shape[1]), np.float32)
saved = {}


def hook(l, m, C, K21, aux, w):
    global state
    x = features(l, m, C, K21, aux, w)
    if l == 15:
        saved.update(h=final_input(x, w, state, P), sigma=aux["sigma"].copy(), host=m.copy(), x=x.copy())
        old_delta, _ = gnn_step(l, x, aux["sigma"], w, state, P)
        saved["old_delta"] = old_delta
        return np.zeros_like(m)
    delta, state = gnn_step(l, x, aux["sigma"], w, state, P)
    return delta


started = time.time()
with threadpool_limits(limits=4):
    mean, _ = predict(W, hook=hook, gate_floor=.01, collect_states=False)
# Targets are loaded only after every input and all recurrent states are fixed.
truth = np.load(G22 / "truthpost" / f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
saved["target"] = truth[-1]-saved["host"]
saved["mean_prefix"] = mean
assert all(np.all(np.isfinite(v)) for v in saved.values())
np.savez(out / "data.npz", **saved)
record = dict(rank=args.rank, wall_s=time.time()-started,
              host_final_mse=float(np.mean(saved["target"]**2)),
              old_final_mse=float(np.mean((saved["target"]-saved["old_delta"])**2)),
              frozen_prefix_payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),
              provenance="Frozen learned layers 1..14; final inputs captured before correction; no truth feedback anywhere.")
(out / "result.json").write_text(json.dumps(record, indent=2)+"\n")
print(json.dumps(record), flush=True)
