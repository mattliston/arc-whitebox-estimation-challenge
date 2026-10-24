"""Fixed, bounded prefix-gain diagnostic on the current compact estimator.

No oracle feedback, no fitting, and no evaluation-panel data. The two fixed
perturbations test systematic over/under-application of the prefix head.
The final scalar head is unchanged. Existing 1e9 means are used only after
prediction. A useful branch requires a substantial gain on BOTH dev nets;
otherwise do not start a gain sweep or DAgger refit.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

from compact_fullpair_final_reference_v1 import predict
from radial_features import features
from radial_gnn_numpy import gnn_step
from compact_scalar_final_v1 import final_scalar_features
from compact_final_head_v1 import final_input, final_delta

HERE = Path(__file__).resolve().parent
G22 = HERE.parent / "public_g22_operator_features_v1"
STORE = HERE.parent / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank", type=int, choices=(35, 100), required=True)
    args = ap.parse_args()
    out = HERE / f"current_prefix_gain_r{args.rank}_v1.json"
    arrays = out.with_suffix(".npz")
    if out.exists() or arrays.exists():
        raise FileExistsError(out)
    P = dict(np.load(HERE / "payload_compact_feed_fullpair_v1.npz"))
    Q = dict(np.load(HERE / "payload_fullpair_final_head_v1.npz"))
    man = json.loads((STORE / "manifest.json").read_text())["roles"]["observer_fit"]["row_manifests"]
    W = np.load(STORE / Path(man[args.rank]["path"]).parent / "weights.npy", mmap_mode="r")
    truth = np.load(G22 / "truthpost" / f"truthpost_rank{args.rank:04d}.npz")["mc_post"].astype(np.float64)
    reference = np.load(HERE / f"fullpair_scalar_final_closedloop_r{args.rank}_v1.npz")["mean"]
    records, predictions = [], []
    start = time.time()
    with threadpool_limits(limits=4):
        for gain in (1.0, 0.9, 1.1):
            state = np.zeros((1024, P["head"].shape[1]), np.float32)
            delta_rms = []

            def hook(l, m, C, K21, aux, w):
                nonlocal state
                d, state = gnn_step(l, features(l, m, C, K21, aux, w), aux["sigma"], w, state, P)
                d = d.astype(np.float64) * gain
                delta_rms.append(float(np.sqrt(np.mean(d * d))))
                return d

            def final_hook(l, m, aux, w):
                h = final_input(final_scalar_features(l, m, aux, w), w, state, P)
                return final_delta(h, aux["sigma"], Q)

            t0 = time.time()
            mean, _ = predict(W, hook=hook, final_hook=final_hook, collect_states=False)
            mean = np.maximum(mean, 0)
            assert np.isfinite(mean).all()
            if gain == 1.0:
                parity = float(np.max(np.abs(mean - reference)))
                assert parity < 1e-12, parity
            mse = np.mean((mean - truth) ** 2, axis=1)
            row = dict(gain=gain, final_mse=float(mse[-1]), layer_mse=mse.tolist(),
                       prefix_delta_rms=delta_rms, wall_s=time.time()-t0)
            records.append(row)
            predictions.append(mean)
            print(json.dumps(row), flush=True)
    base = records[0]["final_mse"]
    for row in records:
        row["relative_mse"] = row["final_mse"] / base
    receipt = dict(rank=args.rank, records=records, baseline_max_abs=parity,
                   wall_s=time.time()-start,
                   frame="Current compact scalar-final NumPy reference; stored 1e9 post means; fixed development diagnostic, not a panel score or fitted calibration.")
    np.savez(arrays, gain=np.asarray([1.0, .9, 1.1]), mean=np.stack(predictions))
    out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
