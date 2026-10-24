"""MC-FINAL-K5-V1: one-pass MC for the L16 final-layer stats AND the k5-vs-
closure measurement at layers 8/11/14.

TAG: public_oracle_kappa_probe_v1/mc_final_k5.py
Knobs: N_TOTAL=2,000,000 default (speed-cut per lead: per-order shares average
over 1024 units and the dE^2 denominator is noise-corrected downstream, so the
k4 term only needs ~10% per-unit precision; MC_N_TOTAL overrides), BATCH=25,000,
rank via argv (default 35).  Forward 16 layers float32 (x = relu(x @ W)).
Accumulates:
  layers 8/11/14 : z..z^5 power sums -> k5_std = kappa5/sigma^5 vs the
                   emission closure 5*g3*g4 - (40/9)*g3^3 from the same g3/g4
  layer 16 (pre) : z..z^4 power sums -> pre_mean/pre_var/k3/k4 + SEs
  layer 16 (post): relu(z) mean/var + SEs
Outputs: mc_oracle_final_rank{R}.npz (keys analyze_final_sensitivity.py
expects), mc_k5_rank{R}.npz, and K5_MEASURED_RESULT.json (rank-35 receipt).
"""

import json
import os
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16")
STORE = ROOT / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = ROOT / "public_oracle_kappa_probe_v1"

WIDTH = 1024
DEPTH = 16
K5_LAYERS = (8, 11, 14)
N_TOTAL = int(os.environ.get("MC_N_TOTAL", 2_000_000))
BATCH = int(os.environ.get("MC_BATCH", 25_000))


def weights_for_rank(rank):
    manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    weights = np.load((STORE / entry["path"]).parent / "weights.npy")
    return np.ascontiguousarray(weights), int(entry["mlp_id"])


def final_stats(sums, count):
    m1 = sums[..., 0, :] / count
    m2 = sums[..., 1, :] / count
    m3 = sums[..., 2, :] / count
    m4 = sums[..., 3, :] / count
    var = m2 - m1 * m1
    k3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    k4 = (m4 - 4 * m1 * m3 + 6 * m1 * m1 * m2 - 3 * m1 ** 4) - 3 * var * var
    post_mean = sums[..., 4, :] / count
    post_var = sums[..., 5, :] / count - post_mean * post_mean
    return m1, var, k3, k4, post_mean, post_var


def k5_stats(sums, count):
    m1 = sums[..., 0, :] / count
    m2 = sums[..., 1, :] / count
    m3 = sums[..., 2, :] / count
    m4 = sums[..., 3, :] / count
    m5 = sums[..., 4, :] / count
    mu2 = m2 - m1 ** 2
    mu3 = m3 - 3 * m1 * m2 + 2 * m1 ** 3
    mu4 = m4 - 4 * m1 * m3 + 6 * m1 ** 2 * m2 - 3 * m1 ** 4
    mu5 = m5 - 5 * m1 * m4 + 10 * m1 ** 2 * m3 - 10 * m1 ** 3 * m2 + 4 * m1 ** 5
    sigma = np.sqrt(mu2)
    g3 = mu3 / sigma ** 3
    g4 = (mu4 - 3 * mu2 ** 2) / sigma ** 4
    k5_std = (mu5 - 10 * mu2 * mu3) / sigma ** 5
    closure = 5.0 * g3 * g4 - (40.0 / 9.0) * g3 ** 3
    return k5_std, closure, g3, g4


def run_rank(rank):
    weights, mlp_id = weights_for_rank(rank)
    n_batches = N_TOTAL // BATCH
    seed = 3016500 + 1000 * rank + 13
    rng = np.random.default_rng(seed)
    slot = {l: i for i, l in enumerate(K5_LAYERS)}
    k5_sums = np.zeros((n_batches, len(K5_LAYERS), 5, WIDTH), dtype=np.float64)
    fin_sums = np.zeros((n_batches, 6, WIDTH), dtype=np.float64)
    started = time.time()
    print(f"[mc_final_k5] rank={rank} mlp_id={mlp_id} N={N_TOTAL} "
          f"k5_layers={K5_LAYERS} seed={seed}", flush=True)
    for b in range(n_batches):
        x = rng.standard_normal((BATCH, WIDTH), dtype=np.float32)
        for l in range(1, DEPTH + 1):
            z = x @ weights[l - 1]
            if l in slot:
                s = slot[l]
                zz = z * z
                k5_sums[b, s, 0] = z.sum(axis=0, dtype=np.float64)
                k5_sums[b, s, 1] = zz.sum(axis=0, dtype=np.float64)
                k5_sums[b, s, 2] = (zz * z).sum(axis=0, dtype=np.float64)
                k5_sums[b, s, 3] = (zz * zz).sum(axis=0, dtype=np.float64)
                k5_sums[b, s, 4] = (zz * zz * z).sum(axis=0, dtype=np.float64)
            if l == DEPTH:
                zz = z * z
                fin_sums[b, 0] = z.sum(axis=0, dtype=np.float64)
                fin_sums[b, 1] = zz.sum(axis=0, dtype=np.float64)
                fin_sums[b, 2] = (zz * z).sum(axis=0, dtype=np.float64)
                fin_sums[b, 3] = (zz * zz).sum(axis=0, dtype=np.float64)
                post = np.maximum(z, np.float32(0.0))
                fin_sums[b, 4] = post.sum(axis=0, dtype=np.float64)
                fin_sums[b, 5] = (post * post).sum(axis=0, dtype=np.float64)
            else:
                x = np.maximum(z, np.float32(0.0))
        if (b + 1) % 20 == 0:
            print(f"[mc_final_k5] rank={rank} batch {b + 1}/{n_batches} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)

    stats = final_stats(fin_sums.sum(axis=0), float(N_TOTAL))
    b_stats = final_stats(fin_sums, float(BATCH))
    ses = [s.std(axis=0, ddof=1) / np.sqrt(n_batches) for s in b_stats]
    np.savez_compressed(
        OUT / f"mc_oracle_final_rank{rank:04d}.npz",
        rank=rank, mlp_id=mlp_id, n_samples=N_TOTAL, batch=BATCH, seed=seed,
        layer=16,
        pre_mean=stats[0], pre_var=stats[1], k3=stats[2], k4=stats[3],
        post_mean=stats[4], post_var=stats[5],
        se_pre_mean=ses[0], se_pre_var=ses[1], se_k3=ses[2], se_k4=ses[3],
        se_post_mean=ses[4], se_post_var=ses[5],
        elapsed_s=time.time() - started,
    )
    print(f"[mc_final_k5] rank={rank} wrote mc_oracle_final_rank{rank:04d}.npz "
          f"(se_post_mean rms={float(np.sqrt(np.mean(ses[4] ** 2))):.2e})", flush=True)

    k5_std, closure, g3, g4 = k5_stats(k5_sums.sum(axis=0), float(N_TOTAL))
    b_k5 = k5_stats(k5_sums, float(BATCH))[0]
    se_k5 = b_k5.std(axis=0, ddof=1) / np.sqrt(n_batches)
    np.savez_compressed(
        OUT / f"mc_k5_rank{rank:04d}.npz",
        rank=rank, mlp_id=mlp_id, n_samples=N_TOTAL, seed=seed,
        layers=np.array(K5_LAYERS), k5_std=k5_std, closure_k5=closure,
        g3=g3, g4=g4, se_k5=se_k5,
    )
    per_layer = {}
    for l, s in slot.items():
        mask = np.abs(closure[s]) > np.quantile(np.abs(closure[s]), 0.5)
        ratio = k5_std[s][mask] / closure[s][mask]
        batch_meds = [float(np.median(b_k5[b, s][mask] / closure[s][mask]))
                      for b in range(n_batches)]
        per_layer[str(l)] = {
            "k5_std_masked_median": round(float(np.median(k5_std[s][mask])), 5),
            "closure_k5_masked_median": round(float(np.median(closure[s][mask])), 5),
            "ratio_measured_over_closure_masked_median": round(float(np.median(ratio)), 4),
            "ratio_batch_spread_se": round(float(np.std(batch_meds, ddof=1) / np.sqrt(n_batches)), 4),
            "k5_per_unit_se_rms": round(float(np.sqrt(np.mean(se_k5[s] ** 2))), 5),
            "g3_masked_median_abs": round(float(np.median(np.abs(g3[s][mask]))), 5),
            "g4_masked_median": round(float(np.median(g4[s][mask])), 5),
        }
        print(f"[mc_final_k5] L{l}: k5_std={per_layer[str(l)]['k5_std_masked_median']} "
              f"closure={per_layer[str(l)]['closure_k5_masked_median']} "
              f"ratio={per_layer[str(l)]['ratio_measured_over_closure_masked_median']}"
              f"+/-{per_layer[str(l)]['ratio_batch_spread_se']}", flush=True)
    result = {
        "tag": "public_oracle_kappa_probe_v1/k5_measured",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Does the true standardized fifth pre-activation cumulant match the kappa5 emission closure (5 g3 g4 - 40/9 g3^3)?",
        "rank": rank, "n_samples": N_TOTAL, "layers": list(K5_LAYERS),
        "convention": "mask = units with |closure_k5| above its per-layer median; ratio = per-unit measured/closure, masked median; closure from the SAME samples' g3/g4",
        "stakes": "ratio ~1: closure floor holds and push-inexactness is the majority owner of the deep k4 decay; ratio >>1 (e.g. ~3x): intrinsic chaos-4 content, emission truncation owns the ladder",
        "per_layer": per_layer,
        "elapsed_s": round(time.time() - started, 1),
    }
    (OUT / "K5_MEASURED_RESULT.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[mc_final_k5] rank={rank} wrote K5_MEASURED_RESULT.json in "
          f"{time.time() - started:.0f}s total", flush=True)


def main():
    import sys
    ranks = [int(a) for a in sys.argv[1:]] or [35]
    for rank in ranks:
        run_rank(rank)
    print("[mc_final_k5] DONE", flush=True)


if __name__ == "__main__":
    main()
