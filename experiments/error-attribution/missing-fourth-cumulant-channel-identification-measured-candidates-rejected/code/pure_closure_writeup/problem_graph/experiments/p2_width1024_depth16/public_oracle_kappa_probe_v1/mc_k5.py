"""MC-K5-V1: measured standardized fifth pre-activation cumulant vs the
kappa5 emission closure, layers 8 / 11 / 14, rank 35.

TAG: public_oracle_kappa_probe_v1/mc_k5.py
Knobs: N_TOTAL=8,000,000  BATCH=25,000  LAYERS=(8,11,14)  RANK=35
Cumulant ladder (central moments mu_k, sigma = sqrt(mu_2)):
  kappa5 = mu5 - 10*mu2*mu3;  k5_std = kappa5 / sigma^5
  g3 = mu3 / sigma^3;  g4 = (mu4 - 3*mu2^2) / sigma^4
  closure_k5 = 5*g3*g4 - (40/9)*g3^3   (from the SAME samples' g3/g4)
Deliverable: K5_MEASURED_RESULT.json with per-layer masked medians (mask =
units with |closure_k5| above its median, the audit convention) of k5_std,
closure_k5, and the per-unit ratio measured/closure, plus batch-spread SEs.
Decides deep-k4-decay ownership: ratio ~1 = closure floor holds (push-
inexactness is the majority owner); ratio >>1 = intrinsic chaos-4 content
(emission truncation owns the ladder).
"""

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16")
STORE = ROOT / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = ROOT / "public_oracle_kappa_probe_v1"

RANK = 35
TARGET_LAYERS = (8, 11, 14)
WIDTH = 1024
N_TOTAL = 8_000_000
BATCH = 25_000


def weights_for_rank(rank):
    manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    weights = np.load((STORE / entry["path"]).parent / "weights.npy")
    return np.ascontiguousarray(weights[:max(TARGET_LAYERS)]), int(entry["mlp_id"])


def cumulants(sums, count):
    """sums[..., p, :] = sum z^(p+1), p = 0..4."""
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


def masked_median(values, mask):
    return float(np.median(values[mask]))


def main():
    weights, mlp_id = weights_for_rank(RANK)
    n_batches = N_TOTAL // BATCH
    seed = 3016500 + 1000 * RANK + 23
    rng = np.random.default_rng(seed)
    layer_slot = {l: i for i, l in enumerate(TARGET_LAYERS)}
    batch_sums = np.zeros((n_batches, len(TARGET_LAYERS), 5, WIDTH), dtype=np.float64)
    started = time.time()
    print(f"[mc_k5] rank={RANK} mlp_id={mlp_id} N={N_TOTAL} layers={TARGET_LAYERS} seed={seed}", flush=True)
    for b in range(n_batches):
        x = rng.standard_normal((BATCH, WIDTH), dtype=np.float32)
        for l in range(1, max(TARGET_LAYERS) + 1):
            z = x @ weights[l - 1]
            if l in layer_slot:
                s = layer_slot[l]
                zz = z * z
                batch_sums[b, s, 0] = z.sum(axis=0, dtype=np.float64)
                batch_sums[b, s, 1] = zz.sum(axis=0, dtype=np.float64)
                batch_sums[b, s, 2] = (zz * z).sum(axis=0, dtype=np.float64)
                batch_sums[b, s, 3] = (zz * zz).sum(axis=0, dtype=np.float64)
                batch_sums[b, s, 4] = (zz * zz * z).sum(axis=0, dtype=np.float64)
            x = np.maximum(z, np.float32(0.0))
        if (b + 1) % 40 == 0:
            print(f"[mc_k5] batch {b + 1}/{n_batches} elapsed={time.time() - started:.0f}s", flush=True)

    k5_std, closure, g3, g4 = cumulants(batch_sums.sum(axis=0), float(N_TOTAL))
    b_k5, b_closure, _, _ = cumulants(batch_sums, float(BATCH))
    se_k5 = b_k5.std(axis=0, ddof=1) / np.sqrt(n_batches)

    per_layer = {}
    for l, s in layer_slot.items():
        mask = np.abs(closure[s]) > np.quantile(np.abs(closure[s]), 0.5)
        ratio = k5_std[s][mask] / closure[s][mask]
        batch_medians = [float(np.median(b_k5[b, s][mask] / closure[s][mask]))
                         for b in range(n_batches)]
        per_layer[str(l)] = {
            "k5_std_masked_median": round(masked_median(k5_std[s], mask), 5),
            "closure_k5_masked_median": round(masked_median(closure[s], mask), 5),
            "ratio_measured_over_closure_masked_median": round(float(np.median(ratio)), 4),
            "ratio_batch_spread_se": round(float(np.std(batch_medians, ddof=1) / np.sqrt(n_batches)), 4),
            "k5_std_unmasked_median": round(float(np.median(k5_std[s])), 5),
            "k5_per_unit_se_rms": round(float(np.sqrt(np.mean(se_k5[s] ** 2))), 5),
            "g3_masked_median_abs": round(float(np.median(np.abs(g3[s][mask]))), 5),
            "g4_masked_median": round(float(np.median(g4[s][mask])), 5),
        }
        print(f"[mc_k5] L{l}: k5_std med={per_layer[str(l)]['k5_std_masked_median']} "
              f"closure med={per_layer[str(l)]['closure_k5_masked_median']} "
              f"ratio={per_layer[str(l)]['ratio_measured_over_closure_masked_median']}"
              f" +/- {per_layer[str(l)]['ratio_batch_spread_se']}", flush=True)

    np.savez_compressed(
        OUT / f"mc_k5_rank{RANK:04d}.npz",
        rank=RANK, mlp_id=mlp_id, n_samples=N_TOTAL, seed=seed,
        layers=np.array(TARGET_LAYERS), k5_std=k5_std, closure_k5=closure,
        g3=g3, g4=g4, se_k5=se_k5,
    )
    result = {
        "tag": "public_oracle_kappa_probe_v1/k5_measured",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "question": "Does the true standardized fifth pre-activation cumulant match the kappa5 emission closure (5 g3 g4 - 40/9 g3^3), or does intrinsic content exceed it?",
        "rank": RANK, "n_samples": N_TOTAL, "layers": list(TARGET_LAYERS),
        "convention": "mask = units with |closure_k5| above its per-layer median; ratio = per-unit measured/closure, masked median; closure from the SAME samples' g3/g4",
        "per_layer": per_layer,
        "elapsed_s": round(time.time() - started, 1),
    }
    (OUT / "K5_MEASURED_RESULT.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[mc_k5] wrote {OUT / 'K5_MEASURED_RESULT.json'}", flush=True)


if __name__ == "__main__":
    main()
