"""MC-ORACLE-FINAL-V1: MC oracle for the FINAL layer (L16) pre-activation stats.

TAG: public_oracle_kappa_probe_v1/mc_oracle_final.py
Knobs: N_TOTAL=8,000,000  BATCH=25,000  RANKS=(35,201)  float32 forward
Forward 15 layers, then z16 = x15 @ W16: accumulate z16 power sums (pre_mean,
pre_var, k3, k4 + batch-spread SEs) and relu(z16) post_mean/post_var + SEs.
Purpose: final-layer sensitivity decomposition (the one consumption site with
no downstream feedback).  Same seeding discipline as mc_oracle_deep (base
3016500 + 1000*rank, offset +13 for an independent draw).
Output: mc_oracle_final_rank{R:04d}.npz, [1024] arrays.
"""

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16")
STORE = ROOT / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = ROOT / "public_oracle_kappa_probe_v1"

RANKS = (35, 201)
WIDTH = 1024
DEPTH = 16
N_TOTAL = 8_000_000
BATCH = 25_000


def weights_for_rank(rank):
    manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError(f"row_manifests[{rank}] rank mismatch")
    weights = np.load((STORE / entry["path"]).parent / "weights.npy")
    if weights.shape != (DEPTH, WIDTH, WIDTH) or weights.dtype != np.float32:
        raise AssertionError(f"unexpected weights schema at rank {rank}")
    return np.ascontiguousarray(weights), int(entry["mlp_id"])


def stats_from_sums(sums, count):
    m1 = sums[..., 0, :] / count
    m2 = sums[..., 1, :] / count
    m3 = sums[..., 2, :] / count
    m4 = sums[..., 3, :] / count
    var = m2 - m1 * m1
    k3 = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    m4c = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
    k4 = m4c - 3.0 * var * var
    post_mean = sums[..., 4, :] / count
    post_var = sums[..., 5, :] / count - post_mean * post_mean
    return m1, var, k3, k4, post_mean, post_var


def run_rank(rank):
    weights, mlp_id = weights_for_rank(rank)
    n_batches = N_TOTAL // BATCH
    seed = 3016500 + 1000 * rank + 13
    rng = np.random.default_rng(seed)
    batch_sums = np.zeros((n_batches, 6, WIDTH), dtype=np.float64)
    started = time.time()
    print(f"[mc_final] rank={rank} mlp_id={mlp_id} N={N_TOTAL} seed={seed}", flush=True)
    for b in range(n_batches):
        x = rng.standard_normal((BATCH, WIDTH), dtype=np.float32)
        for l in range(DEPTH - 1):
            x = np.maximum(x @ weights[l], np.float32(0.0))
        z = x @ weights[DEPTH - 1]
        zz = z * z
        batch_sums[b, 0] = z.sum(axis=0, dtype=np.float64)
        batch_sums[b, 1] = zz.sum(axis=0, dtype=np.float64)
        batch_sums[b, 2] = (zz * z).sum(axis=0, dtype=np.float64)
        batch_sums[b, 3] = (zz * zz).sum(axis=0, dtype=np.float64)
        post = np.maximum(z, np.float32(0.0))
        batch_sums[b, 4] = post.sum(axis=0, dtype=np.float64)
        batch_sums[b, 5] = (post * post).sum(axis=0, dtype=np.float64)
        if (b + 1) % 40 == 0:
            print(f"[mc_final] rank={rank} batch {b + 1}/{n_batches} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)

    pre_mean, pre_var, k3, k4, post_mean, post_var = stats_from_sums(
        batch_sums.sum(axis=0), float(N_TOTAL)
    )
    b_stats = stats_from_sums(batch_sums, float(BATCH))
    ses = [s.std(axis=0, ddof=1) / np.sqrt(n_batches) for s in b_stats]
    elapsed = time.time() - started
    out = OUT / f"mc_oracle_final_rank{rank:04d}.npz"
    np.savez_compressed(
        out, rank=rank, mlp_id=mlp_id, n_samples=N_TOTAL, batch=BATCH, seed=seed,
        layer=16,
        pre_mean=pre_mean, pre_var=pre_var, k3=k3, k4=k4,
        post_mean=post_mean, post_var=post_var,
        se_pre_mean=ses[0], se_pre_var=ses[1], se_k3=ses[2], se_k4=ses[3],
        se_post_mean=ses[4], se_post_var=ses[5],
        elapsed_s=elapsed,
    )
    print(f"[mc_final] rank={rank} wrote {out} in {elapsed:.0f}s", flush=True)


def main():
    import sys
    ranks = [int(a) for a in sys.argv[1:]] or list(RANKS)
    for rank in ranks:
        run_rank(rank)
    print("[mc_final] DONE", flush=True)


if __name__ == "__main__":
    main()
