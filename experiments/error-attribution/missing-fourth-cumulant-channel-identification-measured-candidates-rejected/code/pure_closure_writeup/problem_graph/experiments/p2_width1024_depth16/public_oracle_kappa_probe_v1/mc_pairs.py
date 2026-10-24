"""MC-PAIRS-V1: Monte Carlo pre-activation covariance at 2000 fixed unit pairs.

TAG: public_oracle_kappa_probe_v1/mc_pairs.py
Knobs: N_TOTAL=8,000,000  BATCH=25,000  LAYERS=15  RANKS=(35,201)
Same forward as mc_oracle.py (float32 x = relu(x @ W), stored weights as-is);
accumulates E[z_i], E[z_j], E[z_i z_j] at the pairs_spec.py pair list per layer,
giving cov_ij = E[z_i z_j] - E[z_i]E[z_j] with batch-spread SEs.
Layer-1 gate: cov at the pairs is exactly (W1^T W1)_ij; asserted to MC precision.
Output: mc_pairs_rank{R:04d}.npz (cov [15,2000], se_cov, pairs_i, pairs_j).
"""

import json
import time
from pathlib import Path

import numpy as np

from pairs_spec import pair_list, N_PAIRS

ROOT = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16")
STORE = ROOT / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = ROOT / "public_oracle_kappa_probe_v1"

RANKS = (35, 201)
LAYERS = 15
WIDTH = 1024
N_TOTAL = 8_000_000
BATCH = 25_000


def weights_for_rank(rank):
    manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError(f"row_manifests[{rank}] rank mismatch")
    weights = np.load((STORE / entry["path"]).parent / "weights.npy")
    return np.ascontiguousarray(weights[:LAYERS]), int(entry["mlp_id"])


def run_rank(rank, pairs_i, pairs_j):
    weights, mlp_id = weights_for_rank(rank)
    n_batches = N_TOTAL // BATCH
    rng = np.random.default_rng(3016500 + 1000 * rank + 7)
    # per-batch sums: stat 0 = z_i, 1 = z_j, 2 = z_i*z_j
    batch_sums = np.zeros((n_batches, LAYERS, 3, N_PAIRS), dtype=np.float64)
    started = time.time()
    print(f"[mc_pairs] rank={rank} mlp_id={mlp_id} N={N_TOTAL} pairs={N_PAIRS} "
          f"seed={3016500 + 1000 * rank + 7}", flush=True)
    for b in range(n_batches):
        x = rng.standard_normal((BATCH, WIDTH), dtype=np.float32)
        for l in range(LAYERS):
            z = x @ weights[l]
            zi = z[:, pairs_i]
            zj = z[:, pairs_j]
            batch_sums[b, l, 0] = zi.sum(axis=0, dtype=np.float64)
            batch_sums[b, l, 1] = zj.sum(axis=0, dtype=np.float64)
            batch_sums[b, l, 2] = (zi * zj).sum(axis=0, dtype=np.float64)
            x = np.maximum(z, np.float32(0.0))
        if (b + 1) % 40 == 0:
            print(f"[mc_pairs] rank={rank} batch {b + 1}/{n_batches} "
                  f"elapsed={time.time() - started:.0f}s", flush=True)

    def cov_from(sums, count):
        return sums[..., 2, :] / count - (sums[..., 0, :] / count) * (sums[..., 1, :] / count)

    cov = cov_from(batch_sums.sum(axis=0), float(N_TOTAL))
    per_batch = cov_from(batch_sums, float(BATCH))
    se_cov = per_batch.std(axis=0, ddof=1) / np.sqrt(n_batches)

    w1 = weights[0].astype(np.float64)
    cov_exact = np.sum(w1[:, pairs_i] * w1[:, pairs_j], axis=0)  # (W1^T W1)_ij
    resid = cov[0] - cov_exact
    z_rms = float(np.sqrt(np.mean((resid / se_cov[0]) ** 2)))
    rel = float(np.sqrt(np.mean(resid ** 2)) / np.sqrt(np.mean(cov_exact ** 2)))
    elapsed = time.time() - started
    print(f"[mc_pairs] rank={rank} LAYER-1 GATE: pair-cov vs (W1^T W1)_ij relRMS={rel:.3e}, "
          f"z-score RMS={z_rms:.2f} (expect ~1)", flush=True)
    if z_rms > 2.0:
        raise AssertionError(f"rank {rank}: pair-cov layer-1 gate failed (z RMS {z_rms})")

    out = OUT / f"mc_pairs_rank{rank:04d}.npz"
    np.savez_compressed(
        out, rank=rank, mlp_id=mlp_id, n_samples=N_TOTAL, batch=BATCH,
        pairs_i=pairs_i, pairs_j=pairs_j, cov=cov, se_cov=se_cov,
        layer1_gate_z_rms=z_rms, layer1_gate_rel_rms=rel, elapsed_s=elapsed,
    )
    print(f"[mc_pairs] rank={rank} wrote {out} in {elapsed:.0f}s", flush=True)


def main():
    pairs_i, pairs_j = pair_list()
    for rank in RANKS:
        run_rank(rank, pairs_i, pairs_j)
    print("[mc_pairs] DONE", flush=True)


if __name__ == "__main__":
    main()
