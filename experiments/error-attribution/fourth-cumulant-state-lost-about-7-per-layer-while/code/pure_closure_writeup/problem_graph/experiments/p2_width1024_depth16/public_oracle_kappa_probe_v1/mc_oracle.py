"""MC-ORACLE-KAPPA-V1: Monte Carlo oracle for pre-activation cumulants, layers 1..6.

TAG: public_oracle_kappa_probe_v1/mc_oracle.py
Knobs: N_TOTAL=2,000,000 default, env MC_N_TOTAL overrides (production run used
8,000,000 because the 2M smoke showed reference SE comparable to the model's
own mean error); BATCH=25,000 (MC_BATCH); LAYERS=6; RANKS=(35,100,201);
float32 forward
Convention (matches whestbench.simulation and the Q24 estimator): x0 ~ N(0, I_1024)
float32; z_l = x_{l-1} @ W_l with W_l the stored per-row weights.npy matrices used
as-is (scale sqrt(2/1024) is baked into the stored weights; no extra /32); x_l =
relu(z_l).  first_pre_covariance in the estimator is W_1^T W_1, so MC layer-1
variance must match diag(W_1^T W_1) to MC precision -- asserted below before any
output is written.

Outputs per rank: mc_oracle_rank{R:04d}.npz with per-layer per-unit
pre_mean, pre_var, k3 (third central moment), k4 (fourth cumulant = m4c - 3 var^2),
post_mean, post_var, and batch-spread standard errors for each.
"""

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16")
STORE = ROOT / "learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3"
OUT = ROOT / "public_oracle_kappa_probe_v1"

RANKS = (35, 100, 201)
LAYERS = int(__import__("os").environ.get("MC_LAYERS", 6))
WIDTH = 1024
N_TOTAL = int(__import__("os").environ.get("MC_N_TOTAL", 2_000_000))
BATCH = int(__import__("os").environ.get("MC_BATCH", 25_000))
PREFIX = __import__("os").environ.get("MC_PREFIX", "mc_oracle_rank")
INV_SQRT_2PI = 0.3989422804014327


def weights_for_rank(rank):
    manifest = json.loads((STORE / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["roles"]["observer_fit"]["row_manifests"][rank]
    if int(entry["role_rank"]) != rank:
        raise AssertionError(f"row_manifests[{rank}] rank mismatch: {entry}")
    weights_path = (STORE / entry["path"]).parent / "weights.npy"
    weights = np.load(weights_path)
    if weights.shape != (16, WIDTH, WIDTH) or weights.dtype != np.float32:
        raise AssertionError(f"unexpected weights schema at rank {rank}")
    return np.ascontiguousarray(weights[:LAYERS]), int(entry["mlp_id"])


def stats_from_sums(sums, count):
    """sums[..., stat, unit]: stat = z, z^2, z^3, z^4, relu(z), relu(z)^2."""
    m1 = sums[..., 0, :] / count
    m2 = sums[..., 1, :] / count
    m3 = sums[..., 2, :] / count
    m4 = sums[..., 3, :] / count
    var = m2 - m1 * m1
    m3c = m3 - 3.0 * m1 * m2 + 2.0 * m1 ** 3
    m4c = m4 - 4.0 * m1 * m3 + 6.0 * m1 * m1 * m2 - 3.0 * m1 ** 4
    k3 = m3c
    k4 = m4c - 3.0 * var * var
    post_mean = sums[..., 4, :] / count
    post_var = sums[..., 5, :] / count - post_mean * post_mean
    return m1, var, k3, k4, post_mean, post_var


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def run_rank(rank):
    weights, mlp_id = weights_for_rank(rank)
    n_batches = N_TOTAL // BATCH
    rng = np.random.default_rng(3016500 + 1000 * rank)
    batch_sums = np.zeros((n_batches, LAYERS, 6, WIDTH), dtype=np.float64)
    started = time.time()
    print(f"[mc_oracle] rank={rank} mlp_id={mlp_id} N={N_TOTAL} batches={n_batches}x{BATCH} seed={3016500 + 1000 * rank}", flush=True)
    for b in range(n_batches):
        x = rng.standard_normal((BATCH, WIDTH), dtype=np.float32)
        for l in range(LAYERS):
            z = x @ weights[l]
            zz = z * z
            batch_sums[b, l, 0] = z.sum(axis=0, dtype=np.float64)
            batch_sums[b, l, 1] = zz.sum(axis=0, dtype=np.float64)
            batch_sums[b, l, 2] = (zz * z).sum(axis=0, dtype=np.float64)
            batch_sums[b, l, 3] = (zz * zz).sum(axis=0, dtype=np.float64)
            x = np.maximum(z, np.float32(0.0))
            batch_sums[b, l, 4] = x.sum(axis=0, dtype=np.float64)
            batch_sums[b, l, 5] = (x * x).sum(axis=0, dtype=np.float64)
        if (b + 1) % 10 == 0:
            print(f"[mc_oracle] rank={rank} batch {b + 1}/{n_batches} elapsed={time.time() - started:.1f}s", flush=True)

    total = batch_sums.sum(axis=0)
    pre_mean, pre_var, k3, k4, post_mean, post_var = stats_from_sums(total, float(N_TOTAL))
    b_mean, b_var, b_k3, b_k4, b_pmean, b_pvar = stats_from_sums(batch_sums, float(BATCH))

    def batch_se(values):
        return values.std(axis=0, ddof=1) / np.sqrt(n_batches)

    se_pre_mean = batch_se(b_mean)
    se_pre_var = batch_se(b_var)
    se_k3 = batch_se(b_k3)
    se_k4 = batch_se(b_k4)
    se_post_mean = batch_se(b_pmean)
    se_post_var = batch_se(b_pvar)

    # Layer-1 gates: z_1 = x0 @ W1 is exactly Gaussian with covariance W1^T W1.
    w1_64 = weights[0].astype(np.float64)
    var_exact = np.einsum("ij,ij->j", w1_64, w1_64)  # diag(W1^T W1)
    var_rel_rms = rms((pre_var[0] - var_exact) / var_exact)
    post_exact = np.sqrt(var_exact) * INV_SQRT_2PI
    post_rel_rms = rms((post_mean[0] - post_exact) / post_exact)
    k3_l1_z = rms(k3[0] / se_k3[0]) if np.all(se_k3[0] > 0) else float("nan")
    elapsed = time.time() - started
    print(f"[mc_oracle] rank={rank} LAYER-1 GATES: var relRMS vs diag(W1^T W1) = {var_rel_rms:.3e} "
          f"(expect ~{np.sqrt(2.0 / N_TOTAL):.1e}); post-mean relRMS vs sigma/sqrt(2pi) = {post_rel_rms:.3e}; "
          f"k3 z-score RMS = {k3_l1_z:.2f} (expect ~1)", flush=True)
    if var_rel_rms > 5.0e-3:
        raise AssertionError(f"rank {rank}: MC layer-1 variance does not match diag(W1^T W1): relRMS={var_rel_rms}")
    if post_rel_rms > 5.0e-3:
        raise AssertionError(f"rank {rank}: MC layer-1 post-mean does not match sigma/sqrt(2pi): relRMS={post_rel_rms}")

    suffix = __import__("os").environ.get("MC_SUFFIX", "")
    out_path = OUT / f"{PREFIX}{rank:04d}{suffix}.npz"
    np.savez_compressed(
        out_path,
        rank=rank,
        mlp_id=mlp_id,
        n_samples=N_TOTAL,
        batch=BATCH,
        seed=3016500 + 1000 * rank,
        pre_mean=pre_mean, pre_var=pre_var, k3=k3, k4=k4,
        post_mean=post_mean, post_var=post_var,
        se_pre_mean=se_pre_mean, se_pre_var=se_pre_var,
        se_k3=se_k3, se_k4=se_k4, se_post_mean=se_post_mean, se_post_var=se_post_var,
        layer1_var_rel_rms=var_rel_rms,
        layer1_post_rel_rms=post_rel_rms,
        layer1_k3_zscore_rms=k3_l1_z,
        elapsed_s=elapsed,
    )
    print(f"[mc_oracle] rank={rank} wrote {out_path} in {elapsed:.1f}s", flush=True)


def main():
    import sys
    ranks = [int(a) for a in sys.argv[1:]] or list(RANKS)
    for rank in ranks:
        run_rank(rank)
    print("[mc_oracle] DONE", flush=True)


if __name__ == "__main__":
    main()
