"""TAG analyst A13: FRESH TRAINING NETWORKS (generated, not from the public dataset) and their Monte Carlo truths.
A network is defined by its seed alone: weights = numpy Generator(PCG64(seed)).standard_normal((16, 1024, 1024)) in float64, times
sqrt(2 / 1024), cast to float32, C order: the dtype, shape and orientation of the store's weights.npy (z_l = y_{l-1} @ W[l], no
biases, input x ~ N(0, I)). The public networks are statistically the same thing (checked on ranks 35, 37, 400: entries iid Gaussian,
sd sqrt(2/1024) to 1e-3, layer means not centred, column norms as iid). Weights are never stored: regenerate them (0.3 s).
Truth files: truth_seed{seed}.npz next to this file, written by mc_truth.py:
  mc_post (16, 1024) float64 = Monte Carlo mean of the post-activations of every layer over N samples (two independent halves pooled),
  mc_post_A, mc_post_B = the two halves, mc_post_var = per-unit variance of the post-activation (so the truth noise variance of
  mc_post is mc_post_var / N), N, N_half, seed, weights_sha256 (sha256 of the float32 weight bytes the truth was computed with).
usage:
  import loader
  w = loader.weights(7100000)                 # (16, 1024, 1024) float32
  t = loader.truth(7100000)                   # dict; t['mc_post'][15] is the final-layer target; checks the weight hash
  seeds = loader.available()                  # seeds with a finished truth file, ascending
"""
import hashlib
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
SEED0 = 7_100_000


def weights(seed):
    w = np.random.Generator(np.random.PCG64(int(seed))).standard_normal((16, 1024, 1024)) * np.sqrt(2.0 / 1024)
    return np.ascontiguousarray(w.astype(np.float32))


def weights_sha256(w):
    assert w.dtype == np.float32 and w.shape == (16, 1024, 1024) and w.flags['C_CONTIGUOUS']
    return hashlib.sha256(w.tobytes()).hexdigest()


def truth_path(seed):
    return HERE / f'truth_seed{int(seed)}.npz'


def truth(seed, check_weights=True):
    with np.load(truth_path(seed), allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    assert int(t['seed']) == int(seed) and t['mc_post'].shape == (16, 1024) and np.isfinite(t['mc_post']).all()
    if check_weights:
        assert str(t['weights_sha256']) == weights_sha256(weights(seed)), 'truth was computed with different weights'
    return t


def available():
    return sorted(int(p.stem[len('truth_seed'):]) for p in HERE.glob('truth_seed*.npz'))
