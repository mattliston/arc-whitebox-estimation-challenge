"""v3 (trainer, 10-12 15:3xZ; lead 15:2xZ: the per-round weight caches add ~12 GB a round): = v2's generated-network paths WITHOUT a disk
cache. Each call regenerates the network's weights from its seed with the analyst's generator (O/analyst/fresh/loader.py weights(), the same
bytes v2 cached), checks them against the truth file's weights_sha256 on the first use of each seed in this process (as v2 did once per seed),
and hands them to the engine as an in-memory .npy stream: the engine's np.load(weights_path, allow_pickle=False) reads it exactly as it read
the cached file. Truth checks (seed, shape, finite, minimum N) as v2. Measured: generation about 2.0 s + first-use sha256 about 0.6 s per eval
(1-2% of a 112-190 s sizing eval), no disk written. v2 stays as it was (running workers import it).
"""
import hashlib
import io

import numpy as np

from otr_generated_v2 import MIN_N, SEED_FLOOR, load_analyst


def make_data_paths_mem(public_paths, allowed_seeds):
    allowed = frozenset(int(seed) for seed in allowed_seeds)
    assert all(seed >= SEED_FLOOR for seed in allowed)
    analyst = load_analyst()
    verified = {}

    def data_paths(rank):
        rank = int(rank)
        if rank < SEED_FLOOR:
            return public_paths(rank)
        assert rank in allowed, 'Generated seed is not in this worker allowlist: %d' % rank
        truth_path = analyst.truth_path(rank)
        assert truth_path.exists(), 'Generated truth is not ready: %s' % truth_path
        weights = analyst.weights(rank)
        assert weights.dtype == np.float32 and weights.shape == (16, 1024, 1024) and weights.flags['C_CONTIGUOUS']
        stamp = (truth_path.stat().st_size, truth_path.stat().st_mtime_ns)
        if verified.get(rank) != stamp:
            weights_sha = hashlib.sha256(memoryview(weights).cast('B')).hexdigest()
            with np.load(truth_path, allow_pickle=False) as truth:
                assert int(truth['seed']) == rank
                assert str(truth['weights_sha256']) == weights_sha, 'Generated weight/truth hash mismatch'
                assert truth['mc_post'].shape == (16, 1024)
                assert np.isfinite(truth['mc_post']).all()
                assert int(truth['N']) >= MIN_N, 'Incomplete generated truth'
            verified[rank] = stamp
        stream = io.BytesIO()
        np.save(stream, weights, allow_pickle=False)
        del weights
        stream.seek(0)
        return stream, truth_path

    return data_paths
