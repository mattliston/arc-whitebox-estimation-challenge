"""v2 (trainer, 10-11 10:08Z): the minimum truth sample count comes from env TRAINER_GENERATED_MIN_N (default 64,000,000 as v1); round Nf uses
A36 truths of 15,990,784 samples (their extra noise only adds variance). Offline generated-network paths for refit sizing and acceptance only.

The analyst's generator and truth files are authoritative. The allowlist is explicit
so untouched judging seeds cannot enter a worker by accident. Original data paths,
truths and trainer modules are unchanged. Cache writes are atomic and never replace
an existing network. This module is not included in submission artifacts.
"""
import fcntl
import hashlib
import importlib.util
from pathlib import Path

import numpy as np

from otr_common import O, publish
MIN_N = int(float(__import__('os').environ.get('TRAINER_GENERATED_MIN_N', '64000000')))

SEED_FLOOR = 7_000_000
LOADER_PATH = O / 'analyst/fresh/loader.py'


def load_analyst():
    spec = importlib.util.spec_from_file_location('otr_analyst_fresh_loader_v1', LOADER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_data_paths(public_paths, allowed_seeds, cache_dir):
    allowed = frozenset(int(seed) for seed in allowed_seeds)
    assert all(seed >= SEED_FLOOR for seed in allowed)
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    analyst = load_analyst()
    verified = {}

    def data_paths(rank):
        rank = int(rank)
        if rank < SEED_FLOOR:
            return public_paths(rank)
        assert rank in allowed, 'Generated seed is not in this worker allowlist: %d' % rank
        truth_path = analyst.truth_path(rank)
        assert truth_path.exists(), 'Generated truth is not ready: %s' % truth_path
        weights_path = cache / ('weights_seed%d.npy' % rank)
        stamp = (truth_path.stat().st_size, truth_path.stat().st_mtime_ns)
        if verified.get(rank) == stamp:
            return weights_path, truth_path
        # Two workers may first encounter the same seed together. Keep the lock
        # file as an audit artifact; only the first holder creates the weights.
        with (cache / ('weights_seed%d.lock' % rank)).open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not weights_path.exists():
                weights = analyst.weights(rank)
                publish(weights_path, lambda stream: np.save(stream, weights, allow_pickle=False), 'wb')
                del weights
            weights = np.load(weights_path, mmap_mode='r', allow_pickle=False)
            assert weights.dtype == np.float32 and weights.shape == (16, 1024, 1024)
            assert weights.flags['C_CONTIGUOUS']
            weights_sha = hashlib.sha256(memoryview(weights).cast('B')).hexdigest()
            with np.load(truth_path, allow_pickle=False) as truth:
                assert int(truth['seed']) == rank
                assert str(truth['weights_sha256']) == weights_sha, 'Generated weight/truth hash mismatch'
                assert truth['mc_post'].shape == (16, 1024)
                assert np.isfinite(truth['mc_post']).all()
                assert int(truth['N']) >= MIN_N, 'Incomplete generated truth'
            del weights
            verified[rank] = stamp
        return weights_path, truth_path

    return data_paths
