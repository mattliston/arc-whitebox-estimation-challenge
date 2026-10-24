"""Explicit, SHA-pinned generated truth plans. Offline only; legacy defaults intact.

A plan is immutable. Unfinished truths require immutable completion receipts
bound to the plan SHA, exact path, sample count, producer source and file hash.
No generated truth is opened before its seed passes the worker allowlist.
"""
import fcntl
import hashlib
import json
from pathlib import Path

SEED_FLOOR = 7_000_000


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


class TruthPlan:
    def __init__(self, path, expected_sha256, allowed_seeds):
        self.path = Path(path).resolve()
        self.sha256 = expected_sha256
        require(digest(expected_sha256) and sha(self.path) == expected_sha256, 'Truth plan SHA mismatch')
        self.plan = json.loads(self.path.read_text())
        require(self.plan.get('schema') == 'analyst_accelerated_truth_plan_v1', 'Unknown truth plan schema')
        self.allowed = frozenset(int(s) for s in allowed_seeds)
        require(all(s >= SEED_FLOOR for s in self.allowed), 'Non-generated seed in allowlist')
        self.entries = {}
        for entry in self.plan['entries']:
            seed = entry['seed']
            require(type(seed) is int and seed >= SEED_FLOOR and seed not in self.entries, 'Invalid/duplicate plan seed')
            require(entry['role'] in ('sizing', 'acceptance', 'final'), 'Unknown truth role')
            require(entry['kind'] in ('existing_full', 'produce'), 'Unknown truth kind')
            require(Path(entry['path']).is_absolute() and Path(entry['completion_receipt']).is_absolute(), 'Truth paths must be absolute')
            require(type(entry['N']) is int and entry['N'] == 2 * entry['N_half'] > 0, 'Invalid exact N')
            require(entry['N_half'] == entry['half_batches'] * self.plan['batch'], 'Truth batch count mismatch')
            require(entry['half_batches'] % 2 == 0, 'Truth must have balanced alternating signs')
            require(entry.get('sha256') is None or digest(entry['sha256']), 'Invalid pinned truth SHA')
            self.entries[seed] = dict(entry)
        require(self.allowed <= self.entries.keys(), 'Allowlist includes seed absent from explicit truth plan')
        self.sources = self.plan.get('source_sha256')
        require(isinstance(self.sources, dict) and self.sources, 'Plan requires explicit source_sha256 map')
        for path, expected in self.sources.items():
            require(Path(path).is_absolute() and digest(expected) and sha(path) == expected, 'Truth producer source SHA mismatch: ' + path)

    def spec(self):
        return {'path': str(self.path), 'sha256': self.sha256}

    def entry(self, seed):
        seed = int(seed)
        require(seed in self.allowed, 'Generated seed is not in this worker allowlist: %d' % seed)
        return self.entries[seed]

    def resolve(self, seed, required=True):
        entry = self.entry(seed)
        truth = Path(entry['path'])
        if not truth.is_file():
            require(not required, 'Generated truth is not ready: ' + str(truth))
            return None
        expected = entry.get('sha256')
        completion = None
        if expected is None:
            receipt = Path(entry['completion_receipt'])
            if not receipt.is_file():
                require(not required, 'Generated completion receipt is not ready: ' + str(receipt))
                return None
            completion = json.loads(receipt.read_text())
            require(completion.get('schema') == 'analyst_truth_completion_v1', 'Unknown completion schema')
            require(completion.get('plan_sha256') == self.sha256, 'Completion plan SHA mismatch')
            for key in ('seed', 'path', 'N', 'N_half', 'half_batches'):
                require(completion.get(key) == entry[key], 'Completion identity mismatch: ' + key)
            require(completion.get('batch') == self.plan['batch'], 'Completion batch mismatch')
            require(completion.get('source_sha256') == self.sources, 'Completion source identity mismatch')
            expected = completion.get('sha256')
            require(digest(expected) and digest(completion.get('weights_sha256')), 'Invalid completion truth/weights SHA')
        return entry, expected, completion

    def ready(self, seed):
        return self.resolve(seed, required=False) is not None


def make_data_paths(public_paths, plan, cache_dir):
    import numpy as np
    from otr_common import publish
    from otr_generated_v1 import load_analyst
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    analyst = load_analyst()
    verified = {}
    metadata = {}

    def data_paths(rank):
        rank = int(rank)
        if rank < SEED_FLOOR:
            return public_paths(rank)
        entry, expected, completion = plan.resolve(rank)
        truth_path = Path(entry['path'])
        weights_path = cache / ('weights_seed%d.npy' % rank)
        stamp = (truth_path.stat().st_size, truth_path.stat().st_mtime_ns, expected)
        if verified.get(rank) == stamp:
            return weights_path, truth_path
        require(sha(truth_path) == expected, 'Generated truth file SHA mismatch')
        with (cache / ('weights_seed%d.lock' % rank)).open('a') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if not weights_path.exists():
                weights = analyst.weights(rank)
                publish(weights_path, lambda stream: np.save(stream, weights, allow_pickle=False), 'wb')
                del weights
            weights = np.load(weights_path, mmap_mode='r', allow_pickle=False)
            require(weights.dtype == np.float32 and weights.shape == (16, 1024, 1024)
                    and weights.flags['C_CONTIGUOUS'], 'Generated weights shape/dtype mismatch')
            weights_sha = hashlib.sha256(memoryview(weights).cast('B')).hexdigest()
            with np.load(truth_path, allow_pickle=False) as truth:
                require(int(truth['seed']) == rank and int(truth['N']) == entry['N']
                        and int(truth['N_half']) == entry['N_half'], 'Generated truth seed/exact N mismatch')
                require(str(truth['weights_sha256']) == weights_sha, 'Generated weight/truth hash mismatch')
                require(truth['mc_post'].shape == (16, 1024) and np.isfinite(truth['mc_post']).all(), 'Invalid generated targets')
            if completion:
                require(completion['weights_sha256'] == weights_sha, 'Completion weights SHA mismatch')
            del weights
            metadata[rank] = dict(plan=plan.spec(), seed=rank, path=str(truth_path), sha256=expected,
                                  N=entry['N'], N_half=entry['N_half'], role=entry['role'],
                                  weights_sha256=weights_sha, source_sha256=plan.sources,
                                  completion_receipt=({'path': entry['completion_receipt'],
                                      'sha256': sha(entry['completion_receipt'])} if completion else None))
            verified[rank] = stamp
        return weights_path, truth_path

    return data_paths, metadata
