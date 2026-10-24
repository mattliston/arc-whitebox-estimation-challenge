"""Numpy-free worker with input/output lifetime bounded by each open budget. No candidate-state or billing-history resets."""
import array
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import types
import flopscope as flops
import flopscope.numpy as fnp

assert importlib.util.find_spec('numpy') is None
HERE = Path(__file__).resolve().parent
P2 = HERE.parent
candidate = Path(sys.argv[1]).resolve()
ranks = [int(x) for x in sys.argv[2].split(',')]
outdir = Path(sys.argv[3]).resolve()
store = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
rows = json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('handle_candidate')
mod.__file__ = str(candidate)
sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec'), mod.__dict__)

class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights = 1024, 16, weights
        self.seed, self.name = 0, f'handle-{rank}'

est = mod.Estimator()
start = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
          api_version='2', scratch_dir=None, submission_dir=None, seed=0))
manifest = dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                ranks=ranks, setup_s=time.monotonic()-start, pb_log=[list(x) for x in mod._PB_LOG],
                purpose=__doc__, server_instrumented=True)
(outdir/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(json.dumps(dict(event='setup', setup_s=manifest['setup_s'], pb_log=manifest['pb_log'])), flush=True)
with (outdir/'rows.jsonl').open('x') as log:
    for rank in ranks:
        fixture = store/Path(rows[rank]['path']).parent/'weights.npy'
        with flops.BudgetContext(flop_budget=2**41, quiet=True, namespace=f'handles-{rank}') as ctx:
            blob = fnp.load(str(fixture))
            weights = [blob[layer] for layer in range(16)]
            start = time.monotonic()
            prediction = est.predict(MLP(weights, rank), 2**41)
            data, shape, dtype = prediction._fetch_data()
            assert tuple(shape) == (16, 1024), shape
            decoded = array.array('f' if '32' in str(dtype) else 'd')
            decoded.frombytes(data)
            assert len(decoded) == 16384 and all(math.isfinite(x) for x in decoded)
            # Release harness-owned remote inputs/output while the session is
            # still open. summary_dict flushes deferred frees in this session.
            del blob, weights, prediction
            summary = ctx.summary_dict()
        record = dict(rank=rank, finite_outputs=len(decoded), flops=int(summary['flops_used']),
                      instrumented_wall_s=time.monotonic()-start,
                      index_cache_entries=[len(getattr(mod, k)) for k in ('_IPT_A','_IPT_B','_IPT_C')])
        assert max(record['index_cache_entries']) <= 480
        if rank in (35, 100):
            (outdir/f'pred_{rank}.bin').write_bytes(data)
            (outdir/f'pred_{rank}.json').write_text(json.dumps(dict(shape=list(shape), dtype=str(dtype)))+'\n')
        log.write(json.dumps(record)+'\n'); log.flush()
        print(json.dumps(record), flush=True)
        del decoded, data
print('ALL_WORKER_NETWORKS_COMPLETE', flush=True)
