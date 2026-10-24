"""TAG opus-release: setup-aware native evaluation (in-process flopscope on numpy/OpenBLAS, one BLAS thread): bill and MSE per
network, one row after every network. Copy of H/hill_climb_e1_ynt_psdb_d5_v1/native_probe.py with the output directory given
explicitly (so everything stays under O/release) and the process's kernel counters added per network (cycles and instructions do
not move with machine load; wall and CPU seconds do, see rusage.py). Truth is used only for offline scoring.
Bills depend on the cache history of the process: compare arms only on the SAME rank list in the SAME order, fresh process each.
usage (OpenBLAS env python, under the guard): native_probe.py <candidate.py> <ranks> <new outdir>"""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

import numpy as np
import flopscope as flops

import sys
sys.path.insert(0,'/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/release')
sys.path.insert(0,'/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/opus_release_v1/pipeline')
import rusage
import captured_setup_replay_generic_v1 as replay

P2 = Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16')
candidate = Path(sys.argv[1]).resolve()
ranks = [int(x) for x in sys.argv[2].split(',')]
assert ranks == [35,100]
assert hashlib.sha256(candidate.read_bytes()).hexdigest()==os.environ['CANDIDATE_SHA256']
out = Path(sys.argv[3]).resolve()
out.mkdir()
store = P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest = json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('candidate_eval')
mod.__file__ = str(candidate)
sys.modules[mod.__name__] = mod
exec(compile(candidate.read_text(), str(candidate), 'exec'), mod.__dict__)
expected_setup = replay.install(mod, os.environ['CAPTURED_SETUP_MANIFEST'])


class MLP:
    def __init__(self, weights, rank):
        self.width, self.depth, self.weights, self.seed, self.name = 1024, 16, weights, 0, f'phase2-{rank}'


est = mod.Estimator()
t0 = time.monotonic()
if hasattr(est, 'setup'):
    est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None,
                                    submission_dir=None, seed=0))
setup = dict(setup_s=time.monotonic() - t0, pb_log=[list(x) for x in getattr(mod, '_PB_LOG', [])])
setup['captured_replay'] = replay.verify(mod, expected_setup)
record = dict(candidate=str(candidate), sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(), ranks=ranks,
              runtime=sys.executable, numpy=np.__version__, setup=setup,
              thread_env={k: os.environ.get(k) for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                                                         'VECLIB_MAXIMUM_THREADS']})
(out/'manifest.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(dict(event='setup', **setup)), flush=True)
if os.environ.get('SETUP_ONLY') == '1':
    print('CAPTURED_SETUP_REPLAY_PASS', flush=True)
    sys.exit(0)
with (out/'rows.jsonl').open('x') as fp:
    for rank in ranks:
        weights = np.load(store/Path(manifest[rank]['path']).parent/'weights.npy')
        t0, u0 = time.monotonic(), rusage.read()
        with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
            y = np.asarray(est.predict(MLP([weights[l] for l in range(16)], rank), 2**41), dtype=np.float64)
        seconds, used = time.monotonic() - t0, rusage.delta(rusage.read(), u0)
        assert y.shape == (16, 1024) and np.isfinite(y).all(), (rank, y.shape)
        s = ctx.summary_dict()
        flops_used = int(s['flops_used'])
        truth = np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'].astype(np.float64)
        mse = float(np.mean((y[15] - truth[15])**2))
        util = flops_used/2**41
        row = dict(rank=rank, mse=mse, all_layer_mse=float(np.mean((y - truth)**2)), util=util, adjusted=mse*max(.1, util),
                   flops=flops_used, wall_s=seconds, cpu_s=used['user_s'] + used['system_s'], user_s=used['user_s'],
                   system_s=used['system_s'], cycles=used['cycles'], instructions=used['instructions'],
                   backend_s=s['flopscope_backend_time_s'], footprint_max=used['footprint_max'])
        np.save(out/f'pred_{rank}.npy', y)
        fp.write(json.dumps(row) + '\n')
        fp.flush()
        print(json.dumps(row), flush=True)
        del weights, y, truth
print('ALL_NETWORKS_COMPLETE', flush=True)
