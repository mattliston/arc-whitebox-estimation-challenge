"""Setup-aware, metered native evaluation with a receipt after every network.

Run under guard.py with one BLAS thread. Files are small and written once;
partial results survive a guard stop. Truth is used only for offline scoring.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import types

import numpy as np
import flopscope as flops

HERE=Path(__file__).resolve().parent.parent
P2=HERE.parent
candidate=Path(sys.argv[1]).resolve()
ranks=[int(x) for x in sys.argv[2].split(',')]
tag=sys.argv[3]
assert Path(tag).name==tag
out=HERE/tag
out.mkdir()
store=P2/'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
manifest=json.loads((store/'manifest.json').read_text())['roles']['observer_fit']['row_manifests']
stub=types.ModuleType('whestbench')
stub.BaseEstimator=type('BaseEstimator',(),{})
sys.modules['whestbench']=stub
mod=types.ModuleType('candidate_eval')
mod.__file__=str(candidate)
sys.modules[mod.__name__]=mod
exec(compile(candidate.read_text(),str(candidate),'exec'),mod.__dict__)
class MLP:
    def __init__(self,weights,rank):
        self.width,self.depth,self.weights,self.seed,self.name=1024,16,weights,0,f'phase2-{rank}'
capture = {}
original_scalar = mod.nonlinear_scalar
original_gnn = mod.gnn_step
def scalar_capture(*args, **kwargs):
    value = original_scalar(*args, **kwargs)
    capture['physical'] = value[0]
    capture['scalar_calls'] = capture.get('scalar_calls', 0) + 1
    return value
def gnn_capture(layer, *args, **kwargs):
    value = original_gnn(layer, *args, **kwargs)
    if layer == 15:
        capture['delta'] = value[0]
        capture['terminal_calls'] = capture.get('terminal_calls', 0) + 1
    return value
mod.nonlinear_scalar = scalar_capture
mod.gnn_step = gnn_capture
est=mod.Estimator()
t0=time.monotonic()
est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,api_version='2',scratch_dir=None,submission_dir=None,seed=0))
setup=dict(setup_s=time.monotonic()-t0,pb_log=[list(x) for x in getattr(mod,'_PB_LOG',[])])
record=dict(candidate=str(candidate),sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),ranks=ranks,
    runtime=sys.executable,numpy=np.__version__,setup=setup,
    thread_env={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']})
(out/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(event='setup',**setup)),flush=True)
with (out/'rows.jsonl').open('x') as fp:
    for rank in ranks:
        weights=np.load(store/Path(manifest[rank]['path']).parent/'weights.npy')
        capture.clear()
        t0=time.monotonic();c0=time.process_time()
        with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
            y=np.asarray(est.predict(MLP([weights[l] for l in range(16)],rank),2**41),dtype=np.float64)
        seconds=time.monotonic()-t0;cpu=time.process_time()-c0
        assert y.shape==(16,1024) and np.isfinite(y).all(), (rank,y.shape)
        flops_used=int(ctx.summary_dict()['flops_used'])
        assert capture['scalar_calls'] == capture['terminal_calls'] == 1
        physical = np.asarray(capture['physical'], dtype=np.float64).copy()
        delta = np.asarray(capture['delta'], dtype=np.float64).copy()
        assert physical.shape == delta.shape == (1024,)
        assert np.isfinite(physical).all() and np.isfinite(delta).all()
        replay = np.maximum(physical + delta, 0.0).astype(np.float32).astype(np.float64)
        assert replay.tobytes() == y[15].tobytes(), 'scale1 terminal replay'
        np.savez(out/f'terminal_{rank}.npz', physical=physical, delta=delta)
        capture.clear()
        truth=np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz')['mc_post'][15].astype(np.float64)
        mse=float(np.mean((y[15]-truth)**2));util=flops_used/2**41
        row=dict(rank=rank,mse=mse,util=util,adjusted=mse*max(.1,util),flops=flops_used,wall_s=seconds,cpu_s=cpu,
                 uvp=dict(getattr(mod,'_UVP_STATS',{})),ks=dict(getattr(mod,'_KS_STATS',{})),p6=dict(getattr(mod,'_P6_STATS',{})),ubc=dict(getattr(mod,'_UBC_STATS',{})),ynt=dict(getattr(mod,'_YNT_STATS',{})))
        np.save(out/f'pred_{rank}.npy',y)
        fp.write(json.dumps(row)+'\n');fp.flush()
        print(json.dumps(row),flush=True)
        del weights,y,truth
print('ALL_NETWORKS_COMPLETE',flush=True)
