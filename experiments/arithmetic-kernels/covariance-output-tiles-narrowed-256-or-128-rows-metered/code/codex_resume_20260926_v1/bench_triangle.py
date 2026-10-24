import json
from pathlib import Path
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
P2 = HERE.parent
src = P2/'fable_fire_v1/fire_candS29/candS29_S28_d7cp_d7sq_mem.py'
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('triangle_bench'); mod.__file__ = str(src); sys.modules[mod.__name__] = mod
exec(compile(src.read_text(), str(src), 'exec'), mod.__dict__)
est = mod.Estimator()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41, api_version='2', scratch_dir=None, submission_dir=None, seed=0))
parent = mod._sym_upper
exec((HERE/'triangle_fragment_v1.py').read_text(), mod.__dict__)
rng = np.random.default_rng(926)
results = []
for m, k in [(1024, 1024), (384, 6144), (320, 5568), (384, 4480)]:
    A = (rng.normal(size=(m,k))/np.sqrt(k)).astype(np.float32)
    B = A.T.copy()  # exact symmetric product; kernel cost depends on shape.
    ref = A.astype(float)@B.astype(float)
    for block in (512,256,128):
        mod._TRI_BLOCK=block
        for repeat in range(2):
            before = time.process_time()
            with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
                y = np.asarray((parent if block==512 else mod._sym_upper)(fnp.asarray(A),fnp.asarray(B)))
            results.append(dict(m=m,k=k,block=block,repeat=repeat,flops=int(ctx.summary_dict()['flops_used']),
                                cpu_s=time.process_time()-before,relative_error=float(np.linalg.norm(y-ref)/np.linalg.norm(ref))))
            print(json.dumps(results[-1]),flush=True)
with (HERE/'triangle_bench.json').open('x') as f: json.dump(results,f,indent=2)
