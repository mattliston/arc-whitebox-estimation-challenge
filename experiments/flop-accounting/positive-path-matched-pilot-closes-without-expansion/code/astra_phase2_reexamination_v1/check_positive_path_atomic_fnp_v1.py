"""Actual isolated official-library arithmetic cost and NumPy parity."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from positive_path_resummation_v1 import factors,positive_factor
from positive_path_resummation_fnp_v1 import positive_path_factor

HERE=Path(__file__).resolve().parent;OUT=HERE/'positive_path_atomic_fnp_v1.json';assert not OUT.exists()
rng=np.random.default_rng(26951945);n=1024
C=rng.normal(size=(n,n))*.01;C=(C+C.T)/2;np.fill_diagonal(C,rng.uniform(.3,1,n))
g=rng.uniform(.02,.98,n);d=rng.uniform(.01,.3,n);eta=rng.normal(size=n)
K21=rng.normal(size=(n,n))*.003;K22=rng.normal(size=(n,n))*.002;radial=.05
B,E=factors(C,g,d,eta,K21,K22,radial);reference,_=positive_factor(B,E)
start=time.time()
with flops.BudgetContext(flop_budget=2**41,wall_time_limit_s=120,quiet=True) as ctx:
    args=[fnp.asarray(v) for v in (C,g,d,eta,K21,K22)]
    conversion=int(ctx.flops_used)
    value=positive_path_factor(*args,radial)
    cost=int(ctx.flops_used)-conversion
value=np.asarray(value)
error=float(np.max(np.abs(value-reference)));relative=float(np.linalg.norm(value-reference)/np.linalg.norm(reference))
assert error<1e-13 and relative<1e-12,(error,relative)
record=dict(width=n,function_flops=cost,input_conversion_flops=conversion,
            old_B_elementwise_flops=2*n*n,incremental_fifteen_layer_flops=15*(cost-2*n*n),
            fraction_of_rank35_bank_bill=15*(cost-2*n*n)/490645310609,
            numpy_max_abs_error=error,numpy_relative_error=relative,wall_s=time.time()-start,
            versions={k:importlib.metadata.version(k) for k in ('flopscope','whestbench','numpy')},
            source_sha256=hashlib.sha256((HERE/'positive_path_resummation_fnp_v1.py').read_bytes()).hexdigest(),
            qualification='Actual isolated full-width function arithmetic. Fifteen-layer incremental projection assumes unchanged selection/cost elsewhere; not a full estimator bill, live-memory, residual-time, or score gate.')
OUT.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2),flush=True)
