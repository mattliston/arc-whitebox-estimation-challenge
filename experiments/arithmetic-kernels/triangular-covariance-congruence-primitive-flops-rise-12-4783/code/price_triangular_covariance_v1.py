"""Optimistic SPD-only covariance primitive price; not a candidate."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_info

HERE=Path(__file__).resolve().parent
parent=HERE/'candidate_old_both_tail_trace_bounded_v6.py'
assert hashlib.sha256(parent.read_bytes()).hexdigest()=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
assert all(x['num_threads']==1 for x in threadpool_info())
stub=types.ModuleType('whestbench');stub.BaseEstimator=type('BaseEstimator',(),{});sys.modules['whestbench']=stub
mod=types.ModuleType('price_parent');mod.__file__=str(parent);sys.modules[mod.__name__]=mod
exec(compile(parent.read_text(),str(parent),'exec'),mod.__dict__)
est=mod.Estimator()
est.setup(types.SimpleNamespace(width=1024,depth=16,flop_budget=2**41,
    api_version='2',scratch_dir=None,submission_dir=None,seed=0))
assert mod._W5_PORT_SCOPE is None
rng=np.random.default_rng(20260926)
A=rng.normal(size=(1024,1024)).astype(np.float32)/32
C0=(A@A.T+np.eye(1024,dtype=np.float32)*.5).astype(np.float32)
W0=(rng.normal(size=(1024,1024))/32).astype(np.float32)
ref=W0.astype(np.float64).T@C0.astype(np.float64)@W0.astype(np.float64)
del A

def upper(a,b):
    n=int(a.shape[0])
    if n<=64:return mod._mm(a,b)
    h=n//2
    top=upper(a[:h,:h],b[:h,:])+mod._mm(a[:h,h:],b[h:,:])
    bottom=upper(a[h:,h:],b[h:,:])
    return fnp.concatenate([top,bottom],axis=0)

def gram(u):
    left=u[:,:512];right=u[:,512:]
    out=fnp.empty((1024,1024),dtype=fnp.float32)
    fnp.copyto(out[:512,:512],fnp.einsum('ki,kj->ij',left,left,optimize=False))
    cross=mod._mm(left.T,right)
    fnp.copyto(out[:512,512:],cross)
    fnp.copyto(out[512:,:512],cross.T)
    fnp.copyto(out[512:,512:],fnp.einsum('ki,kj->ij',right,right,optimize=False))
    return out

with flops.BudgetContext(flop_budget=2**40,quiet=True) as ctx:
    C=fnp.asarray(C0,dtype=fnp.float32);W=fnp.asarray(W0,dtype=fnp.float32)
    methods={'parent':lambda:mod._sym_upper(W.T,mod._mm(C,W)),
             'triangular_cholesky':lambda:gram(upper(fnp.linalg.cholesky(C).T,W))}
    records={}
    for name,fn in methods.items():
        warm=fn();del warm
        seen=[];bills=[];times=[];cpus=[]
        for _ in range(2):
            before=int(ctx.summary_dict()['flops_used'])
            t=time.monotonic();cpu=time.process_time()
            value=fn()
            cpus.append(time.process_time()-cpu);times.append(time.monotonic()-t)
            bills.append(int(ctx.summary_dict()['flops_used'])-before)
            seen.append(np.asarray(value).copy());del value
        assert bills[0]==bills[1],(name,bills)
        assert seen[0].tobytes()==seen[1].tobytes()
        error=float(np.linalg.norm(seen[0].astype(np.float64)-ref)/np.linalg.norm(ref))
        assert error<1e-4,(name,error)
        records[name]=dict(flops=bills[0],wall_s=times,cpu_s=cpus,
                           relative_reference_error=error,repeated_bill_and_bytes_exact=True)
        del seen
ratio=records['triangular_cholesky']['flops']/records['parent']['flops']
record=dict(created_utc=datetime.now(timezone.utc).isoformat(),scope=__doc__,
            parent_sha256=hashlib.sha256(parent.read_bytes()).hexdigest(),methods=records,
            proposed_over_parent_flops=ratio,optimistic_primitive_saving_pct=100*(1-ratio),
            advance_to_current_state_feasibility=bool(ratio<=.75),
            limitations='Synthetic SPD only, isolated warm primitives, no current W5 prediction scope or PSD handling. No network/target access, candidate, full bill or score.',
            guard_expected='7.5GB, one BLAS thread, serial')
with (HERE/'triangular_covariance_price_v1.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
