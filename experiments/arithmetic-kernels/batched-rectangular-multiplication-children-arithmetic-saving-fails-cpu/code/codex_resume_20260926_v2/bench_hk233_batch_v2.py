"""Fixed kernel-only screen: same algebra, fifteen sequential vs batched children."""
from pathlib import Path
import gc
import hashlib
import json
import statistics
import sys
import time
import types
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp

HERE = Path(__file__).resolve().parent
H1 = HERE.parent / 'codex_resume_20260926_v1'
dest = HERE / 'hk233_batch_bench_v2.json'
assert not dest.exists()
stub = types.ModuleType('whestbench'); stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
source = HERE / 'candidate_old_both_tail_trace_bounded_v6.py'
assert hashlib.sha256(source.read_bytes()).hexdigest() == '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
mod = types.ModuleType('hkb_bench'); mod.__file__ = str(source)
sys.modules[mod.__name__] = mod
exec(compile(source.read_text(), str(source), 'exec'), mod.__dict__)
exec((H1 / 'hk233_fragment_v1.py').read_text(), mod.__dict__)
helper = HERE / 'hk233_batch_fragment_v2.py'
exec(helper.read_text(), mod.__dict__)
flops.configure(symmetry_warnings=False)

# All scalar tensor basis pairs verify batching/unpacking and all outputs.
testenv = {'fnp': np}
exec(helper.read_text().split('# Dedicated view caches')[0], testenv)
plain = lambda pairs: [a @ b for a, b in pairs]
for i in range(6):
    for j in range(9):
        a, b = np.zeros((2,3)), np.zeros((3,3))
        a.flat[i] = 1; b.flat[j] = 1
        assert np.array_equal(testenv['_hkb233_product'](a,b,plain,1), a@b)
rng = np.random.default_rng(20260926)
a, b = rng.normal(size=(21,34)), rng.normal(size=(34,19))
with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
    got = mod._hkb233_product(fnp.asarray(a), fnp.asarray(b),
                             lambda pairs:[x@y for x,y in pairs],4)
err = float(np.max(np.abs(np.asarray(got) - a@b)))
assert err < 1e-12, err
del a,b,got,testenv

est=mod.Estimator()
t0=time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
    api_version='2', scratch_dir=None, submission_dir=None, seed=0))
setup_s=time.monotonic()-t0
def direct6(a,b):
    m,k,n=int(a.shape[0]),int(a.shape[1]),int(b.shape[1])
    assert mod._ipt_fits(6,m,k,n)
    mod._TR_LAST[0]=None
    return mod._ipt_product_ab(a,b,fnp.empty((m,n),dtype=fnp.float32),6)

prior=json.loads((H1/'hk233_bench_v3.json').read_text())
shapes=[(576,8576,1024),(768,11776,896),(640,8192,960),(896,9984,960)]
rows=[]; summaries=[]
for m,k,n in shapes:
    orient=min(('hk6','hkt6'),key=lambda method:statistics.mean(
        r['flops'] for r in prior if r['shape']==[m,k,n] and r['method']==method and not r['cold']))
    aa=rng.normal(size=(m,k)).astype(np.float32)
    bb=rng.normal(size=(k,n)).astype(np.float32)
    ref=aa.astype(np.float64)@bb.astype(np.float64)
    norm=np.linalg.norm(ref)
    a,b=fnp.asarray(aa),fnp.asarray(bb)
    del aa,bb
    local=[]
    for rep in range(3):
        methods=('parent','sequential','batch') if rep%2==0 else ('batch','sequential','parent')
        for method in methods:
            mod._TR_LAST[0]=None
            mod._LVF_EAGER[0]=rep==0
            t0=time.process_time()
            with flops.BudgetContext(flop_budget=2**43,quiet=True) as ctx:
                if method=='parent':
                    product=mod._shared_old384_mm(a,b)
                else:
                    x,y=(a,b) if orient=='hk6' else (b.T,a.T)
                    product=(mod._hk233_product(x,y,direct6,64) if method=='sequential'
                             else mod._hkb233_product(x,y,mod._hkb6_many,64))
                    if orient=='hkt6': product=product.T
            cpu=time.process_time()-t0
            mod._LVF_EAGER[0]=False
            got=np.asarray(product,dtype=np.float64)
            row=dict(shape=[m,k,n],orientation=orient,method=method,rep=rep,cold=rep==0,
                flops=int(ctx.flops_used),calls=len(ctx.op_log),cpu_s=cpu,
                backend_s=ctx.flopscope_backend_time_s,
                relative_error=float(np.linalg.norm(got-ref)/norm),
                max_abs_error=float(np.max(np.abs(got-ref))),
                output_sha256=hashlib.sha256(got.tobytes()).hexdigest())
            assert row['relative_error'] < .002, row
            local.append(row); rows.append(row); print(json.dumps(row),flush=True)
            del product,got
    by={name:[r for r in local if r['method']==name and not r['cold']]
        for name in ('parent','sequential','batch')}
    info=dict(shape=[m,k,n],orientation=orient,
        parent_bill=statistics.mean(r['flops'] for r in by['parent']),
        sequential_bill=statistics.mean(r['flops'] for r in by['sequential']),
        batch_bill=statistics.mean(r['flops'] for r in by['batch']),
        parent_cpu=statistics.median(r['cpu_s'] for r in by['parent']),
        sequential_cpu=statistics.median(r['cpu_s'] for r in by['sequential']),
        batch_cpu=statistics.median(r['cpu_s'] for r in by['batch']),
        parent_calls=statistics.mean(r['calls'] for r in by['parent']),
        sequential_calls=statistics.mean(r['calls'] for r in by['sequential']),
        batch_calls=statistics.mean(r['calls'] for r in by['batch']))
    summaries.append(info); print(json.dumps(dict(summary=info)),flush=True)
    del a,b,ref
    mod._TR_LAST[0]=None;gc.collect()
cpu_seq=sum(r['batch_cpu'] for r in summaries)/sum(r['sequential_cpu'] for r in summaries)
cpu_parent=sum(r['batch_cpu'] for r in summaries)/sum(r['parent_cpu'] for r in summaries)
passed=(all(r['batch_bill']<=r['sequential_bill'] and r['batch_bill']<r['parent_bill'] for r in summaries)
        and cpu_seq<=.8 and cpu_parent<2.)
result=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest(),setup_s=setup_s,
    tiny_basis_cases=54,tiny_native_fp64_max_abs=err,rows=rows,summaries=summaries,
    aggregate_batch_cpu_vs_sequential=cpu_seq,aggregate_batch_cpu_vs_parent=cpu_parent,
    primitive_gate_passed=passed,scope='No network/target access; primitive-only, no estimator gain')
with dest.open('x') as stream:json.dump(result,stream,indent=2,allow_nan=False)
print(json.dumps(dict(primitive_gate_passed=passed,cpu_vs_seq=cpu_seq,cpu_vs_parent=cpu_parent)),flush=True)
