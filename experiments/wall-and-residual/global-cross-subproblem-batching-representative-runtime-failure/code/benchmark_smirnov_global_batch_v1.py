"""One frozen representative FNP-only product test, with shared scratch."""
import ast
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from packed_arena_generic import layout
from smirnov_stream8_native_v1 import Arena, Product, prepare
from smirnov_global_batch_native_v1 import NativeGlobalBatch

HERE = Path(__file__).resolve().parent
OUT = HERE/'smirnov_global_batch_benchmark_v1.json'
assert not OUT.exists()
assert importlib.metadata.version('flopscope')=='0.12.0'
assert os.environ['OPENBLAS_NUM_THREADS']==os.environ['VECLIB_MAXIMUM_THREADS']=='1'
static = json.loads((HERE/'smirnov_global_batch_static_v2.json').read_text())
assert static['prerequisite_pass'] and static['static_memory_pass']
trial_plan = static['records'][-1]
placement = trial_plan['placement']
arena_elements = trial_plan['largest_first_arena_bytes']//4
primary = next(row for row in json.loads((HERE/'smirnov_executable_transpose_certificate_v1.json').read_text())['records']
               if row['name']=='signed_executable_0')
assert primary['passed']

rng = np.random.default_rng(930066)
rng.normal(size=(216,216));rng.normal(size=(216,432))
m,k,n = 1024,832,8192
aa = rng.normal(size=(m,k)).astype(np.float32)/np.float32(np.sqrt(k))
bb = rng.normal(size=(k,n)).astype(np.float32)
reference = aa.astype(np.float64)@bb.astype(np.float64)
a,b = fnp.asarray(aa),fnp.asarray(bb)
out = fnp.empty((m,n),dtype=fnp.float32)
shared = fnp.empty(arena_elements,dtype=fnp.float32)
records=[]
started=time.time()


def call(kind,repetition,operation,expected_flops,frozen_output=None):
    with flops.BudgetContext(flop_budget=2**39,wall_time_limit_s=60,quiet=True) as context:
        t0=time.perf_counter()
        operation(a,b,out)
        wall=time.perf_counter()-t0
    value=np.asarray(out).copy()
    error=float(np.linalg.norm(value.astype(np.float64)-reference)/np.linalg.norm(reference))
    row=dict(kind=kind,repetition=repetition,wall_s=wall,flops=int(context.flops_used),
             expected_flops=expected_flops,bill_pass=context.flops_used==expected_flops,
             relative_frobenius=error,finite=bool(np.isfinite(value).all()),
             output_sha256=hashlib.sha256(value.tobytes()).hexdigest(),
             context=context.summary_dict())
    if frozen_output is not None:
        row['relative_to_frozen']=float(np.linalg.norm(value.astype(np.float64)-frozen_output.astype(np.float64))/np.linalg.norm(frozen_output.astype(np.float64)))
        row['byte_equal_frozen']=bool(np.array_equal(value,frozen_output))
    records.append(row)
    print(json.dumps({key:val for key,val in row.items() if key!='context'}),flush=True)
    assert row['bill_pass'] and row['finite']
    return value


# Same bank D4 helpers and same generic layout used by the frozen benchmark;
# scratch storage is shared sequentially with the other algorithms.
bank_source=(HERE/'candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py').read_text()
assert hashlib.sha256(bank_source.encode()).hexdigest()=='3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757'
lines=bank_source.splitlines(keepends=True)
functions={node.name:''.join(lines[node.lineno-1:node.end_lineno]) for node in ast.parse(bank_source).body if isinstance(node,ast.FunctionDef)}
env=dict(fnp=fnp,_PACKED_FRAME_CACHE={})
for name in ('_cp3_bind_arena_views','_packed_frame','_packed_product_cached','_packed_product_4'):
    exec(functions[name],env)
bank_size,specs,_=layout(1024,1024,8192,4)
assert bank_size<=arena_elements
views=env['_cp3_bind_arena_views'](shared,specs)
bank=lambda x,y,z:env['_packed_product_4'](x,y,z,views)
for repetition in ('cold','warm'):
    call('bankD4',repetition,bank,8417538304)

# Avoid allocating a second arena merely to obtain the frozen numerical
# reference. This is setup-equivalent to Arena.__init__, sharing the buffer.
frozen=Product.__new__(Product)
frozen.circuits={key:prepare(primary['circuits'][key]) for key in ('L','R','P')}
frozen.arena=Arena.__new__(Arena)
frozen.arena.capacity=2**28
frozen.arena.array=shared
frozen.arena.free=[(0,2**28)]
frozen.arena.owned={};frozen.arena.views={};frozen.arena.padding={}
frozen.arena.padding_bytes=0;frozen.arena.live=0;frozen.arena.peak_live=0
frozen_value=call('frozen_smirnov','cold',frozen,6834384092)
del frozen
gc.collect()

kernel=NativeGlobalBatch(primary,shared,placement)
for repetition,cost in (('cold',7258335196),('warm',6611249012)):
    call('global_batch',repetition,kernel,cost,frozen_value)
retained=kernel.arena.capacity*4+kernel.arena.padding_bytes
rss=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
ratios={}
for repetition,bound in (('cold',1.5),('warm',1.2)):
    base=next(row for row in records if row['kind']=='bankD4' and row['repetition']==repetition)
    trial=next(row for row in records if row['kind']=='global_batch' and row['repetition']==repetition)
    ratios[repetition]=dict(ratio=trial['wall_s']/base['wall_s'],threshold=bound,
                            passed=trial['wall_s']<=bound*base['wall_s'])
trial_rows=[row for row in records if row['kind']=='global_batch']
numeric=all(row['relative_frobenius']<=2e-4 and row['relative_to_frozen']<=1e-10 for row in trial_rows)
memory=retained<=1168900096 and rss<=2_000_000_000
result=dict(passed=numeric and memory and all(row['passed'] for row in ratios.values()),
            numerical_pass=numeric,memory_pass=memory,timing=ratios,records=records,
            retained_workspace_bytes=retained,arena_bytes=kernel.arena.capacity*4,
            dedicated_padding_bytes=kernel.arena.padding_bytes,
            peak_live_arena_bytes=kernel.arena.peak_live*4,peak_process_rss_bytes=rss,
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            kernel_source_sha256=hashlib.sha256((HERE/'smirnov_global_batch_native_v1.py').read_bytes()).hexdigest(),
            utc_finished=datetime.now(timezone.utc).isoformat(),wall_seconds=time.time()-started,
            qualification='Exactly one fixed representative test under concurrent root MPS training, one BLAS thread, shared sequential scratch. No target/network data or MPS used by this worker. Timing is not isolated service timing; success would still require structured CP, complete candidate bills/panels and persistent total/residual-wall plus memory gates. Failure does not authorize a sweep.')
OUT.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({key:val for key,val in result.items() if key!='records'},indent=2),flush=True)
