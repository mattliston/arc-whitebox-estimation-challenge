"""Meter complete read primitives in the native runtime; no network labels."""
from pathlib import Path
import gc
import hashlib
import json
import resource
import statistics
import sys
import time
import types
import numpy as np
import flopscope as flops

D = Path(__file__).resolve().parent
H = D.parent
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
seal = json.loads((D/'source_seal.json').read_text())
for name, value in seal['files'].items():
    assert sha(D/name) == value
parent = H/'candidate_old_both_tail_trace_bounded_v6.py'
assert sha(parent) == seal['parent_sha256']
stub = types.ModuleType('whestbench')
stub.BaseEstimator = type('BaseEstimator', (), {})
sys.modules['whestbench'] = stub
mod = types.ModuleType('depth7_read_bench')
mod.__file__ = str(parent)
sys.modules[mod.__name__] = mod
exec(compile(parent.read_text(), str(parent), 'exec'), mod.__dict__)
exec(compile((D/'fragment.py').read_text(), str(D/'fragment.py'), 'exec'), mod.__dict__)
candidate_ks = mod._ks_product
base_ks = mod._D7R_OLD_KS
est = mod.Estimator()
start = time.monotonic()
est.setup(types.SimpleNamespace(width=1024, depth=16, flop_budget=2**41,
    api_version='2', scratch_dir=None, submission_dir=None, seed=0))
setup = time.monotonic()-start
assert sum(mod._D7R_REGION) <= int(mod._SCRATCH[0].shape[0])
np_rng = np.random.default_rng(2026092714)


def once(path, data):
    with path.open('x') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')


fixtures = []
for m,k,n,M,K,N,sizes in [(16,28,20,16,28,20,(12,8,8)),
                          (13,27,17,16,28,20,(12,8,8)),
                          (20,5,12,20,8,12,(4,4))]:
    a = np_rng.integers(-2, 3, (m,k)).astype(np.float32)
    b = np_rng.integers(-2, 3, (k,n)).astype(np.float32)
    region = (49*(M//4)*(max(sizes)//4), 49*(max(sizes)//4)*(N//4),
              2*49*(M//4)*(N//4))
    with flops.BudgetContext(flop_budget=2**41, quiet=True) as ctx:
        aa, bb = mod.fnp.asarray(a), mod.fnp.asarray(b)
        out = mod.fnp.empty((m,n), dtype=mod.fnp.float32)
        mod._d7r_product(aa,bb,out,(M,K,N,sizes),L=2,region=region)
    result = np.asarray(out, dtype=np.float64)
    truth = a.astype(np.float64) @ b.astype(np.float64)
    assert np.array_equal(result, truth)
    assert mod._TR_LAST[0] is None
    fixtures.append(dict(shape=[m,k,n], padded=[M,K,N], chunks=sizes, exact=True,
                         flops=int(ctx.summary_dict()['flops_used'])))
once(D/'qualification.json',dict(passed=True,fixtures=fixtures,
    scope='Exact integer reduced-depth composition and padding; production depth checked below',
    source_seal_sha256=sha(D/'source_seal.json')))
print(json.dumps(dict(event='qualified', fixtures=len(fixtures))),flush=True)

shapes = [(1024,4096,1024),(864,9984,1024),(736,8768,896),(640,7360,864)]
records = []
for shape in shapes:
    m,k,n = shape
    a = np_rng.standard_normal((m,k)).astype(np.float32)
    b = np_rng.standard_normal((k,n)).astype(np.float32)
    truth = a.astype(np.float64) @ b.astype(np.float64)
    truth_norm = float(np.linalg.norm(truth))
    plan6 = mod._ks_plan(m,k,n)
    assert plan6 is not None
    plan7 = mod._d7r_plan(m,k,n,plan6)
    assert plan7 is not None, shape
    with flops.BudgetContext(flop_budget=2**41, quiet=True):
        aa, bb = mod.fnp.asarray(a), mod.fnp.asarray(b)
    runs = {'base': [], 'candidate': []}
    cold = {}
    values = {}
    for repetition in [-1,0,1,2]:
        for mode in (('base','candidate') if repetition % 2 == 0 else ('candidate','base')):
            mod._ks_product = base_ks if mode == 'base' else candidate_ks
            mod._TR_LAST[0] = None
            calls_before = dict(mod._D7R_STATS)
            with flops.BudgetContext(flop_budget=2**41,quiet=True) as ctx:
                t0 = time.monotonic()
                c0 = time.process_time()
                out = mod._mm(aa,bb)
                cpu = time.process_time()-c0
                wall = time.monotonic()-t0
            value = np.asarray(out,dtype=np.float64)
            assert value.shape == (m,n) and np.isfinite(value).all()
            if mode in values:
                assert np.array_equal(values[mode],value), (shape,mode,repetition)
            else:
                values[mode] = value.copy()
            summary = ctx.summary_dict()
            ops = summary['operations']
            record = dict(flops=int(summary['flops_used']),cpu_s=cpu,wall_s=wall,
                backend_s=sum(float(v.get('flopscope_backend_time_s',0)) for v in ops.values()),
                operations=sum(int(v.get('calls',v.get('count',0))) for v in ops.values()),
                relative_error=float(np.linalg.norm(value-truth)/truth_norm),
                changed_read_calls=mod._D7R_STATS['calls']-calls_before['calls'],
                changed_chunks=mod._D7R_STATS['chunks']-calls_before['chunks'])
            assert record['changed_read_calls'] == (1 if mode=='candidate' else 0)
            if repetition < 0:
                cold[mode] = record
            else:
                runs[mode].append(record)
            del out,value
    aggregate = {}
    for mode in runs:
        assert len({r['flops'] for r in runs[mode]}) == 1
        aggregate[mode] = {key: statistics.median(r[key] for r in runs[mode])
                           for key in ['flops','cpu_s','wall_s','backend_s','operations','relative_error']}
    ratios = {key: aggregate['candidate'][key]/aggregate['base'][key]
              for key in ['flops','cpu_s','wall_s','backend_s','relative_error']}
    row = dict(shape=shape,parent_plan=plan6,candidate_plan=plan7,cold=cold,
               runs=runs,aggregate=aggregate,ratios=ratios)
    records.append(row)
    print(json.dumps(dict(event='shape',shape=shape,ratios=ratios)),flush=True)
    del a,b,aa,bb,truth,values
    gc.collect()
mod._ks_product = candidate_ks
bill_ratio = sum(r['aggregate']['candidate']['flops'] for r in records)/sum(r['aggregate']['base']['flops'] for r in records)
cpu_ratio = sum(r['aggregate']['candidate']['cpu_s'] for r in records)/sum(r['aggregate']['base']['cpu_s'] for r in records)
gates = dict(each_bill_better=all(r['ratios']['flops']<1 for r in records),
    aggregate_bill=bill_ratio <= .97,cpu=cpu_ratio<2,
    relative_error=all(r['aggregate']['candidate']['relative_error']<2e-4 and r['ratios']['relative_error']<=4 for r in records))
result = dict(scope='Four fixed synthetic primitive shapes; not a whole-estimator gain or platform timing',
    source_seal_sha256=sha(D/'source_seal.json'),records=records,gates=gates,
    passes_primitive_gate=all(gates.values()),aggregate_bill_ratio=bill_ratio,
    aggregate_cpu_ratio=cpu_ratio,all_repeated_predictions_exact=True,
    existing_arena_floats=int(mod._SCRATCH[0].shape[0]),new_region=mod._D7R_REGION,
    extra_arena_bytes=0,setup_s=setup,elapsed_s=time.monotonic()-start,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    training_targets_accessed=False,valid40_accessed=False)
once(D/'bench_result.json',result)
print(json.dumps({k:result[k] for k in ['gates','passes_primitive_gate','aggregate_bill_ratio','aggregate_cpu_ratio','elapsed_s']},indent=2),flush=True)
