"""Fixed native geometry, small qualification, bank and1/4-thread timing."""
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import resource
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from threadpoolctl import threadpool_limits
from smirnov_stream8_native_v1 import Product
from packed_arena_generic import layout

HERE = Path(__file__).resolve().parent
OUT = HERE/'smirnov_stream8_native_benchmark_v1.json'
assert not OUT.exists()
assert importlib.metadata.version('flopscope') == '0.12.0'
certificate = json.loads((HERE/'smirnov_executable_transpose_certificate_v1.json').read_text())
assert certificate['passed']
primary = next(r for r in certificate['records'] if r['name'] == 'signed_executable_0')
assert json.loads((HERE/'executable_border_padding_envelope_v1.json').read_text())['stage1_twenty_percent_gate']
records = []
started = time.time()
rng = np.random.default_rng(930066)
kernel = Product(primary)


def run_product(a, b, out, shape, kind, call, repeat, threads):
    with flops.BudgetContext(flop_budget=2**41, wall_time_limit_s=120, quiet=True) as ctx:
        t0 = time.perf_counter()
        call(a, b, out)
        seconds = time.perf_counter()-t0
    record = dict(kind=kind, shape=shape, repeat=repeat, threads=threads,
                  wall_s=seconds, flops=int(ctx.flops_used))
    print(json.dumps(record), flush=True)
    return record


def save(passed, stage):
    result = dict(passed=passed, stage=stage, records=records,
                  reserved_arena_bytes=kernel.arena.capacity*4,
                  dedicated_padding_bytes=kernel.arena.padding_bytes,
                  retained_workspace_bytes=kernel.arena.capacity*4+kernel.arena.padding_bytes,
                  peak_live_arena_bytes=kernel.arena.peak_live*4,
                  source_sha256=hashlib.sha256((HERE/'smirnov_stream8_native_v1.py').read_bytes()).hexdigest(),
                  wall_s=time.time()-started,
                  peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                  qualification='Fixed prototype only: iid matrices, official-version native FNP, exact same bankD4 product, one/four BLAS threads, cold/warm. Not structured CP accuracy, a complete candidate/14-net score, official service timing or persistent total/residual-wall gate.')
    OUT.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}, indent=2), flush=True)


with threadpool_limits(limits=1):
    m, k, n = shape = (216, 216, 432)
    aa = rng.normal(size=(m, k)).astype(np.float32)/np.float32(np.sqrt(k))
    bb = rng.normal(size=(k, n)).astype(np.float32)
    exact = aa.astype(np.float64)@bb.astype(np.float64)
    a, b = fnp.asarray(aa), fnp.asarray(bb)
    out = fnp.empty((m, n), dtype=fnp.float32)
    row = run_product(a, b, out, shape, 'small_certificate', kernel, 0, 1)
    error = np.asarray(out).astype(np.float64)-exact
    row['relative_frobenius'] = float(np.linalg.norm(error)/np.linalg.norm(exact))
    row['max_abs'] = float(np.max(np.abs(error)))
    row['passed'] = np.isfinite(error).all().item() and row['relative_frobenius'] <= 2e-4
    records.append(row)
    if not row['passed']:
        save(False, 'small_numeric_gate')
        raise SystemExit(0)

# Load only the known bank product helpers, not estimator/model literals.
path = HERE/'candidate_fullpair_runtime_rectd4_t8192_bulkcopies_v1.py'
source = path.read_text()
assert hashlib.sha256(source.encode()).hexdigest() == '3b7f55f9a814789743467bb4bcda0837c3c1aad15d120e4cdab632594bf54757'
lines = source.splitlines(keepends=True)
funcs = {n.name: ''.join(lines[n.lineno-1:n.end_lineno]) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
env = dict(fnp=fnp, _PACKED_FRAME_CACHE={})
for name in ('_cp3_bind_arena_views', '_packed_frame', '_packed_product_cached', '_packed_product_4'):
    exec(funcs[name], env)
size, specs, _ = layout(1024, 1024, 8192, 4)
bank_arena = fnp.empty(size, dtype=fnp.float32)
bank_views = env['_cp3_bind_arena_views'](bank_arena, specs)
bank_call = lambda a, b, out: env['_packed_product_4'](a, b, out, bank_views)
m, k, n = shape = (1024, 832, 8192)
aa = rng.normal(size=(m, k)).astype(np.float32)/np.float32(np.sqrt(k))
bb = rng.normal(size=(k, n)).astype(np.float32)
with threadpool_limits(limits=4):
    exact = aa.astype(np.float64)@bb.astype(np.float64)
a, b = fnp.asarray(aa), fnp.asarray(bb)
out = fnp.empty((m, n), dtype=fnp.float32)
for threads in (1, 4):
    with threadpool_limits(limits=threads):
        for kind, call in (('bankD4', bank_call), ('smirnov_stream8', kernel)):
            for repeat in (0, 1):
                row = run_product(a, b, out, shape, kind, call, repeat, threads)
                error = np.asarray(out).astype(np.float64)-exact
                row['relative_frobenius'] = float(np.linalg.norm(error)/np.linalg.norm(exact))
                row['max_abs'] = float(np.max(np.abs(error)))
                row['finite'] = bool(np.isfinite(error).all())
                records.append(row)
                if kind == 'bankD4':
                    assert row['flops'] == 8417538304
warm = {(r['kind'], r['threads']): r for r in records if r['repeat'] == 1}
for threads in (1, 4):
    trial, bank = warm['smirnov_stream8', threads], warm['bankD4', threads]
    trial['warm_bank_time_ratio'] = trial['wall_s']/bank['wall_s']
    trial['bank_flop_ratio'] = trial['flops']/bank['flops']
passed = (all(r.get('finite', True) and r['relative_frobenius'] <= 2e-4 for r in records) and
          all(warm['smirnov_stream8', t]['warm_bank_time_ratio'] <= 3 for t in (1, 4)) and
          kernel.arena.capacity*4+kernel.arena.padding_bytes <= 1168900096)
save(passed, 'full_fixed_product')
