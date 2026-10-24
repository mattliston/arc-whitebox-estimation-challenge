"""One read-only profile of the frozen tiny Smirnov program."""
import cProfile
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pstats
import resource
import time
import numpy as np
import flopscope as flops
import flopscope.numpy as fnp
from smirnov_stream8_native_v1 import Product

HERE = Path(__file__).resolve().parent
OUT = HERE / 'smirnov_tiny_dispatch_profile_v1.json'
PROF = HERE / 'smirnov_tiny_dispatch_profile_v1.prof'
assert not OUT.exists() and not PROF.exists()
assert importlib.metadata.version('flopscope') == '0.12.0'
assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
assert os.environ['VECLIB_MAXIMUM_THREADS'] == '1'
kernel_file = HERE / 'smirnov_stream8_native_v1.py'
kernel_hash = hashlib.sha256(kernel_file.read_bytes()).hexdigest()
assert kernel_hash == '7c9ca40924410a7d5b6280c40bf500f78d9c935c831914e90047c549de430c74'
certificate = json.loads((HERE / 'smirnov_executable_transpose_certificate_v1.json').read_text())
assert certificate['passed']
primary = next(r for r in certificate['records'] if r['name'] == 'signed_executable_0')
kernel = Product(primary)
rng = np.random.default_rng(930066)
m, k, n = 216, 216, 432
aa = rng.normal(size=(m, k)).astype(np.float32) / np.float32(np.sqrt(k))
bb = rng.normal(size=(k, n)).astype(np.float32)
reference = aa.astype(np.float64) @ bb.astype(np.float64)
a, b = fnp.asarray(aa), fnp.asarray(bb)
out = fnp.empty((m, n), dtype=fnp.float32)
records = []
started = time.time()
profiler = cProfile.Profile()
for name in ('unprofiled_warmup', 'profiled_warm'):
    with flops.BudgetContext(flop_budget=2**36, wall_time_limit_s=60, quiet=True) as context:
        t0 = time.perf_counter()
        if name == 'profiled_warm':
            profiler.enable()
        kernel(a, b, out)
        if name == 'profiled_warm':
            profiler.disable()
        call_wall = time.perf_counter() - t0
    values = np.asarray(out).copy()
    error = float(np.linalg.norm(values.astype(np.float64) - reference) / np.linalg.norm(reference))
    finite = bool(np.isfinite(values).all())
    output_hash = hashlib.sha256(values.tobytes()).hexdigest()
    summary = context.summary_dict()
    record = dict(kind=name, call_wall_s=call_wall, relative_frobenius=error,
                  output_sha256=output_hash, finite=finite, context=summary)
    records.append(record)
    print(json.dumps({k:v for k,v in record.items() if k != 'context'}), flush=True)
    assert finite and error <= 2e-4
assert records[0]['output_sha256'] == records[1]['output_sha256']
profiler.dump_stats(str(PROF))
stats = pstats.Stats(profiler)
functions = []
for (path, line, name), (primitive, total, own, cumulative, callers) in stats.stats.items():
    functions.append(dict(path=path, line=line, name=name, primitive_calls=primitive,
                          total_calls=total, self_s=own, cumulative_s=cumulative))
functions.sort(key=lambda r: -r['self_s'])
kernel_functions = [r for r in functions if r['path'].endswith('smirnov_stream8_native_v1.py')]
arena_functions = [r for r in kernel_functions if r['name'] in ('allocate', 'release', 'padded')]
view_functions = [r for r in functions if 'flopscope' in r['path'] and
                  any(token in r['name'] for token in ('getitem', 'reshape', 'view'))]
rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
result = dict(passed=rss <= 2_000_000_000, shape=(m,k,n), records=records,
              kernel_functions=kernel_functions, arena_functions=arena_functions,
              view_functions=view_functions, top_self_functions=functions[:50],
              top_cumulative_functions=sorted(functions, key=lambda r: -r['cumulative_s'])[:35],
              profile_total_self_s=stats.total_tt,
              kernel_python_self_s=sum(r['self_s'] for r in kernel_functions),
              arena_python_self_s=sum(r['self_s'] for r in arena_functions),
              source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              frozen_kernel_sha256=kernel_hash,
              profile_sha256=hashlib.sha256(PROF.read_bytes()).hexdigest(),
              utc_finished=datetime.now(timezone.utc).isoformat(),
              wall_seconds=time.time()-started, peak_process_rss_bytes=rss,
              qualification='Diagnostic only: fixed tiny frozen program, one unprofiled cold-view warmup and one intrusive profiled warm call. No kernel changes, full geometry, targets, MPS, fit, estimator or performance claim. Cumulative profile function timings overlap; official operation timing buckets are separate. Existing full-size timing failure remains unchanged.')
OUT.write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in
                 ('records','kernel_functions','arena_functions','view_functions','top_self_functions','top_cumulative_functions')}, indent=2), flush=True)
