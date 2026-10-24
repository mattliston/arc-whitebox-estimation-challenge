"""Exact production-client parity, billing and fixed runtime qualification."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
D = HERE.parent
read = lambda p: json.loads(Path(p).read_text())
new = read(HERE / 'read5_boundary/completed.json')
old = read(D / 'cost75_boundary/completed.json')
native = read(HERE / 'dev/result.json')
watch = read(HERE / 'boundary_watch.json')
assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
expected = read(HERE / 'build.json')['candidate_sha256']
assert new['completed'] and new['candidate_sha256'] == native['candidate_sha256'] == expected
assert hashlib.sha256((HERE / 'candidate.py').read_bytes()).hexdigest() == expected
rows = []
for i, rank in enumerate((35, 100, 35)):
    a, b, n = new['records'][i], old['records'][i], native['records'][i]
    assert a['rank'] == b['rank'] == n['rank'] == rank
    result = np.load(HERE / 'read5_boundary' / f'call_{i:02d}_rank{rank:04d}.npy')
    assert np.array_equal(result, np.load(n['output']))
    assert a['summary']['flops_used'] == n['flops']
    context = a['summary']
    backend = sum(o.get('flopscope_backend_time_s', 0) for o in context['operations'].values())
    overhead = sum(o.get('flopscope_overhead_time_s', 0) for o in context['operations'].values())
    rows.append(dict(index=i, rank=rank, native_byte_exact=True, native_flops_exact=True,
                     new_wall_s=a['wall_s'], parent_wall_s=b['wall_s'], wall_ratio=a['wall_s']/b['wall_s'],
                     new_requests=a['requests'], parent_requests=b['requests'],
                     flops=context['flops_used'], backend_s=backend, overhead_s=overhead,
                     residual_s=a['wall_s']-backend-overhead))
assert np.array_equal(np.load(HERE / 'read5_boundary/call_00_rank0035.npy'),
                      np.load(HERE / 'read5_boundary/call_02_rank0035.npy'))
pooled_ratio = sum(r['new_wall_s'] for r in rows)/sum(r['parent_wall_s'] for r in rows)
runtime_passed = all(r['new_wall_s'] <= 45 for r in rows) and pooled_ratio <= 1.10
rec = dict(rows=rows, candidate_sha256=expected, native_byte_exact=True, native_flops_exact=True,
           repeat_byte_exact=True, numpy_free_client=True, watch=watch,
           pooled_wall_ratio=pooled_ratio, runtime_gate_passed=runtime_passed,
           boundary_gate_passed=runtime_passed,
           comparison='Against saved same-day Cost75 client run. Local wall time is not a platform guarantee.')
with (HERE / 'boundary_comparison.json').open('x') as stream:
    json.dump(rec, stream, indent=2)
print(json.dumps(rec), flush=True)
