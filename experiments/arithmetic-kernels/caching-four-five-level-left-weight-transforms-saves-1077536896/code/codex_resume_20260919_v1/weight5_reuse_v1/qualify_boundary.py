"""Fixed exact client parity, full-context timing and actual request audit."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'read5_packet_v1'
read = lambda path: json.loads(Path(path).read_text())
build = read(HERE / 'build.json')
assert read(HERE / 'native_comparison.json')['native_gate_passed']
new = read(HERE / 'weight5_boundary/completed.json')
old = read(PARENT / 'read5_boundary/completed.json')
native = read(HERE / 'dev/result.json')
watch = read(HERE / 'boundary_watch.json')
assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
assert new['completed'] and new['candidate_sha256'] == native['candidate_sha256'] == build['candidate_sha256']
assert hashlib.sha256((HERE / 'candidate.py').read_bytes()).hexdigest() == build['candidate_sha256']
assert old['candidate_sha256'] == build['parent_sha256']
rows = []
for i, rank in enumerate((35, 100, 35)):
    a, b, n = new['records'][i], old['records'][i], native['records'][i]
    assert a['rank'] == b['rank'] == n['rank'] == rank
    result = np.load(HERE / 'weight5_boundary' / f'call_{i:02d}_rank{rank:04d}.npy')
    assert np.array_equal(result, np.load(n['output']))
    assert np.array_equal(result, np.load(PARENT / 'read5_boundary' / f'call_{i:02d}_rank{rank:04d}.npy'))
    assert a['summary']['flops_used'] == n['flops']
    context = a['summary']
    cache = n['weight5_cache']
    rows.append(dict(index=i, rank=rank, native_and_parent_byte_exact=True, native_flops_exact=True,
                     new_wall_s=a['wall_s'], parent_wall_s=b['wall_s'], wall_ratio=a['wall_s'] / b['wall_s'],
                     new_requests=a['requests'], parent_requests=b['requests'],
                     measured_saved_requests=b['requests'] - a['requests'],
                     source_counted_saved_primitives=30 * cache['hits'] - cache['creators'],
                     flops=context['flops_used'], backend_s=context['flopscope_backend_time_s'],
                     overhead_s=context['flopscope_overhead_time_s'], residual_s=context['residual_wall_time_s']))
assert np.array_equal(np.load(HERE / 'weight5_boundary/call_00_rank0035.npy'),
                      np.load(HERE / 'weight5_boundary/call_02_rank0035.npy'))
pooled_ratio = sum(r['new_wall_s'] for r in rows) / sum(r['parent_wall_s'] for r in rows)
runtime = all(r['new_wall_s'] < 45 for r in rows) and pooled_ratio <= .99
receipt = dict(rows=rows, candidate_sha256=build['candidate_sha256'],
               native_and_parent_byte_exact=True, native_flops_exact=True, repeat_byte_exact=True,
               numpy_free_client=True, watch=watch, pooled_wall_ratio=pooled_ratio,
               runtime_gate_passed=runtime, boundary_gate_passed=runtime,
               comparison='Saved same-day Read5 run; elapsed time is local evidence, not a platform guarantee.',
               truth_accessed=False)
with (HERE / 'boundary_comparison.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
