"""Authenticate exact Read5 outputs and the measured cache cost identity."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'read5_packet_v1'
read = lambda path: json.loads(Path(path).read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
build = read(HERE / 'build.json')
assert sha(HERE / 'candidate.py') == build['candidate_sha256']
assert sha(PARENT / 'candidate.py') == build['parent_sha256']
assert read(HERE / 'unit_v1/result.json')['passed']
rows = []
for name, expected_ranks in [('dev', (35, 100, 35)), ('cold100', (100,))]:
    watch = read(HERE / (name + '_watch.json'))
    assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
    a, b = read(HERE / name / 'result.json'), read(PARENT / name / 'result.json')
    assert a['candidate_sha256'] == build['candidate_sha256'] and not a['truth_accessed']
    assert b['candidate_sha256'] == build['parent_sha256']
    assert tuple(r['rank'] for r in a['records']) == expected_ranks
    for new, old in zip(a['records'], b['records']):
        assert new['rank'] == old['rank'] and new['weights_sha256'] == old['weights_sha256']
        assert new['includes_initialization'] == old['includes_initialization']
        assert sha(new['output']) == sha(old['output']) == new['output_sha256'] == old['output_sha256']
        cache = new['weight5_cache']
        assert cache['completed'] and cache['numeric_cache_retired']
        assert [r['layer'] for r in cache['layers']] == list(range(16))
        assert all(0 <= r['creators'] <= 1 for r in cache['layers'])
        assert cache['maximum_cache_bytes'] <= 68841472
        saving = cache['removed_flops'] - cache['copied_flops']
        assert saving > 0 and old['flops'] - new['flops'] == saving
        rows.append(dict(sequence=name, index=new['index'], rank=new['rank'],
                         all16_byte_exact=True, weights_sha256=new['weights_sha256'],
                         new_flops=new['flops'], parent_flops=old['flops'], saved_flops=saving,
                         flop_ratio=new['flops'] / old['flops'],
                         creators=cache['creators'], hits=cache['hits'],
                         copied_flops=cache['copied_flops'], omitted_flops=cache['removed_flops'],
                         source_counted_saved_primitives=30 * cache['hits'] - cache['creators'],
                         max_cache_bytes=cache['maximum_cache_bytes'],
                         new_native_wall_s=new['wall_s'], parent_native_wall_s=old['wall_s']))
dev = read(HERE / 'dev/result.json')['records']
cold100 = read(HERE / 'cold100/result.json')['records'][0]
assert dev[0]['output_sha256'] == dev[2]['output_sha256']
assert dev[1]['output_sha256'] == cold100['output_sha256']
receipt = dict(candidate_sha256=build['candidate_sha256'], rows=rows,
               repeat35_exact=True, warm_cold100_exact=True, all16_byte_exact=True,
               saved_work_identity_exact=True, native_gate_passed=True, truth_accessed=False)
with (HERE / 'native_comparison.json').open('x') as stream:
    json.dump(receipt, stream, indent=2)
print(json.dumps(receipt), flush=True)
