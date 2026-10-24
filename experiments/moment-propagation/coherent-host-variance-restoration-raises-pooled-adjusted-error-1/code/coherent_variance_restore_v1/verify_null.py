import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
build = read(HERE / 'build.json')
watch = read(HERE / 'null_watch.json')
assert watch['completed'] and watch['whole_group_retired'] and watch['max_swap_growth_bytes'] == 0
null = read(HERE / 'null/result.json')
parent = read(HERE.parent / 'lazy_rotation_v1/dev/result.json')
assert null['candidate_sha256'] == build['null_sha256'] == sha(HERE / 'null.py')
assert parent['candidate_sha256'] == build['parent_sha256']
assert [r['rank'] for r in null['records']] == [35, 35]
reference = parent['records'][0]
assert sha(Path(reference['output'])) == reference['output_sha256']
ref = np.load(reference['output'])
for r in null['records']:
    assert sha(Path(r['output'])) == r['output_sha256']
    assert r['weights_sha256'] == reference['weights_sha256']
    assert np.load(r['output']).tobytes() == ref.tobytes()
    assert r['weight5_cache']['numeric_cache_retired']
    assert r['max_rss_bytes'] < 8_000_000_000
assert null['repeat_byte_exact']
out = dict(passed=True, all16_parent_byte_exact=True, repeat_byte_exact=True,
           own_flops=[r['flops'] for r in null['records']],
           cold_extra_flops=null['records'][0]['flops']-reference['flops'],
           warm_extra_flops=null['records'][1]['flops']-parent['records'][2]['flops'])
with (HERE / 'null_verdict.json').open('x') as stream:
    json.dump(out, stream, indent=2)
print(json.dumps(out), flush=True)
