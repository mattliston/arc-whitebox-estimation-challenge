import numpy as np
from common import HERE, CONTROLS, read, write, pin, verify, sha, seal_check

seal_check()
build = read(HERE/'build.json')
unit = read(HERE/'unit.json')
assert unit['passed']
saved = read(CONTROLS/'closeout.json')
saved_pins = {p['path']:p for p in saved['artifact_pins']}
removed = unit['expected_full_call_removed_flops']
rows = []
for stage, parent_stage, field, expected in [
    ('disabled35','parent_triple','disabled',0),
    ('enabled_triple','parent_triple','candidate',removed),
    ('enabled_cold100','parent_cold100','candidate',removed)]:
    own = read(HERE/stage/'result.json')
    parent_path = CONTROLS/parent_stage/'result.json'
    verify(saved_pins[str(parent_path)])
    parent = read(parent_path)
    assert own['candidate_sha256'] == build[field]['sha256']
    assert parent['candidate_sha256'] == build['parent']['sha256']
    assert not own['truth_accessed'] and not parent['truth_accessed']
    parent_rows = parent['records'][:1] if stage=='disabled35' else parent['records']
    assert len(own['records']) == len(parent_rows)
    for a,b in zip(own['records'],parent_rows):
        verify(saved_pins[b['output']])
        x = np.load(a['output']); y = np.load(b['output'])
        assert a['rank'] == b['rank'] and a['includes_initialization'] == b['includes_initialization']
        assert x.shape == y.shape == (16,1024) and x.dtype == y.dtype == np.float32
        assert x.tobytes() == y.tobytes(), (stage,a['rank'],float(np.max(np.abs(x-y))))
        assert np.isfinite(x).all() and (x>=0).all()
        assert b['flops']-a['flops'] == expected, (stage,a['flops'],b['flops'],expected)
        assert a['weight5_cache']['numeric_cache_retired']
        rows.append(dict(stage=stage,rank=a['rank'],index=a['index'],all16_bytes_exact=True,
            parent_flops=b['flops'],candidate_flops=a['flops'],removed_flops=expected,
            own=pin(a['output']),parent=pin(b['output'])))
dev = read(HERE/'enabled_triple/result.json')['records']
cold = read(HERE/'enabled_cold100/result.json')['records'][0]
assert sha(dev[0]['output']) == sha(dev[2]['output'])
assert sha(dev[1]['output']) == sha(cold['output'])
write(HERE/'native_qualified.json',dict(passed=True,rows=rows,candidate=build['candidate'],parent=build['parent'],
    disabled_identity=True,repeat35_exact=True,warm_cold100_exact=True,
    expected_removed_flops=removed,truth_accessed=False))
print(dict(passed=True,all16_bytes_exact=True,removed_flops=removed),flush=True)
