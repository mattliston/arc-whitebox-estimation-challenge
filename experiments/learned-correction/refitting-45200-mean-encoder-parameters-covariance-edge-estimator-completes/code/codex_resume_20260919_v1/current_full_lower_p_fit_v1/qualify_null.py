from endpoint_common import *
endpoint_verify()
done('native_null')
build = read(HERE / 'build.json')
control_path = D / 'full_lower_multilayer_mean_observer_v1/parent_triple/result.json'
control = read(control_path)
null = read(HERE / 'native_null/result.json')
assert control['candidate_sha256'] == build['parent']['sha256']
assert null['source'] == build['null_candidate'] and null['completed'] and not null['truth_accessed']
assert len(null['records']) == len(control['records']) == 3
checks = []
for r, c in zip(null['records'], control['records']):
    assert r['rank'] == c['rank'] and r['includes_initialization'] == c['includes_initialization']
    check(r['prediction'])
    check(r['weights'])
    check(r['bill'])
    assert pin(c['output'])['sha256'] == c['output_sha256'] == r['prediction']['sha256']
    assert r['flops'] == c['flops'] and r['weights']['sha256'] == c['weights_sha256']
    assert r['numeric_cache_retired'] and r['passive_extra_fnp_ops'] == 0
    checks.append(dict(rank=r['rank'], output_all16_byte_exact=True, full_bill_exact=True))
write(HERE / 'native_null_qualification.json', dict(passed=True, checks=checks,
    build=pin(HERE / 'build.json'), acquisition=pin(HERE / 'native_null/result.json'),
    control=pin(control_path), watch=pin(HERE / 'native_null_watch.json'), truth_accessed=False))
print(dict(passed=True, exact_calls=3), flush=True)
