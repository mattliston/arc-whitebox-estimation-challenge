import resource
from common import *

seal_check()
for name in ['disabled', 'enabled', 'cold100']:
    done(name)
disabled = read(HERE / 'disabled/result.json')['records']
enabled = read(HERE / 'enabled/result.json')['records']
cold = read(HERE / 'cold100/result.json')['records']
assert [r['rank'] for r in disabled] == [35]
assert [r['rank'] for r in enabled] == [35, 100, 35]
assert [r['rank'] for r in cold] == [100]
assert enabled[0]['prediction']['sha256'] == enabled[2]['prediction']['sha256']
assert enabled[1]['prediction']['sha256'] == cold[0]['prediction']['sha256']
for key in ['rosters', 'cohorts']:
    assert enabled[0][key] == enabled[2][key] and enabled[1][key] == cold[0][key]
for r in disabled + enabled + cold:
    check(r['prediction']); check(r['weights'])
    assert r['numeric_cache_retired'] and r['input_flops'] == 0 and r['first_parent_row_exact']
    assert r['peak_rss_bytes'] < 8_000_000_000
    if r['full_cubic_support']:
        assert len(r['rosters']) == 30 and all(q['count'] == 1024 for q in r['rosters'])
pairs = []
for c, row in zip([enabled[0], cold[0]], read(HERE / 'inputs.json')['rows']):
    assert c['rank'] == row['rank'] and c['includes_initialization'] and row['parent']['includes_initialization']
    assert c['weights']['sha256'] == row['parent']['weights_sha256']
    check(row['parent_result']); check(row['prediction'])
    pairs.append(dict(rank=c['rank'], candidate=c, parent=row['parent'],
        parent_prediction=row['prediction'], truth=row['truth']))
q = dict(passed=True, repeat35_exact=True, warm_cold100_exact=True,
    all_roster_and_cohort_repeats_exact=True, all_enabled_rosters_full=True,
    disabled_parent_all16_bytes_and_bill_exact=True, enabled_first_row_exact=True,
    native_full_calls=5, native_full_flops=sum(r['flops'] for r in disabled + enabled + cold),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE / 'qualification.json', q)
write(HERE / 'predictions_sealed.json', dict(pairs=pairs, candidate=pin(HERE / 'candidate.py'),
    qualification=pin(HERE / 'qualification.json'),
    result_records=[pin(HERE / x / 'result.json') for x in ['disabled', 'enabled', 'cold100']],
    targets_not_yet_decoded=True))
print(q, flush=True)

