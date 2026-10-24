import resource
from common import *

seal_check()
for name in ['disabled','enabled','cold100']:done(name)
disabled=read(HERE/'disabled/result.json')['records']
enabled=read(HERE/'enabled/result.json')['records']
cold=read(HERE/'cold100/result.json')['records']
assert [r['rank'] for r in disabled]==[35]
assert [r['rank'] for r in enabled]==[35,100,35]
assert [r['rank'] for r in cold]==[100]
for a,b in [(enabled[0],enabled[2]),(enabled[1],cold[0])]:
    for key in ['prediction','shifts','rays']:assert a[key]['sha256']==b[key]['sha256'],key
    for key in ['source_checks','ray_checks','advance_checks','extra_flops','helper_costs']:
        assert a[key]==b[key],key
for r in disabled+enabled+cold:
    for key in ['prediction','shifts','weights']:check(r[key])
    if r['rays']:check(r['rays'])
    assert r['numeric_cache_retired'] and r['input_flops']==0 and r['first_parent_row_exact']
    assert r['underlying_all16_parent_bytes_exact'] and r['underlying_context_bill_exact']
    assert r['peak_rss_bytes']<8_000_000_000
    assert r['flops']-r['extra_flops']==r['baseline_context_flops']
pairs=[]
for c,row in zip([enabled[0],cold[0]],read(HERE/'inputs.json')['rows']):
    assert c['rank']==row['rank'] and c['includes_initialization'] and row['parent']['includes_initialization']
    assert c['baseline_context_flops']==row['parent']['flops']
    check(row['parent_result']);check(row['prediction'])
    pairs.append(dict(rank=c['rank'],candidate=c,parent=row['parent'],parent_prediction=row['prediction'],truth=row['truth']))
q=dict(passed=True,repeat35_exact=True,warm_cold100_exact=True,
    all_source_ray_shift_repeats_exact=True,disabled_parent_all16_bytes_and_bill_exact=True,
    all_enabled_underlying_parent_bytes_and_context_bill_exact=True,
    native_full_calls=5,native_full_flops=sum(r['flops'] for r in disabled+enabled+cold),
    enabled_added_flops=sum(r['extra_flops'] for r in enabled+cold),
    source_comparisons=sum(len(r['source_checks']) for r in enabled+cold),
    ray_comparisons=sum(len(r['ray_checks']) for r in enabled+cold),
    recurrence_comparisons=sum(len(r['advance_checks']) for r in enabled+cold),
    max_source_relative=max(c['relative'] for r in enabled+cold for c in r['source_checks']),
    max_ray_relative=max(c['relative'] for r in enabled+cold for c in r['ray_checks']),
    max_recurrence_relative=max(c['relative'] for r in enabled+cold for c in r['advance_checks']),
    extra_fractions=[r['extra_fraction'] for r in enabled+cold],
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'qualification.json',q)
write(HERE/'predictions_sealed.json',dict(pairs=pairs,candidate=pin(HERE/'candidate.py'),qualification=pin(HERE/'qualification.json'),
    result_records=[pin(HERE/x/'result.json') for x in ['disabled','enabled','cold100']],targets_not_yet_decoded=True))
print(q,flush=True)

