from endpoint_common import *
from lifecycle import qualify

endpoint_verify()
done('native_trained')
old = D/'current_p_native_lifecycle_v1'
result = read(HERE/'native_trained/result.json')
records = result['records']
assert result['completed'] and len(records) == result['full_estimator_calls'] == 41
assert result['source'] == read(HERE/'build.json')['candidate'] and result['role'] == 'trained'
assert [r['rank'] for r in records] == read(HERE/'inputs.json')['valid']+[37]
qualify(records,check,read)
references,retirements = [],[]
for i,r in enumerate(records):
    if i < 30:
        p = old/f'native_trained/call_{i:02d}.json'
        c = read(p)
        for key in ['index','rank','flops','includes_initialization','weights','domain','selectors']:
            assert r[key] == c[key],(i,key)
        check(c['prediction']);check(c['bill'])
        assert r['prediction']['sha256'] == c['prediction']['sha256']
        references.extend([pin(p),c['prediction'],c['bill']])
    p = HERE/f'native_trained/call_{i:02d}_retirement.json'
    x = read(p)
    assert x['complete'] and x['index'] == i and x['rank'] == r['rank']
    assert x['all_tracked_call_objects_released'] and x['tracked_objects'] == 18 and not x['live_objects']
    assert x['after_closed_saved_bill'] and x['saved_bill_unchanged'] and x['allocator_called']
    assert not x['allocator_statistics_called'] and not x['candidate_state_reset']
    check(x['saved_bill']);assert x['saved_bill'] == r['bill']
    retirements.append(pin(p))
write(HERE/'trained_lifecycle_qualified.json',dict(passed=True,role='trained',calls=41,
    historical_prefix_exact_calls=30,original_paid_body_unchanged=True,closed_contexts_released=True,
    saved_bills_unchanged=True,records=pin(HERE/'native_trained/result.json'),watch=pin(HERE/'native_trained_watch.json'),
    lifecycle_pins=[r[k] for r in records for k in ['input_bill','input_reset','lifecycle']],
    original_comparison_pins=references,retirement_pins=retirements,all_retirements_qualified=True,
    allocator_qualification=pin(D/'current_p_native_allocator_qualification_v1/qualification.json'),truth_accessed=False))
print(dict(passed=True,trained_calls=41,historical_prefix_exact=30,retirements_qualified=41),flush=True)
