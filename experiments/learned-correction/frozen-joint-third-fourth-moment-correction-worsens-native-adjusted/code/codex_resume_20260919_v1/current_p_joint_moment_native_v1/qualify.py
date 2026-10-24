"""All development predictions and complete bills precede target decoding."""
from common import *

verify()
roles={}
pins=[]
for role in ('parent','null','trained'):
    clean_watch('native_'+role)
    path=HERE/f'native_{role}/result.json'
    q=read(path)
    assert q['completed'] and q['full_estimator_calls']==3 and q['repeat_output_domain_selector_bytes_exact']
    assert [r['rank'] for r in q['records']]==[35,100,35]
    assert q['all_16_domains_and_45_selectors_qualified'] and q['owned_fnp_input_and_closed_context_lifecycle']
    roles[role]=q
    pins += [pin(path),q['source']]
    for row in q['records']:
        for key in ('prediction','bill','input_bill','input_reset','lifecycle'):
            check(row[key]);pins.append(row[key])
        retirement=HERE/f"native_{role}/call_{row['index']:02d}_retirement.json"
        r=read(retirement)
        assert r['all_tracked_call_objects_released'] and r['tracked_objects']==18 and not r['live_objects']
        assert r['saved_bill_unchanged'] and r['after_closed_saved_bill'] and not r['candidate_state_reset']
        check(r['saved_bill'])
        pins.append(pin(retirement))
for p,n in zip(roles['parent']['records'],roles['null']['records']):
    for key in ('domain','selectors','flops','includes_initialization'):
        assert p[key]==n[key],key
    assert p['prediction']['sha256']==n['prediction']['sha256']
write(HERE/'predictions_sealed.json',dict(completed=True,disabled_adapter_exact=True,all_calls_retired=True,
      before_target_decoding=True,roles={k:pin(HERE/f'native_{k}/result.json') for k in roles},pins=pins))
print(dict(completed=True,disabled_exact=True,calls=9,pins=len(pins)))
