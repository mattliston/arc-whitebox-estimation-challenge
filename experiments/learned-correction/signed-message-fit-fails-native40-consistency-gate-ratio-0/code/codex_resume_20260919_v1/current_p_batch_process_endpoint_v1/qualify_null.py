from common import *
from records import verify_run

admission()
warm,pins = verify_run('native_null','control','null',[35,100,35])
cold,extra = verify_run('native_null_cold100','control','null',[100])
pins += extra + qualify_retirements('native_null',warm) + qualify_retirements('native_null_cold100',cold)
references = read(HERE/'inputs.json')['null_references']
comparisons = []
for actual,expected in zip(warm+cold,references):
    for key in ['index','rank','flops','includes_initialization','weights','domain','selectors']:
        assert actual[key]==expected[key],(actual['rank'],key)
    check(expected['prediction'])
    check(expected['bill'])
    assert actual['prediction']['sha256']==expected['prediction']['sha256']
    pins += [expected['prediction'],expected['bill']]
    comparisons.append(dict(index=actual['index'],rank=actual['rank'],all16_output_bytes_exact=True,
        all_domain_and_selector_bytes_exact=True,own_complete_bill_exact=True,flops=actual['flops']))
assert warm[1]['prediction']['sha256']==cold[0]['prediction']['sha256']
write(HERE/'native_null_qualification.json',dict(passed=True,comparisons=comparisons,
    full_native_calls=4,warm_cold100_exact=True,all_retirements_qualified=True,
    no_score_labels=True,pins=list({p['path']:p for p in pins}.values()),
    source=read(HERE/'build.json')['null_candidate']))
print(dict(passed=True,full_native_calls=4),flush=True)
