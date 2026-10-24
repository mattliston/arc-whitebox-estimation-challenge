from common import *
from records import verify_records
build=admission();meta=read(HERE/'inputs.json')
ref=meta['null_reference'];check(ref['result']);check(ref['watch'])
baseline,pins=verify_records(Path(ref['result']['path']).parent,build['parent'],'parent','development',[35,100,35],ref['watch']['path'])
null,more=verify_records(HERE/'native_null',build['null_candidate'],'null','control',[35,100,35],HERE/'native_null_watch.json');pins+=more
cold,more=verify_records(HERE/'native_null_cold100',build['null_candidate'],'null','control',[100],HERE/'native_null_cold100_watch.json');pins+=more
for a,b in zip(baseline,null):
    assert a['prediction']['sha256']==b['prediction']['sha256']
    assert a['flops']==b['flops'] and a['multiplier']==b['multiplier']
    assert a['domain']==b['domain'] and a['selectors']==b['selectors']
assert cold[0]['prediction']['sha256']==null[1]['prediction']['sha256']
assert cold[0]['domain']==null[1]['domain'] and cold[0]['selectors']==null[1]['selectors']
write(HERE/'null_qualification.json',dict(passed=True,all16_parent_prediction_bytes_exact=True,
    own_complete_parent_bills_exact=True,domain_selector_bytes_exact=True,warm_cold100_output_exact=True,
    all_retirements_qualified=True,targets_decoded=False,pins=list({p['path']:p for p in pins}.values())))
print(dict(passed=True,new_calls=4,parent_reference_calls=3),flush=True)
