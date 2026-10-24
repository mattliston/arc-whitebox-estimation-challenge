import zipfile
from common import *

verify();clean_watch('qualification')
meta=read(HERE/'inputs.json');check(meta['qualification_reference'])
parent=read(meta['qualification_reference']['path'])
candidate=read(HERE/'qualification/result.json')
assert candidate['completed'] and candidate['full_estimator_calls']==3
assert candidate['repeat_output_domain_selector_bytes_exact']
comparisons=[]
for a,b in zip(parent['records'],candidate['records']):
    assert a['rank']==b['rank']
    for key in ('domain','selectors','includes_initialization'):assert a[key]==b[key],key
    assert a['prediction']['sha256']==b['prediction']['sha256']
    check(a['prediction']);check(b['prediction']);check(b['packet'])
    assert b['extra_gaussian_variance_exact'] and b['scalar_complement_max_abs']<=1e-12
    assert b['flops']>a['flops']
    rp=HERE/f"qualification/call_{b['index']:02d}_retirement.json";r=read(rp)
    assert r['all_tracked_call_objects_released'] and not r['live_objects'] and r['tracked_objects']==26
    comparisons.append(dict(rank=a['rank'],parent_bill=a['bill'],capture_bill=b['bill'],
        parent_flops=a['flops'],capture_flops=b['flops'],added_flops=b['flops']-a['flops'],retirement=pin(rp)))
with zipfile.ZipFile(candidate['records'][0]['packet']['path']) as a,zipfile.ZipFile(candidate['records'][2]['packet']['path']) as b:
    assert a.namelist()==b.namelist()
    assert all(a.read(k)==b.read(k) for k in a.namelist())
write(HERE/'qualification.json',dict(passed=True,positive_output_bytes_exact=True,domains_selectors_exact=True,
    dual_packet_repeat_bytes_exact=True,retired_objects_per_call=26,comparisons=comparisons,
    parent=meta['qualification_reference'],capture=pin(HERE/'qualification/result.json'),targets_decoded=False))
print(dict(passed=True,comparisons=comparisons),flush=True)
