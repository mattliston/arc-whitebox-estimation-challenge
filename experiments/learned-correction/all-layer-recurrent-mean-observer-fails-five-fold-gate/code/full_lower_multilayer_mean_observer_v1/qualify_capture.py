import sys
import numpy as np
from common import HERE,read,write,pin,verify,seal_check
seal_check();mode=sys.argv[1]
def compare(c,p):
    verify(c['capture']);verify(c['weights']);verify(p['output']);verify(p['receipt'])
    with np.load(c['capture']['path'],allow_pickle=False) as z:q=z['prediction'].copy()
    a=np.load(p['output']['path'],allow_pickle=False)
    assert a.tobytes()==q.tobytes() and c['flops']==p['flops'] and c['weights']['sha256']==p['weights_sha256']
    return dict(rank=c['rank'],all16_parent_bytes_exact=True,complete_bill_exact=True)
def arrays(c):
    with np.load(c['capture']['path'],allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
if mode=='qualification':
    checks=[];calls=[]
    for cap,raw in [('capture_triple','parent_triple'),('capture_cold100','parent_cold100')]:
        cs=read(HERE/cap/'result.json')['records'];ps=read(HERE/raw/'result.json')['records']
        for c,p in zip(cs,ps):
            checks.append(compare(c,dict(output=pin(p['output']),flops=p['flops'],weights_sha256=p['weights_sha256'],receipt=pin(HERE/raw/'result.json'))));calls.append(c)
    a,b,c,d=map(arrays,calls)
    assert all(a[k].tobytes()==c[k].tobytes() for k in a)
    assert all(b[k].tobytes()==d[k].tobytes() for k in b)
    write(HERE/'capture_qualification.json',dict(passed=True,checks=checks,repeat_all_fields_exact=True,warm_cold_all_fields_exact=True,truth_accessed=False))
elif mode=='train':
    done=read(HERE/'train_captures_completed.json');rows=done['rows'];checks=[]
    for r in rows:
        verify(r['watch']);g=read(r['watch']['path']);assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
        assert r['record']['includes_initialization'];checks.append(compare(r['record'],r['parent']))
    write(HERE/'captures_sealed.json',dict(passed=True,rows=rows,checks=checks,source_seal=pin(HERE/'capture_sources_sealed.json'),qualification=pin(HERE/'capture_qualification.json'),truth_accessed=False))
else:raise ValueError(mode)
print(dict(mode=mode,passed=True,calls=len(checks)),flush=True)
