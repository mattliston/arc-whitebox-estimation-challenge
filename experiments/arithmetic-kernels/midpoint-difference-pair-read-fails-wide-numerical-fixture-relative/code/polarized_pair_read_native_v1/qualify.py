import resource
import numpy as np
from common import *
seal_check()
for n in ('disabled','enabled','cold38'):done(n)
meta=read(HERE/'inputs.json');verify(meta['bank_triple']);parent=read(meta['bank_triple']['path'])['records']
disabled=read(HERE/'disabled/result.json')['records'];on=read(HERE/'enabled/result.json')['records'];cold=read(HERE/'cold38/result.json')['records'][0]
assert [r['rank'] for r in disabled]==[35,100,35] and [r['rank'] for r in on]==[36,38,36] and cold['rank']==38
for old,new in zip(parent,disabled):
    assert sha(old['output'])==old['output_sha256']==new['prediction']['sha256']
    assert old['flops']==new['flops'] and old['includes_initialization']==new['includes_initialization']
    assert old['weights_sha256']==new['weights']['sha256'] and new['disabled'] and not new['helper_calls']
for first,repeat in ((on[0],on[2]),(on[1],cold)):
    assert first['prediction']['sha256']==repeat['prediction']['sha256'] and first['helper_calls']==repeat['helper_calls']
for row in disabled+on+[cold]:
    verify(row['prediction']);verify(row['weights']);assert row['numeric_cache_retired'] and min(row['unfloored_variance_minima'])>=1e-10
    if row['disabled']:continue
    old=meta['parent'][str(row['rank'])];verify(old['prediction'])
    now=np.load(row['prediction']['path'],allow_pickle=False);before=np.load(old['prediction']['path'],allow_pickle=False)
    assert now[0].tobytes()==before[0].tobytes()
write(HERE/'qualification.json',dict(passed=True,disabled_all16_bytes_and_bills_exact=True,enabled_first_row_exact=True,
    own_repeat36_exact=True,warm_cold38_exact=True,helper_metadata_repeats_exact=True,unfloored_variances_positive=True,
    full_native_calls=7,full_native_flops=sum(r['flops'] for r in disabled+on+[cold]),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
pairs=[]
for row in (on[0],cold):
    old=meta['parent'][str(row['rank'])];assert old['includes_initialization'] and row['includes_initialization'] and old['weights_sha256']==row['weights']['sha256']
    pairs.append(dict(rank=row['rank'],candidate=row,parent=old,truth=meta['truths'][str(row['rank'])]))
write(HERE/'predictions_sealed.json',dict(pairs=pairs,candidate=pin(HERE/'candidate.py'),qualification=pin(HERE/'qualification.json'),
    ordinary_1B_targets_not_yet_decoded=True,result_records=[pin(HERE/n/'result.json') for n in ('disabled','enabled','cold38')]))
print(dict(passed=True,full_native_calls=7),flush=True)
