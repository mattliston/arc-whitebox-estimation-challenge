from common import *
seal_check()
for repair_pin in read(HERE/'repair_sources_sealed.json')['pins']:check(repair_pin)
done('unit');done('solver_unit_v2')
inputs=read(HERE/'inputs.json');names=['qual35a','qual100','qual35b']+[f'train{r}' for r in inputs['train']]
records=[]
for name in names:
    done(name);r=read(HERE/f'{name}.json');check(r['file']);check(r['weights'])
    assert r['all16_parent_bytes_and_full_bill_exact'] and r['cache_retired'] and r['input_preserved']
    assert r['includes_initialization'] and not r['targets_decoded']
    assert r['peak_rss_bytes']<8_000_000_000 and r['helper_flops']<=20_000_000_000
    records.append(r)
for key in ['feature_digests','parent_flops','helper_flops','reference_relative','formal_diagonal_relative']:
    assert records[0][key]==records[2][key],key
write(HERE/'features_sealed.json',dict(pins=[pin(HERE/f'{n}.json') for n in names]+[r['file'] for r in records],
    records=records,all16_parent_bytes_and_full_bill_exact=True,repeat35_features_and_bill_exact=True,
    old35_and100_feature_parity=True,targets_not_yet_decoded=True,whole_function_inverse=True))
print(dict(qualified=True,cold_calls=len(records),targets_decoded=False),flush=True)
