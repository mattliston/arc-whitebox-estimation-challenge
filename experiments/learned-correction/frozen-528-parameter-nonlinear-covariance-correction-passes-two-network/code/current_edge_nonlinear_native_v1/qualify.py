import resource
import numpy as np
from common import *
from reference import reference
seal_check()
for n in ('bank_triple','enabled','cold38'):done(n)
meta=read(HERE/'inputs.json');verify(meta['bank_triple'])
parent=read(meta['bank_triple']['path'])['records'];bank=read(HERE/'bank_triple/result.json')['records']
assert [x['rank'] for x in parent]==[x['rank'] for x in bank]==[35,100,35]
for old,new in zip(parent,bank):
    assert sha(old['output'])==old['output_sha256']==new['prediction']['sha256']
    assert old['flops']==new['flops'] and old['includes_initialization']==new['includes_initialization']
    assert old['weights_sha256']==new['weights']['sha256'] and new['bank_parameters']
on=read(HERE/'enabled/result.json')['records'];cold=read(HERE/'cold38/result.json')['records'][0]
assert [r['rank'] for r in on]==[36,38,36] and cold['rank']==38
assert on[0]['prediction']['sha256']==on[2]['prediction']['sha256']
assert on[1]['prediction']['sha256']==cold['prediction']['sha256']
def compare(a,b):
    verify(a);verify(b)
    with np.load(a['path'],allow_pickle=False) as x,np.load(b['path'],allow_pickle=False) as y:
        assert set(x.files)==set(y.files) and all(x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes() for k in x.files)
compare(bank[0]['capture'],bank[2]['capture']);compare(on[0]['capture'],on[2]['capture']);compare(on[1]['capture'],cold['capture'])
with np.load(HERE/'parameters.npz',allow_pickle=False) as z:trained={k:z[k] for k in z.files}
with np.load(FIT/'bank_parameters.npz',allow_pickle=False) as z:original={k:z[k] for k in z.files}
fidelity=[]
for r in bank+on+[cold]:
    verify(r['prediction']);verify(r['capture']);verify(r['weights'])
    assert r['numeric_cache_retired'] and len(r['unfloored_variance_minima'])==16 and min(r['unfloored_variance_minima'])>=1e-10
    assert all(x['full_source_finite_and_symmetric'] for x in r['source_checks'])
    p=original if r['bank_parameters'] else trained
    if not r['bank_parameters']:
        old=meta['parent'][str(r['rank'])];verify(old['prediction'])
        a=np.load(r['prediction']['path'],allow_pickle=False);o=np.load(old['prediction']['path'],allow_pickle=False)
        assert a[:8].tobytes()==o[:8].tobytes()
    with np.load(r['capture']['path'],allow_pickle=False) as z:
        assert np.array_equal(z['indices'],np.arange(16)*67)
        for l in range(8,15):
            aux={k:z[f'{k}_{l:02d}'] for k in ['sigma','alpha','pdf','gate','pre_k3','pre_k4','pre_cov']}
            a=z[f'source_{l:02d}'];m=z[f'm_{l:02d}'];K=z[f'K21_{l:02d}']
            er=relative(a,reference(m,K,aux,p));wide=relative(a,reference(m,K,aux,p,wide=True))
            assert max(er,wide)<=3e-5,(r['rank'],l,er,wide)
            fidelity.append(dict(rank=r['rank'],index=r['index'],bank_parameters=r['bank_parameters'],layer=l,numpy_relative=er,fp64_relative=wide))
q=dict(passed=True,bank_triple_all16_bytes_and_bills_exact=True,enabled_first8_exact=True,repeat36_exact=True,
    warm_cold38_exact=True,all_source_repeats_exact=True,all16_unfloored_variances_positive=True,
    fidelity=fidelity,full_native_calls=7,full_native_flops=sum(r['flops'] for r in bank+on+[cold]),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'qualification.json',q)
pairs=[]
for r in [on[0],cold]:
    old=meta['parent'][str(r['rank'])]
    assert old['includes_initialization'] and r['includes_initialization'] and old['weights_sha256']==r['weights']['sha256']
    pairs.append(dict(rank=r['rank'],candidate=r,parent=old,truth=meta['truths'][str(r['rank'])]))
write(HERE/'predictions_sealed.json',dict(pairs=pairs,candidate=pin(HERE/'candidate.py'),qualification=pin(HERE/'qualification.json'),
    result_records=[pin(HERE/x/'result.json') for x in ('bank_triple','enabled','cold38')],targets_not_yet_decoded=True))
print(dict(passed=True,max_fp64_relative=max(x['fp64_relative'] for x in fidelity),full_native_calls=7),flush=True)
