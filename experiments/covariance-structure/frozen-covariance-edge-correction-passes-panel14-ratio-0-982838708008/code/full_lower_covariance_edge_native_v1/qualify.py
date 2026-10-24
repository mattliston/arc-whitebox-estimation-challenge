import numpy as np
from common import HERE,D,read,write,pin,verify,seal_check,guarded,sha,relative
from reference import reference
seal_check()
for n in ['disabled','enabled','cold38']:guarded(n)
meta=read(HERE/'inputs.json');verify(meta['disabled_parent'])
parent=read(meta['disabled_parent']['path'])['records'];off=read(HERE/'disabled/result.json')['records']
assert [r['rank'] for r in parent]==[r['rank'] for r in off]==[35,100,35]
for p,r in zip(parent,off):
    verify(r['prediction']);assert sha(p['output'])==p['output_sha256']==r['prediction']['sha256'] and p['flops']==r['flops']
on=read(HERE/'enabled/result.json')['records'];cold=read(HERE/'cold38/result.json')['records'][0]
assert [r['rank'] for r in on]==[36,38,36] and cold['rank']==38
assert on[0]['prediction']['sha256']==on[2]['prediction']['sha256']
assert on[1]['prediction']['sha256']==cold['prediction']['sha256']
for r in on+[cold]:
    p=meta['parent'][str(r['rank'])];verify(p['capture']);verify(r['prediction'])
    assert p['includes_initialization'] and p['weights']['sha256']==r['weights']['sha256']
    with np.load(p['capture']['path'],allow_pickle=False) as z:old=z['prediction']
    new=np.load(r['prediction']['path'],allow_pickle=False)
    assert old[:8].tobytes()==new[:8].tobytes()
def compare(a,b):
    verify(a);verify(b)
    with np.load(a['path'],allow_pickle=False) as x,np.load(b['path'],allow_pickle=False) as y:
        assert set(x.files)==set(y.files) and all(x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes() for k in x.files)
compare(on[0]['capture'],on[2]['capture']);compare(on[1]['capture'],cold['capture'])
b=read(HERE/'build.json');verify(b['parameters'])
with np.load(b['parameters']['path'],allow_pickle=False) as z:p={k:z[k] for k in z.files}
fidelity=[]
for r in on+[cold]:
    with np.load(r['capture']['path'],allow_pickle=False) as z:
        assert np.array_equal(z['indices'],np.arange(16)*67)
        for l in range(8,15):
            aux={k:z[f'{k}_{l:02d}'] for k in ['sigma','alpha','pdf','gate','pre_k3','pre_k4','pre_cov']}
            a=z[f'source_{l:02d}'];m=z[f'm_{l:02d}'];K=z[f'K21_{l:02d}']
            err=relative(a,reference(m,K,aux,p));rerr=relative(a,reference(m,K,aux,p,wide=True))
            assert max(err,rerr)<=3e-5,(r['rank'],l,err,rerr)
            fidelity.append(dict(rank=r['rank'],index=r['index'],layer=l,numpy_relative=err,fp64_relative=rerr))
q=dict(passed=True,disabled_all16_bytes_and_bills_exact=True,enabled_first8_exact=True,repeat36_exact=True,
    warm_cold38_exact=True,all_sampled_source_repeats_exact=True,fidelity=fidelity,candidate=b['candidate'],truth_accessed=False)
write(HERE/'qualification.json',q)
pairs=[dict(rank=36,candidate=on[0],parent=meta['parent']['36']),dict(rank=38,candidate=cold,parent=meta['parent']['38'])]
assert all(r['candidate']['includes_initialization'] for r in pairs)
write(HERE/'predictions_sealed.json',dict(pairs=pairs,qualification=pin(HERE/'qualification.json'),candidate=b['candidate'],
    inputs=pin(HERE/'inputs.json'),means_not_yet_decoded=True))
print(dict(passed=True,max_numpy_relative=max(r['numpy_relative'] for r in fidelity),max_fp64_relative=max(r['fp64_relative'] for r in fidelity)),flush=True)
