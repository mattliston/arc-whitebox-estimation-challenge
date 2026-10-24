import importlib.util
import numpy as np
from common import HERE,OLD,MODEL,read,write,pin,verify,seal_check,guarded,relative

seal_check()
for name in ['disabled','enabled','cold100']:guarded(name)
meta=read(HERE/'inputs.json')
for p in [meta['parent_dev'],meta['parent_cold100']]:verify(p)
parent=read(meta['parent_dev']['path'])['records']
oldcold=read(meta['parent_cold100']['path'])['records'][0]
off=read(HERE/'disabled/result.json')['records']
on=read(HERE/'enabled/result.json')['records']
cold=read(HERE/'cold100/result.json')['records'][0]
assert [r['rank'] for r in parent]==[r['rank'] for r in off]==[r['rank'] for r in on]==[35,100,35]
checks=[]
for i,(p,a,b) in enumerate(zip(parent,off,on)):
    verify(a['prediction']);verify(b['prediction'])
    from common import sha
    assert sha(p['output'])==p['output_sha256']
    pa=np.load(p['output'],allow_pickle=False);aa=np.load(a['prediction']['path'],allow_pickle=False)
    ba=np.load(b['prediction']['path'],allow_pickle=False)
    assert pa.tobytes()==aa.tobytes() and a['flops']==p['flops']
    assert pa[:8].tobytes()==ba[:8].tobytes()
    assert not a['source_enabled'] and b['source_enabled']
    checks.append(dict(rank=b['rank'],index=i,disabled_all16_and_bill_exact=True,enabled_first8_exact=True,
        parent_flops=p['flops'],candidate_flops=b['flops']))
assert off[0]['prediction']['sha256']==off[2]['prediction']['sha256']
assert on[0]['prediction']['sha256']==on[2]['prediction']['sha256']
assert cold['prediction']['sha256']==on[1]['prediction']['sha256']
assert sha(oldcold['output'])==oldcold['output_sha256']
assert np.load(oldcold['output'],allow_pickle=False)[:8].tobytes()==np.load(cold['prediction']['path'],allow_pickle=False)[:8].tobytes()
def same_capture(a,b):
    verify(a);verify(b)
    with np.load(a['path'],allow_pickle=False) as x,np.load(b['path'],allow_pickle=False) as y:
        assert set(x.files)==set(y.files)
        assert all(x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes() for k in x.files)
same_capture(on[0]['capture'],on[2]['capture'])
same_capture(on[1]['capture'],cold['capture'])
spec=importlib.util.spec_from_file_location('old_mean_model',OLD/'model.py')
model=importlib.util.module_from_spec(spec);spec.loader.exec_module(model)
with np.load(MODEL,allow_pickle=False) as z:p={k:z[k] for k in z.files}
fidelity=[]
for record in on+[cold]:
    verify(record['capture'])
    with np.load(record['capture']['path'],allow_pickle=False) as z:
        for l in range(8,16):
            f=z[f'features_{l:02d}'];h=z[f'hidden_{l:02d}'];s=z[f'sigma_{l:02d}'];a=z[f'source_{l:02d}']
            assert f.shape==(1024,30) and h.shape==(1024,64) and s.shape==a.shape==(1024,)
            assert f.dtype==h.dtype==np.float32 and s.dtype==a.dtype==np.float64
            x=model.inputs(f,h,l,p['mu'],p['sd'],np)
            expected=model.forward(x,p,np)*p['label_rms']*s
            x64=np.concatenate([(np.concatenate([f,h],axis=1).astype(np.float64)-p['mu'])/p['sd'],
                np.eye(8)[np.full(1024,l-8)]],axis=1)
            ref=(np.maximum(x64@p['W1'].astype(np.float64)+p['b1'],0)@p['W2'].astype(np.float64)+float(p['b2']))*float(p['label_rms'])*s
            err=relative(a,expected);rerr=relative(a,ref)
            assert max(err,rerr)<=3e-5,(record['rank'],l,err,rerr)
            fidelity.append(dict(rank=record['rank'],index=record['index'],layer=l,numpy_relative=err,
                fp64_relative=rerr,source_mse=float(np.mean(a*a))))
result=dict(passed=True,checks=checks,fidelity=fidelity,repeat35_exact=True,warm_cold100_exact=True,
    all_capture_repeats_exact=True,disabled_no_source_arithmetic=True,candidate=pin(HERE/'candidate.py'),
    development_targets_accessed=False)
write(HERE/'qualification.json',result)
pairs=[dict(rank=35,candidate=on[0],parent=parent[0]),dict(rank=100,candidate=cold,parent=oldcold)]
for pair in pairs:
    a=pair['candidate'];b=pair['parent']
    assert a['includes_initialization'] and b['includes_initialization']
    assert a['weights']['sha256']==b['weights_sha256']
write(HERE/'predictions_sealed.json',dict(pairs=pairs,qualification=pin(HERE/'qualification.json'),
    candidate=pin(HERE/'candidate.py'),inputs=pin(HERE/'inputs.json'),means_not_yet_decoded=True))
print(dict(passed=True,checks=checks,max_numpy_relative=max(x['numpy_relative'] for x in fidelity),
    max_fp64_relative=max(x['fp64_relative'] for x in fidelity)),flush=True)
