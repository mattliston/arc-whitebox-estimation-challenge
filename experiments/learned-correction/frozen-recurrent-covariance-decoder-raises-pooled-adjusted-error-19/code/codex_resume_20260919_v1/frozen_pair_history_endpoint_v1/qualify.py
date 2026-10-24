import importlib.util,resource
import numpy as np
from common import *
from reference import covariance
seal_check()
for n in ('disabled','enabled','cold38'):done(n)
meta=read(HERE/'inputs.json');verify(meta['bank_triple']);parent=read(meta['bank_triple']['path'])['records']
disabled=read(HERE/'disabled/result.json')['records'];on=read(HERE/'enabled/result.json')['records'];cold=read(HERE/'cold38/result.json')['records'][0]
assert [r['rank'] for r in disabled]==[35,100,35] and [r['rank'] for r in on]==[36,38,36] and cold['rank']==38
for old,new in zip(parent,disabled):
    assert sha(old['output'])==old['output_sha256']==new['prediction']['sha256']
    assert old['flops']==new['flops'] and old['includes_initialization']==new['includes_initialization']
    assert old['weights_sha256']==new['weights']['sha256'] and new['disabled'] and not new['history_layers']
assert on[0]['prediction']['sha256']==on[2]['prediction']['sha256'] and on[1]['prediction']['sha256']==cold['prediction']['sha256']
def compare(a,b):
    verify(a);verify(b)
    with np.load(a['path'],allow_pickle=False) as x,np.load(b['path'],allow_pickle=False) as y:
        assert set(x.files)==set(y.files) and all(x[k].dtype==y[k].dtype and x[k].tobytes()==y[k].tobytes() for k in x.files)
compare(disabled[0]['capture'],disabled[2]['capture']);compare(on[0]['capture'],on[2]['capture']);compare(on[1]['capture'],cold['capture'])
with np.load(HERE/'parameters.npz',allow_pickle=False) as z:q={k:z[k] for k in z.files}
with np.load(P0,allow_pickle=False) as z:p={k:z[k] for k in z.files if k!='first'}
sp=importlib.util.spec_from_file_location('original_history_gnn',A/'radial_gnn_numpy.py');gnn=importlib.util.module_from_spec(sp);sp.loader.exec_module(gnn)
fidelity=[];history=[]
for row in disabled+on+[cold]:
    verify(row['prediction']);verify(row['capture']);verify(row['weights'])
    assert row['numeric_cache_retired'] and min(row['unfloored_variance_minima'])>=1e-10
    if row['disabled']:continue
    old=meta['parent'][str(row['rank'])];verify(old['prediction'])
    now=np.load(row['prediction']['path'],allow_pickle=False);before=np.load(old['prediction']['path'],allow_pickle=False)
    assert now[:8].tobytes()==before[:8].tobytes()
    W=np.load(row['weights']['path'],allow_pickle=False);state=np.zeros((1024,64),np.float32)
    with np.load(row['capture']['path'],allow_pickle=False) as z:
        assert np.array_equal(z['indices'],np.arange(16)*67)
        for l in range(15):
            if l==0:assert not np.any(z[f'before_{l:02d}'])
            else:assert z[f'before_{l:02d}'].tobytes()==z[f'after_{l-1:02d}'].tobytes()
            _,state=gnn.gnn_step(l,z[f'x_{l:02d}'],z[f'sigma_{l:02d}'],W[l],state,p)
            er=relative(z[f'after_{l:02d}'],state);assert er<=3e-5,(row['rank'],l,er)
            history.append(dict(rank=row['rank'],index=row['index'],layer=l,relative=er))
        for l in range(8,15):
            h=z[f'kernel_state_{l:02d}'];s=z[f'kernel_sigma_{l:02d}'];R=z[f'kernel_pre_cov_{l:02d}']/np.outer(s,s)
            value=z[f'kernel_source_{l:02d}'];errors=[relative(value,covariance(h,R,s,q,wide=v)) for v in (False,True)]
            assert max(errors)<=3e-5,(row['rank'],l,errors)
            assert h.tobytes()==z[f'after_{l:02d}'][z['indices']].tobytes()
            fidelity.append(dict(rank=row['rank'],index=row['index'],layer=l,numpy_relative=errors[0],fp64_relative=errors[1]))
    del W,state
qualified=dict(passed=True,disabled_all16_bytes_and_bills_exact=True,enabled_first8_bytes_exact=True,
    own_repeat36_exact=True,warm_cold38_exact=True,all_capture_repeats_exact=True,
    all_unfloored_variances_positive=True,history_fidelity=history,kernel_fidelity=fidelity,
    full_native_calls=7,full_native_flops=sum(x['flops'] for x in disabled+on+[cold]),
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'qualification.json',qualified)
pairs=[]
for row in (on[0],cold):
    old=meta['parent'][str(row['rank'])]
    assert old['includes_initialization'] and row['includes_initialization'] and old['weights_sha256']==row['weights']['sha256']
    pairs.append(dict(rank=row['rank'],candidate=row,parent=old,truth=meta['truths'][str(row['rank'])]))
write(HERE/'predictions_sealed.json',dict(pairs=pairs,candidate=pin(HERE/'candidate.py'),qualification=pin(HERE/'qualification.json'),
    ordinary_1B_targets_not_yet_decoded=True,result_records=[pin(HERE/x/'result.json') for x in ('disabled','enabled','cold38')]))
print(dict(passed=True,max_history_relative=max(x['relative'] for x in history),max_kernel_relative=max(x['fp64_relative'] for x in fidelity)),flush=True)
