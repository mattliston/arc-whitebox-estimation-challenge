import resource
import numpy as np
from common import *
from reference import reference
stage=sys.argv[1];assert stage in ('panel','confirmation');seal_check()
seal=read(HERE/f'{stage}_predictions_sealed.json');roster=PANEL if stage=='panel' else CONFIRM
assert [r['rank'] for r in seal['rows']]==roster
with np.load(NATIVE/'parameters.npz',allow_pickle=False) as z:p={k:z[k] for k in z.files}
fidelity=[]
for row in seal['rows']:
    rank=row['rank'];done(f'{stage}_{rank}');c=row['call'];parent=row['parent']
    for pinrow in (c['prediction'],c['capture'],c['weights'],parent['prediction'],parent['receipt'],parent['source']):verify(pinrow)
    assert c['includes_initialization'] and parent['includes_initialization'] and not c['bank_parameters']
    assert c['weights']['sha256']==parent['weights_sha256'] and c['numeric_cache_retired']
    assert len(c['unfloored_variance_minima'])==16 and min(c['unfloored_variance_minima'])>=1e-10
    a=np.load(c['prediction']['path'],allow_pickle=False);b=np.load(parent['prediction']['path'],allow_pickle=False)
    assert a.shape==b.shape==(16,1024) and a.dtype==b.dtype==np.float32 and np.isfinite(a).all() and (a>=0).all()
    assert a[:8].tobytes()==b[:8].tobytes()
    with np.load(c['capture']['path'],allow_pickle=False) as z:
        assert np.array_equal(z['indices'],np.arange(16)*67)
        for l in range(8,15):
            aux={k:z[f'{k}_{l:02d}'] for k in ('sigma','alpha','pdf','gate','pre_k3','pre_k4','pre_cov')}
            m=z[f'm_{l:02d}'];K=z[f'K21_{l:02d}'];value=z[f'source_{l:02d}']
            e=relative(value,reference(m,K,aux,p));f=relative(value,reference(m,K,aux,p,wide=True))
            assert max(e,f)<=3e-5
            fidelity.append(dict(rank=rank,layer=l,numpy_relative=e,fp64_relative=f))
write(HERE/f'{stage}_qualified.json',dict(passed=True,first8_exact_all=True,all16_unfloored_variances_positive=True,
    source_fidelity=fidelity,cold_initialization_all=True,targets_not_decoded=True,
    predictions=pin(HERE/f'{stage}_predictions_sealed.json'),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(stage=stage,passed=True,networks=len(roster),max_source_relative=max(r['fp64_relative'] for r in fidelity)),flush=True)
