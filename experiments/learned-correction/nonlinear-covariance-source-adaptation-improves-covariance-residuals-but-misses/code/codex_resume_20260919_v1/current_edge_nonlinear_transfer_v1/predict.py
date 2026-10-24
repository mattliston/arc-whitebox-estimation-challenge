import resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from model import forward,reference
verify();verify('data_sealed.json');verify('models_sealed.json')
assert all(t['num_threads']==1 for t in threadpool_info());start=time.monotonic();rows=[]
with np.load(HERE/'bank_parameters.npz',allow_pickle=False) as z:p0={k:z[k] for k in KEYS}
for arm in ('output_only','nonlinear'):
    done(f'fit_{arm}')
    with np.load(HERE/f'{arm}_parameters.npz',allow_pickle=False) as z:p={k:z[k] for k in KEYS}
    q={k:v.astype(np.float64) for k,v in p.items()};q0={k:v.astype(np.float64) for k,v in p0.items()}
    for rank in VALID:
        values=[];checks=[]
        # Do not decode U here: only model inputs and the fixed baseline.
        with np.load(HERE/f'data_{rank:04d}.npz',allow_pickle=False) as z:
            a={k:z[k] for k in ('F','R','S','pf','gf','sf','base','factor')}
        for j,l in enumerate(range(8,15)):
            d={k:(v if k=='factor' else v[j]) for k,v in a.items()};native_proxy=forward(d,p);delta=native_proxy-d['base']
            ref=reference(d,q);base_ref=reference(d,q0);delta_ref=ref-base_ref
            er=float(np.linalg.norm(native_proxy-ref)/np.linalg.norm(ref));dr=float(np.linalg.norm(delta-delta_ref)/max(np.linalg.norm(delta_ref),1e-30))
            assert er<=3e-5 and dr<=1e-4,(arm,rank,l,er,dr)
            assert np.isfinite(delta).all() and np.array_equal(delta,delta.T)
            values.append(delta);checks.append(dict(layer=l,source_relative=er,delta_relative=dr))
        path=HERE/f'predictions_{arm}_{rank:04d}.npz'
        with path.open('xb') as f:np.savez(f,delta=np.stack(values))
        rows.append(dict(arm=arm,rank=rank,predictions=pin(path),checks=checks))
        print(dict(arm=arm,rank=rank,maximum_delta_reference_relative=max(x['delta_relative'] for x in checks)),flush=True)
write(HERE/'predictions_sealed.json',dict(completed=True,rows=rows,pins=[r['predictions'] for r in rows],
    held_targets_not_decoded=True,models=pin(HERE/'models_sealed.json'),wall_s=time.monotonic()-start,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
