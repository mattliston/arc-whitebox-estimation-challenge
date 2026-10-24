import resource
import numpy as np
from threadpoolctl import threadpool_info
from common import *
from model import forward,reference
seal_check();done('unit');assert all(t['num_threads']==1 for t in threadpool_info());meta=read(HERE/'inputs.json')
with np.load(meta['data']['path'],allow_pickle=False) as z:a={k:z[k] for k in ('F','R','S','pf','gf','sf','base','factor')}
with np.load(meta['bank_parameters']['path'],allow_pickle=False) as z:p0={k:z[k] for k in KEYS}
q0={k:v.astype(np.float64) for k,v in p0.items()};rows=[];arrays={}
for arm in ARMS:
    with np.load(meta['models'][arm]['path'],allow_pickle=False) as z:p={k:z[k] for k in KEYS}
    q={k:v.astype(np.float64) for k,v in p.items()}
    for j,l in enumerate(range(8,15)):
        d={k:(v if k=='factor' else v[j]) for k,v in a.items()}
        assert forward(d,p0).tobytes()==d['base'].tobytes()
        value=forward(d,p);delta=value-d['base'];ref=reference(d,q);refdelta=ref-reference(d,q0)
        se=relative(value,ref);de=relative(delta,refdelta);assert se<=3e-5 and de<=1e-4,(arm,l,se,de)
        assert delta.dtype==np.float32 and np.isfinite(delta).all() and np.array_equal(delta,delta.T)
        arrays[f'{arm}_{l:02d}']=delta;rows.append(dict(arm=arm,layer=l,source_relative=se,delta_relative=de))
path=HERE/'predictions.npz'
with path.open('xb') as f:np.savez(f,**arrays)
write(HERE/'predictions_sealed.json',dict(completed=True,predictions=pin(path),models=meta['models'],checks=rows,
    references_not_decoded_in_this_stage=True,no_new_fit=True,rank35_is_training=True,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(completed=True,predictions=14,max_delta_relative=max(r['delta_relative'] for r in rows)),flush=True)
