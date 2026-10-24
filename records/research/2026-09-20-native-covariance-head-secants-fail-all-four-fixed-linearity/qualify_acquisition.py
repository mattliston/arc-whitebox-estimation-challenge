import numpy as np
from common import HERE,PARENT,TRAIN,read,write,pin,verify,seal_check,sha
seal_check();done=read(HERE/'acquisition_completed.json');rows=[]
assert [x['rank'] for x in done['rows']]==TRAIN
for row in done['rows']:
    rank=row['rank'];base=row['parent'];verify(base['source']);verify(base['receipt'])
    assert base['source']['sha256']==sha(PARENT)
    bc=base['call'];x=np.load(bc['output']);prior=row['saved_parent'];verify(prior['output'])
    assert x.shape==(16,1024) and x.dtype==np.float32 and x.tobytes()==np.load(prior['output']['path']).tobytes()
    assert bc['rank']==rank and bc['flops']==prior['flops'] and bc['includes_initialization']
    assert [a['coordinate'] for a in row['secants']]==list(range(16))
    for a in row['secants']:
        for key in ['source','receipt','watch']:verify(a[key])
        c=a['call'];y=np.load(c['output']);assert y.shape==x.shape and y.dtype==x.dtype
        assert np.isfinite(y).all() and (y>=0).all() and y[:8].tobytes()==x[:8].tobytes()
        assert c['rank']==rank and c['includes_initialization'] and c['weights_sha256']==bc['weights_sha256']
        assert c['flops']<2**41 and c['weight5_cache']['numeric_cache_retired']
    rows.append(dict(rank=rank,base_output=pin(bc['output']),base_flops=bc['flops'],
        columns=[dict(coordinate=a['coordinate'],output=pin(a['call']['output']),flops=a['call']['flops'],epsilon=a['epsilon']) for a in row['secants']],
        truth=row['saved_parent']['truth']))
write(HERE/'acquisition_qualified.json',dict(passed=True,rows=rows,all16_columns_per_rank=True,
    all_first8_bytes_exact=True,all_packed_base_bytes_and_bills_exact=True,truth_arrays_accessed=False,
    acquisition=pin(HERE/'acquisition_completed.json')))
print(dict(passed=True,networks=len(rows),columns=96,truth_arrays_accessed=False),flush=True)
