from common import *
from records import verify_records
build=admission();done('qualify_null');q=read(HERE/'null_qualification.json');assert q['passed']
for p in q['pins']:check(p)
trained,pins=verify_records(HERE/'native_trained_control',build['candidate'],'trained','control',[35,100,35],HERE/'native_trained_control_watch.json')
base=read(HERE/'native_null/result.json')['records']
import numpy as np
for a,b in zip(base,trained):
    x=np.load(a['prediction']['path'],allow_pickle=False);y=np.load(b['prediction']['path'],allow_pickle=False)
    assert x.shape==y.shape==(16,1024) and x.dtype==y.dtype==np.float32
    assert x[:8].tobytes()==y[:8].tobytes()
    assert a['gnn_states'][:8]==b['gnn_states'][:8]
write(HERE/'control_qualification.json',dict(passed=True,null=pin(HERE/'null_qualification.json'),
    trained_repeat_bytes_exact=True,analytic_layer0_parent_exact=True,first8_parent_and_gnn_states_exact=True,all_retirements_qualified=True,
    targets_decoded=False,pins=list({p['path']:p for p in q['pins']+pins}.values())))
print(dict(passed=True,new_control_calls=7,targets_decoded=False),flush=True)
