from common import *
from records import verify_records
build=admission();done('qualify_controls');control=read(HERE/'control_qualification.json');assert control['passed']
for p in control['pins']:check(p)
meta=read(HERE/'inputs.json');rows=[];pins=control['pins']+[pin(HERE/'control_qualification.json'),pin(HERE/'build.json')]
import numpy as np
for rank in meta['valid']:
    row=dict(rank=rank,roles={})
    for role in ('parent','trained'):
        name=f'{role}_{rank}'
        records,more=verify_records(HERE/name,build['parent' if role=='parent' else 'candidate'],role,'validation',[rank],HERE/f'{name}_watch.json')
        assert len(records)==1 and records[0]['includes_initialization']
        row['roles'][role]=records[0];pins+=more
    x=np.load(row['roles']['parent']['prediction']['path'],allow_pickle=False)
    y=np.load(row['roles']['trained']['prediction']['path'],allow_pickle=False)
    assert x.shape==y.shape==(16,1024) and x.dtype==y.dtype==np.float32
    assert x[:1].tobytes()==y[:1].tobytes()
    rows.append(row)
write(HERE/'native40_predictions_sealed.json',dict(completed=True,before_validation_targets=True,
    scored_ranks=meta['valid'],rows=rows,pins=list({p['path']:p for p in pins}.values()),
    own_complete_bills=True,matched_cold_initialization=True,analytic_layer0_parent_exact=True,
    all_retirements_qualified=True,control_calls_excluded=True,
    comparison_gate=dict(adjusted_ratio_strictly_below=1,wins_at_least=24)))
print(dict(sealed=True,parent_calls=40,trained_calls=40,validation_targets_decoded=False),flush=True)
