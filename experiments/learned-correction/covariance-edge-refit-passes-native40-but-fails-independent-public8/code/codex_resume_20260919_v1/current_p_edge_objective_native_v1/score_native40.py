from common import *
from scoring import summarize
admission();done('seal_predictions')
sealed=read(HERE/'native40_predictions_sealed.json')
assert sealed['completed'] and sealed['before_validation_targets'] and sealed['matched_cold_initialization']
assert sealed['own_complete_bills'] and sealed['control_calls_excluded'] and sealed['first8_parent_layers_exact']
for p in sealed['pins']:check(p)
meta=read(HERE/'inputs.json');assert sealed['scored_ranks']==meta['valid']
import numpy as np
assert np.__version__=='2.4.6'
predictions={role:[np.load(r['roles'][role]['prediction']['path'],allow_pickle=False) for r in sealed['rows']] for role in ('parent','trained')}
for x,y in zip(predictions['parent'],predictions['trained']):
    assert x.shape==y.shape==(16,1024) and x.dtype==y.dtype==np.float32 and x[:8].tobytes()==y[:8].tobytes()
rows=[]
for index,rank in enumerate(meta['valid']):
    tp=meta['targets'][str(rank)];check(tp)
    with np.load(tp['path'],allow_pickle=False) as z:target=z['mc_post'].astype(np.float64)
    assert target.shape==(16,1024) and np.isfinite(target).all()
    row=dict(rank=rank,target=tp,roles={})
    for role in ('parent','trained'):
        r=sealed['rows'][index]['roles'][role];assert r['rank']==rank and r['includes_initialization']
        mse=np.mean((predictions[role][index].astype(np.float64)-target)**2,axis=1)
        assert np.isfinite(mse).all() and (mse>=0).all()
        row['roles'][role]=dict(final_raw=float(mse[-1]),final_adjusted=float(mse[-1]*r['multiplier']),
            per_layer_raw=mse.tolist(),flops=r['flops'],multiplier=r['multiplier'],prediction=r['prediction'])
    rows.append(row)
summary=summarize(rows)
write(HERE/'native40_score.json',dict(completed=True,integrity_passed=True,**summary,rows=rows,
    first8_layers_byte_exact=True,all_predictions_and_bills_verified_before_validation_targets=True,
    prediction_seal=pin(HERE/'native40_predictions_sealed.json'),matched_cold_initialization=True,
    native_score_measured=True,native_derivative_certified=False,reused_VALID40=True,
    advances_to_downstream_checks=summary['passed'],candidate=read(HERE/'build.json')['candidate'],
    official_changed=False,goal_achieved=False))
print(dict(completed=True,**summary),flush=True)
