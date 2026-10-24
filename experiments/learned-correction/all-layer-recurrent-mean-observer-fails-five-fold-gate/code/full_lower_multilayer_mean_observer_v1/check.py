"""Fixed CHECK forecast, sealed before decoding either CHECK mean target."""
import numpy as np
from common import HERE,CHECK,read,write,pin,verify,seal_check
from observer_math import raw_features,direct
seal_check('fit_sources_sealed_v2.json');assert read(HERE/'cv_result.json')['passed']
fit=read(HERE/'full_fit.json');verify(fit['model'])
with np.load(fit['model']['path'],allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
inputs={r['rank']:r for r in read(HERE/'inputs.json')['rows']};designs={r['rank']:r for r in read(HERE/'designs_sealed.json')['rows']};forecasts=[]
for rank in CHECK:
    c=designs[rank]['record'];verify(c['capture']);verify(c['weights'])
    with np.load(c['capture']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    w=np.load(c['weights']['path'],mmap_mode='r',allow_pickle=False)[8:]
    delta=direct(raw_features(a),a['sigma'][8:],a['features'][8:,:,5].astype(np.float64),w,model['mean'],model['scale'],model['beta'])
    prediction=a['prediction'].copy();prediction[-1]=np.maximum(prediction[-1].astype(np.float64)+delta[-1],0).astype(np.float32)
    assert np.isfinite(prediction).all() and (prediction>=0).all() and prediction[:15].tobytes()==a['prediction'][:15].tobytes()
    path=HERE/f'check_rank{rank:04d}.npz';np.savez(path,prediction=prediction,delta=delta)
    forecasts.append(dict(rank=rank,prediction=pin(path),parent=c,truth=inputs[rank]['truth']))
write(HERE/'check_predictions_sealed.json',dict(rows=forecasts,model=fit['model'],check_targets_not_decoded=True))
scores=[]
for r in forecasts:
    verify(r['truth'])
    with np.load(r['truth']['path'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
    with np.load(r['parent']['capture']['path'],allow_pickle=False) as z:p=z['prediction'][-1].astype(np.float64)
    with np.load(r['prediction']['path'],allow_pickle=False) as z:q=z['prediction'][-1].astype(np.float64)
    pr=float(np.mean((p-truth)**2));qr=float(np.mean((q-truth)**2));c=r['parent']
    pa=pr*c['multiplier'];qa=qr*max(.1,(c['flops']+1_000_000_000)/2**41)
    scores.append(dict(rank=r['rank'],parent_raw=pr,candidate_raw=qr,parent_adjusted=pa,candidate_adjusted=qa,adjusted_ratio=qa/pa))
ratio=sum(r['candidate_adjusted'] for r in scores)/sum(r['parent_adjusted'] for r in scores)
passed=ratio<=.97 and all(r['adjusted_ratio']<=1.02 for r in scores)
result=dict(passed=passed,pooled_adjusted_ratio=ratio,rows=scores,provisional_added_flops=1_000_000_000,actual_native_added_cost_qualified=False,official_score=False,pristine_holdout=False)
write(HERE/'check_result.json',result);write(HERE/'check_outcome.json',dict(stage='check',passed=passed,study_closed=not passed,native_port_pending=passed,uploaded=False));print(result,flush=True)
