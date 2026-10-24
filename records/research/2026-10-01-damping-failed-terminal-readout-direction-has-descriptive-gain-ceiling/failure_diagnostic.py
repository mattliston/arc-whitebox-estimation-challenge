"""Post-failure diagnostic only: fixed OOF direction's quadratic loss terms."""
from pathlib import Path
import json,sys
from prepare_capture import *
import numpy as np
p=pin(D/'protocol.json');pp,c=plan(p['path'],p['sha256']);environment(c)
cv=read(D/'cv_result.json');assert not cv['admitted'] and cv['fit_count']==5
inputs=capture_inputs(c,pp);assert cv['inputs']==inputs
records=[]
for i,(item,fixture) in enumerate(zip(inputs,c['data'])):
    with np.load(checked(item['row']['packet']),allow_pickle=False) as z:
        host=z['host'].astype(np.float64);base=z['prediction'][-1].astype(np.float64)
        v=np.asarray(cv['folds'][i%5]['coefficient_vector'],dtype=np.float32)
        extra=((z['sigma'].astype(np.float32)*np.float32(z['rms']))*(z['h128']@v))
        extra=(extra*z['effective_gain']).astype(np.float32)
        delta=(z['delta']+extra).astype(np.float32)
        pred=np.maximum(host+delta.astype(np.float64),0).astype(np.float32).astype(np.float64)
    with np.load(checked(fixture['truth']),allow_pickle=False) as z:y=z['mc_post'][-1].astype(np.float64)
    e=base-y;direction=pred-base
    records.append(dict(rank=fixture['rank'],fold=i%5,E=float(e@e),G=float(e@direction),H=float(direction@direction)))
def stats(rows):
    E=sum(x['E'] for x in rows);G=sum(x['G'] for x in rows);K=sum(x['H'] for x in rows)
    alpha=min(1.,max(0.,-G/K))
    return dict(E=E,G=G,H=K,linear_pct=200*G/E,quadratic_pct=100*K/E,
      direction_helpful_at_zero=G<0,alpha_quadratic_min_in_0_1=alpha,
      descriptive_secant_gain_bound_pct=-100*(2*alpha*G+alpha*alpha*K)/E)
out=dict(protocol=pp,cv_result=pin(D/'cv_result.json'),source=pin(__file__),
    pooled=stats(records),folds=[stats([r for r in records if r['fold']==f]) for f in range(5)],
    records=records,diagnostic_only=True,no_refit=True,no_candidate=True,
    limitation='Post-failure descriptive quadratic for the saved rounded OOF direction, not validation or a new fitted endpoint.')
once(D/'failure_diagnostic.json',out);print(json.dumps({k:out[k] for k in ('pooled','folds','limitation')}))

