"""Independent scalar aggregation, packet pins, saved endpoints; no model fit."""
import math
import numpy as np
from common import *
verify();r=read(HERE/'fit_result.json');frozen=read(HERE/'model_frozen.json');forecast=read(HERE/'predictions_frozen.json')
for k in ('model','normalizers'):assert pin(frozen[k]['path'])==frozen[k]==r[k]
assert pin(forecast['predictions']['path'])==forecast['predictions']
for p in read(HERE/'before_train_targets.json')['packets']:assert pin(p['path'])==p
for p in r['targets']:assert pin(p['path'])==p
errors=[];rows=[]
with np.load(HERE/'predictions.npz') as z:
    for rank in TRAIN+CHECK:
        path=P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz'
        with np.load(path) as a:truth=a['mc_post'][15].astype(np.float64)
        source=HERE/('capture_exact_00' if rank in TRAIN[:2] else 'capture_exact_01')/f'row_{rank:04d}.npz'
        with np.load(source) as a:parent=a['prediction'][-1]
        base=math.fsum((float(x)-float(y))**2 for x,y in zip(parent,truth))/len(truth)
        mse=math.fsum((float(x)-float(y))**2 for x,y in zip(z[str(rank)],truth))/len(truth)
        expected=next(x for x in r['rows'] if x['rank']==rank)
        errors.extend((abs(base-expected['parent_mse']),abs(mse-expected['mse']),abs(mse/base-expected['raw_ratio'])))
        rows.append(dict(rank=rank,parent=base,mse=mse,ratio=mse/base))
held=[x for x in rows if x['rank'] in CHECK];pooled=math.fsum(x['mse'] for x in held)/math.fsum(x['parent'] for x in held)
assert max(errors)<1e-12 and abs(pooled-r['check_pooled_raw_ratio'])<1e-12
assert all(x['minimum_variance']>0 for x in forecast['diagnostics'])
passed=pooled<=.975 and sum(x['ratio']<1 for x in held)>=6 and max(x['ratio'] for x in held)<=1.05
assert passed==r['passed']
once(HERE/'verification.json',dict(passed=True,maximum_aggregation_difference=max(errors),model_sha256=r['model']['sha256'],checked_packets=20,independent_check_gate_passed=passed,pooled_check_raw_ratio=pooled))
print(json.dumps(dict(verified=True,model_gate_passed=passed,pooled=pooled)),flush=True)
