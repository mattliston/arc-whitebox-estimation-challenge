"""Fixed row-preserving diagonal completion; no truth or fitted coefficient."""
from pathlib import Path
import hashlib
import json
import resource
import time
import numpy as np

D = Path(__file__).resolve().parent
O = D.parents[2]
P = O.parent
S = P / 'learned_residual_observer_design/v4_feature_store_v3/sealed_target_blind_feature_store_v3'
planes_path = O / 'engineer/a33_starsum/planes/s38_rank0651.npz'
weights_path = S / 'rows/observer_fit/0651/weights.npy'
aux_path = O / 'engineer/a36_carrier/state/s38_rank0651.npz'
prereg = dict(design='Ahat=a*v.T; d=(diag(A)-a*v)/(1-1/n); DeltaA=diag(d)-d*ones.T/n',
             beta=.85, ranks=[651], layers=list(range(1,16)),
             coefficient=1, target='exact saved star transported diagonal, not labels',
             scope='Historical S38 planes, current v17 rank1 beta. Geometry falsifier only; no current-native accuracy claim.',
             no_fit=True, no_grid=True)
(D/'PREREG_v1.json').write_text(json.dumps(prereg,indent=2)+'\n')
start=time.monotonic()
weights=np.load(weights_path)
planes=np.load(planes_path)
aux=np.load(aux_path)
rows=[]
for l in range(1,16):
    A=planes[f'star_into_layer_{l:02d}'].astype(np.float64)
    W=weights[l].astype(np.float64)
    n=len(A)
    a=A.sum(axis=1)
    v=.85*(a@A)/(a@a)+.15/n
    approx=(W.T@a)*((W*W).T@v)
    exact=((W*W)*(A.T@W)).sum(axis=0)
    d=(np.diag(A)-a*v)/(1-1/n)
    correction=(W*W*W).T@d-(W*W).mean(axis=0)*(W.T@d)
    err=approx-exact
    after=err+correction
    row=dict(layer=l, error2=float(err@err), after_error2=float(after@after),
             recovery_pct=float(100*(1-after@after/(err@err))),
             correction_cosine=float((-err@correction)/np.linalg.norm(err)/np.linalg.norm(correction)),
             correction_to_error_norm=float(np.linalg.norm(correction)/np.linalg.norm(err)),
             diag_a_relative=float(np.linalg.norm(np.diag(A))/np.linalg.norm(A)),
             row_sum_residual=float(abs(v.sum()-1)))
    if l<15:
        g,unused,sigma,alpha,pdf=aux[f'aux_{l:02d}']
        sensitivity=-alpha*pdf/(6*sigma*sigma)
        we=err*sensitivity; wa=after*sensitivity
        row.update(weighted_error2=float(we@we), weighted_after_error2=float(wa@wa),
                   mean_sensitivity_recovery_pct=float(100*(1-wa@wa/(we@we))))
    rows.append(row)
e=sum(x['error2'] for x in rows); a2=sum(x['after_error2'] for x in rows)
we=sum(x.get('weighted_error2',0) for x in rows);wa=sum(x.get('weighted_after_error2',0) for x in rows)
result=dict(prereg=prereg, rows=rows, total_recovery_pct=100*(1-a2/e),
            mean_sensitivity_total_recovery_pct=100*(1-wa/we),
            inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (planes_path,weights_path,aux_path)},
            seconds=time.monotonic()-start, maxrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
(D/'RESULT_v1.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','prereg')},indent=2))
