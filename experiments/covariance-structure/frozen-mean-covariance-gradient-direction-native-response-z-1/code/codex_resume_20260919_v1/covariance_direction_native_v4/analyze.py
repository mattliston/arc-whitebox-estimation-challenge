"""Predeclared response qualification, then one scalar constrained fit."""
import sys,resource
from common import HERE,FIT,H,check,meta,read,sha,write
check()
import numpy as np
from threadpoolctl import threadpool_info
assert all(x['num_threads']==1 for x in threadpool_info())
ranks=[int(x) for x in sys.argv[1].split(',')];assert ranks in ([35],list(FIT))
rows=[];responses=[];parents=[];inputs=read(HERE/'inputs.json')['fit']
for rank in ranks:
    item=next(x for x in inputs if x['rank']==rank)
    with np.load(item['capture'],allow_pickle=False) as z:parent=z['prediction'].astype(np.float64)
    values={};receipts={}
    for label in ('plus','minus','halfplus','halfminus'):
        folder=HERE/f'native_rank{rank:04d}_{label}';r=read(folder/'result.json')
        g=read(HERE/f'native_rank{rank:04d}_{label}_watch.json')
        assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
        assert r['completed'] and r['trust_envelope_passed'] and sha(r['prediction']['path'])==r['prediction']['sha256']
        values[label]=np.load(r['prediction']['path'],allow_pickle=False).astype(np.float64)[-1]
        receipts[label]=meta(folder/'result.json')
    y0=parent[-1];full=(values['plus']-values['minus'])/(2*H)
    half=(values['halfplus']-values['halfminus'])/H
    norm=float(np.linalg.norm(half));assert norm>0
    stability=float(np.linalg.norm(full-half)/norm)
    signal=float(np.sqrt(np.mean((H*half)**2)))
    curvature=float(np.linalg.norm(values['plus']+values['minus']-2*y0)/np.linalg.norm(values['plus']-values['minus']))
    passed=stability<=.05 and signal>=5e-6 and curvature<=.10
    rows.append(dict(rank=rank,stability=stability,signal_rms=signal,curvature=curvature,
                     passed=bool(passed),native_receipts=receipts))
    responses.append(half);parents.append(y0)
qualified=all(x['passed'] for x in rows)
result=dict(completed=True,ranks=ranks,response_qualified=qualified,response_rows=rows,
    truth_accessed=False,capacity_bound=False,native_gradient_qualified=False,
    source_seal=meta(HERE/'sources_sealed.json'))
if qualified and ranks==list(FIT):
    numerator=denominator=baseline=0.;errors=[];weights=[]
    for rank,r,parent in zip(ranks,responses,parents):
        item=next(x for x in inputs if x['rank']==rank)
        with np.load(item['truth'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        e=parent-truth;w=item['parent_multiplier'];errors.append(e);weights.append(w)
        numerator+=w*float(np.mean(r*e));denominator+=w*float(np.mean(r*r));baseline+=w*float(np.mean(e*e))
    unconstrained=-numerator/(1.001*denominator);t=float(np.clip(unconstrained,-H,H))
    mse=[float(np.mean((e+t*r)**2)) for e,r in zip(errors,responses)]
    individual=[m/float(np.mean(e*e)) for m,e in zip(mse,errors)]
    ratio=sum(w*m for w,m in zip(weights,mse))/baseline
    result.update(truth_accessed=True,scalar_fit=True,unconstrained_t=unconstrained,t=t,
        radius=H,individual_predicted_ratios=individual,pooled_predicted_ratio=ratio,
        information_passed=bool(ratio<=.80 and max(individual)<=1.),
        predicted_only=True,actual_nonlinear_score=False)
    np.savez(HERE/'linear_response_fit.npz',response=np.stack(responses),parent=np.stack(parents),
             error=np.stack(errors),weight=np.asarray(weights),t=np.asarray(t))
name='qualification35.json' if ranks==[35] else 'information_result.json'
result['peak_rss_bytes']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
write(HERE/name,result);print(result,flush=True)
