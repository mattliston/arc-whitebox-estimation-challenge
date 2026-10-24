"""Rebuild ridge by augmented least squares and scores by scalar fsum."""
import math
import numpy as np
from common import *
from basis import actions
verify();r=read(HERE/'result.json');assert r['validity']
for name in ('unit_guard.json','run_guard.json'):
    g=read(HERE/name);assert g['completed'] and g['whole_group_retired'] and not g['remaining_rows'] and g['max_swap_growth_bytes']==0
for e in read(HERE/'before_targets.json')['packets']+r['targets']:assert pin(e['path'])==e
for name,key in (('before_targets.json','responses'),('model_frozen.json','model'),('predictions_frozen.json','predictions')):
    e=read(HERE/name)[key];assert pin(e['path'])==e
assert not read(HERE/'model_frozen.json')['check_targets_accessed']
assert not read(HERE/'predictions_frozen.json')['check_targets_accessed']
with np.load(HERE/'responses.npz') as z:responses={k:z[k] for k in z.files}
with np.load(HERE/'model.npz') as z:model={k:z[k] for k in z.files}
with np.load(HERE/'predictions.npz') as z:predictions={k:z[k] for k in z.files}
truth={};base={}
for rank in TRAIN+CHECK:
    with np.load(P2/'fable_refit_v1/truth688'/f'truthpost_rank{rank:04d}.npz') as z:truth[rank]=z['mc_post'][15].astype(float)
    err=(float(a)-float(b) for a,b in zip(responses[f'parent_{rank}'],truth[rank]))
    base[rank]=math.fsum(e*e for e in err)/len(truth[rank])
scale=np.sqrt(np.array([math.fsum(math.fsum(float(x)*float(x) for x in responses[f'response_{rank}'][:,j])/len(truth[rank])/base[rank] for rank in TRAIN)/len(TRAIN) for j in range(15)]))
assert np.allclose(scale,model['scale'],rtol=1e-12,atol=0)
fit_errors=[]
for arm,count in (('local',5),('augmented',15)):
    X=np.vstack([responses[f'response_{rank}'][:,:count]/(math.sqrt(base[rank])*scale[:count]) for rank in TRAIN])
    y=np.concatenate([(truth[rank]-responses[f'parent_{rank}'])/math.sqrt(base[rank]) for rank in TRAIN])
    Xa=np.vstack((X,math.sqrt(.01*len(y))*np.eye(count)));ya=np.concatenate((y,np.zeros(count)))
    beta=np.linalg.lstsq(Xa,ya,rcond=None)[0]
    error=float(np.max(np.abs(beta-model[arm+'_beta'])));assert error<1e-10;fit_errors.append(error)
    assert np.allclose(model[arm+'_coeff'][:count],beta/scale[:count],rtol=1e-9,atol=1e-12)
rows=[];differences=[]
for arm in ('local','augmented'):
    for rank in TRAIN+CHECK:
        diff=(float(a)-float(b) for a,b in zip(predictions[f'{arm}_{rank}'],truth[rank]));mse=math.fsum(x*x for x in diff)/len(truth[rank])
        old=next(x for x in r['rows'] if x['arm']==arm and x['rank']==rank)
        differences.extend((abs(mse/old['mse']-1),abs(base[rank]/old['parent_mse']-1)))
        rows.append(dict(arm=arm,rank=rank,split='TRAIN' if rank in TRAIN else 'CHECK',mse=mse,parent_mse=base[rank],raw_ratio=mse/base[rank]))
summary={}
for arm in ('local','augmented'):
    rr=[x for x in rows if x['arm']==arm and x['split']=='CHECK'];tr=[x for x in rows if x['arm']==arm and x['split']=='TRAIN']
    summary[arm]=dict(check_pooled_raw_ratio=math.fsum(x['mse'] for x in rr)/math.fsum(x['parent_mse'] for x in rr),check_wins=sum(x['raw_ratio']<1 for x in rr),check_worst=max(x['raw_ratio'] for x in rr),train_pooled_raw_ratio=math.fsum(x['mse'] for x in tr)/math.fsum(x['parent_mse'] for x in tr))
    differences.extend(abs(val-r['summary'][arm][key]) for key,val in summary[arm].items())
aug=summary['augmented'];relative=aug['check_pooled_raw_ratio']/summary['local']['check_pooled_raw_ratio']
gates=dict(pooled=aug['check_pooled_raw_ratio']<=.95,wins=aug['check_wins']>=6,worst=aug['check_worst']<=1.05,feature_value=relative<=.975)
assert gates==r['gates'] and all(gates.values())==r['passed'] and max(differences)<1e-12
query_error=[]
for rank in TRAIN[:2]:
    with np.load(packet(rank)) as z:a={k:z[k] for k in ('C','K','s14','F','W')}
    for j,(name,D) in enumerate(actions(a['C'],a['K'],a['s14'],a['F'][:,5].astype(float),a['F'][:,6].astype(float))):
        for col in (0,37,999):
            w=a['W'][:,col].astype(float)
            actual=float(np.einsum('i,ij,j->',w,D,w,optimize=False))
            error=abs(actual-responses[f'variance_{rank}'][col,j]);query_error.append(error)
            assert error<1e-11*max(1.,abs(actual))
once(HERE/'verification.json',dict(passed=True,study_passed=r['passed'],gates=gates,maximum_ridge_beta_difference=max(fit_errors),maximum_score_relative_difference=max(differences),maximum_variance_witness_absolute=max(query_error),witness_count=len(query_error),result_sha256=sha(HERE/'result.json'),verifier=pin(Path(__file__))))
print(json.dumps(dict(verified=True,study_passed=r['passed'],ridge_max=max(fit_errors),score_max=max(differences),variance_witness_max=max(query_error),summary=summary)))
