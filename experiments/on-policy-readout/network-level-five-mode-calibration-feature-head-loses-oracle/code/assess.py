"""Open development labels after all four fixed models are sealed."""
import json
import numpy as np
from common import D, bind, sha, once, load, response

cfg,records=bind()
pred={m:{} for m in ('intercept','network')};models=[]
for fold in range(4):
    guard=json.loads((D/f'fold_{fold}_guard.json').read_text())
    assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
    folder=D/f'fold_{fold}'; result=json.loads((folder/'result.json').read_text())
    assert result['test_ranks']==cfg['folds'][fold]
    assert not result['test_labels_accessed'] and not result['check32_accessed'] and not result['valid40_accessed']
    assert result['source_seal_sha256']==sha(D/'source_seal.json')
    models.append(result)
    for mode in pred:
        model=result['models'][mode]
        assert sha(folder/f'{mode}_model.npz')==model['sha256']
        path=folder/f'{mode}_prediction.npy';assert sha(path)==model['prediction_sha256']
        prediction=np.load(path);assert prediction.shape==(32,1024)
        pred[mode].update(zip(result['test_ranks'],prediction))
once(D/'preassessment_seal.json',dict(models=models,assessment_labels_accessed=False,
    source_seal_sha256=sha(D/'source_seal.json')))
ranks=cfg['ranks']
x,sigma,base,truth,targets=load(ranks,records,cfg,True,ranks)
mse={'bank':np.mean((base-truth)**2,axis=1)}
for mode in pred:mse[mode]=np.mean((np.stack([pred[mode][r] for r in ranks])-truth)**2,axis=1)
phi=response(x,sigma)
oracle=[]
for index,rank in enumerate(ranks):
    coeff,_,_,_=np.linalg.lstsq(phi[index],truth[index]-base[index],rcond=None)
    residual=base[index]+phi[index]@coeff-truth[index]
    oracle.append(float(np.mean(residual**2)))
mse['oracle']=np.array(oracle)
rng=np.random.default_rng(2026092702)
indices=rng.integers(0,len(ranks),size=(50000,len(ranks)))
def compare(a,b):
    ratio=float(a.mean()/b.mean())
    interval=np.quantile(a[indices].mean(1)/b[indices].mean(1),[.025,.975]).tolist()
    return dict(raw_ratio=ratio,change_pct=100*(ratio-1),ci95_ratio=interval,
                ci95_pct=[100*(v-1) for v in interval],wins=int((a<b).sum()))
comparisons={m:compare(mse[m],mse['bank']) for m in ('intercept','network','oracle')}
contrast=compare(mse['network'],mse['intercept'])
fold_scores={};gates={}
for mode in pred:
    ratios=[]
    for fold in cfg['folds']:
        ix=[ranks.index(r) for r in fold]
        ratios.append(float(mse[mode][ix].mean()/mse['bank'][ix].mean()))
    fold_scores[mode]=ratios;c=comparisons[mode]
    gates[mode]=dict(raw=c['raw_ratio']<=.97,confidence=c['ci95_ratio'][1]<.98,
                     wins=c['wins']>=96,all_folds=all(v<1 for v in ratios))
gates['network']['incremental']=contrast['raw_ratio']<=.99
eligible={m:all(v.values()) for m,v in gates.items()}
survivor='network' if eligible['network'] else ('intercept' if eligible['intercept'] else None)
result=dict(scope='Reused FIT128 development; weights separated by network fold, no pristine validation claim',
    protocol_sha256=sha(D/'protocol.json'),source_seal_sha256=sha(D/'source_seal.json'),
    comparisons_vs_bank=comparisons,network_vs_intercept=contrast,fold_raw_ratios=fold_scores,
    gates=gates,survivor=survivor,original_check32_accessed=False,valid40_accessed=False,
    native_qualified=False,submission_qualified=False,
    oracle_scope='Unattainable per-network fitted correction in the five fixed response modes',
    per_network=[dict(rank=r,**{m:float(v[i]) for m,v in mse.items()}) for i,r in enumerate(ranks)],
    target_files=targets,decision='Advance fixed survivor' if survivor else 'Close fixed common-mode fit without rescue')
once(D/'assessment.json',result)
print(json.dumps({k:result[k] for k in ('comparisons_vs_bank','network_vs_intercept','fold_raw_ratios','gates','survivor','decision')},indent=2),flush=True)
