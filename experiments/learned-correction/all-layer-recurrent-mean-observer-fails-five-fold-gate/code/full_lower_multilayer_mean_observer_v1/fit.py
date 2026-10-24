import resource
import numpy as np
from common import HERE,TRAIN,read,write,pin,verify,seal_check
from observer_math import raw_features,local_design,normalize_design,normalizers,solve_gram,direct

seal_check('fit_sources_sealed.json');designs=read(HERE/'designs_sealed.json');assert designs['passed'] and not designs['targets_accessed']
inputs={x['rank']:x for x in read(HERE/'inputs.json')['rows']};packets={};raw={};rows={r['rank']:r for r in designs['rows']}
for rank in TRAIN:
    r=rows[rank];verify(r['record']['capture']);verify(r['design'])
    with np.load(r['record']['capture']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
    packets[rank]=a;raw[rank]=raw_features(a)
folds=[]
for fold in range(5):
    fit_ranks=[r for i,r in enumerate(TRAIN) if i%5!=fold];held=[r for i,r in enumerate(TRAIN) if i%5==fold]
    mu,scale=normalizers([raw[r] for r in fit_ranks]);folds.append(dict(fold=fold,fit_ranks=fit_ranks,held_ranks=held,mean=mu,scale=scale))
write(HERE/'fit_inputs_sealed_before_targets.json',dict(designs=pin(HERE/'designs_sealed.json'),source_seal=pin(HERE/'fit_sources_sealed.json'),folds=[dict(fold=f['fold'],fit=f['fit_ranks'],held=f['held_ranks']) for f in folds],target_arrays_decoded=False,ridge=1e-3,provisional_added_flops=1_000_000_000))
truth={}
for rank in TRAIN:
    verify(inputs[rank]['truth']);assert inputs[rank]['weights']['sha256']==rows[rank]['record']['weights']['sha256']
    with np.load(inputs[rank]['truth']['path'],allow_pickle=False) as z:truth[rank]=z['mc_post'].astype(np.float64)
    assert truth[rank].shape==(16,1024) and np.isfinite(truth[rank]).all()

def fit_model(ranks,mu,scale,role):
    gram=np.zeros((102,102));rhs=np.zeros(102);total=sum(rows[r]['record']['multiplier']*8192 for r in ranks)
    for rank in ranks:
        a=packets[rank]
        raw_b=np.load(rows[rank]['design']['path'],allow_pickle=False) if role=='recurrent' else local_design(raw[rank],a['sigma'][8:])
        b=normalize_design(raw_b,mu,scale).reshape(-1,102);y=(truth[rank][8:]-a['prediction'][8:].astype(np.float64)).reshape(-1)
        weight=rows[rank]['record']['multiplier']/total
        gram+=weight*(b.T@b);rhs+=weight*(b.T@y)
    return solve_gram(gram,rhs)

def predict(rank,mu,scale,beta,role):
    a=packets[rank];w=np.load(rows[rank]['record']['weights']['path'],mmap_mode='r',allow_pickle=False)[8:]
    delta=direct(raw[rank],a['sigma'][8:],a['features'][8:,:,5].astype(np.float64),w,mu,scale,beta,role=='recurrent')
    return np.maximum(a['prediction'][-1].astype(np.float64)+delta[-1],0).astype(np.float32)

predictions={k:{} for k in ['recurrent','local']};fits=[]
for f in folds:
    for role in ['recurrent','local']:
        beta,cert=fit_model(f['fit_ranks'],f['mean'],f['scale'],role)
        path=HERE/f'fold{f["fold"]}_{role}.npz';np.savez(path,mean=f['mean'],scale=f['scale'],beta=beta)
        fits.append(dict(fold=f['fold'],role=role,coefficients=pin(path),equations=cert,fit_ranks=f['fit_ranks'],held_ranks=f['held_ranks']))
        for rank in f['held_ranks']:predictions[role][rank]=predict(rank,f['mean'],f['scale'],beta,role)
        print(dict(fold=f['fold'],role=role,equation_error=cert['normal_equation_relative_error']),flush=True)
paths={}
for role in predictions:
    path=HERE/f'oof_{role}.npy';np.save(path,np.stack([predictions[role][r] for r in TRAIN]));paths[role]=pin(path)
write(HERE/'oof_predictions_sealed.json',dict(predictions=paths,fits=fits,selection_used=False))
scores=[]
for i,rank in enumerate(TRAIN):
    c=rows[rank]['record'];p=float(np.mean((packets[rank]['prediction'][-1].astype(np.float64)-truth[rank][-1])**2));own=max(.1,(c['flops']+1_000_000_000)/2**41)
    entry=dict(rank=rank,fold=i%5,parent_raw=p,parent_adjusted=p*c['multiplier'],parent_flops=c['flops'],provisional_added_flops=1_000_000_000)
    for role in predictions:
        mse=float(np.mean((predictions[role][rank].astype(np.float64)-truth[rank][-1])**2));entry[role]=dict(raw=mse,adjusted=mse*own,adjusted_ratio=mse*own/entry['parent_adjusted'])
    scores.append(entry)
p=sum(x['parent_adjusted'] for x in scores);r=sum(x['recurrent']['adjusted'] for x in scores);c=sum(x['local']['adjusted'] for x in scores)
fold_scores=[dict(fold=f,ratio=sum(x['recurrent']['adjusted'] for x in scores if x['fold']==f)/sum(x['parent_adjusted'] for x in scores if x['fold']==f)) for f in range(5)]
wins=sum(x['recurrent']['adjusted_ratio']<1 for x in scores);fold_wins=sum(x['ratio']<1 for x in fold_scores);worst=max(x['recurrent']['adjusted_ratio'] for x in scores)
passed=r/p<=.95 and r/c<=.99 and fold_wins>=4 and wins>=12 and worst<=1.20
result=dict(passed=passed,pooled_adjusted_ratio=r/p,recurrent_over_local=r/c,fold_wins=fold_wins,wins=wins,worst_adjusted_ratio=worst,folds=fold_scores,rows=scores,cv_fits=10,provisional_added_flops=1_000_000_000,actual_native_added_cost_qualified=False,official_score=False,pristine_holdout=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
write(HERE/'cv_result.json',result);print({k:v for k,v in result.items() if k not in ['rows','folds']},flush=True)
if not passed:
    write(HERE/'outcome.json',dict(stage='cv',passed=False,study_closed=True,full_fit=False,uploaded=False));raise SystemExit(0)
mu,scale=normalizers([raw[r] for r in TRAIN]);beta,cert=fit_model(TRAIN,mu,scale,'recurrent')
path=HERE/'terminal.npz';np.savez(path,mean=mu,scale=scale,beta=beta)
write(HERE/'full_fit.json',dict(completed=True,model=pin(path),equations=cert,train=TRAIN,total_fits=11))
write(HERE/'outcome.json',dict(stage='cv',passed=True,study_closed=False,full_fit=True,CHECK_pending=True,uploaded=False))
