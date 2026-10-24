"""Fixed five-fold ridge fit; only saved readout coordinates may change."""
import resource,time
import numpy as np
from threadpoolctl import threadpool_info
from common import HERE,TRAIN,TARGET_MANIFEST,read,sha,pin,write,verify,seal_check

RIDGE=1e-3

def solve(B,y,w):
    w=w/np.sum(w)
    scale=np.sqrt(np.sum(w[:,None]*B*B,axis=0))
    assert np.isfinite(scale).all() and np.min(scale)>1e-20
    X=B/scale
    gram=X.T@(w[:,None]*X)+RIDGE*np.eye(B.shape[1])
    rhs=X.T@(w*y)
    beta=np.linalg.solve(gram,rhs)
    error=float(np.linalg.norm(gram@beta-rhs)/max(np.linalg.norm(rhs),1e-30))
    assert error<1e-9
    return beta/scale,dict(normal_equation_relative_error=error,condition=float(np.linalg.cond(gram)))

def replay(a,coeff):
    h=np.asarray(coeff[:64],dtype=np.float32);b=np.float32(coeff[64])
    first=a['sigma'].astype(np.float32)*np.float32(a['rms'])
    middle=(a['hidden']@h+b).astype(np.float32)
    last=(np.float32(1)+np.float32(.25)*np.tanh(np.asarray(a['gain'],dtype=np.float32))).astype(np.float32)
    delta=((first*middle)*last).astype(np.float32)
    return np.maximum(a['host'].astype(np.float64)+delta.astype(np.float64),0).astype(np.float32)

def main():
    seal_check();assert all(t['num_threads']==1 for t in threadpool_info())
    began=time.monotonic();manifest=read(HERE/'captures_sealed.json')
    assert manifest['completed'] and not manifest['truth_accessed'] and manifest['train']==TRAIN
    rng=np.random.Generator(np.random.PCG64DXSM(2026092062))
    B=rng.normal(size=(113,7));y=rng.normal(size=113);w=np.linspace(.1,1,113)
    coeff,cert=solve(B,y,w)
    wn=w/w.sum();norm=np.sqrt(np.sum(wn[:,None]*B*B,axis=0))
    expected=np.linalg.lstsq(np.vstack([np.sqrt(wn)[:,None]*B/norm,np.sqrt(RIDGE)*np.eye(7)]),np.r_[np.sqrt(wn)*y,np.zeros(7)],rcond=None)[0]/norm
    assert np.allclose(coeff,expected,rtol=1e-10,atol=1e-12)
    data=[];max_replay=0
    for row in manifest['rows']:
        verify(row['watch']);verify(row['result']);verify(row['record']['capture']);verify(row['record']['weights'])
        guard=read(row['watch']['path']);assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
        with np.load(row['record']['capture']['path'],allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        old=np.r_[a['head'],a['head_b']].astype(np.float64)
        pred=replay(a,old);err=float(np.max(np.abs(pred-a['prediction'][-1])));max_replay=max(max_replay,err)
        assert err<=2e-7,(row['rank'],err)
        scale=(a['sigma'].astype(np.float32)*np.float32(a['rms'])).astype(np.float64)*(np.float32(1)+np.float32(.25)*np.tanh(np.asarray(a['gain'],dtype=np.float32))).astype(np.float64)
        basis=scale[:,None]*np.column_stack([a['hidden'],np.ones(1024)]).astype(np.float64)
        data.append(dict(row=row,a=a,old=old,B=basis,parent=a['prediction'][-1].astype(np.float64),unclipped=a['host'].astype(np.float64)+a['delta'].astype(np.float64)))
    assert all(np.array_equal(d['old'],data[0]['old']) for d in data)
    targets={r['rank']:r for r in read(TARGET_MANIFEST)['target_sources']}
    write(HERE/'fit_inputs_sealed_before_targets.json',dict(captures=pin(HERE/'captures_sealed.json'),
        source_seal=pin(HERE/'sources_sealed.json'),target_manifest=pin(TARGET_MANIFEST),
        solver_certificate=cert,solver_independent_max_error=float(np.max(np.abs(coeff-expected))),
        original_readout_max_error=max_replay,train=TRAIN,ridge=RIDGE,targets_decoded=False))
    for item in data:
        rank=item['row']['rank'];source=targets[rank]
        assert sha(source['target_file'])==source['target_sha256']
        assert sha(source['weight_file'])==item['row']['record']['weights']['sha256']
        with np.load(source['target_file'],allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        item.update(truth=truth,residual=truth-item['unclipped'])
    fits=[];predictions=[None]*20
    for fold in range(5):
        indices=[i for i in range(20) if i%5!=fold]
        B=np.concatenate([data[i]['B'] for i in indices]);y=np.concatenate([data[i]['residual'] for i in indices])
        w=np.concatenate([np.full(1024,data[i]['row']['record']['multiplier']) for i in indices])
        delta,eq=solve(B,y,w);head=(data[0]['old']+delta).astype(np.float32)
        for i in range(20):
            if i%5==fold:predictions[i]=replay(data[i]['a'],head)
        p=HERE/f'fold{fold}_coefficients.npz';np.savez(p,delta=delta,head=head)
        fits.append(dict(fold=fold,fit_ranks=[TRAIN[i] for i in indices],held_ranks=[TRAIN[i] for i in range(20) if i%5==fold],equations=eq,coefficients=pin(p)))
    p=HERE/'oof_predictions.npy';np.save(p,np.stack(predictions))
    write(HERE/'oof_sealed_before_aggregate.json',dict(predictions=pin(p),fits=fits,selection_used=False))
    rows=[]
    for i,item in enumerate(data):
        pmse=float(np.mean((item['parent']-item['truth'])**2));cmse=float(np.mean((predictions[i].astype(np.float64)-item['truth'])**2));mult=item['row']['record']['multiplier']
        rows.append(dict(rank=TRAIN[i],fold=i%5,parent_raw=pmse,candidate_raw=cmse,parent_adjusted=pmse*mult,candidate_adjusted=cmse*mult,ratio=cmse/pmse,flops=item['row']['record']['flops']))
    folds=[]
    for f in range(5):
        part=[r for r in rows if r['fold']==f];p=float(np.mean([r['parent_adjusted'] for r in part]));c=float(np.mean([r['candidate_adjusted'] for r in part]));folds.append(dict(fold=f,parent=p,candidate=c,ratio=c/p))
    parent=float(np.mean([r['parent_adjusted'] for r in rows]));candidate=float(np.mean([r['candidate_adjusted'] for r in rows]));ratio=candidate/parent
    wins=sum(f['ratio']<1 for f in folds);passed=ratio<1 and wins>=3 and max(f['ratio'] for f in folds)<=1.05
    result=dict(completed=True,cv_gate_passed=passed,pooled_parent_adjusted=parent,pooled_candidate_adjusted=candidate,pooled_adjusted_ratio=ratio,fold_wins=wins,folds=folds,rows=rows,ridge=RIDGE,source_seal=pin(HERE/'sources_sealed.json'),captures=pin(HERE/'captures_sealed.json'),fit_count=5,official_score=False,not_pristine_holdout=True,changed_native_bill_qualified=False,original_readout_max_error=max_replay,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,elapsed_s=time.monotonic()-began)
    write(HERE/'cv_result.json',result)
    print({k:v for k,v in result.items() if k not in ['rows','source_seal','captures']},flush=True)

if __name__=='__main__':main()
