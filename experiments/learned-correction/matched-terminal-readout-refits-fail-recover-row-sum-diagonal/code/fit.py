"""Fixed matched five-fold ridge, then separately admitted full-data export."""
import sys, time
from common import *


def prepare(pp,c):
    seal=dict(protocol=pp,fit_source=pin(__file__),ridge=.001,cv_gates=GATES,
              target_arrays_decoded=False,source_frozen_before_target_arithmetic=True)
    once(D/'fit_seal.json',seal)
    jobs={}
    for mode in ('cv','export'):
        cmd=[c['runtime'],'-B',str(Path(__file__).resolve()),'--'+mode,pp['path'],pp['sha256']]
        jobs[mode]=dict(command=cmd,guard_command=[c['light'],'-B',c['guard_source']['path'],'--record',str(D/(mode+'_guard.json')),
                       '--seconds','120','--rss-limit','7500000000',*cmd],log=str(D/(mode+'.log')))
    once(D/'fit_commands.json',dict(protocol=pp,seal=pin(D/'fit_seal.json'),jobs=jobs,thread_env=dict.fromkeys(THREADS,'1')))
    print(__import__('json').dumps(seal))


def perform(mode,pp,c):
    environment(c);prerequisites(c);seal=read(D/'fit_seal.json')
    assert seal['protocol']==pp and seal['fit_source']==pin(__file__) and seal['cv_gates']==GATES
    commands=read(D/'fit_commands.json')
    assert commands['protocol']==pp and commands['seal']==pin(D/'fit_seal.json')
    import numpy as np
    from threadpoolctl import threadpool_info
    assert np.__version__=='2.4.6' and all(x['num_threads']==1 for x in threadpool_info())
    replay=load(c['sources']['readout']['path'],'fixed_native_readout').replay
    editor=load(c['sources']['source_edit']['path'],'fixed_terminal_head_editor')
    inputs=capture_inputs(c,pp);data={arm:[] for arm in ARMS};started=time.monotonic()
    def solve(B,y,w):
        w=w/w.sum();scale=np.sqrt(np.sum(w[:,None]*B*B,axis=0))
        assert np.isfinite(scale).all() and np.min(scale)>1e-20
        X=B/scale;gram=X.T@(w[:,None]*X)+.001*np.eye(65);rhs=X.T@(w*y)
        coefficient=np.linalg.solve(gram,rhs)
        error=float(np.linalg.norm(gram@coefficient-rhs)/max(np.linalg.norm(rhs),1e-30));assert error<1e-9
        return coefficient/scale,scale,dict(normal_equation_relative_error=error,condition=float(np.linalg.cond(gram)))
    # Same fixed solver certificate as the existing terminal-head fitter, now 65 columns.
    rng=np.random.Generator(np.random.PCG64DXSM(2026092062));B=rng.normal(size=(113,65));y=rng.normal(size=113);w=np.linspace(.1,1,113)
    got,scale,certificate=solve(B,y,w);wn=w/w.sum()
    independent=np.linalg.lstsq(np.vstack((np.sqrt(wn)[:,None]*B/scale,np.sqrt(.001)*np.eye(65))),
                               np.r_[np.sqrt(wn)*y,np.zeros(65)],rcond=None)[0]/scale
    np.testing.assert_allclose(got,independent,rtol=1e-10,atol=1e-12)
    for arm in ARMS:
        literal=np.asarray(editor.terminal_head(Path(c['arms'][arm]['path']).read_text()),dtype=np.float32)
        for i,item in enumerate(inputs[arm]):
            row=item['row'];assert row['rank']==TRAIN[i] and row['weights']==c['data'][i]['weights']
            assert row['source']==c['arms'][arm] and not row['targets_accessed'] and row['readout_replay_exact']
            if i==0:assert row['independent_all16_and_bill_exact'] is True
            with np.load(checked(row['packet']),allow_pickle=False) as z:packet={k:z[k].copy() for k in z.files}
            old=np.r_[packet['head'],packet['head_b']].astype(np.float32);assert old.tobytes()==literal.tobytes()
            prediction,delta=replay(packet,old)
            assert delta.tobytes()==packet['delta'].tobytes() and prediction.astype(np.float64).tobytes()==packet['prediction'][-1].tobytes()
            physical_scale=(packet['sigma'].astype(np.float32)*np.float32(packet['rms'])).astype(np.float64)*packet['effective_gain'].astype(np.float64)
            B=physical_scale[:,None]*np.column_stack((packet['hidden'],np.ones(1024))).astype(np.float64)
            assert B.shape==(1024,65) and np.isfinite(B).all()
            data[arm].append(dict(packet=packet,old=old,B=B,unclipped=packet['host'].astype(np.float64)+delta.astype(np.float64),
                                  multiplier=row['multiplier'],flops=row['flops']))
    if mode=='cv':
        once(D/'features_frozen_before_targets.json',dict(protocol=pp,fit_source=pin(__file__),inputs=inputs,
             solver_certificate=certificate,incumbent_readout_replay_exact=True,targets_decoded_in_this_process=False))
    else:
        cvjob=commands['jobs']['cv']
        assert cvjob['command']==[c['runtime'],'-B',str(Path(__file__).resolve()),'--cv',pp['path'],pp['sha256']]
        guard(D/'cv_guard.json',cvjob['command'],120);cv=read(D/'cv_result.json')
        assert cv['completed'] and cv['protocol']==pp and cv['fit_source']==pin(__file__)
        assert cv['cv_gates']==GATES and cv['ridge']==.001 and cv['all_finite'] and cv['fit_count']==10
        assert cv['physical_admitted'] or cv['control_admitted']
    targets=[]
    for fixture in c['data']:
        with np.load(checked(fixture['truth']),allow_pickle=False) as z:truth=z['mc_post'][-1].astype(np.float64)
        assert truth.shape==(1024,) and np.isfinite(truth).all();targets.append(truth)
    def fit(arm,indices):
        a=data[arm]
        delta,scale,eq=solve(np.concatenate([a[i]['B'] for i in indices]),
             np.concatenate([targets[i]-a[i]['unclipped'] for i in indices]),
             np.concatenate([np.full(1024,a[i]['multiplier']) for i in indices]))
        head=(a[0]['old'].astype(np.float64)+delta).astype(np.float32);assert np.isfinite(head).all()
        return head,delta,scale,eq
    if mode=='export':
        exports={}
        for arm in ARMS:
            if arm=='k4' and not cv['physical_admitted']:continue
            head,delta,scale,eq=fit(arm,list(range(20)))
            source=Path(c['arms'][arm]['path']).read_text();candidate,disabled,patches,old=editor.edit(source,head)
            assert disabled==source.encode() and np.asarray(old,dtype=np.float32).tobytes()==data[arm][0]['old'].tobytes()
            path=D/f'candidate_{arm}_head.py'
            with path.open('xb') as f:f.write(candidate)
            model=D/f'model_{arm}.npz'
            with model.open('xb') as f:np.savez(f,head=head,delta=delta,old=data[arm][0]['old'],column_scale=scale)
            exports[arm]=dict(candidate=pin(path),source=c['arms'][arm],model=pin(model),equations=eq,
                 eligible_standalone=cv['physical_admitted'] if arm=='k4' else cv['control_admitted'],
                 inverse_whole_source_exact=True,changed_parameters=65,added_inference_operations=0)
        result=dict(completed=True,protocol=pp,cv_result=pin(D/'cv_result.json'),exports=exports,fit_source=pin(__file__),
                    ridge=.001,train=TRAIN,all_other_parameter_bytes_exact=True,elapsed_s=time.monotonic()-started)
        once(D/'fit_export.json',result);print(__import__('json').dumps(result),flush=True);return
    predictions={arm:[None]*20 for arm in ARMS};folds=[]
    for fold in range(5):
        train=[i for i in range(20) if i%5!=fold];held=[i for i in range(20) if i%5==fold]
        for arm in ARMS:
            head,_,_,eq=fit(arm,train)
            for i in held:
                predictions[arm][i]=replay(data[arm][i]['packet'],head)[0].astype(np.float64)
                assert np.isfinite(predictions[arm][i]).all()
            folds.append(dict(arm=arm,fold=fold,train=[TRAIN[i] for i in train],held=[TRAIN[i] for i in held],head=head.tolist(),equations=eq))
    rows=[]
    for i,rank in enumerate(TRAIN):
        row=dict(rank=rank,fold=i%5)
        for arm in ARMS:
            for suffix,pred in [('original',data[arm][i]['packet']['prediction'][-1]),('fitted',predictions[arm][i])]:
                raw=float(np.mean((pred-targets[i])**2));assert np.isfinite(raw)
                row[arm+'_'+suffix]=dict(raw=raw,adjusted=raw*data[arm][i]['multiplier'],flops=data[arm][i]['flops'])
        rows.append(row)
    def comparison(candidate,baseline):
        raw=float(np.mean([r[candidate]['raw'] for r in rows])/np.mean([r[baseline]['raw'] for r in rows])-1)*100
        adjusted=float(np.mean([r[candidate]['adjusted'] for r in rows])/np.mean([r[baseline]['adjusted'] for r in rows])-1)*100
        fold_pct=[]
        for fold in range(5):
            a=[r for r in rows if r['fold']==fold]
            fold_pct.append(float(np.mean([r[candidate]['adjusted'] for r in a])/np.mean([r[baseline]['adjusted'] for r in a])-1)*100)
        return dict(raw_pct=raw,adjusted_pct=adjusted,fold_adjusted_pct=fold_pct,fold_wins=sum(x<0 for x in fold_pct))
    comparisons=dict(physical_vs_original=comparison('k4_fitted','rrpi_original'),
                     physical_vs_fitted_control=comparison('k4_fitted','rrpi_fitted'),
                     fitted_control_vs_original=comparison('rrpi_fitted','rrpi_original'),
                     physical_head_effect=comparison('k4_fitted','k4_original'),
                     physical_unfitted_vs_original=comparison('k4_original','rrpi_original'))
    p=comparisons['physical_vs_original'];q=comparisons['physical_vs_fitted_control'];b=comparisons['fitted_control_vs_original']
    physical=p['adjusted_pct']<=-.2 and q['adjusted_pct']<=-.2 and q['fold_wins']>=4
    control=b['adjusted_pct']<=-.2 and b['fold_wins']>=4
    result=dict(completed=True,protocol=pp,fit_source=pin(__file__),physical_admitted=physical,control_admitted=control,
                comparisons=comparisons,rows=rows,folds=folds,cv_gates=GATES,ridge=.001,fit_count=10,
                all_finite=True,actual_bill_weighting=True,selection_or_tuning=False,official_score=False,
                historically_reused_train_not_pristine=True,elapsed_s=time.monotonic()-started)
    once(D/'cv_result.json',result);print(__import__('json').dumps({k:result[k] for k in ('physical_admitted','control_admitted','comparisons')}),flush=True)


if __name__=='__main__':
    assert len(sys.argv)==4 and sys.argv[1] in ('--prepare','--cv','--export')
    pp,c=plan(sys.argv[2],sys.argv[3])
    if sys.argv[1]=='--prepare':prepare(pp,c)
    else:perform(sys.argv[1][2:],pp,c)
