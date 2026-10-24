import time,gc,resource
import numpy as np
from common import *
from transport import carry,faces,query
from data import network,cell,consumer
seal=verify();u=read(HERE/'unit_result.json');g=read(HERE/'unit_guard.json')
assert u['passed'] and g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
assert u['seal_sha256']==sha(HERE/'seal.json')
rows=[];pins={};start=time.monotonic();names=('mean','covariance','next_variance_mean','next_total_mean')
for ordinal in TRAIN:
    info=network(ordinal);prev=cell(info,8)
    for l in LAYERS:
        c=cell(info,l+1);s=prev['sigma'];A=prev['Rs']*np.outer(s,s);B=prev['D']*np.outer(s,s)
        W=prev['Wnext'].astype(np.float64);gate=prev['gate'];d=prev['pdf']/s
        Ag,Bg,T=carry(A,B,gate,d,W);physical=faces(Ag,Bg)
        errs=[]
        for i,j in ((0,0),(17,93),(505,971),(1023,5)):
            direct=query(A,B,gate,d,W,i,j)
            errs.extend(abs(x-y) for x,y in zip(direct,(physical[1][i,j],physical[2][i,j],T[i,j])))
        witness=max(errs);assert witness<=2e-11,(ordinal,l,witness)
        sn=c['sigma'];fourth=(physical[0]/sn**4,physical[1]/np.outer(sn**3,sn),physical[2]/np.outer(sn*sn,sn*sn));odd=T/np.outer(sn*sn,sn)
        fn=consumer(c);actual_full,e1=fn(c['S'],c['k4'],c['q31'],c['q22']);actual_fourth,e2=fn(None,c['k4'],c['q31'],c['q22'])
        actual=dict(full=actual_full,fourth=actual_fourth);errors=[e1,e2];outputs={}
        for arm,source in (('fourth',(None,*fourth)),('odd',(odd,None,None,None)),('coupled',(odd,*fourth))):
            outputs[arm],error=fn(*source);errors.append(error)
        metrics={}
        for target,truth in actual.items():
            for arm,pred in outputs.items():
                group={}
                for name in names:
                    y=truth[name];z=pred[name];base=float(np.sum(y*y));energy=float(np.sum(z*z));alignment=float(np.sum(y*z));res=float(np.sum((y-z)**2))
                    assert base>0 and np.isfinite(base+res) and abs(res-(base+energy-2*alignment))<=1e-11*max(base,res,1e-30)
                    group[name]=dict(baseline_energy=base,prediction_energy=energy,alignment=alignment,residual_energy=res,ratio=res/base)
                metrics[target+'_'+arm]=group
        r=dict(ordinal=ordinal,source_layer=l,target_layer=l+1,metrics=metrics,query_witness_max_abs=witness,finite_consumer_max_abs=max(errors),
            initial_D_rms=float(np.sqrt(np.mean(prev['D']**2))),next_fourth_diagonal_rms=float(np.sqrt(np.mean(fourth[0]**2))),next_odd_diagonal_rms=float(np.sqrt(np.mean(np.diag(odd)**2))))
        rows.append(r)
        with (HERE/'cells.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        for p in prev['pins']+c['pins']:pins[p['path']]=p
        print(json.dumps(dict(ordinal=ordinal,transition=l,full_coupled={k:v['ratio'] for k,v in metrics['full_coupled'].items()},fourth_carry={k:v['ratio'] for k,v in metrics['fourth_fourth'].items()},elapsed_s=time.monotonic()-start)),flush=True)
        prev=c;del fn,outputs,actual,actual_full,actual_fourth,A,B,Ag,Bg,T,physical,fourth,odd;gc.collect()
assert len(rows)==36
groups=tuple(rows[0]['metrics']);pooled={};networks={}
for group in groups:
    layer_base={l:{name:sum(r['metrics'][group][name]['baseline_energy'] for r in rows if r['source_layer']==l)/6 for name in names} for l in LAYERS}
    pooled[group]={name:sum(r['metrics'][group][name]['residual_energy']/layer_base[r['source_layer']][name] for r in rows)/36 for name in names}
    networks[group]=[]
    for ordinal in TRAIN:
        rr=[r for r in rows if r['ordinal']==ordinal]
        ratio={name:sum(r['metrics'][group][name]['residual_energy']/layer_base[r['source_layer']][name] for r in rr)/sum(r['metrics'][group][name]['baseline_energy']/layer_base[r['source_layer']][name] for r in rr) for name in names}
        networks[group].append(dict(ordinal=ordinal,ratios=ratio))
primary=('covariance','next_total_mean')
gates=dict(coupled=all(pooled['full_coupled'][k]<=.85 for k in primary),each_network=all(all(r['ratios'][k]<=.95 for k in primary) for r in networks['full_coupled']),fourth=all(pooled['fourth_fourth'][k]<=.80 for k in primary))
once(HERE/'result.json',dict(passed=all(gates.values()),gates=gates,pooled=pooled,networks=networks,rows=rows,external_inputs=list(pins.values()),elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,seal_sha256=sha(HERE/'seal.json'),single_reference_noise_removed=False,mini14_71_accessed=False,final1B_accessed=False,model_fitted=False,candidate_built=False))
print(json.dumps(dict(event='complete',passed=all(gates.values()),gates=gates,pooled=pooled)),flush=True)
