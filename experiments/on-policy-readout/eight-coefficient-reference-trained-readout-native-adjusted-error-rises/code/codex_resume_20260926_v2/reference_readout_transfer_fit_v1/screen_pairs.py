import subprocess
from common import *

assert read(HERE/'native_train_result.json')['passed']
guard('assess_train');guard('native_fit');guard('native_parent')
LIGHT=ROOT/'.venv/bin/python';NATIVE=P2/'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD=P2/'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
once(HERE/'pairs_seal.json',dict(sources=[pin(HERE/x) for x in ['candidate.py','fit_result.json','native_train_result.json',
    'screen_pairs.py','check_pair_native.py','common.py']]+[pin(H/'native_eval.py'),pin(H/'compare_native.py')],
    coefficients_frozen=True,selection_networks_excluded_from_fit=True))
def run(name,script,args,runtime):
    print(json.dumps(dict(event='launch',name=name)),flush=True)
    with (HERE/(name+'.log')).open('x') as f:
        p=subprocess.run([str(LIGHT),str(GUARD),'--record',str(HERE/(name+'_guard.json')),
            '--seconds','300',str(runtime),str(script),*args],stdout=f,stderr=subprocess.STDOUT)
    g=guard(name);assert p.returncode==0
    print(json.dumps(dict(event='retired',name=name,seconds=g['elapsed_s'],peak_rss=g['max_rss_bytes'])),flush=True)
base='reference_readout_parent_observer2_v1';fit='reference_readout_fit_observer2_v1'
run('observer_parent',H/'native_eval.py',[str(H/'candidate_old_both_tail_trace_bounded_v6.py'),'35,100',base],NATIVE)
run('observer_fit',H/'native_eval.py',[str(HERE/'candidate.py'),'35,100',fit],NATIVE)
cmp='reference_readout_observer2_comparison_v1.json'
run('observer_compare',H/'compare_native.py',[base,fit,cmp],LIGHT)
r=read(H/cmp);observer_passed=r['adjusted_wins']==2 and r['adjusted_pct']<=-2
once(HERE/'observer_pair_result.json',dict(passed=observer_passed,comparison=pin(H/cmp)))
print(json.dumps(dict(event='observer_pair_score',passed=observer_passed,adjusted_pct=r['adjusted_pct'],wins=r['adjusted_wins'])),flush=True)
if not observer_passed:
    once(HERE/'pairs_result.json',dict(passed=False,stage='observer_pair',check_targets_read=False))
else:
    run('check_parent',HERE/'check_pair_native.py',['parent'],NATIVE)
    run('check_fit',HERE/'check_pair_native.py',['fit'],NATIVE)
    p=read(HERE/'check_parent/complete.json')['rows'];c=read(HERE/'check_fit/complete.json')['rows'];rows=[]
    for b,a in zip(p,c):
        assert b['mini']==a['mini'];bp=packet(b['packet']);ap=packet(a['packet'])
        assert bp['target'].tobytes()==ap['target'].tobytes()
        bm=math.fsum(float(x)**2 for x in bp['prediction'][-1]-bp['target'])/1024
        am=math.fsum(float(x)**2 for x in ap['prediction'][-1]-ap['target'])/1024
        assert math.isclose(bm,b['mse'],rel_tol=1e-14) and math.isclose(am,a['mse'],rel_tol=1e-14)
        rows.append(dict(mini=a['mini'],raw_ratio=am/bm,adjusted_ratio=a['adjusted']/b['adjusted']))
    ratio=sum(x['adjusted'] for x in c)/sum(x['adjusted'] for x in p)
    passed=ratio<=.98 and all(x['adjusted_ratio']<1 for x in rows)
    once(HERE/'pairs_result.json',dict(passed=passed,stage='CHECK_pair',observer_pair_passed=True,
        pooled_adjusted_ratio=ratio,rows=rows,check_targets_read=True,coefficients_unchanged=True))
    print(json.dumps(dict(event='CHECK_pair_score',passed=passed,adjusted_ratio=ratio,rows=rows)),flush=True)
