"""HELD scheduling-only continuation of fixed native six; two immutable cases reused."""
import argparse,hashlib,importlib.util,json,math,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;O=R.parent.parent
ORIGINAL=R/'run_native_price_early_vread_astra_v1.py'
ORIGINAL_SHA='d925d8f717212a9a8e38f0723492fd3b2c4f95b10da7f902c525000fbe1f7cc8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(ORIGINAL)==ORIGINAL_SHA
spec=importlib.util.spec_from_file_location('source_pinned_early_price_v1',ORIGINAL)
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
u=v.u
RUN=R/'native_price_run_astra_v2';OLD=R/'native_price_run_astra_v1'
CONT=R/'NATIVE_PRICE_CONTINUATION_SPEC_astra_v2.json'
PLAN=((7100136,'on'),(7100137,'parent'),(7100137,'off'),(7100137,'on'))

def config(review):
    assert os.getpriority(os.PRIO_PROCESS,0)==0,'controller must start at nice0; guard adds nice10 once'
    c=u.read(CONT);r=u.read(review)
    assert r['approved'] is True and r['reviewed_by']=='root' and r['native_slot_granted'] is True
    assert r['scheduling_only_continuation'] is True
    assert r['launcher_sha256']==sha(__file__) and r['continuation_spec_sha256']==sha(CONT)
    assert r['prereg_sha256']==sha(v.PREREG) and r['native_price_spec_sha256']==sha(v.SPEC)
    assert c['fixed_remaining_plan']==[{'seed':seed,'route':route} for seed,route in PLAN]
    assert c['fixed_reused_plan']==[{'seed':7100136,'route':'parent'},{'seed':7100136,'route':'off'}]
    assert c['original_launcher_sha256']==sha(ORIGINAL)
    for e in c['evidence']:assert sha(e['path'])==e['sha256'],e['path']
    retired=u.read(c['retirement_receipt'])
    assert retired['status']=='two_cases_complete_controller_retired' and retired['numerical_children_signalled'] is False
    assert retired['controller_pid']==30910 and retired['remaining_numerical_evaluations']==4
    assert retired['all_completed_guards_clean_retired'] is True and retired['controller_retired'] is True
    s=v.config(Path(c['original_review']))
    return s,c

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root-granted',action='store_true');ap.add_argument('--review',type=Path,required=True);args=ap.parse_args()
    assert args.root_granted and args.review.resolve().parent==O/'lead'
    for k in u.THREADS:assert os.environ.get(k)=='1',k
    s,c=config(args.review);gate=u.admission();RUN.mkdir(exist_ok=False)
    u.write(RUN/'launch_receipt_astra_v2.json',dict(utc=u.stamp(),launcher=u.evidence(__file__),original_launcher=u.evidence(ORIGINAL),prereg=u.evidence(v.PREREG),spec=u.evidence(v.SPEC),continuation=u.evidence(CONT),review=u.evidence(args.review),gate=gate,one_slot=True,controller_nice=0,guard_nice=0,native_nice=10,no_fresh_or_oos_read=True))
    all_results={};checks=[]
    try:
        for seed in v.SEEDS:
            reuse=next(x for x in s['baseline_reuse'] if x['seed']==seed)
            all_results[seed]={'base':u.result(Path(reuse['cert']),Path(reuse['guard']),s['base'],seed,reused=True)}
        for route in ('parent','off'):
            rs=all_results[7100136]
            rs[route]=u.result(OLD/f'{route}_7100136_astra_v1.json',OLD/f'{route}_7100136_astra_v1_guard.json',s['candidates'][route],7100136)
            check=dict(seed=7100136,route=route,**v.gates(route,rs));checks.append(check)
            retired=u.read(c['retirement_receipt'])
            assert check==next(x for x in retired['completed_checks'] if x['route']==route)
        for seed,route in PLAN:
            config(args.review);gate=u.admission();source=s['candidates'][route];rs=all_results[seed]
            cert=RUN/f'{route}_{seed}_astra_v2.json';guard=RUN/f'{route}_{seed}_astra_v2_guard.json';log=RUN/f'{route}_{seed}_astra_v2.log'
            envargs=['PB_ALL=1','STATS='+','.join(u.STATS+v.EXTRA)]+[k+'=1' for k in u.THREADS]
            cmd=[u.GUARD_PY,'-B',str(u.GUARD),'--record',str(guard),'--seconds','900','--rss-backstop','7500000000','/usr/bin/env',*envargs,u.NATIVE_PY,'-B',str(u.CERT),source['path'],str(seed),str(cert)]
            u.write(RUN/f'{route}_{seed}_admission_astra_v2.json',dict(gate=gate,command=cmd,controller_nice=os.getpriority(os.PRIO_PROCESS,0)))
            print(json.dumps(dict(event='early_price_continuation_start',utc=u.stamp(),route=route,seed=seed)),flush=True)
            with log.open('xb') as out:subprocess.run(cmd,cwd=O/'scout_native',env=os.environ.copy(),stdout=out,stderr=subprocess.STDOUT,check=True)
            rs[route]=u.result(cert,guard,source,seed)
            check=dict(seed=seed,route=route,**v.gates(route,rs));checks.append(check)
            u.write(RUN/f'{route}_{seed}_checks_astra_v2.json',check)
            print(json.dumps(dict(event='early_price_continuation_done',utc=u.stamp(),route=route,seed=seed,flops=rs[route]['row']['flops'])),flush=True)
        assert len(checks)==6
        pairs=[dict(seed=seed,base_path=str(all_results[seed]['base']['cert']),base_sha256=sha(all_results[seed]['base']['cert']),arm_path=str(all_results[seed]['on']['cert']),arm_sha256=sha(all_results[seed]['on']['cert'])) for seed in v.SEEDS]
        worst=max(math.log(all_results[x]['on']['row']['flops']/all_results[x]['base']['row']['flops']) for x in v.SEEDS)
        u.write(RUN/'PRICE_FOR_ROOT_FREEZE_astra_v2.json',dict(status='ready_for_root_freeze_before_fresh_summary',utc=u.stamp(),prereg_sha256=sha(v.PREREG),aggregation='max_pair_log_ratio',candidate_source_sha256={'base':s['base']['sha256'],'arm':s['candidates']['on']['sha256']},pairs=pairs,max_pair_log_ratio=worst,all_noop_resource_route_gates_passed=True,checks=checks,evidence=[item for rs in all_results.values() for result in rs.values() for item in result['evidence']],all_guards_retired=True,no_fresh_or_oos_content_read=True,evaluations=6,reused_completed_evaluations=2,new_evaluations=4,scheduling_continuation=u.evidence(CONT)))
        print(json.dumps(dict(event='early_price_complete_slot_released',utc=u.stamp(),max_pair_log_ratio=worst)),flush=True)
    except BaseException as e:
        u.write(RUN/'STOPPED_astra_v2.json',dict(utc=u.stamp(),error=repr(e),completed_checks=checks,no_retry_or_next_evaluation_authorized=True));raise
if __name__=='__main__':main()
