"""HELD six-case native pricing. One separately granted slot; no scientific fresh reads."""
import argparse,hashlib,importlib.util,json,math,os,subprocess
from pathlib import Path
R=Path(__file__).resolve().parent;S=R.parent;O=S.parent
LIB=S/'run_rank_native_price_astra_v1.py'
LIB_SHA='e0609577eb5e1f9eaa61fad26018cfe7cc0456712b45cc835a063eaf8d393277'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(LIB)==LIB_SHA
spec=importlib.util.spec_from_file_location('source_pinned_native_metadata',LIB)
u=importlib.util.module_from_spec(spec);spec.loader.exec_module(u)
H=S/'capacity_early_vread_held_astra_v1_v3';RUN=R/'native_price_run_astra_v1'
PREREG=R/'PREREG_v1.json';SPEC=R/'native_price_spec_astra_v1.json'
ROUTES=('parent','off','on');SEEDS=(7100136,7100137)
EXTRA=('_ASTRA_EVR_STATS','_ASTRA_RLF_STATS','_ASTRA_RLS_STATS')

def config(review):
    d=u.read(PREREG);r=u.read(review);s=u.read(SPEC)
    assert d['status']=='root_registered_before_own_evaluations' and d['joint_allocation_registered'] is True
    assert d['native_price_seeds']==list(SEEDS) and d['native_price_spec_sha256']==sha(SPEC)
    assert r['approved'] is True and r['reviewed_by']=='root' and r['baseline_reuse_approved'] is True
    assert r['prereg_sha256']==sha(PREREG) and r['launcher_sha256']==sha(__file__)
    assert r['native_price_spec_sha256']==sha(SPEC)
    assert r['guard_sha256']==sha(u.GUARD) and r['cert_runner_sha256']==sha(u.CERT)
    assert s['seeds']==list(SEEDS) and s['routes']==list(ROUTES)
    for x in [s['base'],*s['candidates'].values()]:assert sha(x['path'])==x['sha256']
    for x in s['baseline_reuse']:
        for p in x['evidence']:assert sha(p['path'])==p['sha256']
    return s

def gates(route,rs):
    current=rs[route];a=current['row'];stats=a['stats']
    assert all(x in stats for x in ('_ASTRA_RLF_STATS','_ASTRA_RLS_STATS'))
    if route=='parent':return {'parent_complete':True}
    p=rs['parent'];b=p['row']
    assert stats['_ASTRA_COMPUTE_STATS']==b['stats']['_ASTRA_COMPUTE_STATS']
    if route=='off':
        assert current['prediction'].tobytes()==p['prediction'].tobytes() and a['mse_layers']==b['mse_layers']
        assert (a['flops'],a['ops'])==(b['flops'],b['ops'])
        assert stats['_ASTRA_EVR_STATS']=={'calls':0,'events':[]}
        assert stats['_ASTRA_RLF_STATS']==b['stats']['_ASTRA_RLF_STATS'] and stats['_ASTRA_RLS_STATS']==b['stats']['_ASTRA_RLS_STATS']
        return {'all16_bytes_mse_bill_ops_exact':True,'parent_read_counters_exact':True}
    assert current['prediction'][:10].tobytes()==p['prediction'][:10].tobytes()
    assert a['mse_layers'][:10]==b['mse_layers'][:10]
    assert current['prediction'][15].tobytes()!=p['prediction'][15].tobytes()
    e=stats['_ASTRA_EVR_STATS'];assert e['calls']==4 and [x['layer'] for x in e['events']]==[10,11,12,13]
    for x in e['events']:
        assert x['full_state_rank']==x['prefix_rank']==384 and x['prefix_v_read_rank']==320 and x['common_rank']==256
        assert x['lazy_prefix']==(x['layer']==13) and x['trailing_rank']==(384 if x['layer']==13 else 320)
    ar,br=stats['_ASTRA_RLF_STATS'],b['stats']['_ASTRA_RLF_STATS']
    for key in ('fused','parent','leaf_add_elements'):assert ar[key]==br[key],key
    assert len(ar['shapes'])==len(br['shapes'])
    for x,y in zip(ar['shapes'],br['shapes']):
        assert x[:3]+x[4:]==y[:3]+y[4:] and (x[3]==y[3] or (x[3],y[3])==(320,384))
    assert stats['_ASTRA_RLS_STATS']==b['stats']['_ASTRA_RLS_STATS']
    return {'unchanged_layers0_to9':True,'final_changed':True,'four_early_events_correct':True,'original_young_dispatch_scaling_unwind_retained':True,'copy_counters_exact':True}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root-granted',action='store_true');ap.add_argument('--review',type=Path,required=True);args=ap.parse_args()
    assert args.root_granted and args.review.resolve().parent==O/'lead'
    for k in u.THREADS:os.environ[k]='1'
    s=config(args.review);gate=u.admission();RUN.mkdir(exist_ok=False)
    u.write(RUN/'launch_receipt_astra_v1.json',dict(utc=u.stamp(),launcher=u.evidence(__file__),prereg=u.evidence(PREREG),spec=u.evidence(SPEC),review=u.evidence(args.review),gate=gate,one_slot=True,no_fresh_or_oos_read=True))
    all_results={};checks=[]
    try:
        for seed in SEEDS:
            reuse=next(x for x in s['baseline_reuse'] if x['seed']==seed)
            rs={'base':u.result(Path(reuse['cert']),Path(reuse['guard']),s['base'],seed,reused=True)};all_results[seed]=rs
            for route in ROUTES:
                config(args.review);gate=u.admission();source=s['candidates'][route]
                cert=RUN/f'{route}_{seed}_astra_v1.json';guard=RUN/f'{route}_{seed}_astra_v1_guard.json';log=RUN/f'{route}_{seed}_astra_v1.log'
                stats=u.STATS+EXTRA
                envargs=['PB_ALL=1','STATS='+','.join(stats)]+[k+'=1' for k in u.THREADS]
                cmd=['/usr/bin/nice','-n','10',u.GUARD_PY,'-B',str(u.GUARD),'--record',str(guard),'--seconds','900','--rss-backstop','7500000000','/usr/bin/env',*envargs,u.NATIVE_PY,'-B',str(u.CERT),source['path'],str(seed),str(cert)]
                u.write(RUN/f'{route}_{seed}_admission_astra_v1.json',dict(gate=gate,command=cmd))
                print(json.dumps(dict(event='early_price_start',utc=u.stamp(),route=route,seed=seed)),flush=True)
                with log.open('xb') as out:subprocess.run(cmd,cwd=O/'scout_native',env=os.environ.copy(),stdout=out,stderr=subprocess.STDOUT,check=True)
                rs[route]=u.result(cert,guard,source,seed)
                check=dict(seed=seed,route=route,**gates(route,rs));checks.append(check)
                u.write(RUN/f'{route}_{seed}_checks_astra_v1.json',check)
                print(json.dumps(dict(event='early_price_done',utc=u.stamp(),route=route,seed=seed,flops=rs[route]['row']['flops'])),flush=True)
        pairs=[dict(seed=seed,base_path=str(all_results[seed]['base']['cert']),base_sha256=sha(all_results[seed]['base']['cert']),arm_path=str(all_results[seed]['on']['cert']),arm_sha256=sha(all_results[seed]['on']['cert'])) for seed in SEEDS]
        worst=max(math.log(all_results[x]['on']['row']['flops']/all_results[x]['base']['row']['flops']) for x in SEEDS)
        u.write(RUN/'PRICE_FOR_ROOT_FREEZE_astra_v1.json',dict(status='ready_for_root_freeze_before_fresh_summary',utc=u.stamp(),prereg_sha256=sha(PREREG),aggregation='max_pair_log_ratio',candidate_source_sha256={'base':s['base']['sha256'],'arm':s['candidates']['on']['sha256']},pairs=pairs,max_pair_log_ratio=worst,all_noop_resource_route_gates_passed=True,checks=checks,evidence=[item for rs in all_results.values() for result in rs.values() for item in result['evidence']],all_guards_retired=True,no_fresh_or_oos_content_read=True,evaluations=6))
        print(json.dumps(dict(event='early_price_complete_slot_released',utc=u.stamp(),max_pair_log_ratio=worst)),flush=True)
    except BaseException as e:
        u.write(RUN/'STOPPED_astra_v1.json',dict(utc=u.stamp(),error=repr(e),completed_checks=checks,no_retry_or_next_evaluation_authorized=True));raise
if __name__=='__main__':main()
