"""Metadata-only complete six-case retirement/freeze proposal audit; no evaluation."""
import argparse,hashlib,importlib.util,json,math,subprocess
from pathlib import Path
E=Path(__file__).resolve().parent;OLD=E/'native_price_run_astra_v1';RUN=E/'native_price_run_astra_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=E/'run_native_price_early_vread_astra_v2.py'
assert sha(p)=='3ed22cecfbb2d34dcf96010c334a4bb804975f1e3d14bdc4b8ec144fe98fe36f'
s=importlib.util.spec_from_file_location('pinned_price_metadata_only',p);c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
u=c.u;v=c.v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--managed-session',type=int,required=True);ap.add_argument('--managed-exit-code',type=int,required=True);a=ap.parse_args()
    assert a.managed_session==11812 and a.managed_exit_code==0
    active=u.read(RUN/'ACTIVE_SESSION_astra_v2.json');assert active['managed_session_id']==a.managed_session
    s,_=c.config(Path(active['root_review']['path']))
    price=RUN/'PRICE_FOR_ROOT_FREEZE_astra_v2.json';d=u.read(price)
    assert d['evaluations']==6 and d['reused_completed_evaluations']==2 and d['new_evaluations']==4
    assert d['status']=='ready_for_root_freeze_before_fresh_summary'
    assert not (RUN/'STOPPED_astra_v2.json').exists()
    all_results={};checks=[];guards=[];pids={30910,30933,33672,33959,39736,39776,39781}
    for seed in v.SEEDS:
        b=next(x for x in s['baseline_reuse'] if x['seed']==seed)
        rs={'base':u.result(Path(b['cert']),Path(b['guard']),s['base'],seed,reused=True)};all_results[seed]=rs
        for route in v.ROUTES:
            old=seed==7100136 and route in ('parent','off');r=OLD if old else RUN;version='v1' if old else 'v2'
            cert=r/f'{route}_{seed}_astra_{version}.json';guard=r/f'{route}_{seed}_astra_{version}_guard.json'
            rs[route]=u.result(cert,guard,s['candidates'][route],seed)
            checks.append(dict(seed=seed,route=route,**v.gates(route,rs)))
            g=u.read(guard);pids.add(g['pid'])
            guards.append(dict(seed=seed,route=route,**u.evidence(guard),max_rss_bytes=g['max_rss_bytes'],max_footprint_bytes=g['max_footprint_bytes'],completed=g['completed'],exit_code=g['exit_code'],whole_group_retired=g['whole_group_retired'],remaining_rows=g['remaining_rows']))
    assert checks==d['checks']
    worst=max(math.log(all_results[x]['on']['row']['flops']/all_results[x]['base']['row']['flops']) for x in v.SEEDS)
    assert worst==d['max_pair_log_ratio']
    for e in d['evidence']:assert sha(e['path'])==e['sha256']
    for pid in pids:assert not subprocess.run(['ps','-p',str(pid),'-o','pid='],capture_output=True,text=True).stdout.strip(),pid
    rows=subprocess.check_output(['ps','-axo','pid=,ppid=,command='],text=True).splitlines()
    owned=[x for x in rows if ('run_cert_v2.py' in x or 'guard_footprint_v3.py' in x or 'run_native_price_early_vread_astra_v2.py --root-granted' in x) and (str(RUN) in x or 'run_native_price_early_vread_astra_v2.py --root-granted' in x)]
    assert not owned,owned
    pairs=[dict(seed=seed,base_bill=all_results[seed]['base']['row']['flops'],parent_bill=all_results[seed]['parent']['row']['flops'],off_bill=all_results[seed]['off']['row']['flops'],on_bill=all_results[seed]['on']['row']['flops'],on_vs_base_log_pct=100*math.log(all_results[seed]['on']['row']['flops']/all_results[seed]['base']['row']['flops']),on_vs_parent_log_pct=100*math.log(all_results[seed]['on']['row']['flops']/all_results[seed]['parent']['row']['flops'])) for seed in v.SEEDS]
    out=RUN/'FULL_SIX_RETIREMENT_astra_v2.json'
    u.write(out,dict(status='all_six_complete_all_guards_and_continuation_owner_retired',utc=u.stamp(),managed_session_id=a.managed_session,managed_exit_code=a.managed_exit_code,active=u.evidence(RUN/'ACTIVE_SESSION_astra_v2.json'),continuation=u.evidence(c.CONT),original_owner_retirement=u.evidence(OLD/'CONTROLLER_FINAL_RETIREMENT_astra_v1.json'),price_proposal=u.evidence(price),root_review=active['root_review'],checks=checks,guards=guards,absent_owned_pids=sorted(pids),absent_owned_command_rows=True,pairs=pairs,max_pair_log_ratio=worst,max_bill_log_pct=100*worst,max_rss_bytes=max(x['max_rss_bytes'] for x in guards),max_footprint_bytes=max(x['max_footprint_bytes'] for x in guards),no_fresh_or_oos_read=True,no_next_evaluation_granted=True))
    print(json.dumps(dict(receipt=u.evidence(out),pairs=pairs,max_bill_log_pct=100*worst,max_rss_bytes=max(x['max_rss_bytes'] for x in guards),max_footprint_bytes=max(x['max_footprint_bytes'] for x in guards))))
if __name__=='__main__':main()
