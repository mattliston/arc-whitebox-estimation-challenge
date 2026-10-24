"""Fixed protected base difference read224 smoke, at most two new native calls."""
import argparse,hashlib,importlib.util,json,math,os,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DONOR=O/'lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py'
assert sha(DONOR)=='1af0cb77eacc49a1a6b206bfd87fd190dfc24667de47c627834be4e4d4e56b91'
spec=importlib.util.spec_from_file_location('_prd224_owned_donor',DONOR)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
guard=q.guard;write=q.write
ORDER=[('ON',7100136),('ON',7100137)]
RULE=dict(first_seed_early_close_adjusted_ratio_gte=1.02,completed_pair_mean_log_net_lt=0.0,
          implementation_failure_closes=True,no_retry=True,no_followup_authority=True)

# Prospective scheduling-only amendment; native live guard limits are unchanged.
q.RESOURCE=dict(q.RESOURCE,entry_counted_max_exclusive=10,logical_total_max=10)

# Scheduling-only priority for the two already-running GOP2 fresh controllers.
# PLAN is immutable. Inspect process identity/lifecycle only; never result bodies.
GOP_FRESH_PLANS = []  # Earlier GOP controllers retired; no stale PID priority.

def gop_fresh_waiting():
 rows=[]
 for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,comm='],text=True).splitlines():
  fields=line.strip().split(None,2)
  if len(fields)==3:rows.append((int(fields[0]),int(fields[1]),Path(fields[2]).name.lower()))
 live={p for p,_,_ in rows};waiting=[]
 for path,digest,pid in GOP_FRESH_PLANS:
  assert sha(path)==digest
  plan=json.loads(Path(path).read_text());assert plan['controller_pid']==pid and plan['phase']=='fresh'
  # The pinned GOP driver only launches a Python guard as a direct Python child.
  # Its transient resource-check ps child is not a native/guard reservation.
  if pid in live and not any(pp==pid and name.startswith('python') for _,pp,name in rows):waiting.append(pid)
 return waiting

def admit(run,role,seed):
 started=time.monotonic();attempt=0;last_status=None
 while True:
  state=q.guard.resources();waiting=gop_fresh_waiting();elapsed=time.monotonic()-started;attempt+=1
  okay=state['free_percent']>=60 and state['counted']<10 and not waiting and elapsed<=1800
  record=dict(role=role,seed=seed,controller_pid=os.getpid(),attempt=attempt,elapsed_seconds=elapsed,
              admitted=okay,gop_fresh_waiting_controllers=waiting,**state)
  q.guard.event(run/'admission_wait.jsonl',record)
  if okay:q.guard.event(run/'admission.jsonl',record);return record
  if last_status is None or elapsed-last_status>=30:
   print(json.dumps(dict(status='waiting_for_headroom_or_GOP_fresh_priority_no_native_child',role=role,seed=seed,
      free_percent=state['free_percent'],counted=state['counted'],gop_fresh_waiting_controllers=waiting,
      elapsed_seconds=elapsed,slot_reserved=True)),flush=True)
   last_status=elapsed
  assert elapsed<1800,'Bounded admission wait expired; no automatic retry'
  time.sleep(min(15,1800-elapsed))

q.admit=admit

def admissions(h,sequence,controller):
 records=[json.loads(x)for x in (h/'admission.jsonl').read_text().splitlines()]
 assert [(x['role'],x['seed'])for x in records]==sequence
 assert all(x['controller_pid']==controller and x['free_percent']>=60 and x['counted']<10 and x['gop_fresh_waiting_controllers']==[] and x['admitted']is True and 0<=x['elapsed_seconds']<=1800 for x in records)
 attempts=[json.loads(x)for x in (h/'admission_wait.jsonl').read_text().splitlines()]
 assert [x for x in attempts if x['admitted']]==records
 assert all(x['controller_pid']==controller and (x['role'],x['seed'])in sequence for x in attempts)
 return dict(path=str(h/'admission.jsonl'),sha256=sha(h/'admission.jsonl'),records=records,wait_path=str(h/'admission_wait.jsonl'),wait_sha256=sha(h/'admission_wait.jsonl'))

# Adapt the approved admission receipt helper to the inherited one-argument interface.
q.admissions=lambda h:admissions(h,q.ORDER,os.getpid())


def config(path,digest,grant_path):
    assert sha(path)==digest
    c=json.loads(Path(path).read_text());r=json.loads(Path(grant_path).read_text())
    assert c['purpose']=='prd224_diagnostic2' and c['fixed_order']==[list(x)for x in ORDER]
    assert c['rule']==RULE and c['new_native_calls_max']==2 and c['resource_policy']==q.RESOURCE and c['admission_wait_policy']==q.WAIT
    assert c['run_name']=='run_prd224_diagnostic_v1'
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert c['source_pins'][str(DONOR)]==sha(DONOR)
    assert set(c['candidates'])=={'parent','ON'} and all(x['mode']==1 for x in c['candidates'].values())
    parent=c['candidates']['parent'];arm=c['candidates']['ON']
    assert parent['sha256']=='4e6e4daee89b6d7c8812e145e627896fb59afc511b3b61da4a797fe54774dc76'
    assert arm['sha256']=='10a063dee7a34e2267d18384af69da5ec3f42c2fb8f41c0b0a1029b62503c5be'
    build_path=D/'build_prd224_source_HELD_v1.py'
    bs=importlib.util.spec_from_file_location('_prd224_frozen_source',build_path)
    build=importlib.util.module_from_spec(bs);bs.loader.exec_module(build)
    comp=c['composition_parent']
    assert sha(comp['path'])==comp['sha256']=='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
    assert Path(comp['path']).read_bytes()==Path(parent['path']).read_bytes()+Path(c['setup_repair_fragment']).read_bytes()
    assert Path(arm['path']).read_bytes()==build.transform(Path(comp['path']).read_bytes())
    assert r['composition_parent_sha256']==comp['sha256'] and r['PB_ALL_control_reuse_approved']is True
    guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
    reuse=c['parent_reuse'];root=json.loads(Path(reuse['acceptance_path']).read_text())
    owner=json.loads(Path(reuse['retirement_path']).read_text());decision=json.loads(Path(reuse['complete_path']).read_text())
    assert root['approved']is True and root['reviewed_by']=='root' and root['candidate_sha256']==parent['sha256']
    assert root['actual_exit_code']==0 and root['all_owned_PID_PPID_PGID_absent']is True
    assert root['owner_path']==reuse['retirement_path'] and root['owner_sha256']==sha(reuse['retirement_path'])
    assert root['complete_path']==reuse['complete_path'] and root['complete_sha256']==sha(reuse['complete_path'])
    assert owner['actual_managed_exit_observed']is True and owner['exit_code']==0 and owner['all_owned_PID_PPID_PGID_absent']is True
    assert owner['all_guards_clean']is True and owner['candidate_sha256']==parent['sha256']
    assert decision['status']=='PASS_EXACT_V1_IDENTITY_AND_LOWER_BILL' and decision['complete_calls']==2
    assert c['native_reuse']==owner['evaluations'] and [x['seed']for x in c['native_reuse']]==[7100136,7100137]
    for path in reuse.values():assert c['source_pins'][path]==sha(path)
    for item in c['native_reuse']:
        for key in ('certificate','guard'):assert c['source_pins'][item[key]]==item[key+'_sha256']
        p=Path(item['certificate']);assert c['source_pins'][str(p.with_name(p.stem+'_pred.npz'))]==item['prediction_npz_sha256']
    assert r['approved'] is True and r['reviewed_by']=='root' and r['purpose']==c['purpose']
    assert r['protocol_sha256']==digest and r['runner_sha256']==sha(__file__) and r['numerical_slots']==1
    assert r['parent_reuse_approved'] is True and r['parent_reuse']==reuse
    assert r['fixed_order']==c['fixed_order'] and r['rule']==RULE and r['resource_policy']==q.RESOURCE
    assert r['admission_wait_policy']==q.WAIT and r['run_name']==c['run_name']
    assert all(os.environ.get(k)=='1' for k in guard.THREADS) and os.getpriority(os.PRIO_PROCESS,0)==0
    assert [x['seed']for x in c['truth_metadata']]==[7100136,7100137]
    for x in c['truth_metadata']:assert sha(x['path'])==x['sha256']
    return c

def check(c,item,seed):
    base=q.current(dict(c,candidates={'ON':c['candidates']['parent']}),next(x for x in c['native_reuse']if x['seed']==seed),seed)
    arm=q.current(c,item,seed);q.setup_equal(base,arm)
    assert arm['cert']['native_bill_comparable'] and base['cert']['native_bill_comparable']
    assert arm['cert']['python_version']==base['cert']['python_version']
    a,b=arm['row'],base['row'];fhm=a['stats']['_ASTRA_FHM_STATS']
    assert a['stats']['_YR7_ACTIVE']==[False] and a['stats']['_YR7_BIND_STATS']==dict(keys=17,retired=True)
    ys=a['stats']['_YR7_STATS'];assert ys['retired']is True and ys['value_cache_entries']==0
    assert all(tuple(x)in {tuple(z)for z in c['shape_allowlist']}for x in ys['routed_shapes'])
    assert arm['pred'][:10].tobytes()==base['pred'][:10].tobytes() and a['mse_layers'][:10]==b['mse_layers'][:10]
    assert fhm['retired'] is True and fhm['layers']==list(range(1,15))
    st=a['stats']['_PRD_STATS']
    assert st['retired']is True and len(st['builds'])==1
    z=st['builds'][0]
    assert len(z)==5 and z[0]==10 and 0<z[1]<=1024 and z[2]==320 and 4096<z[3]<=6144 and z[4]==224
    assert [x[0]for x in st['reads']]==list(range(10,15))
    assert all(len(x)==4 and 0<x[1]<=1024 and x[2]==z[3] and x[3]==224 for x in st['reads'])
    assert all(type(x)is int and x==0 for x in a['stats']['_ASTRA_IOP_STATS'].values())
    lm=math.log(a['mse_final']/b['mse_final']);lb=math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
    return dict(seed=seed,implementation_pass=True,prefix_first10_exact=True,all16_finite=True,
        comparator='accepted actual YR7v2 controls; diagnostic only',separate_prd224_gain_banked=False,fresh_requires_actual_YR7_parent=True,
        log_mse_ratio=lm,log_bill_ratio=lb,log_net=lm+lb,adjusted_change_pct=100*math.expm1(lm+lb),
        parent_mse=b['mse_final'],arm_mse=a['mse_final'],parent_bill=b['flops'],arm_bill=a['flops'],
        parent_ops=b['ops'],arm_ops=a['ops'],prd224_stats=st,shape_route_stats={k:a['stats'][k]for k in ('_ASTRA_RLF_STATS','_UBX_RC_STATS')})

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-authorized',action='store_true');p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha256',required=True);p.add_argument('--grant',type=Path,required=True);a=p.parse_args();assert a.run_authorized
    c=config(a.protocol,a.protocol_sha256,a.grant);run=D/c['run_name'];run.mkdir(exist_ok=False)
    write(run/'PLAN_v1.json',dict(protocol_path=str(a.protocol.resolve()),protocol_sha256=a.protocol_sha256,
        root_grant_path=str(a.grant.resolve()),root_grant_sha256=sha(a.grant),runner_sha256=sha(__file__),controller_pid=os.getpid(),
        fixed_order=c['fixed_order'],rule=RULE,expected_possible_guards=[str(run/f'native_{s}_{r}_guard.json')for r,s in ORDER],
        reuse=c['parent_reuse'],resource_policy=q.RESOURCE))
    evaluations=[];checks=[];early=False
    try:
        for role,seed in ORDER:
            config(a.protocol,a.protocol_sha256,a.grant)
            env=dict(os.environ,PB_ALL='1',STATS=','.join(q.STATS+('_YR7_STATS','_YR7_ACTIVE','_YR7_BIND_STATS','_PRD_STATS')),GOP2_MODE='1',PYTHONDONTWRITEBYTECODE='1')
            q.admit(run,role,seed)
            item=guard.run_native(run,dict(pinned_files=c['source_pins'],candidates=c['candidates']),role,seed,env)
            evaluations.append(dict(role=role,seed=seed,**item));report=check(c,item,seed);checks.append(report)
            write(run/f'CHECK_{seed}_v1.json',report)
            print(json.dumps(dict(completed_calls=len(evaluations),seed=seed,implementation_pass=True)),flush=True)
            if seed==7100136 and report['log_net']>=math.log(1.02):early=True;break
        q.ORDER=[(x['role'],x['seed'])for x in evaluations]
        mean=sum(x['log_net']for x in checks)/len(checks)
        status='FAIL_EARLY_GROSS_DEGRADATION' if early else ('SURVIVES_TO_ROOT_FRESH_REVIEW' if mean<0 else 'FAIL_NONNEGATIVE_PAIRED_NET')
        write(run/'DIAGNOSTIC_RETIRED_v1.json',dict(status=status,protocol_sha256=a.protocol_sha256,root_grant_sha256=sha(a.grant),
            plan_sha256=sha(run/'PLAN_v1.json'),ownership_sha256=sha(run/'ownership.jsonl'),admission_evidence=q.admissions(run),
            evaluations=evaluations,checks=checks,completed_calls=len(evaluations),mean_log_net=mean,adjusted_change_pct=100*math.expm1(mean),
            all_guards_retired=True,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
            no_followup_authorization=True,production_qualified=False,root_actual_controller_retirement_required=True,
            comparator='accepted actual YR7v2 controls; diagnostic only',separate_prd224_gain_banked=False,fresh_requires_actual_YR7_parent=True))
    except BaseException as e:
        write(run/'STOPPED_v1.json',dict(status='IMPLEMENTATION_OR_OWNER_FAILURE_CLOSED',error=repr(e),evaluations=evaluations,checks=checks,
            no_retry=True,owned_group_retirement_verified=guard.OWNED_RETIREMENT_CONFIRMED,
            slot_reservation_retained=not guard.OWNED_RETIREMENT_CONFIRMED,root_controller_retirement_required=True))
        raise

def interrupted(signum,frame):raise KeyboardInterrupt('Owned PRD controller signal%s'%signum)
if __name__=='__main__':
    previous=signal.signal(signal.SIGTERM,interrupted)
    try:main()
    finally:signal.signal(signal.SIGTERM,previous)
