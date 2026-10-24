"""Fixed read-action common-basis diagnostic, at most two new native calls."""
import argparse,hashlib,importlib.util,json,math,os,signal,subprocess,time
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
DONOR=O/'lead/runs/astra_gop2_current_delivery_v1/finite_gop2_current_qualification_HELD_v1.py'
assert sha(DONOR)=='1af0cb77eacc49a1a6b206bfd87fd190dfc24667de47c627834be4e4d4e56b91'
spec=importlib.util.spec_from_file_location('_rap4_owned_donor',DONOR)
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
guard=q.guard;write=q.write
ORDER=[('ON',7100136),('ON',7100137)]
RULE=dict(either_seed_gross_log_net_gte=math.log(1.02),otherwise_pair_mean_log_net_lt=0.0)

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
    assert c['purpose']=='rap4_diagnostic2' and c['fixed_order']==[list(x)for x in ORDER]
    assert c['rule']==RULE and c['new_native_calls_max']==2 and c['resource_policy']==q.RESOURCE and c['admission_wait_policy']==q.WAIT
    assert c['run_name']=='run_rap4_diagnostic_v1'
    assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
    assert c['source_pins'][str(DONOR)]==sha(DONOR)
    assert set(c['candidates'])=={'parent','ON'} and all(x['mode']==1 for x in c['candidates'].values())
    parent=c['candidates']['parent'];arm=c['candidates']['ON']
    assert parent['sha256']=='4e6e4daee89b6d7c8812e145e627896fb59afc511b3b61da4a797fe54774dc76'
    assert arm['sha256']=='f65e34965d60400a368c3d1c03f2ace4abcc5b1f0793864c2635faddf018f006'
    build_path=D/'build_rap4_source_HELD_v2.py'
    bs=importlib.util.spec_from_file_location('_rap4_frozen_source',build_path)
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
    st=a['stats']['_UC8_STATS']
    assert st['retired'] is True and st['builds']
    assert all(x[0] in (10,13) and x[1] in (320,384) and x[2]>0 and x[3:]==[248,72,8] for x in st['builds'])
    for kind in ('reconstruction','read'):
        assert set(x[0] for x in st[kind])==set(range(10,15))
        assert all(len(x)==3 and 0<x[1]<=1024 and x[2]>0 for x in st[kind])
    rab=a['stats']['_RAP4_STATS']
    assert rab['retired']is True and len(rab['builds'])==len(st['builds'])
    assert all(len(x)==5 and x[0]in(10,13) and 72<=x[1]<=1024 and x[2]in(320,384) and x[3]>0 and x[4]==4 for x in rab['builds'])
    assert len(rab['probe_rows'])==len(rab['builds']) and all(len(x)==4 and len(set(x))==4 and all(type(i)is int and 0<=i<1024 for i in x)for x in rab['probe_rows'])
    assert all(type(x)is int and x==0 for x in a['stats']['_ASTRA_IOP_STATS'].values())
    lm=math.log(a['mse_final']/b['mse_final']);lb=math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
    return dict(seed=seed,implementation_pass=True,prefix_first10_exact=True,all16_finite=True,
        comparator='accepted actual YR7v2 controls; diagnostic only',separate_rap4_gain_banked=False,fresh_requires_actual_YR7_parent=True,
        log_mse_ratio=lm,log_bill_ratio=lb,log_net=lm+lb,adjusted_change_pct=100*math.expm1(lm+lb),
        parent_mse=b['mse_final'],arm_mse=a['mse_final'],parent_bill=b['flops'],arm_bill=a['flops'],
        parent_ops=b['ops'],arm_ops=a['ops'],uc8_stats=st,rap4_stats=rab,shape_route_stats={k:a['stats'][k]for k in ('_ASTRA_RLF_STATS','_UBX_RC_STATS')})
