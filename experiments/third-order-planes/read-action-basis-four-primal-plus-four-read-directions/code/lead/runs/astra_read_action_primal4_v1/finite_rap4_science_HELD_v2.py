"""Held frozen RAP4 ON-only pilot8 with accepted d401 controls; remaining fresh8+OOS8 paired; four fixed shards.
One managed controller per shard owns one sequential guarded native slot.
Complete all four shards and actual root retirement before any phase means.
"""
import argparse,hashlib,importlib.util,json,math,os,signal,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
QP=D/'finite_rap4_interfaces_HELD_v1.py'
assert sha(QP)=='470ec165c841416dac19f5ebbe057972c7cb226bcee1b6fc65ad50a0fc950a48'
s=importlib.util.spec_from_file_location('private_rap4_native_science',QP);smoke=importlib.util.module_from_spec(s);s.loader.exec_module(smoke)
q=smoke.q
q.RESOURCE=dict(q.RESOURCE,logical_total_max=7)
SCIENCE_JUDGE=D/'native_rap4_science_judge_HELD_v1.py'
q.guard.JUDGE=SCIENCE_JUDGE
q.STATS=q.STATS+('_YR7_STATS','_YR7_ACTIVE','_YR7_BIND_STATS','_UC8_STATS','_RAP4_STATS')
def current(c,item,role,seed):
 assert role in ('parent','ON')
 cached=role=='parent'and seed in c['identity_groups']['pilot']
 original=q.guard.JUDGE
 if cached:q.guard.JUDGE=Path(c['parent_control_judge'])
 try:x=q.current(dict(c,candidates={'ON':c['candidates'][role]}),item,seed)
 finally:q.guard.JUDGE=original
 a=x['row'];st=a['stats']
 assert st['_YR7_ACTIVE']==[False]and st['_YR7_BIND_STATS']==dict(keys=17,retired=True)
 ys=st['_YR7_STATS'];assert ys['retired']is True and ys['value_cache_entries']==0
 assert all(tuple(v)in {tuple(z)for z in c['shape_allowlist']}for v in ys['routed_shapes'])
 if role=='ON':
  uc=st['_UC8_STATS'];assert uc['retired']is True and uc['builds']
  assert all(v[0]in(10,13)and v[1]in(320,384)and v[2]>0 and v[3:]==[248,72,8]for v in uc['builds'])
  for kind in('reconstruction','read'):
   assert set(v[0]for v in uc[kind])==set(range(10,15))
   assert all(len(v)==3 and 0<v[1]<=1024 and v[2]>0 for v in uc[kind])
  rab=st['_RAP4_STATS'];assert rab['retired']is True and len(rab['builds'])==len(uc['builds'])
  assert all(len(v)==5 and v[0]in(10,13)and 72<=v[1]<=1024 and v[2]in(320,384)and v[3]>0 and v[4]==4 for v in rab['builds'])
  assert len(rab['probe_rows'])==len(rab['builds'])and all(len(v)==4 and len(set(v))==4 and all(type(i)is int and 0<=i<1024 for i in v)for v in rab['probe_rows'])
 else:assert st.get('_UC8_STATS')is None and st.get('_RAP4_STATS')is None
 return x

def gate_ON(base,arm):
 q.setup_equal(base,arm)
 assert base['cert']['python_version']==arm['cert']['python_version']
 assert base['pred'][:10].tobytes()==arm['pred'][:10].tobytes()
 b,a=base['row'],arm['row'];assert a['mse_layers'][:10]==b['mse_layers'][:10]
 logbill=math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
 return dict(first10_layers_exact=True,setup_identical=True,rap4_fixed_route=True,log_bill_ratio=logbill,ops_delta=a['ops']-b['ops'],no_accuracy_or_price_sign_gate=True)

SEEDS=None # Bound only from a registered, separately root-allocated protocol.
SHARDS=(0,1,2,3)
def validated_seeds(c):
 x=c['identity_groups'];assert set(x)=={'pilot','fresh_tail','oos'}
 assert all(isinstance(x[k],list)and len(x[k])==8 for k in x)
 allseeds=x['pilot']+x['fresh_tail']+x['oos']
 assert len(set(allseeds))==24 and all(type(v)is int and 7100000<=v<7200000 for v in allseeds)
 assert c['phase_seeds']=={'pilot':x['pilot'],'remaining':x['fresh_tail']+x['oos']}
 return c['phase_seeds']

def load_config(path,digest):
 global SEEDS
 assert sha(path)==digest;c=json.loads(Path(path).read_text())
 assert c['status']=='REGISTERED'and c['purpose']=='rap4_pilot8_fresh16_oos8'
 SEEDS=validated_seeds(c)
 assert c['shards']==list(SHARDS)and c['native_calls_max']==40 and len(c['native_control_reuse'])==8
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 assert c['resource_policy']==q.RESOURCE and c['admission_wait_policy']==q.WAIT
 assert c['candidates']['parent']['sha256']=='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
 assert c['candidates']['ON']['sha256']=='f65e34965d60400a368c3d1c03f2ace4abcc5b1f0793864c2635faddf018f006'
 assert all(x['mode']==1 for x in c['candidates'].values())
 q.guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 c['_protocol_path']=str(Path(path).resolve());c['_protocol_sha256']=digest
 allocation(c);control_metadata(c)
 return c

def allocation(c):
 p=Path(c['root_allocation_path']);assert sha(p)==c['root_allocation_sha256']
 a=json.loads(p.read_text())
 assert a['approved']is True and a['reviewed_by']=='root'and a['purpose']==c['purpose']
 assert all(a[k]==c['identity_groups'][k]for k in ('pilot','fresh_tail','oos'))
 assert a['baseline_sha256']==c['candidates']['parent']['sha256']and a['candidate_sha256']==c['candidates']['ON']['sha256']
 assert c['source_pins'][str(p.resolve())]==sha(p)
 assert a['no_evaluated_seed_reclassified']is True and a['no_requests']is True and a['numerical_authorization']is False
 assert a['truth_metadata']==c['truth_metadata']
 assert [x['seed']for x in c['truth_metadata']]==SEEDS['pilot']+SEEDS['remaining']
 for t in c['truth_metadata']:assert t['N']>=15990784 and sha(t['path'])==t['sha256']

def control_metadata(c):
 p=Path(c['parent_controls_acceptance_path']);assert sha(p)==c['parent_controls_acceptance_sha256']
 a=json.loads(p.read_text())
 assert a['approved']is True and a['reviewed_by']=='root'and a['candidate_sha256']==c['candidates']['parent']['sha256']
 assert a['status']=='four_controllers_exited0_and_all_expected_guards_retired'
 assert a['all_actual_managed_exits_observed']is True and a['actual_exit_codes']==[0,0,0,0]
 assert a['all_owned_PID_PPID_PGID_absent']is True and a['all_guards_clean']is True and a['all16_finite']is True
 assert a['completed_parent_calls']==8 and a['completed_candidate_calls']==0 and a['seeds']==c['identity_groups']['pilot']
 assert a['evaluations']==c['native_control_reuse']and a['truth_metadata']==c['truth_metadata'][:8]
 assert a['PB_ALL']=='1' and a['PB_WALL_S']==1e9 and a['prediction_dtype']=='float32'
 assert a['protocol_path']==c['parent_controls_protocol_path']and a['protocol_sha256']==c['parent_controls_protocol_sha256']
 assert sha(a['protocol_path'])==a['protocol_sha256']and sha(a['owner_path'])==a['owner_sha256']
 for file,h in a['evidence_sha256'].items():assert sha(file)==h
 return a

def qualification(c,g):
 rp=Path(g['qualification_root_acceptance_path']);assert sha(rp)==g['qualification_root_acceptance_sha256']==c['qualification_root_retirement_sha256']
 assert rp==Path(c['qualification_root_retirement_path'])
 root=json.loads(rp.read_text());assert root['approved']is True and root['reviewed_by']=='root'
 assert root['candidate_sha256']==c['candidates']['ON']['sha256']
 assert root['both_actual_managed_exits_observed']is True and root['all_owned_PID_PPID_PGID_absent']is True
 retirement=Path(root['retirement_path']);assert sha(retirement)==root['retirement_sha256']
 rr=json.loads(retirement.read_text());assert rr['approved']is True and rr['reviewed_by']=='root'and rr['all_owned_PID_PPID_PGID_absent']is True
 dp=Path(root['complete_path']);assert sha(dp)==root['complete_sha256']
 decision=json.loads(dp.read_text())
 assert decision['status']=='SURVIVES_TO_ROOT_FRESH_REVIEW'and decision['completed_calls']==2 and decision['mean_log_net']<0
 assert decision['candidate_sha256']==c['candidates']['ON']['sha256']and decision['seeds']==[7100136,7100137]
 assert decision['root_retirement_path']==str(retirement)and decision['root_retirement_sha256']==sha(retirement)
 assert len(decision['checks'])==2 and [v['seed']for v in decision['checks']]==[7100136,7100137]
 assert all(v['implementation_pass']is True and math.isfinite(v['log_net'])and v['log_net']<math.log(1.02)for v in decision['checks'])
 pp=Path(c['qualification_protocol_path']);assert sha(pp)==c['qualification_protocol_sha256'];qc=json.loads(pp.read_text())
 assert qc['candidates']['ON']==c['candidates']['ON']
 assert g['parent_controls_reuse_approved']is True and g['parent_controls_acceptance_sha256']==c['parent_controls_acceptance_sha256']
 return dict(path=str(rp),sha256=sha(rp),smoke_decision_path=str(dp),smoke_decision_sha256=sha(dp))

def science_current(c,item,role,seed):
 x=current(c,item,role,seed)
 cached=role=='parent'and seed in c['identity_groups']['pilot']
 assert x['cert']['science_protocol_sha256']==(c['parent_controls_protocol_sha256']if cached else c['_protocol_sha256'])
 assert x['cert']['science_protocol_path']==(c['parent_controls_protocol_path']if cached else c['_protocol_path'])
 return x

def home(c,phase,shard):
 h=D/c['run_names'][phase][str(shard)]
 assert h.parent==D and h.name.startswith('run_rap4_science_')
 return h

def order(phase,shard):return [(role,seed)for seed in SEEDS[phase][shard::len(SHARDS)]for role in(('ON',)if phase=='pilot'else('parent','ON'))]

def admissions(h,sequence,controller):
 records=[json.loads(x)for x in (h/'admission.jsonl').read_text().splitlines()]
 assert [(x['role'],x['seed'])for x in records]==sequence
 assert all(x['controller_pid']==controller and x['free_percent']>=60 and x['counted']<10 and x['admitted']is True and 0<=x['elapsed_seconds']<=1800 for x in records)
 attempts=[json.loads(x)for x in (h/'admission_wait.jsonl').read_text().splitlines()]
 assert [x for x in attempts if x['admitted']]==records
 assert all(x['controller_pid']==controller and (x['role'],x['seed'])in sequence for x in attempts)
 return dict(path=str(h/'admission.jsonl'),sha256=sha(h/'admission.jsonl'),records=records,wait_path=str(h/'admission_wait.jsonl'),wait_sha256=sha(h/'admission_wait.jsonl'))

def run(c,digest,phase,shard,gp):
 assert shard in SHARDS
 g=json.loads(gp.read_text());assert g['approved']is True and g['reviewed_by']=='root'
 assert g['purpose']==c['purpose'] and g['phase']==phase and g['protocol_sha256']==digest
 assert g['runner_sha256']==sha(__file__) and g['numerical_slots']==4 and g['shards']==list(SHARDS)
 assert g['resource_policy']==c['resource_policy'] and g['admission_wait_policy']==c['admission_wait_policy']
 assert g['max_concurrent_shards']==4 and g['logical_total_reserved']<=7
 assert g['run_names']==c['run_names'][phase] and all(os.environ.get(k)=='1'for k in q.guard.THREADS)
 assert os.getpriority(os.PRIO_PROCESS,0)==0
 allocation(c);accepted=qualification(c,g)
 if phase=='remaining':
  fp=D/c['decision_names']['pilot'];assert sha(fp)==g['fresh_decision_sha256']
  f=json.loads(fp.read_text());assert f['protocol_sha256']==digest and f['phase']=='pilot' and f['passed']is True
  freshgot,_=pool(c,digest,'pilot',Path(f['barrier']['root_retirement_path']))
  freshrows,freshmean,freshse,freshpass=summarize(freshgot,SEEDS['pilot'])
  assert freshrows==f['rows'] and freshmean==f['mean_log_net'] and freshpass
 h=home(c,phase,shard);h.mkdir(exist_ok=False);sequence=order(phase,shard)
 q.write(h/'PLAN_v1.json',dict(protocol_sha256=digest,root_grant_path=str(gp.resolve()),root_grant_sha256=sha(gp),phase=phase,shard=shard,controller_pid=os.getpid(),resource_policy=c['resource_policy'],admission_wait_policy=c['admission_wait_policy'],fixed_order=sequence,qualification_acceptance=accepted,fresh_decision_sha256=g.get('fresh_decision_sha256'),expected_guards=[str(h/f'native_{seed}_{role}_guard.json')for role,seed in sequence]))
 proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);evaluations=[]
 try:
  for role,seed in sequence:
   q.guard.pins(proto)
   env=dict(os.environ,PB_ALL='1',STATS=','.join(q.STATS),GOP2_MODE=str(c['candidates'][role]['mode']),GOP2_SCIENCE_PROTOCOL_PATH=c['_protocol_path'],GOP2_SCIENCE_PROTOCOL_SHA256=c['_protocol_sha256'],PYTHONDONTWRITEBYTECODE='1')
   q.admit(h,role,seed)
   item=q.guard.run_native(h,proto,role,seed,env)
   science_current(c,item,role,seed) # only finite/profile/source/custody; no comparative science
   evaluations.append(dict(role=role,seed=seed,**item))
   q.write(h/f'COUNT_{len(evaluations):02d}_v1.json',dict(phase=phase,shard=shard,completed=len(evaluations),expected=len(sequence)))
   print(json.dumps(dict(phase=phase,shard=shard,completed=len(evaluations),expected=len(sequence))),flush=True)
  admission_evidence=admissions(h,sequence,os.getpid())
  q.write(h/'NATIVE_SHARD_READY_v1.json',dict(status='all_calls_and_guards_complete_await_controller_retirement',protocol_sha256=digest,root_grant_sha256=sha(gp),phase=phase,shard=shard,controller_pid=os.getpid(),qualification_acceptance=accepted,fresh_decision_sha256=g.get('fresh_decision_sha256'),evaluations=evaluations,admission_evidence=admission_evidence,all_guards_retired=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,scientific_means_computed=False,automatic_followup=False))
 except BaseException as e:
  q.write(h/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,no_retry=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,slot_reservation_retained=not q.guard.OWNED_RETIREMENT_CONFIRMED))
  raise

def pool(c,digest,phase,rp):
 root=json.loads(rp.read_text())
 assert root['approved']is True and root['reviewed_by']=='root'
 assert root['status']=='four_controllers_exited0_and_all_expected_guards_retired'
 assert root['protocol_sha256']==digest and root['phase']==phase
 assert root['all_owned_PID_PPID_PGID_absent']is True
 assert [x['shard']for x in root['shards']]==list(SHARDS)
 assert len({x['controller_pid']for x in root['shards']})==4
 ready=[];pins={}
 # Establish ALL FOUR full fixed shard barriers before any scientific result JSON.
 for shard,owner in zip(SHARDS,root['shards']):
  h=home(c,phase,shard);p=h/'NATIVE_SHARD_READY_v1.json'
  assert owner['ready_path']==str(p) and owner['ready_sha256']==sha(p)
  assert owner['actual_managed_exit_observed']is True and owner['exit_code']==0
  pp=h/'PLAN_v1.json';op=h/'ownership.jsonl'
  assert owner['plan_sha256']==sha(pp) and owner['ownership_sha256']==sha(op)
  r=json.loads(p.read_text());plan=json.loads(pp.read_text())
  assert plan['protocol_sha256']==digest and plan['phase']==phase and plan['shard']==shard
  assert plan['root_grant_sha256']==r['root_grant_sha256']
  assert plan['qualification_acceptance']==r['qualification_acceptance']
  assert plan['fresh_decision_sha256']==r['fresh_decision_sha256']
  gp=Path(plan['root_grant_path']);assert sha(gp)==plan['root_grant_sha256']
  grant=json.loads(gp.read_text())
  assert grant['approved']is True and grant['reviewed_by']=='root' and grant['purpose']==c['purpose']
  assert grant['phase']==phase and grant['protocol_sha256']==digest and grant['runner_sha256']==sha(__file__)
  assert grant['numerical_slots']==4 and grant['shards']==list(SHARDS) and grant['run_names']==c['run_names'][phase]
  assert r['qualification_acceptance']['sha256']==grant['qualification_root_acceptance_sha256']
  assert r['qualification_acceptance']['path']==grant['qualification_root_acceptance_path']
  assert sha(r['qualification_acceptance']['path'])==r['qualification_acceptance']['sha256']
  assert grant['resource_policy']==c['resource_policy'] and grant['admission_wait_policy']==c['admission_wait_policy']
  assert grant['max_concurrent_shards']==4 and grant['logical_total_reserved']<=7
  assert plan['resource_policy']==c['resource_policy'] and plan['admission_wait_policy']==c['admission_wait_policy']
  owners=[json.loads(x)for x in op.read_text().splitlines()if x.strip()]
  assert len(owners)==len(order(phase,shard))
  evidence=admissions(h,order(phase,shard),r['controller_pid'])
  assert r['admission_evidence']==evidence
  assert owner['admission_sha256']==evidence['sha256'] and owner['admission_wait_sha256']==evidence['wait_sha256']

  assert r['status']=='all_calls_and_guards_complete_await_controller_retirement'
  assert r['protocol_sha256']==digest and r['phase']==phase and r['shard']==shard
  assert r['controller_pid']==owner['controller_pid']==plan['controller_pid']
  assert r['all_guards_retired']is True and r['owned_group_retirement_verified']is True
  assert [(x['role'],x['seed'])for x in r['evaluations']]==order(phase,shard)
  assert plan['fixed_order']==[list(x)for x in order(phase,shard)]
  assert plan['expected_guards']==[x['guard']for x in r['evaluations']]
  for own,e in zip(owners,r['evaluations']):
   role,seed=e['role'],e['seed'];source=c['candidates'][role]
   assert own['controller_pid']==r['controller_pid'] and own['controller_nice']==0 and own['guard_native_nice']==10
   assert own['kind']==role and own['seed']==seed and own['guard_record']==e['guard']
   assert type(own['owned_guard_pid'])is int and own['owned_guard_pid']>0
   assert own['candidate_sha256']==source['sha256']
   assert own['environment']=={**{k:'1'for k in q.guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':source['sha256']}
   assert own['command']==[str(q.guard.PY),'-B',str(q.guard.GUARD),'--record',e['guard'],'--seconds','900','--footprint-limit','7500000000','--rss-backstop','7500000000',str(q.guard.RT),'-B',str(SCIENCE_JUDGE),source['path'],str(seed),e['certificate']]
   for kind in('certificate','guard'):assert sha(e[kind])==e[kind+'_sha256'];pins[e[kind]]=e[kind+'_sha256']
   pp=Path(e['certificate']);pred=pp.with_name(pp.stem+'_pred.npz');assert sha(pred)==e['prediction_npz_sha256'];pins[str(pred)]=e['prediction_npz_sha256']
  ready.append(r)
 assert all(r['qualification_acceptance']==ready[0]['qualification_acceptance'] for r in ready)
 assert all(r['root_grant_sha256']==ready[0]['root_grant_sha256'] for r in ready)
 assert all(r['fresh_decision_sha256']==ready[0]['fresh_decision_sha256'] for r in ready)
 got={}
 for r in ready:
  for e in r['evaluations']:got[(e['role'],e['seed'])]=science_current(c,e,e['role'],e['seed'])
 if phase=='pilot':
  control_metadata(c)
  for e in c['native_control_reuse']:
   got[('parent',e['seed'])]=science_current(c,e,'parent',e['seed'])
   for kind in('certificate','guard'):pins[e[kind]]=e[kind+'_sha256']
   pp=Path(e['certificate']);pins[str(pp.with_name(pp.stem+'_pred.npz'))]=e['prediction_npz_sha256']
 return got,dict(root_retirement_path=str(rp),root_retirement_sha256=sha(rp),result_guard_prediction_pins=pins,qualification_acceptance=ready[0]['qualification_acceptance'],fresh_decision_sha256=ready[0]['fresh_decision_sha256'])

def summarize(got,seeds):
 rows=[]
 for seed in seeds:
  base,arm=got[('parent',seed)],got[('ON',seed)]
  try:check=gate_ON(base,arm);valid=True;error=None
  except AssertionError as e:check=None;valid=False;error=repr(e)
  a,b=arm['row'],base['row'];mse=math.log(a['mse_final']/b['mse_final']);bill=math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
  rows.append(dict(seed=seed,implementation_gates_pass=valid,error=error,log_MSE_ratio=mse,log_bill_ratio=bill,log_net=mse+bill,implementation_check=check))
 mean=math.fsum(x['log_net']for x in rows)/len(rows)
 se=statistics.stdev(x['log_net']for x in rows)/math.sqrt(len(rows))
 passed=all(x['implementation_gates_pass'] and math.isfinite(x['log_net'])for x in rows)and mean<0
 return rows,mean,se,bool(passed)

def grade(c,digest,phase,rp):
 got,barrier=pool(c,digest,phase,rp)
 if phase=='pilot':
  rows,mean,se,passed=summarize(got,SEEDS['pilot'])
  result=dict(protocol_sha256=digest,phase=phase,rows=rows,barrier=barrier,mean_log_net=mean,SE_log_net=se,strict_negative_pool=mean<0,passed=passed,unique_networks=8,no_followup_authorization=True)
 else:
  fp=D/c['decision_names']['pilot'];assert sha(fp)==barrier['fresh_decision_sha256']
  pilot=json.loads(fp.read_text());assert pilot['phase']=='pilot'and pilot['protocol_sha256']==digest and pilot['passed']is True
  pgot,_=pool(c,digest,'pilot',Path(pilot['barrier']['root_retirement_path']))
  prows,pmean,pse,ppass=summarize(pgot,SEEDS['pilot']);assert prows==pilot['rows']and pmean==pilot['mean_log_net']and ppass
  allgot={**pgot,**got};groups=c['identity_groups']
  frows,fmean,fse,fpass=summarize(allgot,groups['pilot']+groups['fresh_tail'])
  orows,omean,ose,opass=summarize(allgot,groups['oos'])
  arows,amean,ase,apass=summarize(allgot,groups['pilot']+groups['fresh_tail']+groups['oos'])
  result=dict(protocol_sha256=digest,phase=phase,rows=arows,barrier=barrier,pilot_decision_sha256=sha(fp),fresh16_mean_log_net=fmean,fresh16_SE_log_net=fse,fresh16_passed=fpass,oos8_mean_log_net=omean,oos8_SE_log_net=ose,oos8_passed=opass,all24_mean_log_net=amean,all24_SE_log_net=ase,passed=bool(ppass and fpass and opass and apass),unique_networks=24,no_followup_authorization=True)
 q.write(D/c['decision_names'][phase],result)
 print(json.dumps(dict(phase=phase,passed=result['passed'],mean_log_net=result.get('mean_log_net'),all24_mean_log_net=result.get('all24_mean_log_net'))))


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','grade'));p.add_argument('--phase',choices=('pilot','remaining'),required=True)
 p.add_argument('--shard',type=int,choices=SHARDS);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
 p.add_argument('--run-authorized',action='store_true');p.add_argument('--grant',type=Path);p.add_argument('--root-retirement',type=Path)
 a=p.parse_args();c=load_config(a.protocol,a.protocol_sha256)
 if a.mode=='run':assert a.run_authorized and a.grant and a.shard is not None;run(c,a.protocol_sha256,a.phase,a.shard,a.grant)
 else:assert a.root_retirement;grade(c,a.protocol_sha256,a.phase,a.root_retirement)

def interrupted(signum,frame):raise KeyboardInterrupt('Owned RAP4 science controller signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,old)
