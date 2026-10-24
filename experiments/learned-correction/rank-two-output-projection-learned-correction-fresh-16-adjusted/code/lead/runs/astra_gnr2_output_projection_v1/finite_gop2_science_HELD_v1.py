"""Held fixed native fixed rank-two E2/E3 energy science:16fresh+16OOS, two fixed8-seed shards.
One managed controller per shard owns one sequential guarded native slot.
Complete both shards and actual root retirement before any phase means.
"""
import argparse,hashlib,importlib.util,json,math,os,signal,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
QP=D/'finite_gop2_smoke_HELD_v1.py'
assert sha(QP)=='707ad550b18b2dbc9e45d3c8f0da2a38d5eb7e3a3e08ff0ed6c278d758d706a3'
s=importlib.util.spec_from_file_location('private_gop2_native_science',QP);q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
QUAL_JUDGE=q.guard.JUDGE
PROBE_CURRENT=q.current
SCIENCE_JUDGE=D/'native_gop2_science_judge_HELD_v1.py'
q.guard.JUDGE=SCIENCE_JUDGE
def current(c,item,role,seed):
 for k in ('certificate','guard'):assert sha(item[k])==item[k+'_sha256']
 p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
 d=json.loads(p.read_text());g=json.loads(Path(item['guard']).read_text());src=c['candidates'][role]
 assert d['candidate']==src['path']and d['sha256']==src['sha256']and d['gop2_mode']==src['mode']
 assert d['judge_sha256']==c['source_pins'][str(q.guard.JUDGE)]and d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
 assert d['environment']=={**{k:'1'for k in q.guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':src['sha256'],'GOP2_MODE':str(src['mode'])}
 assert g['command']==[str(q.guard.RT),'-B',str(q.guard.JUDGE),src['path'],str(seed),str(p)]
 assert g['completed']is True and g['exit_code']==0 and g['error']is None and g['whole_group_retired']is True and g['remaining_rows']==[]
 assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
 assert d['setup']['pb_wall_s']==1e9 and math.isfinite(d['setup']['wall_s'])and d['setup']['wall_s']>=0
 y=q.prediction(p,seed);assert d['seeds']==[seed]and len(d['rows'])==1
 r=d['rows'][0];assert r['seed']==seed and type(r['flops'])is int and 0<r['flops']<=2**41 and type(r['ops'])is int and r['ops']>0
 assert len(r['mse_layers'])==16 and all(math.isfinite(v)and v>0 for v in r['mse_layers'])and r['mse_final']==r['mse_layers'][-1]
 assert hashlib.sha256(y.tobytes()).hexdigest()[:16]==r['pred_sha']
 assert r['gop2_cleanup']==dict(active_clear=True,post_cov32_absent=True,hcf_ctx_clear=True,hcf_rows_clear=True,k4l_final_clear=True)
 q.old.contexts(r['stats'],'parent')
 if role=='ON':
  assert r['stats']['_GOP2_STATS']==dict(calls=15,layers=list(range(1,16)),energy_blocks=45,message_misses=15,retired=True,value_cache_entries=0)
  assert r['stats']['_GOP2_ACTIVE']==[False]
 else:
  assert role=='parent'and r['stats']['_GOP2_STATS']is None and r['stats']['_GOP2_ACTIVE']is None
 t=next(t for t in c['truth_metadata']if t['seed']==seed)
 assert r['weights_before_sha256']==r['weights_after_sha256']==t['weights_sha256']
 return dict(cert=d,row=r,pred=y,evidence=item)

def gate_ON(base,arm):
 q.setup_equal(base,arm)
 assert base['cert']['python_version']==arm['cert']['python_version']
 assert base['pred'][0].tobytes()==arm['pred'][0].tobytes()
 b,a=base['row'],arm['row']
 logbill=math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
 return dict(first_layer_exact=True,setup_identical=True,fifteen_energy_routes=True,log_bill_ratio=logbill,ops_delta=a['ops']-b['ops'],no_accuracy_or_price_sign_gate=True)

SEEDS=None # Bound only from a registered, separately root-allocated protocol.
SHARDS=(0,1)
def validated_seeds(c):
 x=c['phase_seeds'];assert set(x)=={'fresh','oos'}
 assert all(isinstance(x[k],list)and len(x[k])==16 for k in ('fresh','oos'))
 allseeds=x['fresh']+x['oos']
 assert len(set(allseeds))==32 and all(type(v)is int and 7100000<=v<7200000 for v in allseeds)
 return {k:list(x[k])for k in ('fresh','oos')}

def load_config(path,digest):
 global SEEDS
 assert sha(path)==digest;c=json.loads(Path(path).read_text())
 assert c['status']=='REGISTERED' and c['purpose']=='gnn_output_projection_rank2_direct_native_fresh16_oos16'
 assert c['direct_native_hypothesis']is True
 SEEDS=validated_seeds(c)
 assert c['shards']==list(SHARDS) and c['native_calls_max']==64 and c['native_control_reuse']==[]
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 assert c['resource_policy']==dict(entry_free_percent_min=60,entry_counted_max_exclusive=8,logical_total_max=8,live_free_percent_min=50,live_counted_max=10,native_rss_footprint_max=7500000000)
 assert c['admission_wait_policy']==dict(poll_seconds=15,status_seconds=30,max_wait_seconds=1800,no_native_child_while_waiting=True,no_automatic_retry=True)
 q.guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 c['_protocol_path']=str(Path(path).resolve());c['_protocol_sha256']=digest
 allocation(c)
 return c

def allocation(c):
 p=Path(c['root_allocation_path']);assert sha(p)==c['root_allocation_sha256']
 a=json.loads(p.read_text())
 assert a['approved']is True and a['reviewed_by']=='root' and a['purpose']==c['purpose']
 assert a['fresh']==SEEDS['fresh'] and a['oos']==SEEDS['oos']
 assert a['baseline_sha256']==c['candidates']['parent']['sha256']
 assert c['source_pins'][str(p.resolve())]==sha(p)
 assert a['no_evaluated_seed_reclassified']is True and a['no_requests']is True and a['numerical_authorization']is False
 assert a['truth_metadata']==c['truth_metadata']
 assert [x['seed']for x in c['truth_metadata']]==SEEDS['fresh']+SEEDS['oos']
 for t in c['truth_metadata']:assert t['N']>=15990784 and sha(t['path'])==t['sha256']

def qualification(c,g):
 rp=Path(g['qualification_root_acceptance_path']);assert sha(rp)==g['qualification_root_acceptance_sha256']==c['qualification_root_retirement_sha256']
 assert rp==Path(c['qualification_root_retirement_path'])
 root=json.loads(rp.read_text());assert root['approved']is True and root['reviewed_by']=='root'
 assert root['actual_exit_code']==0 and root['all_owned_PID_PPID_PGID_absent']is True
 owner=Path(root['owner_path']);assert sha(owner)==root['owner_sha256'];o=json.loads(owner.read_text())
 assert o['actual_managed_exit_observed']is True and o['exit_code']==0 and o['all_owned_PID_PPID_PGID_absent']is True
 assert o['protocol_sha256']==c['qualification_protocol_sha256']and o['owned_absent_ids']==root['owned_ids']
 pp=Path(c['qualification_protocol_path']);assert sha(pp)==c['qualification_protocol_sha256'];qc=json.loads(pp.read_text())
 assert qc['candidates']['ON']==c['candidates']['ON']
 dp=Path(o['decision_path']);assert sha(dp)==o['decision_sha256'];decision=json.loads(dp.read_text())
 assert decision['status']=='PASS_TO_REQUEST_INDEPENDENT_FRESH'and decision['fixed_two_seed_complete']is True and decision['complete_calls']==2
 assert decision['eligible_to_request_fresh']is True and decision['mean_log_adjusted_ratio']<0
 return dict(path=str(rp),sha256=sha(rp),smoke_decision_path=str(dp),smoke_decision_sha256=sha(dp))

def science_current(c,item,role,seed):
 x=current(c,item,role,seed)
 assert x['cert']['science_protocol_sha256']==c['_protocol_sha256']
 assert x['cert']['science_protocol_path']==c['_protocol_path']
 return x

def home(c,phase,shard):
 h=D/c['run_names'][phase][str(shard)]
 assert h.parent==D and h.name.startswith('run_gop2_science_')
 return h

def order(phase,shard):return [(role,seed)for seed in SEEDS[phase][shard::len(SHARDS)]for role in('parent','ON')]

def admissions(h,sequence,controller):
 records=[json.loads(x)for x in (h/'admission.jsonl').read_text().splitlines()]
 assert [(x['role'],x['seed'])for x in records]==sequence
 assert all(x['controller_pid']==controller and x['free_percent']>=60 and x['counted']<8 and x['admitted']is True and 0<=x['elapsed_seconds']<=1800 for x in records)
 attempts=[json.loads(x)for x in (h/'admission_wait.jsonl').read_text().splitlines()]
 assert [x for x in attempts if x['admitted']]==records
 assert all(x['controller_pid']==controller and (x['role'],x['seed'])in sequence for x in attempts)
 return dict(path=str(h/'admission.jsonl'),sha256=sha(h/'admission.jsonl'),records=records,wait_path=str(h/'admission_wait.jsonl'),wait_sha256=sha(h/'admission_wait.jsonl'))

def run(c,digest,phase,shard,gp):
 assert shard in SHARDS
 g=json.loads(gp.read_text());assert g['approved']is True and g['reviewed_by']=='root'
 assert g['purpose']==c['purpose'] and g['phase']==phase and g['protocol_sha256']==digest
 assert g['runner_sha256']==sha(__file__) and g['numerical_slots']==2 and g['shards']==list(SHARDS)
 assert g['resource_policy']==c['resource_policy'] and g['admission_wait_policy']==c['admission_wait_policy']
 assert g['max_concurrent_shards']==2 and g['logical_total_reserved']<=8
 assert g['run_names']==c['run_names'][phase] and all(os.environ.get(k)=='1'for k in q.guard.THREADS)
 assert os.getpriority(os.PRIO_PROCESS,0)==0
 allocation(c);accepted=qualification(c,g)
 if phase=='oos':
  fp=D/c['decision_names']['fresh'];assert sha(fp)==g['fresh_decision_sha256']
  f=json.loads(fp.read_text());assert f['protocol_sha256']==digest and f['phase']=='fresh' and f['passed']is True
  freshgot,_=pool(c,digest,'fresh',Path(f['barrier']['root_retirement_path']))
  freshrows,freshmean,freshse,freshpass=summarize(freshgot,SEEDS['fresh'])
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
   q.write(h/f'COUNT_{len(evaluations):02d}_v1.json',dict(phase=phase,shard=shard,completed=len(evaluations),expected=16))
   print(json.dumps(dict(phase=phase,shard=shard,completed=len(evaluations),expected=16)),flush=True)
  admission_evidence=admissions(h,sequence,os.getpid())
  q.write(h/'NATIVE_SHARD_READY_v1.json',dict(status='all16_calls_and_guards_complete_await_controller_retirement',protocol_sha256=digest,root_grant_sha256=sha(gp),phase=phase,shard=shard,controller_pid=os.getpid(),qualification_acceptance=accepted,fresh_decision_sha256=g.get('fresh_decision_sha256'),evaluations=evaluations,admission_evidence=admission_evidence,all_guards_retired=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,scientific_means_computed=False,automatic_followup=False))
 except BaseException as e:
  q.write(h/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,no_retry=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,slot_reservation_retained=not q.guard.OWNED_RETIREMENT_CONFIRMED))
  raise

def pool(c,digest,phase,rp):
 root=json.loads(rp.read_text())
 assert root['approved']is True and root['reviewed_by']=='root'
 assert root['status']=='two_controllers_exited0_and_all32_guards_retired'
 assert root['protocol_sha256']==digest and root['phase']==phase
 assert root['all_owned_PID_PPID_PGID_absent']is True
 assert [x['shard']for x in root['shards']]==list(SHARDS)
 assert len({x['controller_pid']for x in root['shards']})==2
 ready=[];pins={}
 # Establish BOTH full fixed shard barriers before any scientific result JSON.
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
  assert grant['numerical_slots']==2 and grant['shards']==list(SHARDS) and grant['run_names']==c['run_names'][phase]
  assert r['qualification_acceptance']['sha256']==grant['qualification_root_acceptance_sha256']
  assert r['qualification_acceptance']['path']==grant['qualification_root_acceptance_path']
  assert sha(r['qualification_acceptance']['path'])==r['qualification_acceptance']['sha256']
  assert grant['resource_policy']==c['resource_policy'] and grant['admission_wait_policy']==c['admission_wait_policy']
  assert grant['max_concurrent_shards']==2 and grant['logical_total_reserved']<=8
  assert plan['resource_policy']==c['resource_policy'] and plan['admission_wait_policy']==c['admission_wait_policy']
  owners=[json.loads(x)for x in op.read_text().splitlines()if x.strip()]
  assert len(owners)==16
  evidence=admissions(h,order(phase,shard),r['controller_pid'])
  assert r['admission_evidence']==evidence
  assert owner['admission_sha256']==evidence['sha256'] and owner['admission_wait_sha256']==evidence['wait_sha256']

  assert r['status']=='all16_calls_and_guards_complete_await_controller_retirement'
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
 got,barrier=pool(c,digest,phase,rp);rows,mean,se,passed=summarize(got,SEEDS[phase])
 result=dict(protocol_sha256=digest,phase=phase,rows=rows,barrier=barrier,mean_log_net=mean,SE_log_net=se,strict_negative_pool=mean<0,passed=passed,unique_networks=16,direct_native_gnn_output_projection_rank2_hypothesis=True,no_followup_authorization=True)
 if phase=='oos':
  fp=D/c['decision_names']['fresh'];assert sha(fp)==barrier['fresh_decision_sha256']
  fresh=json.loads(fp.read_text());assert fresh['phase']=='fresh' and fresh['protocol_sha256']==digest and fresh['passed']is True
  fgot,_=pool(c,digest,'fresh',Path(fresh['barrier']['root_retirement_path']))
  frows,fmean,fse,fpass=summarize(fgot,SEEDS['fresh']);assert frows==fresh['rows'] and fmean==fresh['mean_log_net'] and fpass
  allrows=frows+rows;allmean=math.fsum(x['log_net']for x in allrows)/32
  result.update(all32_mean_log_net=allmean,all32_SE_log_net=statistics.stdev(x['log_net']for x in allrows)/math.sqrt(32),unique_networks=32,fresh_decision_sha256=sha(fp),passed=bool(passed and fpass and allmean<0))
 q.write(D/c['decision_names'][phase],result)
 print(json.dumps(dict(phase=phase,passed=result['passed'],mean_log_net=mean,all32_mean_log_net=result.get('all32_mean_log_net'))))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','grade'));p.add_argument('--phase',choices=('fresh','oos'),required=True)
 p.add_argument('--shard',type=int,choices=SHARDS);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
 p.add_argument('--run-authorized',action='store_true');p.add_argument('--grant',type=Path);p.add_argument('--root-retirement',type=Path)
 a=p.parse_args();c=load_config(a.protocol,a.protocol_sha256)
 if a.mode=='run':assert a.run_authorized and a.grant and a.shard is not None;run(c,a.protocol_sha256,a.phase,a.shard,a.grant)
 else:assert a.root_retirement;grade(c,a.protocol_sha256,a.phase,a.root_retirement)

def interrupted(signum,frame):raise KeyboardInterrupt('Owned GOP2 science controller signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,old)
