"""Held fixed native placement science:16fresh+16OOS, four fixed4-seed shards.
One managed controller per shard owns one sequential guarded native slot.
Complete all four shards and actual root retirement before any phase means.
"""
import argparse,hashlib,importlib.util,json,math,os,signal,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
QP=D/'finite_lkc_qualification_UNVALIDATED_v1.py'
assert sha(QP)=='2b002d3d8bdcbba02286a4cfc8ed0395ed05b365d19b169ca242d2005d4760bc'
s=importlib.util.spec_from_file_location('private_lkc_native_science',QP);q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
QUAL_JUDGE=q.guard.JUDGE
SCIENCE_JUDGE=D/'native_lkc_science_judge_HELD_v1.py'
q.guard.JUDGE=SCIENCE_JUDGE
SEEDS={'fresh':list(range(7101672,7101688)),'oos':list(range(7101688,7101704))}
SHARDS=(0,1,2,3)
def load_config(path,digest):
 assert sha(path)==digest;c=json.loads(Path(path).read_text())
 assert c['status']=='REGISTERED' and c['purpose']=='native_late_kc_science32'
 assert c['direct_native_hypothesis']is True and c['phase_seeds']==SEEDS
 assert c['shards']==list(SHARDS) and c['native_calls_max']==64 and c['native_control_reuse']==[]
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 q.guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 return c

def allocation(c):
 delta=c['allocation_delta_clearance'];cp=Path(delta['path']);assert sha(cp)==delta['sha256']
 clearance=json.loads(cp.read_text())
 assert clearance['approved']is True and clearance['reviewed_by']=='root' and clearance['no_collision']is True
 assert clearance['allocation_sha256']==c['root_allocation_sha256']
 p=Path(c['root_allocation_path']);assert sha(p)==c['root_allocation_sha256']
 a=json.loads(p.read_text())
 assert a['approved']is True and a['purpose']=='native_late_kc_science32'
 assert a['phase_seeds']==SEEDS
 assert a['explicit_obsolete_S54_reserve_release']==SEEDS['fresh']+SEEDS['oos']
 assert a['no_evaluated_seed_reclassified']is True and a['no_models_or_requests_authorized']is True
 assert a['truth_metadata']==c['truth_metadata']
 assert [x['seed']for x in c['truth_metadata']]==SEEDS['fresh']+SEEDS['oos']
 for t in c['truth_metadata']:assert t['N']>=15990784 and sha(t['path'])==t['sha256']

def qualification(c,g):
 pp=Path(c['qualification_protocol_path']);assert sha(pp)==c['qualification_protocol_sha256']
 qc=json.loads(pp.read_text());assert qc['candidates']==c['candidates']
 rp=Path(g['qualification_root_acceptance_path']);assert sha(rp)==g['qualification_root_acceptance_sha256']
 root=json.loads(rp.read_text())
 assert root['approved']is True and root['reviewed_by']=='root'
 assert root['status']=='root_qualification6_price_accepted'
 assert root['protocol_sha256']==c['qualification_protocol_sha256']
 assert root['actual_managed_exit_observed']is True and root['exit_code']==0 and root['all_owned_PID_PPID_PGID_absent']is True
 receipt=Path(root['retirement_path']);assert sha(receipt)==root['retirement_sha256']
 r=json.loads(receipt.read_text())
 assert r['status']=='qualification6_complete_all_guards_retired' and r['protocol_sha256']==c['qualification_protocol_sha256']
 assert r['all_guards_retired']is True and r['owned_group_retirement_verified']is True
 assert [(x['role'],x['seed'])for x in r['evaluations']]==q.ORDER
 got={};prices=[];q.guard.JUDGE=QUAL_JUDGE
 try:
  for item in r['evaluations']:
   role,seed=item['role'],item['seed'];x=q.current(qc,item,role,seed);got[(role,seed)]=x
   if role=='OFF':q.gate_OFF(got[('parent',seed)],x)
   elif role=='ON':prices.append(q.gate_ON(got[('OFF',seed)],x)['log_bill_ratio'])
 finally:q.guard.JUDGE=SCIENCE_JUDGE
 assert max(prices)==r['max_log_bill_ratio']==root['max_log_bill_ratio']
 return dict(path=str(rp),sha256=sha(rp),max_log_bill_ratio=max(prices))

def home(c,phase,shard):
 h=D/c['run_names'][phase][str(shard)]
 assert h.parent==D and h.name.startswith('run_lkc_science_')
 return h

def order(phase,shard):return [(role,seed)for seed in SEEDS[phase][shard::len(SHARDS)]for role in('parent','ON')]

def run(c,digest,phase,shard,gp):
 assert shard in SHARDS
 g=json.loads(gp.read_text());assert g['approved']is True and g['reviewed_by']=='root'
 assert g['purpose']==c['purpose'] and g['phase']==phase and g['protocol_sha256']==digest
 assert g['runner_sha256']==sha(__file__) and g['numerical_slots']==4 and g['shards']==list(SHARDS)
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
 q.write(h/'PLAN_v1.json',dict(protocol_sha256=digest,root_grant_path=str(gp.resolve()),root_grant_sha256=sha(gp),phase=phase,shard=shard,controller_pid=os.getpid(),fixed_order=sequence,qualification_acceptance=accepted,fresh_decision_sha256=g.get('fresh_decision_sha256'),expected_guards=[str(h/f'native_{seed}_{role}_guard.json')for role,seed in sequence]))
 proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);evaluations=[]
 try:
  for role,seed in sequence:
   q.guard.pins(proto)
   env=dict(os.environ,PB_ALL='1',STATS=','.join(q.STATS),LKC_MODE=str(c['candidates'][role]['mode']),PYTHONDONTWRITEBYTECODE='1')
   item=q.guard.run_native(h,proto,role,seed,env)
   q.current(c,item,role,seed) # only finite/profile/source/custody; no comparative science
   evaluations.append(dict(role=role,seed=seed,**item))
   q.write(h/f'COUNT_{len(evaluations):02d}_v1.json',dict(phase=phase,shard=shard,completed=len(evaluations),expected=8))
   print(json.dumps(dict(phase=phase,shard=shard,completed=len(evaluations),expected=8)),flush=True)
  q.write(h/'NATIVE_SHARD_READY_v1.json',dict(status='all8_calls_and_guards_complete_await_controller_retirement',protocol_sha256=digest,root_grant_sha256=sha(gp),phase=phase,shard=shard,controller_pid=os.getpid(),qualification_acceptance=accepted,fresh_decision_sha256=g.get('fresh_decision_sha256'),evaluations=evaluations,all_guards_retired=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,scientific_means_computed=False,automatic_followup=False))
 except BaseException as e:
  q.write(h/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,no_retry=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,slot_reservation_retained=not q.guard.OWNED_RETIREMENT_CONFIRMED))
  raise

def pool(c,digest,phase,rp):
 root=json.loads(rp.read_text())
 assert root['approved']is True and root['reviewed_by']=='root'
 assert root['status']=='four_controllers_exited0_and_all32_guards_retired'
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
  owners=[json.loads(x)for x in op.read_text().splitlines()if x.strip()]
  assert len(owners)==8

  assert r['status']=='all8_calls_and_guards_complete_await_controller_retirement'
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
  for e in r['evaluations']:got[(e['role'],e['seed'])]=q.current(c,e,e['role'],e['seed'])
 return got,dict(root_retirement_path=str(rp),root_retirement_sha256=sha(rp),result_guard_prediction_pins=pins,qualification_acceptance=ready[0]['qualification_acceptance'],fresh_decision_sha256=ready[0]['fresh_decision_sha256'])

def summarize(got,seeds):
 rows=[]
 for seed in seeds:
  base,arm=got[('parent',seed)],got[('ON',seed)]
  try:check=q.gate_ON(base,arm);valid=True;error=None
  except AssertionError as e:check=None;valid=False;error=repr(e)
  a,b=arm['row'],base['row'];mse=math.log(a['mse_final']/b['mse_final']);bill=math.log(a['flops']/b['flops'])
  rows.append(dict(seed=seed,implementation_gates_pass=valid,error=error,log_MSE_ratio=mse,log_bill_ratio=bill,log_net=mse+bill,implementation_check=check))
 mean=math.fsum(x['log_net']for x in rows)/len(rows)
 se=statistics.stdev(x['log_net']for x in rows)/math.sqrt(len(rows))
 passed=all(x['implementation_gates_pass'] and math.isfinite(x['log_net'])for x in rows)and mean<0
 return rows,mean,se,bool(passed)

def grade(c,digest,phase,rp):
 got,barrier=pool(c,digest,phase,rp);rows,mean,se,passed=summarize(got,SEEDS[phase])
 result=dict(protocol_sha256=digest,phase=phase,rows=rows,barrier=barrier,mean_log_net=mean,SE_log_net=se,strict_negative_pool=mean<0,passed=passed,unique_networks=16,direct_native_placement_hypothesis=True,no_followup_authorization=True)
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

def interrupted(signum,frame):raise KeyboardInterrupt('Owned lateKC science controller signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,old)
