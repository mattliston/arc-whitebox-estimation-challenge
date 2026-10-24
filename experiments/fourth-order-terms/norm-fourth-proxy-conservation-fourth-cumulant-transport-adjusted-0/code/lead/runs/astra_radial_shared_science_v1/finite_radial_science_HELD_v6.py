"""HELD native C/Q4/CENTRAL fixed shared-control science, no automatic followup.

Each shard is one managed sequential owned slot. All phase owners must retire
before any comparative science. Enrollment is fixed before the first call.
"""
import argparse,hashlib,importlib.util,json,math,os,signal,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
QP=O/'lead/runs/astra_raw_norm_fourth_v1/finite_q4_qualification_UNVALIDATED_v1.py'
assert sha(QP)=='cc57094422ef9d19c70998c6abe8808dda85ad45f1f83e5a51b2daac82752fed'
spec=importlib.util.spec_from_file_location('private_radial_guard',QP);q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)
CP=O/'scout_card/central_fourth_normalization_held_astra_v1/finite_cfn_qualification_UNVALIDATED_v3.py'
assert sha(CP)=='b57c31ff5e11ebaf750015e1b98a8a69d85a3e3dec917f0fe314c25c3c4245d8'
cs=importlib.util.spec_from_file_location('private_radial_cfn_gates',CP);central=importlib.util.module_from_spec(cs);cs.loader.exec_module(central)
JUDGE=D/'native_radial_science_judge_HELD_v1.py';q.guard.JUDGE=JUDGE
SEEDS={'fresh':list(range(7101704,7101720)),'oos':list(range(7101720,7101736))}
ARMS=('Q4','CENTRAL');SHARDS=(0,1,2)
STATS=q.STATS+('_ASTRA_CFN_STATS','_ASTRA_CFN_ACTIVE','_ASTRA_CFN_CONTEXT')
write=q.write;prediction=q.prediction

def config(path,digest):
 assert sha(path)==digest;c=read(path)
 assert c['status']=='REGISTERED' and c['purpose']=='native_radial_shared_alternatives32'
 assert c['arms']==list(ARMS) and c['phase_seeds']==SEEDS and c['shards']==list(SHARDS)
 assert c['native_calls_max']==96 and c['native_control_reuse']==[]
 assert c['admission_count_limit']==8
 assert c['numerical_slots']==c['max_concurrent_shards']==2 and c['staged_launch_order']==[[0,1],[2]]
 assert c['stage_release_names']=={phase:'ROOT_'+phase.upper()+'_STAGE2_RELEASE_v1.json'for phase in SEEDS}
 assert c['science_net']=='paired_actual_logMSE_plus_logbill' and c['strict_sign_gates']==['fresh','oos','all32']
 assert c['exact_tie_preference']=='Q4' and c['alternatives_not_additive']is True
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 assert c['candidates']['C']==dict(path=str(O/'scout_native/k4liab_s51k2tw/candidate.py'),sha256='6a145fadb3224501e683e95424bd9ada0ff571041850c7d482969b742d8f8f41')
 assert set(c['candidates'])=={'C',*ARMS}
 q.guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 return c

def allocation(c):
 p=Path(c['allocation']['path']);assert sha(p)==c['allocation']['sha256'];a=read(p)
 assert a['approved']is True and a['reviewed_by']=='root'
 assert a['purpose']=='native_radial_shared_alternatives32' and a['phase_seeds']==SEEDS and a['arms']==list(ARMS)
 assert a['no_evaluated_seed_reclassified']is True
 assert a['truth_metadata']==c['truth_metadata']
 assert [t['seed']for t in c['truth_metadata']]==SEEDS['fresh']+SEEDS['oos']
 for t in c['truth_metadata']:assert t['N']>=15990784 and sha(t['path'])==t['sha256']
 delta=c['allocation_delta_clearance'];assert sha(delta['path'])==delta['sha256'];d=read(delta['path'])
 assert d['approved']is True and d['reviewed_by']=='root' and d['no_collision']is True
 assert d['allocation_sha256']==c['allocation']['sha256']

def qualifications(c,g):
 out={}
 for arm in ARMS:
  binding=c['qualifications'][arm];accept=g['qualification_acceptances'][arm]
  assert sha(accept['path'])==accept['sha256'];r=read(accept['path'])
  assert r['approved']is True and r['reviewed_by']=='root'
  assert r['status']=='root_qualification6_price_accepted' and r['protocol_sha256']==binding['protocol_sha256']
  assert r['actual_managed_exit_observed']is True and r['exit_code']==0 and r['all_owned_PID_PPID_PGID_absent']is True
  assert sha(binding['helper_path'])==binding['helper_sha256'] and sha(binding['protocol_path'])==binding['protocol_sha256']
  s=importlib.util.spec_from_file_location('private_radial_qualification_'+arm,binding['helper_path']);h=importlib.util.module_from_spec(s);s.loader.exec_module(h)
  qc=read(binding['protocol_path']);h.guard.pins(dict(pinned_files=qc['source_pins'],candidates=qc['candidates']))
  assert {k:qc['candidates']['parent'][k]for k in('path','sha256')}==c['candidates']['C']
  assert {k:qc['candidates']['ON'][k]for k in('path','sha256')}==c['candidates'][arm]
  assert sha(r['retirement_path'])==r['retirement_sha256'];ret=read(r['retirement_path'])
  assert ret['status']=='qualification6_complete_all_guards_retired' and ret['protocol_sha256']==binding['protocol_sha256']
  assert ret['all_guards_retired']is True and ret['owned_group_retirement_verified']is True
  assert [(e['role'],e['seed'])for e in ret['evaluations']]==h.ORDER
  got={};cost=[]
  for e in ret['evaluations']:
   role,seed=e['role'],e['seed'];z=h.current(qc,e,role,seed);got[role,seed]=z
   if role=='OFF':h.gate_OFF(got['parent',seed],z)
   if role=='ON':cost.append(h.gate_ON(got['OFF',seed],z)['log_bill_ratio'])
  assert max(cost)==ret['max_log_bill_ratio']==r['max_log_bill_ratio']
  out[arm]=dict(**accept,max_log_bill_ratio=max(cost))
 return out

def contexts(st,role):
 q.contexts(st,'ON'if role=='Q4'else'parent')
 if role=='CENTRAL':
  assert st['_ASTRA_CFN_ACTIVE']==[False] and st['_ASTRA_CFN_CONTEXT']==[None]
  assert isinstance(st['_ASTRA_CFN_STATS'],dict) and st['_ASTRA_CFN_STATS']['retired']is True
 else:assert all(st[k]is None for k in('_ASTRA_CFN_STATS','_ASTRA_CFN_ACTIVE','_ASTRA_CFN_CONTEXT'))

def current(c,item,role,seed):
 import numpy as np
 for key in('certificate','guard'):assert sha(item[key])==item[key+'_sha256']
 p=Path(item['certificate']);assert sha(p.with_name(p.stem+'_pred.npz'))==item['prediction_npz_sha256']
 d=read(p);g=read(item['guard']);source=c['candidates'][role]
 assert d['candidate']==source['path'] and d['sha256']==source['sha256'] and d['radial_arm']==role
 assert d['judge_sha256']==c['source_pins'][str(JUDGE)]
 assert d['loader_sha256']==c['source_pins'][str(O/'analyst/fresh/loader.py')]
 assert d['environment']=={**{k:'1'for k in q.guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':source['sha256'],'RADIAL_ARM':role}
 assert g['command']==[str(q.guard.RT),'-B',str(JUDGE),source['path'],str(seed),str(p)]
 assert g['completed']is True and g['exit_code']==0 and g['error']is None
 assert g['whole_group_retired']is True and g['remaining_rows']==[]
 assert 0<g['max_rss_bytes']<7500000000 and 0<=g['max_footprint_bytes']<7500000000
 assert d['setup']['pb_wall_s']==1e9 and math.isfinite(d['setup']['wall_s'])and d['setup']['wall_s']>=0
 y=prediction(p,seed);assert d['seeds']==[seed] and len(d['rows'])==1
 row=d['rows'][0];assert row['seed']==seed
 assert type(row['flops'])is int and .1*2**41<=row['flops']<=2**41
 assert type(row['ops'])is int and row['ops']>0
 assert len(row['mse_layers'])==16 and all(math.isfinite(v)and v>0 for v in row['mse_layers'])
 assert row['mse_final']==row['mse_layers'][-1] and hashlib.sha256(y.tobytes()).hexdigest()[:16]==row['pred_sha']
 for name in('_ASTRA_COMPUTE_STATS','_ASTRA_UBOWN_STATS','_ASTRA_CLUSTER_STATS','_ASTRA_BCC_STATS','_ASTRA_QL_STATS'):assert row['stats'][name]is None
 assert row['radial_cleanup']==dict(post_cov32_absent=True,q4_active=False,central_active=False,central_context_clear=True)
 contexts(row['stats'],role)
 truth=next(t for t in c['truth_metadata']if t['seed']==seed)
 assert row['weights_before_sha256']==row['weights_after_sha256']==truth['weights_sha256']
 return dict(cert=d,row=row,pred=y,path=str(p),evidence=item)

def gate(base,arm,role):
 assert role in ARMS
 return (q if role=='Q4'else central).gate_ON(base,arm)

def home(c,phase,shard):
 p=D/c['run_names'][phase][str(shard)];assert p.parent==D and p.name.startswith('run_radial_');return p

def order(phase,shard,arms):return [(role,seed)for seed in SEEDS[phase][shard::len(SHARDS)]for role in('C',*arms)]

def fresh_decision(c,digest,binding):
 p=D/c['decision_names']['fresh'];assert sha(p)==binding['sha256'] and str(p)==binding['path'];f=read(p)
 assert f['protocol_sha256']==digest and f['phase']=='fresh'
 got,_=pool(c,digest,'fresh',Path(f['barrier']['root_retirement_path']))
 arms=summarize(got,SEEDS['fresh'],ARMS)
 assert f['arms']==arms and f['survivors']==[a for a in ARMS if arms[a]['passed']]
 assert f['survivors'];return f

def admit(h,role,seed):
 state=q.guard.resources()
 q.guard.event(h/'admission.jsonl',dict(role=role,seed=seed,limit=8,**state))
 assert state['free_percent']>=60 and state['counted']<8,'Prospective admission requires free>=60 and fewer than8 counted workers before new native call'

def stage_release(c,digest,phase,shard,g,gp):
 assert g['numerical_slots']==g['max_concurrent_shards']==2 and g['staged_launch_order']==[[0,1],[2]]
 rp=D/c['stage_release_names'][phase]
 assert g['stage_release_path']==str(rp)
 if shard in(0,1):return None
 assert shard==2
 r=read(rp)
 assert r['approved']is True and r['reviewed_by']=='root'
 assert r['status']=='one_initial_shard_exited0_and_owned_groups_retired'
 assert r['protocol_sha256']==digest and r['phase']==phase
 assert r['root_grant_sha256']==sha(gp)and r['released_for_shard']==2
 assert r['initial_shards_launched']==[0,1] and r['max_concurrent_shards']==2
 assert r['actual_managed_exit_observed']is True and r['exit_code']==0
 assert r['all_owned_PID_PPID_PGID_absent']is True and r['slot_reassignment_approved']is True
 prior=r['retired_shard'];assert prior in(0,1)
 for initial in(0,1):
  ip=read(home(c,phase,initial)/'PLAN_v1.json')
  assert ip['protocol_sha256']==digest and ip['phase']==phase and ip['shard']==initial and ip['root_grant_sha256']==sha(gp)
 h=home(c,phase,prior);pp=h/'PLAN_v1.json';op=h/'ownership.jsonl';rp0=h/'NATIVE_SHARD_READY_v1.json'
 assert r['plan_sha256']==sha(pp)and r['ownership_sha256']==sha(op)
 assert r['ready_path']==str(rp0)and r['ready_sha256']==sha(rp0)
 plan=read(pp);ready=read(rp0)
 assert not(h/'STOPPED_v1.json').exists()
 assert ready['status']=='all_fixed_calls_and_guards_complete_await_controller_retirement'
 for key,val in(('protocol_sha256',digest),('phase',phase),('shard',prior),('active_arms',g['active_arms']),('root_grant_sha256',sha(gp))):
  assert plan[key]==ready[key]==val
 assert r['controller_pid']==plan['controller_pid']==ready['controller_pid']
 assert ready['all_guards_retired']is True and ready['owned_group_retirement_verified']is True
 sequence=order(phase,prior,g['active_arms'])
 assert [(e['role'],e['seed'])for e in ready['evaluations']]==sequence and plan['fixed_order']==[list(x)for x in sequence]
 assert plan['expected_guards']==[e['guard']for e in ready['evaluations']]
 assert sorted(h.glob('native_*_guard.json'))==sorted(Path(e['guard'])for e in ready['evaluations'])
 for e in ready['evaluations']:
  assert sha(e['guard'])==e['guard_sha256'];record=read(e['guard'])
  assert record['completed']is True and record['exit_code']==0 and record['error']is None
  assert record['whole_group_retired']is True and record['remaining_rows']==[]
 return dict(path=str(rp),sha256=sha(rp))

def run(c,digest,phase,shard,gp):
 g=read(gp);assert g['approved']is True and g['reviewed_by']=='root'
 assert g['purpose']==c['purpose']and g['phase']==phase and g['protocol_sha256']==digest
 assert g['admission_count_limit']==8
 assert g['runner_sha256']==sha(__file__)and g['numerical_slots']==2and g['shards']==list(SHARDS)
 assert g['run_names']==c['run_names'][phase]and all(os.environ.get(k)=='1'for k in q.guard.THREADS)
 assert os.getpriority(os.PRIO_PROCESS,0)==0
 allocation(c);accepted=qualifications(c,g)
 arms=list(ARMS)if phase=='fresh'else fresh_decision(c,digest,g['fresh_decision'])['survivors']
 assert g['active_arms']==arms
 staged=stage_release(c,digest,phase,shard,g,gp)
 h=home(c,phase,shard);h.mkdir(exist_ok=False);sequence=order(phase,shard,arms)
 write(h/'PLAN_v1.json',dict(protocol_sha256=digest,root_grant_path=str(gp.resolve()),root_grant_sha256=sha(gp),phase=phase,shard=shard,controller_pid=os.getpid(),active_arms=arms,stage_release=staged,fixed_order=sequence,qualification_acceptances=accepted,fresh_decision=g.get('fresh_decision'),expected_guards=[str(h/f'native_{seed}_{role}_guard.json')for role,seed in sequence]))
 proto=dict(pinned_files=c['source_pins'],candidates=c['candidates']);evaluations=[]
 try:
  for role,seed in sequence:
   q.guard.pins(proto)
   env=dict(os.environ,PB_ALL='1',STATS=','.join(STATS),RADIAL_ARM=role,PYTHONDONTWRITEBYTECODE='1')
   admit(h,role,seed)
   item=q.guard.run_native(h,proto,role,seed,env);current(c,item,role,seed)
   evaluations.append(dict(role=role,seed=seed,**item))
   write(h/f'COUNT_{len(evaluations):02d}_v1.json',dict(phase=phase,shard=shard,completed=len(evaluations),expected=len(sequence)))
   print(json.dumps(dict(phase=phase,shard=shard,completed=len(evaluations),expected=len(sequence))),flush=True)
  write(h/'NATIVE_SHARD_READY_v1.json',dict(status='all_fixed_calls_and_guards_complete_await_controller_retirement',protocol_sha256=digest,root_grant_sha256=sha(gp),phase=phase,shard=shard,controller_pid=os.getpid(),active_arms=arms,stage_release=staged,qualification_acceptances=accepted,fresh_decision=g.get('fresh_decision'),evaluations=evaluations,all_guards_retired=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,scientific_means_computed=False,automatic_followup=False,admission_sha256=sha(h/'admission.jsonl')))
 except BaseException as e:
  write(h/'STOPPED_v1.json',dict(error=repr(e),evaluations=evaluations,no_retry=True,owned_group_retirement_verified=q.guard.OWNED_RETIREMENT_CONFIRMED,slot_reservation_retained=not q.guard.OWNED_RETIREMENT_CONFIRMED));raise

def pool(c,digest,phase,rp):
 root=read(rp)
 assert root['approved']is True and root['reviewed_by']=='root'
 assert root['status']=='all_fixed_controllers_exited0_and_guards_retired'
 assert root['protocol_sha256']==digest and root['phase']==phase and root['all_owned_PID_PPID_PGID_absent']is True
 assert [x['shard']for x in root['shards']]==list(SHARDS)and len({x['controller_pid']for x in root['shards']})==len(SHARDS)
 ready=[];pins={};arms=root['active_arms']
 assert (arms==list(ARMS)if phase=='fresh'else bool(arms)and arms==[a for a in ARMS if a in arms])
 for shard,owner in zip(SHARDS,root['shards']):
  h=home(c,phase,shard);p=h/'NATIVE_SHARD_READY_v1.json';pp=h/'PLAN_v1.json';op=h/'ownership.jsonl'
  assert owner['ready_path']==str(p)and owner['ready_sha256']==sha(p)
  assert owner['actual_managed_exit_observed']is True and owner['exit_code']==0
  assert owner['plan_sha256']==sha(pp)and owner['ownership_sha256']==sha(op)
  r=read(p);plan=read(pp)
  assert not(h/'STOPPED_v1.json').exists()
  assert r['status']=='all_fixed_calls_and_guards_complete_await_controller_retirement'
  for key,val in(('protocol_sha256',digest),('phase',phase),('shard',shard),('active_arms',arms)):
   assert r[key]==plan[key]==val
  assert r['controller_pid']==plan['controller_pid']==owner['controller_pid']
  assert r['all_guards_retired']is True and r['owned_group_retirement_verified']is True
  assert r['root_grant_sha256']==plan['root_grant_sha256']and r['qualification_acceptances']==plan['qualification_acceptances']and r['fresh_decision']==plan['fresh_decision']
  gp=Path(plan['root_grant_path']);assert sha(gp)==plan['root_grant_sha256'];g=read(gp)
  assert g['approved']is True and g['reviewed_by']=='root'and g['purpose']==c['purpose']
  assert g['protocol_sha256']==digest and g['runner_sha256']==sha(__file__)and g['phase']==phase
  assert g['admission_count_limit']==8
  assert g['numerical_slots']==2and g['shards']==list(SHARDS)and g['active_arms']==arms and g['run_names']==c['run_names'][phase]
  assert g.get('fresh_decision')==r['fresh_decision']
  assert r['stage_release']==plan['stage_release']==stage_release(c,digest,phase,shard,g,gp)
  for arm in ARMS:
   a=r['qualification_acceptances'][arm];assert {k:a[k]for k in('path','sha256')}==g['qualification_acceptances'][arm]and sha(a['path'])==a['sha256']
  sequence=order(phase,shard,arms)
  ap=h/'admission.jsonl';assert sha(ap)==r['admission_sha256']
  admissions=[json.loads(x)for x in ap.read_text().splitlines()if x.strip()]
  assert [(x['role'],x['seed'])for x in admissions]==sequence
  assert all(x['limit']==8 and x['free_percent']>=60 and x['counted']<8 for x in admissions)
  assert [(e['role'],e['seed'])for e in r['evaluations']]==sequence and plan['fixed_order']==[list(x)for x in sequence]
  assert plan['expected_guards']==[e['guard']for e in r['evaluations']]
  assert sorted(h.glob('native_*_guard.json'))==sorted(Path(e['guard'])for e in r['evaluations'])
  owners=[json.loads(x)for x in op.read_text().splitlines()if x.strip()];assert len(owners)==len(sequence)
  for own,e in zip(owners,r['evaluations']):
   role,seed=e['role'],e['seed'];source=c['candidates'][role]
   assert own['controller_pid']==r['controller_pid']and own['controller_nice']==0 and own['guard_native_nice']==10
   assert own['kind']==role and own['seed']==seed and own['guard_record']==e['guard']and type(own['owned_guard_pid'])is int
   assert own['candidate_sha256']==source['sha256']
   assert own['environment']=={**{k:'1'for k in q.guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':source['sha256']}
   assert own['command']==[str(q.guard.PY),'-B',str(q.guard.GUARD),'--record',e['guard'],'--seconds','900','--footprint-limit','7500000000','--rss-backstop','7500000000',str(q.guard.RT),'-B',str(JUDGE),source['path'],str(seed),e['certificate']]
   for key in('certificate','guard'):assert sha(e[key])==e[key+'_sha256'];pins[e[key]]=e[key+'_sha256']
   guard_record=read(e['guard'])
   assert guard_record['completed']is True and guard_record['exit_code']==0 and guard_record['error']is None
   assert guard_record['whole_group_retired']is True and guard_record['remaining_rows']==[]
   assert 0<guard_record['max_rss_bytes']<7500000000 and 0<=guard_record['max_footprint_bytes']<7500000000
   cp=Path(e['certificate']);yp=cp.with_name(cp.stem+'_pred.npz');assert sha(yp)==e['prediction_npz_sha256'];pins[str(yp)]=e['prediction_npz_sha256']
  ready.append(r)
 assert all(r['root_grant_sha256']==ready[0]['root_grant_sha256']and r['qualification_acceptances']==ready[0]['qualification_acceptances']and r['fresh_decision']==ready[0]['fresh_decision']for r in ready)
 # All fixed owners/guards/artifact hashes verified before any science records.
 got={}
 for r in ready:
  for e in r['evaluations']:got[e['role'],e['seed']]=current(c,e,e['role'],e['seed'])
 return got,dict(root_retirement_path=str(rp),root_retirement_sha256=sha(rp),result_guard_prediction_pins=pins,active_arms=arms,qualification_acceptances=ready[0]['qualification_acceptances'],fresh_decision=ready[0]['fresh_decision'])

def summarize(got,seeds,arms):
 result={}
 for arm in arms:
  rows=[]
  for seed in seeds:
   base,z=got['C',seed],got[arm,seed]
   try:check=gate(base,z,arm);valid=True;error=None
   except AssertionError as e:check=None;valid=False;error=repr(e)
   mse=math.log(z['row']['mse_final']/base['row']['mse_final']);bill=math.log(z['row']['flops']/base['row']['flops'])
   rows.append(dict(seed=seed,implementation_gates_pass=valid,error=error,log_MSE_ratio=mse,log_bill_ratio=bill,log_net=mse+bill,implementation_check=check))
  mean=math.fsum(x['log_net']for x in rows)/len(rows)
  result[arm]=dict(rows=rows,mean_log_net=mean,SE_log_net=statistics.stdev(x['log_net']for x in rows)/math.sqrt(len(rows)),passed=bool(mean<0 and all(x['implementation_gates_pass']and math.isfinite(x['log_net'])for x in rows)))
 return result

def grade(c,digest,phase,rp):
 allocation(c)
 if phase=='oos':
  root=read(rp);assert root['phase']=='oos'and root['protocol_sha256']==digest
  f=fresh_decision(c,digest,root['fresh_decision']);assert root['active_arms']==f['survivors']
 got,barrier=pool(c,digest,phase,rp)
 if phase=='oos':assert barrier['fresh_decision']==root['fresh_decision']and barrier['active_arms']==f['survivors']
 arms=summarize(got,SEEDS[phase],barrier['active_arms'])
 result=dict(protocol_sha256=digest,phase=phase,arms=arms,barrier=barrier,survivors=[a for a in ARMS if a in arms and arms[a]['passed']],alternatives_not_additive=True,no_followup_authorization=True)
 if phase=='oos':
  f=fresh_decision(c,digest,barrier['fresh_decision']);assert barrier['active_arms']==f['survivors']
  eligible=[]
  for arm in barrier['active_arms']:
   rows=f['arms'][arm]['rows']+arms[arm]['rows'];mean=math.fsum(x['log_net']for x in rows)/32
   arms[arm].update(all32_mean_log_net=mean,all32_SE_log_net=statistics.stdev(x['log_net']for x in rows)/math.sqrt(32),passed=bool(arms[arm]['passed']and f['arms'][arm]['passed']and mean<0))
   if arms[arm]['passed']:eligible.append(arm)
  result['survivors']=eligible
  result['selected_alternative']=min(eligible,key=lambda a:(arms[a]['all32_mean_log_net'],ARMS.index(a)))if eligible else None
  result['production_qualified']=False
 write(D/c['decision_names'][phase],result)
 print(json.dumps(dict(phase=phase,survivors=result['survivors'],selected_alternative=result.get('selected_alternative'))))

def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','grade'));p.add_argument('--phase',choices=SEEDS,required=True)
 p.add_argument('--shard',type=int,choices=SHARDS);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
 p.add_argument('--run-authorized',action='store_true');p.add_argument('--grant',type=Path);p.add_argument('--root-retirement',type=Path)
 a=p.parse_args();c=config(a.protocol,a.protocol_sha256)
 if a.mode=='run':assert a.run_authorized and a.grant and a.shard is not None;run(c,a.protocol_sha256,a.phase,a.shard,a.grant)
 else:assert a.root_retirement;grade(c,a.protocol_sha256,a.phase,a.root_retirement)

def interrupted(signum,frame):raise KeyboardInterrupt('Owned shared radial controller signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,old)
