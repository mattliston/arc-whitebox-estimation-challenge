"""Frozen RTP ON-only pilot8 with accepted d401 controls; four fixed shards; no followup authority.
One managed controller per shard owns one sequential guarded native slot.
Complete all four shards and actual root retirement before any phase means.
"""
import argparse,hashlib,importlib.util,json,math,os,signal,statistics
from pathlib import Path
D=Path(__file__).resolve().parent;O=D.parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
QP=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_common_u_compact32_v1/finite_uc32_interfaces_HELD_v1.py')
assert sha(QP)=='8bd0cb93f83ddb22223b8b8f1e8f1017dcb458661a0934637107437a732c15ae'
s=importlib.util.spec_from_file_location('private_rtp_native_science',QP);smoke=importlib.util.module_from_spec(s);s.loader.exec_module(smoke)
q=smoke.q
q.RESOURCE=dict(q.RESOURCE,logical_total_max=7)
SCIENCE_JUDGE=D/'native_rtp_science_judge_HELD_v1.py'
q.guard.JUDGE=SCIENCE_JUDGE
q.STATS=q.STATS+('_YR7_STATS','_YR7_ACTIVE','_YR7_BIND_STATS','_RTT_STATS','_RTT_ACTIVE','_RTT_CONTEXT')
def current(c,item,role,seed):
 assert role in ('parent','reference','ON')
 cached=role in ('parent','reference')and seed in c['identity_groups']['pilot']
 original=q.guard.JUDGE
 if cached:q.guard.JUDGE=Path(c['reference_control_judge']if role=='reference'else c['parent_control_judge'])
 try:x=q.current(dict(c,candidates={'ON':c['candidates'][role]}),item,seed)
 finally:q.guard.JUDGE=original
 a=x['row'];st=a['stats']
 assert st['_YR7_ACTIVE']==[False]and st['_YR7_BIND_STATS']==dict(keys=17,retired=True)
 ys=st['_YR7_STATS'];assert ys['retired']is True and ys['value_cache_entries']==0
 allowed={tuple(z)for z in c['shape_allowlist']}
 if role in ('ON','reference'):
  rs=st['_RTT_STATS'];assert rs['retired']is True
  rows=rs['reads'];assert [v['layer']for v in rows]==list(range(1,15))
  assert st['_RTT_ACTIVE']is False and st['_RTT_CONTEXT']==[None]
  widened=set()
  for v in rows:
   assert 0<v['m']<=1024 and v['k']>0 and v['k']%32==0 and 0<v['c15']<=v['c2']<=1024 and v['c15']%32==v['c2']%32==0
   assert type(v['changed'])is bool and v['changed']==(v['c15']<v['c2'])and type(v['forced_d7'])is bool
   assert v['old_rank']in(0,320,384)
   if v['changed']:
    assert v['basis_rank']in(0,1,2)and v['old_inner_calls']==int(v['old_rank']>0)
    if role=='ON':
     assert v['refine_rank_in']==v['basis_rank']and v['refine_rank_out']==v['basis_rank']
     assert type(v['refine_fallback'])is bool and v['refine_qr_rank']in(0,1,2)
     assert v['refine_old_rank']==v['old_rank']
     assert v['refine_fallback']==(v['refine_qr_rank']<v['refine_rank_in'])
    original=(v['m'],v['k'],v['c2'],v['old_rank'])
    expected=original in allowed or(v['old_rank']==0 and original[:3]in{(1024,2048,1024),(1024,4096,1024),(992,6144,1024)})
    assert v['forced_d7']==expected
    if expected:widened.add((v['m'],v['k'],v['c15'],v['old_rank']))
   else:
    assert v['basis_rank']is None and v['old_inner_calls']==0 and not v['forced_d7']
    if role=='ON':assert all(v[k]is None for k in('refine_rank_in','refine_rank_out','refine_qr_rank','refine_fallback','refine_old_rank'))
  assert all(tuple(v)in allowed|widened for v in ys['routed_shapes'])
 else:
  assert st.get('_RTT_STATS')is None and st.get('_RTT_ACTIVE')is None and st.get('_RTT_CONTEXT')is None
  assert all(tuple(v)in allowed for v in ys['routed_shapes'])

 return x

def gate_ON(base,arm):
 q.setup_equal(base,arm)
 assert base['cert']['python_version']==arm['cert']['python_version']
 changed=[v['layer']for v in arm['row']['stats']['_RTT_STATS']['reads']if v['changed']]
 first=min(changed)if changed else 16
 assert base['pred'][:first].tobytes()==arm['pred'][:first].tobytes()
 b,a=base['row'],arm['row'];assert a['mse_layers'][:first]==b['mse_layers'][:first]
 logbill=math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
 return dict(first_affected_read_layer=first,unchanged_prefix_exact=True,setup_identical=True,rtp_fixed_scope=True,log_bill_ratio=logbill,ops_delta=a['ops']-b['ops'],no_accuracy_or_price_sign_gate=True)


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
 assert c['status']=='REGISTERED'and c['purpose']=='rtp_pilot8_fresh16_oos8'
 SEEDS=validated_seeds(c)
 assert c['shards']==list(SHARDS)and c['native_calls_max']==8 and len(c['native_control_reuse'])==8
 assert c['source_pins'][str(Path(__file__).resolve())]==sha(__file__)
 assert c['resource_policy']==q.RESOURCE and c['admission_wait_policy']==q.WAIT
 assert c['candidates']['parent']['sha256']=='d401532639bdc3d1b294518828f4af03a873727e510e0d4ec91399b1fc169d5a'
 assert c['candidates']['ON']['sha256']=='b10fc51d9055082567e43e80c413679726b07f1cc39182cba2e84816f0f6c010'
 assert c['candidates']['reference']['sha256']=='000e82aa32de4a4e09aba78274efbe65b9332d2a183e88f353e0bbc4be630776'
 assert all(x['mode']==1 for x in c['candidates'].values())
 q.guard.pins(dict(pinned_files=c['source_pins'],candidates=c['candidates']))
 c['_protocol_path']=str(Path(path).resolve());c['_protocol_sha256']=digest
 allocation(c);control_metadata(c);reference_metadata(c)
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

def reference_metadata(c):
 p=Path(c['reference_controls_acceptance_path']);assert sha(p)==c['reference_controls_acceptance_sha256']
 r=json.loads(p.read_text());assert r['approved']is True and r['reviewed_by']=='root'
 assert r['status']=='four_controllers_exited0_and_all_expected_guards_retired'and r['phase']=='pilot'
 assert r['candidate_sha256']==c['candidates']['reference']['sha256']
 assert r['all_owned_PID_PPID_PGID_absent']is True and r['all_guards_clean']is True and r['all16_finite']is True
 assert r['completed_candidate_calls']==8 and r['completed_parent_calls']==0
 assert r['protocol_sha256']==c['reference_controls_protocol_sha256']
 assert sha(c['reference_controls_protocol_path'])==c['reference_controls_protocol_sha256']
 assert [v['shard']for v in r['shards']]==list(SHARDS)
 rows=[]
 for sh in r['shards']:
  assert sh['actual_managed_exit_observed']is True and sh['exit_code']==0
  ready=Path(sh['ready_path']);assert sha(ready)==sh['ready_sha256']
  rows.extend(json.loads(ready.read_text())['evaluations'])
 rows.sort(key=lambda e:e['seed'])
 assert rows==c['reference_control_reuse']and [e['seed']for e in rows]==c['identity_groups']['pilot']
 assert all(e['role']=='ON'for e in rows)
 for path,digest in r['evidence_sha256'].items():assert sha(path)==digest
 return r

def qualification(c,g):
 rp=Path(g['qualification_root_acceptance_path']);assert sha(rp)==g['qualification_root_acceptance_sha256']==c['qualification_root_retirement_sha256']
 assert rp==Path(c['qualification_root_retirement_path'])
 r=json.loads(rp.read_text());assert r['approved']is True and r['reviewed_by']=='root'
 assert r['candidate_sha256']==c['candidates']['ON']['sha256']and r['scope']=='prefix_power_source_tiny_no_native_smoke'
 assert r['fixed_proposal_sha256']==sha(D/'FIXED_PROPOSAL_astra_v1.json')
 assert r['tiny_sha256']==sha(D/'INTEGRATED_TINY_AND_IMPORT_astra_v2.json')
 assert g['parent_controls_reuse_approved']is True and g['reference_controls_reuse_approved']is True
 assert g['parent_controls_acceptance_sha256']==c['parent_controls_acceptance_sha256']
 assert g['reference_controls_acceptance_sha256']==c['reference_controls_acceptance_sha256']
 assert g['post_submission_slot_handoff']is True
 reference_metadata(c)
 return dict(path=str(rp),sha256=sha(rp))

def science_current(c,item,role,seed):
 x=current(c,item,role,seed)
 cached=role in ('parent','reference')and seed in c['identity_groups']['pilot']
 assert x['cert']['science_protocol_sha256']==((c['reference_controls_protocol_sha256']if role=='reference'else c['parent_controls_protocol_sha256'])if cached else c['_protocol_sha256'])
 assert x['cert']['science_protocol_path']==((c['reference_controls_protocol_path']if role=='reference'else c['parent_controls_protocol_path'])if cached else c['_protocol_path'])
 return x

def home(c,phase,shard):
 h=D/c['run_names'][phase][str(shard)]
 assert h.parent==D and h.name.startswith('run_rtp_science_')
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
 assert shard in SHARDS and phase=='pilot'
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
 if phase=='pilot':
  reference_metadata(c)
  for e in c['reference_control_reuse']:
   got[('reference',e['seed'])]=science_current(c,e,'reference',e['seed'])
   for kind in('certificate','guard'):pins[e[kind]]=e[kind+'_sha256']
   pp=Path(e['certificate']);pins[str(pp.with_name(pp.stem+'_pred.npz'))]=e['prediction_npz_sha256']
 return got,dict(root_retirement_path=str(rp),root_retirement_sha256=sha(rp),result_guard_prediction_pins=pins,qualification_acceptance=ready[0]['qualification_acceptance'],fresh_decision_sha256=ready[0]['fresh_decision_sha256'])

def summarize(got,seeds):
 rows=[]
 for seed in seeds:
  parent,reference,arm=got[('parent',seed)],got[('reference',seed)],got[('ON',seed)]
  checks={};valid=True;error=None
  try:
   checks['absolute']=gate_ON(parent,arm);checks['incremental']=gate_ON(reference,arm)
  except AssertionError as e:valid=False;error=repr(e)
  a=arm['row'];comparisons={}
  for name,base in(('absolute',parent),('incremental',reference)):
   b=base['row'];mse=math.log(a['mse_final']/b['mse_final']);bill=math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
   comparisons[name]=dict(log_MSE_ratio=mse,log_bill_ratio=bill,log_net=mse+bill)
  rows.append(dict(seed=seed,implementation_gates_pass=valid,error=error,implementation_checks=checks,**comparisons))
 means={k:math.fsum(x[k]['log_net']for x in rows)/len(rows)for k in('absolute','incremental')}
 ses={k:statistics.stdev(x[k]['log_net']for x in rows)/math.sqrt(len(rows))for k in means}
 passed=all(x['implementation_gates_pass']and all(math.isfinite(x[k]['log_net'])for k in means)for x in rows)and all(v<0 for v in means.values())
 return rows,means,ses,bool(passed)

def grade(c,digest,phase,rp):
 assert phase=='pilot'
 got,barrier=pool(c,digest,phase,rp)
 rows,means,ses,passed=summarize(got,SEEDS['pilot'])
 result=dict(protocol_sha256=digest,phase=phase,rows=rows,barrier=barrier,mean_log_net=means,SE_log_net=ses,
             absolute_vs_d401_negative=means['absolute']<0,incremental_vs_RTT2_negative=means['incremental']<0,
             passed=passed,unique_networks=8,no_followup_authorization=True)
 q.write(D/c['decision_names'][phase],result)
 print(json.dumps(dict(phase=phase,passed=passed,mean_log_net=means)),flush=True)


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=('run','grade'));p.add_argument('--phase',choices=('pilot',),required=True)
 p.add_argument('--shard',type=int,choices=SHARDS);p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha256',required=True)
 p.add_argument('--run-authorized',action='store_true');p.add_argument('--grant',type=Path);p.add_argument('--root-retirement',type=Path)
 a=p.parse_args();c=load_config(a.protocol,a.protocol_sha256)
 if a.mode=='run':assert a.run_authorized and a.grant and a.shard is not None;run(c,a.protocol_sha256,a.phase,a.shard,a.grant)
 else:assert a.root_retirement;grade(c,a.protocol_sha256,a.phase,a.root_retirement)

def interrupted(signum,frame):raise KeyboardInterrupt('Owned RTP science controller signal%s'%signum)
if __name__=='__main__':
 old=signal.signal(signal.SIGTERM,interrupted)
 try:main()
 finally:signal.signal(signal.SIGTERM,old)
