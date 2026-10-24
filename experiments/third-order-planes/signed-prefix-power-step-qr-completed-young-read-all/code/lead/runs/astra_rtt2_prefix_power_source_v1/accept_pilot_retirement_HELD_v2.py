import datetime,importlib.util,json,subprocess,sys
from pathlib import Path
D=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1/lead/runs/astra_rtp_prefix_power_source_v1')
sp=importlib.util.spec_from_file_location('_rtp_retirement',D/'finite_rtp_science_HELD_v2.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
phase=sys.argv[1];assert phase in('pilot','remaining');stage=phase.upper()
pp=D/'SCIENCE_PROTOCOL_REGISTERED_v1.json';digest=m.sha(pp);c=m.load_config(pp,digest)
read=lambda p:json.loads(Path(p).read_text())
exits=read(D/f'SCIENCE_{stage}_ACTUAL_EXITS_v1.json')['workers']
managed=read(D/f'SCIENCE_{stage}_MANAGED_SESSIONS_v1.json')['workers']
assert [x['shard']for x in exits]==list(m.SHARDS)and all(x['exit_code']==0 and x['actual_managed_exit_observed']is True for x in exits)
pins={str(p):m.sha(p)for p in [pp,D/f'ROOT_SCIENCE_{stage}_GRANT_v1.json',D/f'SCIENCE_{stage}_ACTUAL_EXITS_v1.json',D/f'SCIENCE_{stage}_MANAGED_SESSIONS_v1.json',D/f'SCIENCE_{stage}_PROCESS_SNAPSHOTS_v1.jsonl']}
shards=[];ids=set();groups=set();results=[]
for i,(e,handle)in enumerate(zip(exits,managed)):
 assert e['shard']==handle['shard']==i and e['session_id']==handle['session_id']
 h=m.home(c,phase,i);rp=h/'NATIVE_SHARD_READY_v1.json';r=read(rp);plan=read(h/'PLAN_v1.json');owners=[json.loads(x)for x in(h/'ownership.jsonl').read_text().splitlines()]
 assert r['controller_pid']==plan['controller_pid']==handle['controller_pid']and r['phase']==phase and r['shard']==i
 assert r['protocol_sha256']==plan['protocol_sha256']==digest
 assert r['all_guards_retired']is True and r['owned_group_retirement_verified']is True
 assert [(x['role'],x['seed'])for x in r['evaluations']]==m.order(phase,i)and len(owners)==len(m.order(phase,i))
 assert plan['expected_guards']==[x['guard']for x in r['evaluations']]
 assert plan['fixed_order']==[list(x)for x in m.order(phase,i)]
 assert plan['root_grant_sha256']==r['root_grant_sha256']==m.sha(D/f'ROOT_SCIENCE_{stage}_GRANT_v1.json')
 admission=m.admissions(h,m.order(phase,i),r['controller_pid']);assert admission==r['admission_evidence']
 ids.add(r['controller_pid']);groups.add(r['controller_pid'])
 for own,item in zip(owners,r['evaluations']):
  guard=read(item['guard']);seed=item['seed'];role=item['role'];src=c['candidates'][role]
  assert own['controller_pid']==r['controller_pid']and own['controller_nice']==0 and own['guard_native_nice']==10
  assert own['candidate_sha256']==src['sha256']and own['kind']==role and own['seed']==seed
  assert own['environment']=={**{k:'1'for k in m.q.guard.THREADS},'PB_ALL':'1','CANDIDATE_SHA256':src['sha256']}
  assert own['command']==[str(m.q.guard.PY),'-B',str(m.q.guard.GUARD),'--record',item['guard'],'--seconds','900','--footprint-limit','7500000000','--rss-backstop','7500000000',str(m.q.guard.RT),'-B',str(m.SCIENCE_JUDGE),src['path'],str(seed),item['certificate']]
  assert own['guard_record']==item['guard']and own['command'][11:]==guard['command']
  assert guard['completed']is True and guard['exit_code']==0 and guard['error']is None and guard['whole_group_retired']is True and guard['remaining_rows']==[]
  assert guard['footprint_trigger_bytes']==guard['rss_backstop_bytes']==7500000000 and guard['max_rss_bytes']<7500000000 and guard['max_footprint_bytes']<7500000000
  ids.update([own['owned_guard_pid'],guard['pid']]);groups.add(guard['pid'])
  for k in('certificate','guard'):assert m.sha(item[k])==item[k+'_sha256'];pins[item[k]]=item[k+'_sha256']
  pred=Path(item['certificate']).with_name(Path(item['certificate']).stem+'_pred.npz');assert m.sha(pred)==item['prediction_npz_sha256'];pins[str(pred)]=item['prediction_npz_sha256']
  results.append(item)
 for p in [rp,h/'PLAN_v1.json',h/'ownership.jsonl',h/'admission.jsonl',h/'admission_wait.jsonl']:pins[str(p)]=m.sha(p)
 shards.append(dict(**e,controller_pid=r['controller_pid'],ready_path=str(rp),ready_sha256=m.sha(rp),plan_sha256=m.sha(h/'PLAN_v1.json'),ownership_sha256=m.sha(h/'ownership.jsonl'),admission_sha256=admission['sha256'],admission_wait_sha256=admission['wait_sha256']))
expected=8 if phase=='pilot'else 32
assert len(ids)==4+2*expected and len(results)==expected
ps=[tuple(map(int,line.split()))for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()]
assert not[x for x in ps if x[0]in ids or x[1]in ids or x[2]in groups]
# All owner/guard barriers are now satisfied; validate full outputs without comparative means.
for item in results:m.science_current(c,item,item['role'],item['seed'])
root=dict(status='four_controllers_exited0_and_all_expected_guards_retired',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_sha256=digest,phase=phase,shards=shards,all_owned_PID_PPID_PGID_absent=True,owned_pids=sorted(ids),owned_pgids=sorted(groups),all_guards_clean=True,all16_finite=True,completed_candidate_calls=sum(x['role']=='ON'for x in results),completed_parent_calls=sum(x['role']=='parent'for x in results),candidate_sha256=c['candidates']['ON']['sha256'],evidence_sha256=pins,scientific_means_computed=False)
m.q.write(D/f'SCIENCE_{stage}_OWNER_RETIREMENT_v1.json',root)
root.update(approved=True,reviewed_by='root',executed_by='capacity',conditional_root_authorization=read(D/f'ROOT_SCIENCE_{stage}_GRANT_v1.json')['conditional_root_authorization'],owner_path=str(D/f'SCIENCE_{stage}_OWNER_RETIREMENT_v1.json'),owner_sha256=m.sha(D/f'SCIENCE_{stage}_OWNER_RETIREMENT_v1.json'))
m.q.write(D/f'ROOT_{stage}_RETIREMENT_v1.json',root)
print(json.dumps(dict(root_retirement_sha256=m.sha(D/f'ROOT_{stage}_RETIREMENT_v1.json'),owner_sha256=root['owner_sha256'],completed_calls=expected,all_owned_absent=True)))
