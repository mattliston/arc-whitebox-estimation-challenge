"""Metadata-only whole-capture retirement; actual terminal receipts required."""
from pathlib import Path
import hashlib,json,subprocess
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
c=json.loads((D/'CAPTURE_PROTOCOL_REGISTERED_v1.json').read_text())
for path,digest in c['source_pins'].items():assert sha(path)==digest
for v in c['candidates'].values():assert sha(v['path'])==v['sha256']
ids=set();items=[];evidence={};actual=[]
def pin(p):evidence[str(p)]=sha(p)
for shard in range(4):
 run=D/f'run_capture_s{shard}_v1';plan=json.loads((run/'PLAN_v1.json').read_text());ready=json.loads((run/'CAPTURE_SHARD_READY_v1.json').read_text());term=json.loads((run/'ACTUAL_MANAGED_EXIT_v1.json').read_text());managed=json.loads((run/'MANAGED_SESSION_v1.json').read_text())
 assert term['actual_exit_observed'] and term['exit_code']==0 and term['terminal_session_present'] is False and term['managed_session_id']==managed['managed_session_id']
 assert plan['controller_pid']==managed['controller_pid'] and plan['protocol_sha256']==sha(D/'CAPTURE_PROTOCOL_REGISTERED_v1.json')
 assert ready['shard']==shard and ready['complete_calls']==8 and ready['all_guards_clean'] and ready['owned_group_retirement_verified']
 assert [x['seed']for x in ready['evaluations']]==c['shards'][str(shard)]
 ids.add(plan['controller_pid']);actual.append(term)
 owners=[json.loads(x)for x in (run/'ownership.jsonl').read_text().splitlines()];assert [x['seed']for x in owners]==c['shards'][str(shard)]
 for owner in owners:ids.add(owner['owned_guard_pid'])
 for item in ready['evaluations']:
  for k in ('certificate','guard'):assert sha(item[k])==item[k+'_sha256'];pin(Path(item[k]))
  cert=json.loads(Path(item['certificate']).read_text());g=json.loads(Path(item['guard']).read_text());ids.add(g['pid'])
  assert g['completed'] and g['exit_code']==0 and g['whole_group_retired'] and g['remaining_rows']==[] and not g['error']
  assert g['max_rss_bytes']<7500000000 and g['max_footprint_bytes']<7500000000
  assert cert['sha256']==c['candidates']['capture']['sha256'] and cert['seeds']==[item['seed']] and cert['all16_finite'] and cert['terminal254_complete'] and cert['all16_parity_measured']
  assert cert['weights_before_sha256']==cert['weights_after_sha256']==c['network_metadata'][item['seed']]['weights_raw_sha256']
  assert sha(item['capture_path'])==item['capture_sha256'];pin(Path(item['capture_path']))
  p=Path(item['certificate']);p=p.with_name(p.stem+'_pred.npz');assert sha(p)==item['prediction_npz_sha256'];pin(p)
 items.extend(ready['evaluations'])
 for n in ('PLAN_v1.json','CAPTURE_SHARD_READY_v1.json','ACTUAL_MANAGED_EXIT_v1.json','MANAGED_SESSION_v1.json','ownership.jsonl','admission.jsonl'):pin(run/n)
rows=[]
for line in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid=,stat=,comm='],text=True).splitlines():
 v=line.split(None,4)
 if len(v)==5:rows.append(dict(pid=int(v[0]),ppid=int(v[1]),pgid=int(v[2]),stat=v[3],comm=v[4]))
assert not [r for r in rows if {r['pid'],r['ppid'],r['pgid']}&ids]
assert sorted(x['seed']for x in items)==list(range(32))
(D/'POST_EXIT_PS_v1.json').write_text(json.dumps(rows,indent=2)+'\n');pin(D/'POST_EXIT_PS_v1.json')
record=dict(status='TRAIN32_CAPTURE_COMPLETE_RETIRED',complete_calls=32,candidate_sha256=c['candidates']['capture']['sha256'],protocol_sha256=sha(D/'CAPTURE_PROTOCOL_REGISTERED_v1.json'),actual_managed_exits_observed=True,actual_exits=actual,all_owned_PID_PPID_PGID_absent=True,all_guards_clean=True,all16_finite=True,owned_ids=sorted(ids),evaluations=sorted(items,key=lambda x:x['seed']),evidence_sha256=evidence,no_fit_executed=True,executed_by='compute_hunt')
f=D/'OWNER_CAPTURE_RETIREMENT_v1.json';assert not f.exists();f.write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(dict(owner_path=str(f),sha256=sha(f),complete_calls=32,owned_ids=len(ids))))
