"""Fixed pilot/all24 signs, strictly after actual owned retirement."""
from pathlib import Path
import hashlib,json,math,statistics,subprocess,sys
import numpy as np
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
phase=sys.argv[1];assert phase in ('pilot','remaining')
p=D/'PROTOCOL_REGISTERED_astra_v1.json';c=json.loads(p.read_text());evaluations=[];ids=set();pins={};terminals=[];owners=[]
for shard in range(4):
 h=D/f'run_table_r7_{phase}_s{shard}_v1';r=json.loads((h/'READY_v1.json').read_text());t=json.loads((h/'ACTUAL_MANAGED_EXIT_v1.json').read_text());m=json.loads((h/'MANAGED_SESSION_v1.json').read_text())
 assert r['all_guards_retired']and r['protocol_sha256']==sha(p)
 assert t['actual_exit_observed']and t['exit_code']==0 and not t['terminal_session_present']and t['managed_session_id']==m['managed_session_id']
 assert m['controller_pid']==r['controller_pid'];ids.add(r['controller_pid']);terminals.append(t)
 expected=[(seed,role)for seed in c['phase_seeds'][phase][shard::4]for role in (('parent','ON')if seed%2==0 else('ON','parent'))]
 es=r['evaluations'];assert [(e['seed'],e['role'])for e in es]==expected
 own=[json.loads(s)for s in (h/'ownership.jsonl').read_text().splitlines()];assert len(own)==len(es)
 for a,e in zip(own,es):
  assert a['controller_pid']==r['controller_pid']and a['seed']==e['seed'];ids.add(a['owned_guard_pid']);g=json.loads(Path(e['guard']).read_text());ids.add(g['pid'])
  assert g['completed']and g['exit_code']==0 and g['error']is None and g['whole_group_retired']and not g['remaining_rows']
  assert g['max_rss_bytes']<7500000000 and g['max_footprint_bytes']<7500000000
 evaluations.extend(es);owners.append(dict(shard=shard,controller_pid=r['controller_pid'],ready_sha256=sha(h/'READY_v1.json')))
 for n in ('READY_v1.json','ACTUAL_MANAGED_EXIT_v1.json','MANAGED_SESSION_v1.json','PLAN_v1.json','ownership.jsonl'):pins[str(h/n)]=sha(h/n)
live=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()if s.strip()]
assert not any(set(row)&ids for row in live)
for path,digest in c['source_pins'].items():assert sha(path)==digest
for e in evaluations:
 for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256'];pins[e[key]]=e[key+'_sha256']
 cp=Path(e['certificate']);pp=cp.with_name(cp.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256'];pins[str(pp)]=sha(pp)
ret=dict(candidate_sha256=c['candidates']['ON']['sha256'],protocol_sha256=sha(p),phase=phase,all_actual_retired=True,all_actual_exits_observed=True,all_owned_PID_PPID_PGID_absent=True,all_guards_clean=True,owned_ids=sorted(ids),actual_terminals=terminals,owners=owners,evaluations=evaluations,evidence_sha256=pins)
rp=D/f'{phase.upper()}_RETIREMENT_v1.json';rp.open('x').write(json.dumps(ret,indent=2)+'\n')
if phase=='remaining':
 pp=D/'PILOT_DECISION_v1.json';pr=D/'PILOT_RETIREMENT_v1.json';pd=json.loads(pp.read_text());prior=json.loads(pr.read_text())
 assert pd['passed']and pd['all_actual_retired']and pd['retirement_sha256']==sha(pr)and prior['all_actual_retired']
 assert pd['candidate_sha256']==c['candidates']['ON']['sha256'];evaluations=prior['evaluations']+evaluations;pins[str(pp)]=sha(pp);pins[str(pr)]=sha(pr)
def read(e):
 for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256'];pins[e[key]]=e[key+'_sha256']
 cp=Path(e['certificate']);pp=cp.with_name(cp.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256'];pins[str(pp)]=sha(pp)
 with np.load(pp,allow_pickle=False)as z:y=z[str(e['seed'])].copy()
 cert=json.loads(cp.read_text());assert cert['sha256']==c['candidates'][e['role']]['sha256'];assert y.shape==(16,1024)and np.isfinite(y).all()
 return cert,y
rows=[]
for seed in sorted(set(e['seed']for e in evaluations)):
 pair={role:next(e for e in evaluations if(e['seed'],e['role'])==(seed,role))for role in ('parent','ON')}
 ac,ay=read(pair['ON']);bc,by=read(pair['parent']);a,b=ac['rows'][0],bc['rows'][0]
 assert ay[:1].tobytes()==by[:1].tobytes()and a['mse_layers'][:1]==b['mse_layers'][:1]
 assert a['weights_before_sha256']==a['weights_after_sha256']==b['weights_before_sha256']==b['weights_after_sha256']
 assert ac['setup']['pb_single_sha256']==bc['setup']['pb_single_sha256']
 mse=100*math.log(a['mse_final']/b['mse_final']);bill=100*math.log(max(a['flops']/2**41,.1)/max(b['flops']/2**41,.1))
 rows.append(dict(seed=seed,MSE_log_pct=mse,bill_log_pct=bill,net_log_pct=mse+bill,candidate_flops=a['flops'],parent_flops=b['flops'],candidate_MSE=a['mse_final'],parent_MSE=b['mse_final'],first1_exact=True,candidate=pair['ON'],parent=pair['parent']))
expected=c['phase_seeds']['pilot']if phase=='pilot'else c['phase_seeds']['pilot']+c['phase_seeds']['remaining'];assert [r['seed']for r in rows]==expected
def summary(rr):
 return dict(n=len(rr),**{k:statistics.mean(r[k]for r in rr)for k in ('MSE_log_pct','bill_log_pct','net_log_pct')},net_SE=statistics.stdev(r['net_log_pct']for r in rr)/math.sqrt(len(rr)))
means=summary(rows);oos=summary([r for r in rows if r['seed']in c['identity_groups']['oos']])if phase=='remaining'else None
passed=means['net_log_pct']<0 and(phase=='pilot'or oos['net_log_pct']<0)
out=dict(passed=passed,phase=phase,candidate_sha256=c['candidates']['ON']['sha256'],reference_sha256=c['candidates']['parent']['sha256'],all_actual_retired=True,all_first1_prediction_bytes_and_layerMSE_equal=True,n=len(rows),means=means,OOS8=oos,rows=rows,continuation='Unchanged remaining16 allowed only by root conditional grant and pilot PASS.'if phase=='pilot'else'No release authority; root reviews all24/OOS8.',retirement_path=str(rp),retirement_sha256=sha(rp),protocol_sha256=sha(p),evidence_sha256={**pins,str(p):sha(p),str(rp):sha(rp)})
op=D/('PILOT_DECISION_v1.json'if phase=='pilot'else'FINAL24_DECISION_v1.json');op.open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ('rows','evidence_sha256')}));print('decision',sha(op),'retirement',sha(rp))
