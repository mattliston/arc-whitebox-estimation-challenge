"""Fixed current-best adjusted signs; e406 is report-only, after actual owned terminal/guard barriers."""
from pathlib import Path
import hashlib,json,math,statistics,subprocess,sys
import numpy as np
D=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();phase=sys.argv[1];assert phase in ('pilot','remaining')
p=D/'PROTOCOL_REGISTERED_astra_v1.json';c=json.loads(p.read_text());owners=[];evaluations=[];ids=set();pins={};terminals=[]
for i in range(4):
 h=D/f'run_re23_{phase}_s{i}_v1';r=json.loads((h/'READY_v1.json').read_text());t=json.loads((h/'ACTUAL_MANAGED_EXIT_v1.json').read_text());m=json.loads((h/'MANAGED_SESSION_v1.json').read_text())
 assert r['all_guards_retired'] is True and r['protocol_sha256']==sha(p)
 assert t['actual_exit_observed'] and t['exit_code']==0 and t['terminal_session_present'] is False and t['managed_session_id']==m['managed_session_id']
 assert m['controller_pid']==r['controller_pid'];ids.add(r['controller_pid']);terminals.append(t)
 es=r['evaluations'];assert [e['seed']for e in es]==c['phase_seeds'][phase][i::4]
 own=[json.loads(z)for z in (h/'ownership.jsonl').read_text().splitlines()];assert len(own)==len(es)
 for a,e in zip(own,es):
  assert a['controller_pid']==r['controller_pid'] and a['seed']==e['seed'];ids.add(a['owned_guard_pid']);g=json.loads(Path(e['guard']).read_text());ids.add(g['pid'])
  assert g['completed'] is True and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] is True and not g['remaining_rows']
  assert g['max_rss_bytes']<7500000000 and g['max_footprint_bytes']<7500000000
 evaluations.extend(es)
 owners.append(dict(shard=i,controller=r['controller_pid'],ready_sha256=sha(h/'READY_v1.json')))
 for n in ('READY_v1.json','ACTUAL_MANAGED_EXIT_v1.json','MANAGED_SESSION_v1.json','PLAN_v1.json','ownership.jsonl'):pins[str(h/n)]=sha(h/n)
live=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()if s.strip()]
assert not any(set(row)&ids for row in live)
assert sorted(x['seed']for x in evaluations)==c['phase_seeds'][phase]
for e in evaluations:
 for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256'];pins[e[key]]=e[key+'_sha256']
 p0=Path(e['certificate']);pp=p0.with_name(p0.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256'];pins[str(pp)]=e['prediction_npz_sha256']
ret=dict(approved=True,reviewed_by='root',executed_by='bigger_gain_hunt',authority='Root delegated fixedpilot and conditionalremaining plus actualretirement/current-best grade; no tuning.',candidate_sha256=c['candidates']['ON']['sha256'],protocol_sha256=sha(p),all_actual_exits_observed=True,actual_terminals=terminals,all_owned_PID_PPID_PGID_absent=True,all_guards_clean=True,owned_ids=sorted(ids),owners=owners,evaluations=evaluations,evidence_sha256=pins)
retp=D/(phase.upper()+'_RETIREMENT_astra_v1.json');retp.open('x').write(json.dumps(ret,indent=2)+'\n')
if phase=='remaining':
 prior=json.loads((D/'PILOT_DECISION_astra_v1.json').read_text());assert prior['passed'] and prior['all_actual_retired']
 for i in range(4):evaluations.extend(json.loads((D/f'run_re23_pilot_s{i}_v1/READY_v1.json').read_text())['evaluations'])
assert len(evaluations)==(8 if phase=='pilot' else 24)
def read(e):
 for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256']
 p0=Path(e['certificate']);pp=p0.with_name(p0.stem+'_pred.npz');assert sha(pp)==e['prediction_npz_sha256']
 a=json.loads(p0.read_text())
 with np.load(pp,allow_pickle=False)as z:pred=z[str(e['seed'])].copy()
 return a,a['rows'][0],pred
rows=[]
for e in sorted(evaluations,key=lambda z:z['seed']):
 a,ar,ap=read(e);assert a['sha256']==c['candidates']['ON']['sha256'];row=dict(seed=e['seed'],comparisons={})
 for role,key in [('PRIMARY','stack_control_reuse'),('e406','reference_control_reuse')]:
  control=next(x for x in c[key]if x['seed']==e['seed']);b,br,bp=read(control)
  assert a['setup']['pb_single_sha256']==b['setup']['pb_single_sha256'] and a['setup']['pb_single_count']==b['setup']['pb_single_count']
  assert ap.shape==bp.shape==(16,1024) and np.isfinite(ap).all() 
  if role=='PRIMARY':assert ap[:7].tobytes()==bp[:7].tobytes()
  mse=100*math.log(ar['mse_final']/br['mse_final']);bill=100*math.log(max(ar['flops']/2**41,.1)/max(br['flops']/2**41,.1))
  row['comparisons'][role]=dict(MSE_log_pct=mse,bill_log_pct=bill,net_log_pct=mse+bill,ops_delta=ar['ops']-br['ops'],prediction_max_abs_diff=float(np.max(np.abs(ap.astype(np.float64)-bp.astype(np.float64)))))
 rows.append(row)
def summary(rs):
 return {role:{k:statistics.mean(r['comparisons'][role][k]for r in rs)for k in ('MSE_log_pct','bill_log_pct','net_log_pct','ops_delta')}for role in ('PRIMARY','e406')}
means=summary(rows);ses={role:statistics.stdev(r['comparisons'][role]['net_log_pct']for r in rows)/len(rows)**.5 for role in means};groups={}
if phase=='pilot':passed=means['PRIMARY']['net_log_pct']<0
else:
 for name,seeds in [('fresh16',c['identity_groups']['pilot']+c['identity_groups']['fresh_tail']),('OOS8',c['identity_groups']['oos'])]:groups[name]=summary([r for r in rows if r['seed']in seeds])
 passed=all(s['PRIMARY']['net_log_pct']<0 for s in (means,groups['OOS8']))
out=dict(phase=phase,passed=passed,all_actual_retired=True,source=c['candidates']['ON'],n=len(rows),means=means,groups=groups,SE_pct=ses,rows=rows,all_implementation_pass=True,retirement_path=str(retp),retirement_sha256=sha(retp),remaining16_authorized=(phase=='pilot'and passed))
p=D/('PILOT_DECISION_astra_v1.json'if phase=='pilot'else'ALL24_DECISION_astra_v1.json');p.open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k!='rows'}));print('decision',sha(p),'retirement',sha(retp))
