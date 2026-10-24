"""One fixed model: actual retirement, first-ten identity, actual whole bill."""
from pathlib import Path
import hashlib,json,math,subprocess
import numpy as np
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
p=D/'PROTOCOL_REGISTERED_astra_v1.json';c=json.loads(p.read_text());h=D/'run_full384_bill_s0_v1'
r=json.loads((h/'READY_v1.json').read_text());t=json.loads((h/'ACTUAL_MANAGED_EXIT_v1.json').read_text());m=json.loads((h/'MANAGED_SESSION_v1.json').read_text())
assert r['all_guards_retired'] and r['protocol_sha256']==sha(p)
assert t['actual_exit_observed'] and t['exit_code']==0 and not t['terminal_session_present'] and t['managed_session_id']==m['managed_session_id']
assert m['controller_pid']==r['controller_pid']
es=r['evaluations'];assert len(es)==1 and es[0]['seed']==7103248
own=[json.loads(s)for s in (h/'ownership.jsonl').read_text().splitlines()];assert len(own)==1
e=es[0];a=own[0];g=json.loads(Path(e['guard']).read_text())
assert a['controller_pid']==r['controller_pid'] and a['seed']==7103248
assert g['completed'] and g['exit_code']==0 and g['error'] is None and g['whole_group_retired'] and not g['remaining_rows']
assert g['max_rss_bytes']<7500000000 and g['max_footprint_bytes']<7500000000
ids={r['controller_pid'],a['owned_guard_pid'],g['pid']}
live=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()if s.strip()]
assert not any(set(row)&ids for row in live)
pins={str(h/n):sha(h/n)for n in ('READY_v1.json','ACTUAL_MANAGED_EXIT_v1.json','MANAGED_SESSION_v1.json','PLAN_v1.json','ownership.jsonl')}
for path,digest in c['source_pins'].items():assert sha(path)==digest
def read(rec):
 for key in ('certificate','guard'):assert sha(rec[key])==rec[key+'_sha256'];pins[rec[key]]=rec[key+'_sha256']
 z=Path(rec['certificate']);pp=z.with_name(z.stem+'_pred.npz');assert sha(pp)==rec['prediction_npz_sha256'];pins[str(pp)]=sha(pp)
 with np.load(pp,allow_pickle=False)as f:y=f['7103248'].copy()
 return json.loads(z.read_text()),y
bref=c['stack_control_reuse'][0];ac,ay=read(e);bc,by=read(bref)
assert ac['sha256']==c['candidates']['ON']['sha256'] and bc['sha256']==c['candidates']['stack']['sha256']
assert ay.shape==by.shape==(16,1024) and np.isfinite(ay).all() and ay[:10].tobytes()==by[:10].tobytes()
ar,br=ac['rows'][0],bc['rows'][0]
assert ar['mse_layers'][:10]==br['mse_layers'][:10]
assert ar['weights_before_sha256']==ar['weights_after_sha256']==br['weights_before_sha256']
from full384_source_contract_v1 import gate
gate(ar)
ret=dict(candidate_sha256=ac['sha256'],protocol_sha256=sha(p),all_actual_retired=True,all_actual_exits_observed=True,all_owned_PID_PPID_PGID_absent=True,all_guards_clean=True,owned_ids=sorted(ids),actual_terminal=t,evaluations=es,evidence_sha256=pins)
rp=D/'BILL1_RETIREMENT_v1.json';rp.open('x').write(json.dumps(ret,indent=2)+'\n')
mse=100*math.log(ar['mse_final']/br['mse_final']);bill=100*math.log(max(ar['flops']/2**41,.1)/max(br['flops']/2**41,.1))
out=dict(implementation_pass=True,candidate_sha256=ac['sha256'],reference_sha256=bc['sha256'],seed=7103248,first10_prediction_bytes_equal=True,first10_layerMSE_equal=True,all16_finite=True,all_actual_retired=True,flops=ar['flops'],reference_flops=br['flops'],delta_flops=ar['flops']-br['flops'],bill_ratio=ar['flops']/br['flops'],bill_arithmetic_pct=100*(ar['flops']/br['flops']-1),MSE_log_pct=mse,bill_log_pct=bill,net_log_pct=mse+bill,mse_final=ar['mse_final'],reference_mse_final=br['mse_final'],charged_ops_delta=ar['ops']-br['ops'],peak_rss_bytes=g['max_rss_bytes'],peak_footprint_bytes=g['max_footprint_bytes'],retirement_path=str(rp),retirement_sha256=sha(rp),no_followup_authorization=True,decision='One fixed model only; root decides whether to complete fixed pilot8.',evidence_sha256={**pins,str(p):sha(p),str(rp):sha(rp)})
op=D/'BILL1_DECISION_v1.json';op.open('x').write(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k!='evidence_sha256'}));print('decision',sha(op),'retirement',sha(rp))
