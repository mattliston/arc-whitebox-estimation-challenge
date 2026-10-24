"""Root-authorized correction of coverage expectation; no source change or model call."""
from pathlib import Path
import hashlib,json,math,subprocess,ast
import numpy as np
D=Path(__file__).resolve().parent;h=D/'run_uniform64_bill_s0_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def put(p,d):
    with p.open('x')as f:json.dump(d,f,indent=2);f.write('\n')
p=D/'PROTOCOL_REGISTERED_astra_v1.json';c=read(p)
for f,digest in c['source_pins'].items():assert sha(f)==digest
t=read(h/'ACTUAL_MANAGED_EXIT_v1.json');m=read(h/'MANAGED_SESSION_v1.json');plan=read(h/'PLAN_v1.json');stop=read(h/'STOPPED_v1.json')
assert t['actual_exit_observed'] and t['exit_code']==1 and not t['terminal_session_present'] and t['managed_session_id']==m['managed_session_id']
assert stop==dict(error='AssertionError()',evaluations=[],owned_retirement=True)
own=[json.loads(s)for s in (h/'ownership.jsonl').read_text().splitlines()];assert len(own)==1
o=own[0];g=read(o['guard_record']);assert o['controller_pid']==plan['controller_pid']==m['controller_pid']
assert g['completed'] and g['exit_code']==0 and g['error']is None and g['whole_group_retired'] and not g['remaining_rows']
assert g['max_rss_bytes']<7500000000 and g['max_footprint_bytes']<7500000000
ids={m['controller_pid'],o['owned_guard_pid'],g['pid']}
live=[tuple(map(int,s.split()))for s in subprocess.check_output(['/bin/ps','-axo','pid=,ppid=,pgid='],text=True).splitlines()if s.strip()]
assert not any(set(r)&ids for r in live)
cp=h/'native_7103248_ON.json';pp=h/'native_7103248_ON_pred.npz';ac=read(cp)
bc_rec=c['stack_control_reuse'][0];bp=Path(bc_rec['certificate']);bpp=bp.with_name(bp.stem+'_pred.npz')
assert sha(bp)==bc_rec['certificate_sha256'] and sha(bpp)==bc_rec['prediction_npz_sha256'];bc=read(bp)
assert ac['sha256']==c['candidates']['ON']['sha256'] and bc['sha256']==c['candidates']['stack']['sha256']
ar,br=ac['rows'][0],bc['rows'][0]
from uniform64_source_contract_v1 import _uniform64_parent_gate
_uniform64_parent_gate(ar)
events=ar['stats']['_ASTRA_READ64_STATS']['events'];expected=[(10,320),(11,320),(12,320),(13,320),(13,384),(14,320),(14,384)]
assert [(e[0],e[3])for e in events]==expected
assert all(len(e)==6 and e[4]==e[3]-64 and e[5]==256 and e[1]>0 and e[2]>0 for e in events)
assert ar['stats']['_ASTRA_READ64_STATS']['retired']is True
assert ar['sm40_lifecycle_before']['prebind_count']==14
assert ar['weights_before_sha256']==ar['weights_after_sha256']==br['weights_before_sha256']
with np.load(pp,allow_pickle=False)as z:ay=z['7103248'].copy()
with np.load(bpp,allow_pickle=False)as z:by=z['7103248'].copy()
assert ay.shape==by.shape==(16,1024) and np.isfinite(ay).all()
assert ay[:7].tobytes()==by[:7].tobytes() and ar['mse_layers'][:7]==br['mse_layers'][:7]
s=Path(c['candidates']['ON']['path']).read_text();tree=ast.parse(s)
assert '_LCR_LAYERS = (10, 11, 12, 13, 14)'in s and '_UVP_ONLY_LAYERS = _LCR_LAYERS + (15,)'in s
fs=[ast.get_source_segment(s,n)for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cp_slr_inner_sq']
assert any('if _UVP_CUR[0] not in _UVP_ONLY_LAYERS:\n        return _UVP_ORIG_READ(old_left, lam_old, state)'in f and 'return _uvp_inner(old_left, lam_old, uv[1])'in f for f in fs)
pins={str(f):sha(f)for f in [p,Path(__file__),h/'PLAN_v1.json',h/'STOPPED_v1.json',h/'ACTUAL_MANAGED_EXIT_v1.json',h/'MANAGED_SESSION_v1.json',h/'ownership.jsonl',Path(o['guard_record']),cp,pp,bp,bpp]}
cov=dict(status='ROOT_ACCEPTED_ADMINISTRATIVE_COVERAGE_CORRECTION',candidate_sha256=ac['sha256'],source_unchanged=True,no_rerun=True,original_expected_events=10,actual_events=events,actual_affected_layers=[10,11,12,13,14],early_layers_7_8_9_not_executed=True,source_reason='_UVP_ONLY_LAYERS is (10,11,12,13,14,15); early reads return _UVP_ORIG_READ before _uvp_inner.',first7_prediction_bytes_and_layerMSE_equal=True,current_parent_lifecycle_pass=True,original_controller_exit1_preserved=True,original_stop_path=str(h/'STOPPED_v1.json'),forecast_0663_pct_invalid_for_actual_scope=True,root_authority='2026-10-16 root message: Route interpretation accepted; seven events actual scope. Version coverage receipt and grade one; no rerun or eight-model grant.',evidence_sha256=pins)
cv=D/'ACTUAL_COVERAGE_ACCEPTANCE_v2.json';put(cv,cov)
ret=dict(candidate_sha256=ac['sha256'],protocol_sha256=sha(p),all_actual_retired=True,all_owned_PID_PPID_PGID_absent=True,all_guards_clean=True,owned_ids=sorted(ids),actual_terminal=t,guard_exit_code=0,controller_exit1_administrative_coverage_assertion=True,coverage_receipt=str(cv),coverage_sha256=sha(cv),evidence_sha256=pins)
rp=D/'BILL1_RETIREMENT_v2.json';put(rp,ret)
mse=100*math.log(ar['mse_final']/br['mse_final']);bill=100*math.log(max(ar['flops']/2**41,.1)/max(br['flops']/2**41,.1))
out=dict(implementation_pass_for_actual_scope=True,candidate_sha256=ac['sha256'],reference_sha256=bc['sha256'],seed=7103248,first7_prediction_bytes_equal=True,first7_layerMSE_equal=True,all16_finite=True,all_actual_retired=True,actual_cut_events=7,actual_affected_layers=[10,11,12,13,14],flops=ar['flops'],reference_flops=br['flops'],delta_flops=ar['flops']-br['flops'],bill_arithmetic_pct=100*(ar['flops']/br['flops']-1),MSE_log_pct=mse,bill_log_pct=bill,net_log_pct=mse+bill,mse_final=ar['mse_final'],reference_mse_final=br['mse_final'],charged_ops_delta=ar['ops']-br['ops'],peak_rss_bytes=g['max_rss_bytes'],peak_footprint_bytes=g['max_footprint_bytes'],retirement_path=str(rp),retirement_sha256=sha(rp),coverage_path=str(cv),coverage_sha256=sha(cv),no_followup_authorization=True,evidence_sha256=pins)
op=D/'BILL1_DECISION_v2.json';put(op,out)
print(json.dumps({k:v for k,v in out.items()if k!='evidence_sha256'}));print('decision',sha(op),'retirement',sha(rp),'coverage',sha(cv))
