"""Root-delegated metadata-only exact-view transfer and release confirmation.
No models, source changes, retries, private8 extension, uploads or submission.
"""
from pathlib import Path
import hashlib,json,tarfile
D=Path(__file__).resolve().parent;O=D.parents[2];R=O/'lead/runs';N=R/'astra_prefix22_uvp_split_native_ten_v1';S=R/'astra_prefix_row_energy22_science_v1';Z=D/'seal'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p))
def put(p,v):
 with p.open('x')as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
r=read(D/'DELIVERY_READY_v2.json');src=Path(r['candidate']['path']);h=r['candidate']['sha256'];assert sha(src)==h
reference='7b1266280f75edd8e44800e6afdcc5981d9f5d27f1a8226e73cac6f1f1abc84a'
np=N/'NATIVE_TEN_PARITY_v1.json';n=read(np);nr=Path(n['retirement_path']);ret=read(nr)
assert n['passed']and n['approved']and n['reviewed_by']=='root'and n['candidate_sha256']==h and n['reference_sha256']==reference
assert n['n']==10 and n['all_actual_retired']and n['all16_byte_identical']and n['all_flops_identical']and n['retirement_sha256']==sha(nr)
assert [x['seed']for x in n['rows']]==list(range(7103248,7103258))
assert ret['all_actual_exits_observed']and ret['all_owned_PID_PPID_PGID_absent']and ret['all_guards_clean']
for p,v in ret['evidence_sha256'].items():assert sha(p)==v
na=D/'boundary/ROOT_NATIVE_TEN_ACCEPTANCE_v1.json';a=read(na)
assert a['approved']and a['reviewed_by']=='root'and a['candidate_sha256']==h and a['all_16_prediction_bytes_equal']and a['all_FLOPs_equal']and a['all_actual_retired']
for p,v in a['evidence_sha256'].items():assert sha(p)==v
production=D/'boundary/ROOT_PRODUCTION_ACCEPTANCE_v1.json';p=read(production)
assert p['approved']and p['reviewed_by']=='root'and p['candidate_sha256']==h
assert all(p[k]is True for k in('normal_pass','slow_pass','science24_accepted','all_owned_PID_PPID_PGID_absent','release_gate_passed'))
for path,v in p['evidence_sha256'].items():assert sha(path)==v
science=S/'ALL24_DECISION_astra_v1.json';d=read(science);assert d['passed']and d['n']==24 and d['source']['sha256']==reference and d['all_actual_retired']and d['all_implementation_pass']
assert d['means']['PRIMARY']['net_log_pct']<0 and d['groups']['OOS8']['PRIMARY']['net_log_pct']<0
prior=R/'astra_prefix_energy22_delivery_v1/seal/ROOT_OOS8_REUSE_AUTHORITY_v1.json';old=read(prior);assert old['approved']and old['reviewed_by']=='root'and old['candidate_sha256']==reference and old['seeds']==list(range(7103264,7103272))
assert old['decision']['sha256']==sha(science)
for dirname in('astra_uvp_view_hold_v1','astra_uvp_split_views_v1'):
 q=R/dirname/'SOURCE_HELD_v1.json';v=read(q);assert v.get('exact_inverse',v.get('exact_inverse_parent'));assert sha(v['candidate']['path'])==v['candidate']['sha256']and sha(v['parent']['path'])==v['parent']['sha256']
stage=O/'release/runs'/('a_'+r['tag']);arc=stage/'submission.tar.gz';a=read(stage/'stage_a.json');assert a['passed']and a['candidate_sha256']==h
with tarfile.open(arc)as t:assert sorted(t.getnames())==['estimator.py','manifest.json']and t.extractfile('estimator.py').read()==src.read_bytes()
gate=D/'RELEASE_GATE_v4.json';assert read(gate)['passed']is True
proofs=[np,nr,na,production,science,prior,src,D/'ROOT_VIEW_REPAIR_AUTHORITY_v1.json',D/'STAGE_A_EXACT_ARCHIVE_v1.json',gate,stage/'stage_a.json',arc]
for dirname in('astra_uvp_view_hold_v1','astra_uvp_split_views_v1'):proofs.extend((R/dirname/'SOURCE_HELD_v1.json',R/dirname/'FRESH_ALGEBRA_SOURCE_PEER_v1.json'))
evidence={str(p):sha(p)for p in proofs};Z.mkdir()
transfer=Z/'ROOT_OOS8_SEMANTICS_TRANSFER_v1.json';put(transfer,dict(approved=True,reviewed_by='root',executed_by='larger_accuracy',authority='Root explicitly permits direct24/OOS reuse after exact source repair and native10 all16bytes/FLOPs; no additional private8 needed. Root directs immediate first fireable RPE release; root alone fires.',candidate_sha256=h,measurement_candidate_sha256=reference,native_ten_all16_bytes_equal=True,native_ten_FLOPs_equal=True,prediction_and_bill_semantics_unchanged=True,original_rule_preserved=True,seeds=old['seeds'],additional_models=0,evidence_sha256=evidence))
decision=Z/'OOS8_DECISION_v1.json';put(decision,dict(verdict='PASS',candidate_sha256=h,measurement_candidate_sha256=reference,seeds=old['seeds'],n=8,mean_log_net=d['groups']['OOS8']['PRIMARY']['net_log_pct']/100.,all24_mean_log_net=d['means']['PRIMARY']['net_log_pct']/100.,all24_SE_log=d['SE_pct']['PRIMARY']/100.,original_rule_preserved=True,semantics_transfer=pin(transfer),additional_models=0))
evidence.update({str(transfer):sha(transfer),str(decision):sha(decision)})
argv=['/Users/matt/arc-white-box-estimation/.venv/bin/python','-B',str(O/'fire/fire_once.py'),'prefix22uvpsplit_20261015_astra_v1',str(arc),'Paid-prefix row energy repair with exact UVP split-view request reduction; all24/OOS science improvement, PRIMARY runtime PASS, exact native-ten bytes/FLOPs.']
confirmation=D/'RELEASE_CONFIRMED_v1.json';put(confirmation,dict(status='CONFIRMED',approved=True,reviewed_by='root',executed_by='larger_accuracy',authority='Root delegated fixed-condition release preparation; root alone fires.',candidate_path=str(src),candidate_sha256=h,archive_path=str(arc),archive_sha256=sha(arc),stage_a_pass=True,original_PRIMARY_pass=True,original_preregistered_OOS8_pass=True,science24_net_log_pct=d['means']['PRIMARY']['net_log_pct'],OOS8_net_log_pct=d['groups']['OOS8']['PRIMARY']['net_log_pct'],exact_native_ten_all16_byte_and_FLOP_parity=True,science24_scope='Accepted7b126 all24/OOS8 preserved by exact view source proof and repaired native ten; fresh PRIMARY normal/slow passed.',fire_argv_for_root_only=argv,fire_cwd=str(O),submitted=False,evidence_sha256=evidence))
print(json.dumps(dict(status='CONFIRMED',confirmation=pin(confirmation),archive=pin(arc),science24_net_log_pct=d['means']['PRIMARY']['net_log_pct'],OOS8_net_log_pct=d['groups']['OOS8']['PRIMARY']['net_log_pct'],fire_argv_for_root_only=argv)))
