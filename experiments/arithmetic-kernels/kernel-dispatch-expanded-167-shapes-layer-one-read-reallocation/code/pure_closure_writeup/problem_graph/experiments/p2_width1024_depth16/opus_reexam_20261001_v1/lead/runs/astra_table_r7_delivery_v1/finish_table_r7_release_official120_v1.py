"""opus-release: lead-delegated final release seal for b709 (table167+R7). Metadata only; lead alone submits.
Science is DIRECT on the exact source (no cross-source native-ten transfer); captured-setup native/production parity required."""
from pathlib import Path
import hashlib,json,tarfile
D=Path(__file__).resolve().parent;O=D.parents[2];R=O/'lead/runs';B=D/'boundary';S=R/'astra_smirnov_table_r7_science_v1';X=R/'astra_table_r7_native_ten_export_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
def pin(p):return dict(path=str(p),sha256=sha(p))
def put(p,x):
 with p.open('x')as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def verify_evidence(x):
 for p,h in x.get('evidence_sha256',{}).items():assert sha(p)==h,(p,h)
r=read(D/'DELIVERY_READY_v1.json');src=Path(r['candidate']['path']);h=r['candidate']['sha256'];assert sha(src)==h=='b7090c35a185309c73633d66dbaa06d48b2d173c3f6b25ff941a54c224b2ba04'
parent='2b10843a76fc033325ceee9dd30703eff29a1fa535cdc094c5feef72023a0c3c'
science=S/'FINAL24_DECISION_v1.json';d=read(science);assert d['passed']and d['n']==24 and d['candidate_sha256']==h and d['reference_sha256']==parent and d['all_actual_retired']and d['all_first1_prediction_bytes_and_layerMSE_equal']
assert d['means']['net_log_pct']<0 and d['OOS8']['net_log_pct']<0
retire=S/'REMAINING_RETIREMENT_v1.json';assert d['retirement_sha256']==sha(retire)
grant=S/'ROOT_SCIENCE_ACCEPTANCE_RELEASE_GRANT_v1.json';g=read(grant);assert g['approved']and g['reviewed_by']=='root'and g['candidate_sha256']==h and g['science_sha256']==sha(science)
export=X/'NATIVE_TEN_SCIENCE_EXPORT_v1.json';e=read(export);assert e['status']=='PASS_SOURCE_IDENTICAL_EXPORT'and e['source_sha256']==h and e['all_actual_retired'];verify_evidence(e)
native=D/'NATIVE35_100_CAPTURED_SETUP_PARITY_v1.json';p=read(native);assert p['passed']and p['candidate_sha256']==h and p['all16_prediction_bytes_equal']and p['all_FLOPs_equal']and p['all_actual_retired']and p['normal_and_slow_actual_setup_states_exact'];verify_evidence(p)
metadata=D/'OFFICIAL120_METADATA_COMPLETE_v1.json';m=read(metadata);assert m['release_gate_passed']and m['candidate_sha256']==h and m['all_other_gates_unchanged']and m['old100_FAIL_preserved'];verify_evidence(m)
policy=O/'lead/ROOT_PHASE2_OFFICIAL_WALL_POLICY_20261015_v1.json';assert m['official_wall_policy']==pin(policy);policydata=read(policy);assert policydata['approved']and policydata['reviewed_by']=='root'
for mode in ('NORMAL','SLOW'):
 x=read(B/('OFFICIAL_'+mode+'_v1')/'early_gate.json');assert x['passed'];verify_evidence(x)
ret=B/'ROOT_COLLECTORS_RETIREMENT_v1.json';t=read(ret);assert t['all_owned_PID_PPID_PGID_absent']and t['both_actual_exits_observed_via_guard_and_launcher'];verify_evidence(t)
stage=O/'release/runs'/('a_'+r['tag']);arc=stage/'submission.tar.gz';stagea=read(stage/'stage_a.json');assert stagea['passed']and stagea['candidate_sha256']==h and stagea['new_einsum_signatures']==[]
with tarfile.open(arc)as f:assert sorted(f.getnames())==['estimator.py','manifest.json']and f.extractfile('estimator.py').read()==src.read_bytes()
gate=D/'RELEASE_GATE_OFFICIAL120_v1.json';assert read(gate)['passed']and read(gate)['candidate_sha256']==h
paths=[science,retire,grant,export,native,metadata,policy,ret,src,gate,stage/'stage_a.json',arc,D/'ROOT_VECTOR_EINSUM_REVIEW_v1.json',D/'DELIVERY_READY_v1.json']
evidence={str(x):sha(x)for x in paths}
production=B/'ROOT_PRODUCTION_ACCEPTANCE_OFFICIAL120_v1.json';put(production,dict(approved=True,reviewed_by='opus_lead',executed_by='opus-release',authority='Lead (opus) delegated official120 release finalization after unchanged risk/setup/resource gates and captured-setup native bytes/bill parity; direct same-source science24/OOS8. Lead alone fires.',candidate_sha256=h,normal_pass=True,slow_pass=True,science24_accepted=True,all_owned_PID_PPID_PGID_absent=True,release_gate_passed=True,native35_100_both_modes_bytes_and_bill_exact=True,official_wall_policy=pin(policy),hard_wall_seconds=120,advisory_wall_seconds=100,advisory_wall_pass=False,evidence_sha256=evidence))
evidence[str(production)]=sha(production)
argv=['/Users/matt/arc-white-box-estimation/.venv/bin/python','-B',str(O/'fire/fire_once.py'),r['tag'],str(arc),'Expanded Smirnov exact read-circuit table (167 fixed dispatches) plus R7 first-read setup reallocation on 336342; same-source science24 net -0.210% (OOS8 -0.172%), official120 PRIMARY runtime PASS, captured-setup native/production bytes and bills exact.']
confirmation=D/'RELEASE_CONFIRMED_v1.json';put(confirmation,dict(status='CONFIRMED',approved=True,reviewed_by='opus_lead',executed_by='opus-release',authority='Lead delegated fixed-condition release preparation under official120 policy; lead alone fires.',candidate_path=str(src),candidate_sha256=h,parent_sha256=parent,archive_path=str(arc),archive_sha256=sha(arc),stage_a_pass=True,PRIMARY_pass_under_explicit_official120_policy=True,original100_advisory_exceeded=True,official_wall_policy=pin(policy),science24_net_log_pct=d['means']['net_log_pct'],science24_net_SE=d['means']['net_SE'],science24_MSE_log_pct=d['means']['MSE_log_pct'],science24_bill_log_pct=d['means']['bill_log_pct'],OOS8_net_log_pct=d['OOS8']['net_log_pct'],production_native35_100_both_modes_bytes_and_bill_exact=True,science24_scope='Direct same-source b709 fixed fresh24 (7103816-7103839) vs 2b108; OOS8 last8. No transfer.',fire_argv_for_lead_only=argv,fire_cwd=str(O),submitted=False,evidence_sha256=evidence))
print(json.dumps(dict(status='CONFIRMED',confirmation=pin(confirmation),archive=pin(arc),science24=d['means'],OOS8=d['OOS8'],fire_argv_for_lead_only=argv)))
