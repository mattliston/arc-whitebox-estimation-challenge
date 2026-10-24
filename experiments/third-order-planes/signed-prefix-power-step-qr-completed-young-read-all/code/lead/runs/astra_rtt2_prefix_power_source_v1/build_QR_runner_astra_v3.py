import ast,hashlib,json,subprocess
from pathlib import Path
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=O/'lead/runs/astra_rtt2_prefix_power_source_v1';R=O/'lead/runs/astra_rtt2_request_repair_source_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=(D/'finite_rtp_science_HELD_v2.py').read_text();s=old
s=s.replace('Frozen RTP ON-only pilot8 with accepted d401 controls','Frozen QR-power ON-only pilot8 with accepted d401 and exact e406 controls')
s=s.replace("native_rtp_science_judge_HELD_v1.py","native_rtp_science_judge_HELD_v3.py")
s=s.replace("c['reference_control_judge']if role=='reference'else c['parent_control_judge']","item['source_judge_path']if role=='reference'else c['parent_control_judge']")
s=s.replace("expected=original in allowed or(v['old_rank']==0 and original[:3]in{(1024,2048,1024),(1024,4096,1024),(992,6144,1024)})","expected=False  # Exact e406 repair never widens narrowed-prefix D7 eligibility.")
s=s.replace('rtp_pilot8_fresh16_oos8','rtp_qr_e406_pilot8_fresh16_oos8')
s=s.replace('b10fc51d9055082567e43e80c413679726b07f1cc39182cba2e84816f0f6c010','d8967cc36ff90f30e4ae3570b8dc809f1c536532c20e200b76e23ade98bfa9ce')
s=s.replace('000e82aa32de4a4e09aba78274efbe65b9332d2a183e88f353e0bbc4be630776','e406e64b67df590b61f887d2abfb481c7937df4ee069f230b61bed69b324b81c')
def replacefn(text,name,new):
 n=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name==name)
 piece=ast.get_source_segment(text,n);assert text.count(piece)==1
 return text.replace(piece,new.strip())
s=replacefn(s,'reference_metadata','''
def reference_metadata(c):
 p=Path(c['reference_controls_acceptance_path']);assert sha(p)==c['reference_controls_acceptance_sha256']
 r=json.loads(p.read_text());assert r['approved']is True and r['reviewed_by']=='root'
 assert r['status']=='eight_e406_controls_accepted_for_power_pilot'
 assert r['candidate_sha256']==c['candidates']['reference']['sha256']
 assert r['all_owned_PID_PPID_PGID_absent']is True and r['all_guards_clean']is True and r['all16_finite']is True
 assert r['completed_controls']==8 and r['reused_existing_controls']==2 and r['new_controls']==6
 assert r['seeds']==c['identity_groups']['pilot']
 rows=r['evaluations'];assert rows==c['reference_control_reuse']
 assert [e['seed']for e in rows]==c['identity_groups']['pilot']
 assert all(e['role']in ('ON','parent','reference')for e in rows)
 for e in rows:
  for key in ('source_protocol','source_judge'):
   assert sha(e[key+'_path'])==e[key+'_sha256']
   assert c['source_pins'][e[key+'_path']]==e[key+'_sha256']
  for key in ('certificate','guard'):assert sha(e[key])==e[key+'_sha256']
  pp=Path(e['certificate']);assert sha(pp.with_name(pp.stem+'_pred.npz'))==e['prediction_npz_sha256']
 for path,digest in r['evidence_sha256'].items():assert sha(path)==digest
 return r
''')
s=replacefn(s,'science_current','''
def science_current(c,item,role,seed):
 x=current(c,item,role,seed)
 if role=='reference':
  assert item['source_judge_sha256']==sha(item['source_judge_path'])
  expected_path,expected_sha=item['source_protocol_path'],item['source_protocol_sha256']
 elif role=='parent':expected_path,expected_sha=c['parent_controls_protocol_path'],c['parent_controls_protocol_sha256']
 else:expected_path,expected_sha=c['_protocol_path'],c['_protocol_sha256']
 assert x['cert']['science_protocol_sha256']==expected_sha and x['cert']['science_protocol_path']==expected_path
 return x
''')
s=s.replace("r['scope']=='prefix_power_source_tiny_no_native_smoke'","r['scope']=='prefix_power_QR_rebased_source_tiny_no_native_smoke'")
s=s.replace("D/'FIXED_PROPOSAL_astra_v1.json'","D/'FIXED_QR_REBASE_PROPOSAL_astra_v2.json'")
s=s.replace("D/'INTEGRATED_TINY_AND_IMPORT_astra_v2.json'","D/'REPAIRED_QR_TINY_AND_TARIFF_astra_v1.json'")
s=s.replace('incremental_vs_RTT2_negative','incremental_vs_e406_negative')
# No scientific/admission/ownership function changes outside the explicit source/cached-context bindings.
unchanged=['gate_ON','validated_seeds','allocation','control_metadata','home','order','admissions','run','pool','summarize','main','interrupted']
a={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
b={n.name:ast.dump(n,include_attributes=False)for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
assert all(a[k]==b[k]for k in unchanged)
runner=D/'finite_rtp_science_HELD_v3.py';runner.write_text(s) if not runner.exists() else (_ for _ in()).throw(AssertionError('exists'))
judge=D/'native_rtp_science_judge_HELD_v3.py';j=(D/'native_rtp_science_judge_HELD_v1.py').read_text().replace('rtp_pilot8_fresh16_oos8','rtp_qr_e406_pilot8_fresh16_oos8')
with judge.open('x')as f:f.write(j)
proposal=dict(status='HELD_FIXED_SOURCE_NO_MODELS',utc=subprocess.check_output(['/bin/date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),parent_path=str(R/'candidate_rtt2_request_repair_HELD_v2.py'),parent_sha256=sha(R/'candidate_rtt2_request_repair_HELD_v2.py'),candidate_path=str(D/'candidate_prefix_power_QR_ON_HELD_v2.py'),candidate_sha256=sha(D/'candidate_prefix_power_QR_ON_HELD_v2.py'),original_proposal_path=str(D/'FIXED_PROPOSAL_astra_v1.json'),original_proposal_sha256=sha(D/'FIXED_PROPOSAL_astra_v1.json'),one_fixed_step='Y=P-I B^T; V=P^TQ-B(I^TQ); U=PV-I(B^TV); Qnew=orth(U), rank loss retains Q. Same original signed/weighted tail contraction.',QR='Both basis constructions public reduced float64 QR, output float32. Deterministic zero/32eps*m ambiguity fallback to exact inherited twice-reorthogonalized basis. No changes to old prefix/depth dispatch.',native_criteria='Inherited full source/setup/custody/first-affected-layer exact gate; fresh8 actual mean-log-adjusted cost negative versus exact e406 AND d401. Whole runtime/seal/compare_twin_native_v5 criteria retained before any delivery.',control_plan='Reuse exact e4063248/3249 after root acceptance plus six separate controls owned by compute. This runner only8 candidate calls, no extra controls or model qualification.',no_accuracy_claim=True,request_caveat='Typical full-rank saved geometry +4 total requests, not a network bound. Initial/refinement deficient fallback costs more; changed layer geometry can alter total. Production request gate remains mandatory.',reduction_caveat='QR signs/reduction order differ; saved initial-basis projectors exact here. Synthetic refinement and old-prefix subtraction can amplify FP32 cancellation; no whole-network equivalence claim.')
with(D/'FIXED_QR_REBASE_PROPOSAL_astra_v2.json').open('x')as f:json.dump(proposal,f,indent=2);f.write('\n')
c=json.loads((D/'SCIENCE_PROTOCOL_HELD_v2.json').read_text());c['purpose']='rtp_qr_e406_pilot8_fresh16_oos8';c['candidates']['ON'].update(path=str(D/'candidate_prefix_power_QR_ON_HELD_v2.py'),sha256=sha(D/'candidate_prefix_power_QR_ON_HELD_v2.py'));c['candidates']['reference'].update(path=str(R/'candidate_rtt2_request_repair_HELD_v2.py'),sha256=sha(R/'candidate_rtt2_request_repair_HELD_v2.py'))
c['reference_controls_acceptance_path']=str(D.parent/'astra_rtt2_e406_parent6_v1/ROOT_E406_ALL8_ACCEPTANCE_v1.json');c['reference_controls_acceptance_sha256']='PENDING_ROOT_FULL8_ACCEPTANCE';c['reference_control_reuse']=[]
for k in ['reference_controls_protocol_path','reference_controls_protocol_sha256','reference_control_judge']:c.pop(k,None)
c['root_allocation_path']=str(D/'ROOT_SCIENCE_ALLOCATION_QR_v1.json');c['root_allocation_sha256']='PENDING_ROOT'
c['qualification_root_retirement_path']=str(D/'ROOT_SOURCE_ACCEPTANCE_QR_v1.json');c['qualification_root_retirement_sha256']='PENDING_ROOT'
c['run_names']={phase:{str(i):f'run_rtp_science_{phase}_qr_s{i}_v3'for i in range(4)}for phase in ('pilot','remaining')}
c['decision_names']={phase:f'RTP_QR_E406_{phase.upper()}_DECISION_v1.json'for phase in ('pilot','remaining')}
c['source_pins']={p:h for p,h in c['source_pins'].items()if 'astra_read_tail_rank2_native_v1/'not in p and not p.startswith(str(D)+'/')}
# Keep the untouched d401 control/harness dependencies; add only current source/proof dependencies.
for p in [runner,judge,R/'candidate_rtt2_request_repair_HELD_v2.py',D/'candidate_prefix_power_QR_ON_HELD_v2.py',D/'candidate_prefix_power_QR_OFF_HELD_v2.py',D/'fragment_prefix_power_QR_HELD_v2.py',D/'SOURCE_MANIFEST_QR_astra_v2.json',D/'FIXED_QR_REBASE_PROPOSAL_astra_v2.json',D/'REPAIRED_QR_TINY_AND_TARIFF_astra_v1.json',D/'check_repaired_QR_astra_v1.py',D/'FIXED_PROPOSAL_astra_v1.json']:
 c['source_pins'][str(p)]=sha(p)
c['readiness']={'source_frozen':True,'candidate_models_run':0,'missing_before_registration':['root_source_acceptance','root_candidate_allocation','compute_complete8_e406_acceptance_and_item_provenance','postsubmission_actual_four_slot_grant'],'repaired_control_schema':{'status':'eight_e406_controls_accepted_for_power_pilot','completed_controls':8,'reused_existing_controls':2,'new_controls':6,'per_item_extra_keys':['source_protocol_path','source_protocol_sha256','source_judge_path','source_judge_sha256']}}
# Remove stale descriptive metadata which must not claim the former candidate/comparator.
for k in list(c):
 if k not in ('source_pins','native_control_reuse')and isinstance(c[k],str)and ('b10fc' in c[k]or '000e82' in c[k]):c[k]='SUPERSEDED: exact e406 parent plus fixed QR prefix-power'
with(D/'SCIENCE_PROTOCOL_HELD_v3.json').open('x')as f:json.dump(c,f,indent=2);f.write('\n')
root=json.loads((D/'ROOT_SOURCE_ACCEPTANCE_DRAFT_v1.json').read_text());root.update(approved=False,reviewed_by='PENDING_ROOT',candidate_sha256=c['candidates']['ON']['sha256'],scope='prefix_power_QR_rebased_source_tiny_no_native_smoke',fixed_proposal_sha256=sha(D/'FIXED_QR_REBASE_PROPOSAL_astra_v2.json'),tiny_sha256=sha(D/'REPAIRED_QR_TINY_AND_TARIFF_astra_v1.json'))
with(D/'ROOT_SOURCE_ACCEPTANCE_QR_DRAFT_v1.json').open('x')as f:json.dump(root,f,indent=2);f.write('\n')
alloc=json.loads((D/'ROOT_SCIENCE_ALLOCATION_DRAFT_v1.json').read_text());alloc.update(approved=False,reviewed_by='PENDING_ROOT',purpose=c['purpose'],candidate_sha256=c['candidates']['ON']['sha256'])
with(D/'ROOT_SCIENCE_ALLOCATION_QR_DRAFT_v1.json').open('x')as f:json.dump(alloc,f,indent=2);f.write('\n')
g=json.loads((D/'ROOT_PILOT_GRANT_DRAFT_v2.json').read_text());g.update(approved=False,reviewed_by='PENDING_ROOT',purpose=c['purpose'],protocol_sha256='PENDING_REGISTERED_PROTOCOL',runner_sha256=sha(runner),qualification_root_acceptance_path=c['qualification_root_retirement_path'],qualification_root_acceptance_sha256='PENDING_ROOT',reference_controls_acceptance_sha256='PENDING_ROOT_FULL8_ACCEPTANCE',run_names=c['run_names']['pilot'])
with(D/'ROOT_PILOT_GRANT_QR_DRAFT_v1.json').open('x')as f:json.dump(g,f,indent=2);f.write('\n')
# Inert retirement helper: new exact directory, runner and separate outputs, no execution.
ret=(D/'accept_pilot_retirement_HELD_v2.py').read_text().replace('astra_rtp_prefix_power_source_v1','astra_rtt2_prefix_power_source_v1').replace('finite_rtp_science_HELD_v2.py','finite_rtp_science_HELD_v3.py').replace('SCIENCE_PROTOCOL_REGISTERED_v1.json','SCIENCE_PROTOCOL_QR_REGISTERED_v1.json').replace('ROOT_SCIENCE_{stage}_GRANT_v1.json','ROOT_SCIENCE_QR_{stage}_GRANT_v1.json').replace('SCIENCE_{stage}_ACTUAL_EXITS_v1.json','SCIENCE_QR_{stage}_ACTUAL_EXITS_v1.json').replace('SCIENCE_{stage}_MANAGED_SESSIONS_v1.json','SCIENCE_QR_{stage}_MANAGED_SESSIONS_v1.json').replace('SCIENCE_{stage}_PROCESS_SNAPSHOTS_v1.jsonl','SCIENCE_QR_{stage}_PROCESS_SNAPSHOTS_v1.jsonl').replace('SCIENCE_{stage}_OWNER_RETIREMENT_v1.json','SCIENCE_QR_{stage}_OWNER_RETIREMENT_v1.json').replace('ROOT_{stage}_RETIREMENT_v1.json','ROOT_QR_{stage}_RETIREMENT_v1.json')
with(D/'accept_pilot_retirement_QR_HELD_v3.py').open('x')as f:f.write(ret)
for p in [runner,judge,D/'accept_pilot_retirement_QR_HELD_v3.py']:compile(p.read_bytes(),str(p),'exec')
proof=dict(status='HELD_SOURCE_READY_WAIT_EXACT_E406_CONTROLS',unchanged_donor_functions=unchanged,original_judge_numerics_unchanged=True,source_pins_checked=len(c['source_pins']),pending_bindings=c['readiness']['missing_before_registration'],candidate_sha256=c['candidates']['ON']['sha256'],runner_sha256=sha(runner),judge_sha256=sha(judge),protocol_sha256=sha(D/'SCIENCE_PROTOCOL_HELD_v3.json'),no_model_calls=True,no_queues_created=True)
assert all(sha(p)==h for p,h in c['source_pins'].items())
with(D/'QR_RUNNER_SOURCE_READY_astra_v3.json').open('x')as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(proof,indent=2))
