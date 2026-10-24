"""Root-delegated pure-view semantics transfer, private8 grade and confirmation.
No models, source changes, retries, extension, upload or submission.
"""
from pathlib import Path
import hashlib,json,os,subprocess,tarfile
D=Path(__file__).resolve().parent;O=D.parents[2];RUN=O/'lead/runs';OLD=RUN/'astra_qhfl_delivery_v1';N=RUN/'astra_hf_view_repair_native_ten_v1';Z=D/'seal'
PY='/Users/matt/arc-white-box-estimation/.venv/bin/python'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def put(p,x):
 with p.open('x')as f:f.write(x if isinstance(x,str)else json.dumps(x,indent=2,allow_nan=False)+'\n')
def pin(p):return dict(path=str(p),sha256=sha(p))
ready=read(D/'DELIVERY_READY_v1.json');S=Path(ready['candidate']['path']);H=ready['candidate']['sha256'];assert sha(S)==H
oldh='19fc42a4d64b6721a91719f87cd4e81eb3b6f6b54aa24aed92ae4a5facdd8f5f'
np=N/'NATIVE_TEN_PARITY_v1.json';n=read(np);nr=Path(n['retirement_path']);ret=read(nr)
assert n['passed']and n['approved']and n['reviewed_by']=='root'and n['candidate_sha256']==H and n['reference_sha256']==oldh
assert n['n']==10 and n['all_actual_retired']and n['all16_byte_identical']and n['all_flops_identical']and n['retirement_sha256']==sha(nr)
assert [x['seed']for x in n['rows']]==list(range(7103248,7103258))
assert ret['all_actual_exits_observed']and ret['all_owned_PID_PPID_PGID_absent']and ret['all_guards_clean']
for p,h in ret['evidence_sha256'].items():assert sha(p)==h
sp=Path(ready['source_ready']['path']);assert sha(sp)==ready['source_ready']['sha256'];sourceproof=read(sp)
assert sourceproof['candidate_sha256']==H and sourceproof['parent_sha256']==oldh
assert all(sourceproof[k]is True for k in ('exact_inverse_to_19fc','all_tiny_outputs_byte_exact_19fc','arithmetic_flops_exact_19fc','changing_values_in_stable_owners_checked','owner_replacement_and_existing_cache_reset_checked','real_arena_alias_checked'))
assert sourceproof['weight_owners_cached']is False
for p,h in sourceproof['evidence_sha256'].items():assert sha(p)==h
production=D/'boundary/ROOT_PRODUCTION_ACCEPTANCE_v1.json';prod=read(production)
assert prod['approved']and prod['reviewed_by']=='root'and prod['candidate_sha256']==H
assert all(prod[k]is True for k in ('normal_pass','slow_pass','science24_accepted','all_owned_PID_PPID_PGID_absent','release_gate_passed'))
for p,h in prod['evidence_sha256'].items():assert sha(p)==h
evidence={str(p):sha(p)for p in (np,nr,sp,production,S,D/'ROOT_VIEW_REPAIR_AUTHORITY_v1.json',N/'SOURCE_PEER_PASS_v2.json',OLD/'ALL_RELEASE_NUMERICAL_RETIREMENT_v1.json')}
for i in range(4):
 p=OLD/f'seal/ROOT_PHASE_RETIREMENT_initial8_s{i}_v1.json';x=read(p)
 assert x['source_sha256']==oldh and x['all_owned_PID_PPID_PGID_absent']and x['no_means_computed']
 for f,h in x['evidence_sha256'].items():assert sha(f)==h
 evidence[str(p)]=sha(p)
transfer=D/'ROOT_PRIVATE8_SEMANTICS_TRANSFER_v1.json'
put(transfer,dict(approved=True,reviewed_by='root',executed_by='gain_frontier',authority='Root explicitly authorizes carry of immutable19fc private8 after pure basic-view repair source proof and exact native-ten all16/chargedFLOP equality; original sign rule unchanged.',candidate_sha256=H,measurement_candidate_sha256=oldh,pure_basic_view_handle_cache_only=True,native_ten_all16_bytes_equal=True,native_ten_charged_FLOPs_equal=True,prediction_and_bill_semantics_unchanged=True,no_private_outcome_selection=True,private8_was_not_graded_before_repaired_PRIMARY=True,evidence_sha256=evidence))
grade=Z/'grade_transferred_qhfl_seal8_v1.py';grant=Z/'ROOT_TRANSFERRED_SEAL_GRADE_GRANT_v1.json'
put(grant,dict(approved=True,reviewed_by='root',executed_by='gain_frontier',action='grade_fixed_qhfl_private8',numerical_slots=0,controller_sha256=sha(grade),seal_controller_sha256=sha(OLD/'seal/finite_qhfl_seal_arms_v1.py'),candidate_sha256=H,measurement_candidate_sha256=oldh,production_root_acceptance=pin(production),semantics_transfer=pin(transfer),phase_retirements={str(i):pin(OLD/f'seal/ROOT_PHASE_RETIREMENT_initial8_s{i}_v1.json')for i in range(4)}))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','VECLIB_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):env[k]='1'
result=subprocess.run([PY,'-B',str(grade),str(grant)],capture_output=True,text=True,check=True,env=env,timeout=120)
put(Z/'PRIVATE8_GRADE_v1.log',result.stdout+result.stderr);print(result.stdout,flush=True)
decision=Z/'DECISION_initial8_v1.json';seal=read(decision)
assert seal['verdict']=='PASS'and seal['mean_log_net']<0,'Original private8 did not pass; no automatic extension or fire.'
stage=O/'release/runs'/('a_'+ready['tag']);archive=stage/'submission.tar.gz';a=read(stage/'stage_a.json')
assert a['passed']and a['candidate_sha256']==H
with tarfile.open(archive)as t:assert sorted(t.getnames())==['estimator.py','manifest.json']and t.extractfile('estimator.py').read()==S.read_bytes()
gate=D/'RELEASE_GATE_v4.json';assert read(gate)['passed']is True
prefire=RUN/'astra_qhfl_root_fire_v1/ROOT_PREFIRE_VIEW_RECEIPT_v1.json'
assert prefire.exists()
firetag='qhflviewtwin_20261015_astra_v1'
description='Hermite group fusion with pure view-cache request repair; exact24/OOS science improvement, PRIMARY runtime PASS, exact native-ten bytes/FLOPs, original private8 negative.'
argv=[PY,'-B',str(O/'fire/fire_once.py'),firetag,str(archive),description]
for p in (transfer,grant,decision,gate,stage/'stage_a.json',archive,prefire):evidence[str(p)]=sha(p)
confirmation=D/'RELEASE_CONFIRMED_v1.json'
put(confirmation,dict(status='CONFIRMED',approved=True,reviewed_by='root',executed_by='gain_frontier',authority='Root delegated routine fixed-condition release confirmation; root alone fires.',candidate_path=str(S),candidate_sha256=H,archive_path=str(archive),archive_sha256=sha(archive),stage_a_pass=True,original_PRIMARY_pass=True,original_private8_pass=True,sealed_mean_log_net=seal['mean_log_net'],sealed_se_log_net=seal['se_log_net'],exact_native_ten_all16_byte_and_FLOP_parity=True,science24_scope='Accepted19fc all24/OOS8 preserved by pure-view source proof and exact repaired native ten; no oldbase rounding-parity substitute.',fire_argv_for_root_only=argv,fire_cwd=str(O),submitted=False,evidence_sha256=evidence))
stamp=subprocess.check_output(['/bin/date','-u','+%FT%TZ'],text=True).strip()
with (O/'TEAM_LOG.md').open('a')as f:f.write(f'[astra] {stamp} My Hermite view repair1c913b92 release confirmation complete: archive26e3b3c4 exact, ordinary PRIMARY risk19.35% normal/slow accepted, repaired fixedten all16 bytes and FLOPs identical to19fc, original private8 adjusted {100*seal["mean_log_net"]:.6f}% under explicit semantics transfer. All owned numerical processes retired. Root alone submits; evidence {confirmation}.\n')
print(json.dumps(dict(status='CONFIRMED',confirmation_path=str(confirmation),confirmation_sha256=sha(confirmation),private8_pct=100*seal['mean_log_net'],fire_argv_for_root_only=argv)),flush=True)
