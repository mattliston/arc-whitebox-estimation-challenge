from pathlib import Path
import ast,hashlib,json,subprocess
O=Path('/Users/matt/arc-white-box-estimation/pure_closure_writeup/problem_graph/experiments/p2_width1024_depth16/opus_reexam_20261001_v1');D=Path(__file__).resolve().parent
P=O/'engineer/a62_modules/s49/s56ubq1tw_x/candidate.py';old=O/'scout_card/fragment_capacity_ub384_on_UNVALIDATED_astra_v1_v2.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,s):
 with Path(p).open('x') as f:f.write(s)
assert sha(P)=='c149e5800751892eb4d203341154ab2444e824a067ffaf791293a49e035ca0a3'
assert sha(old)=='319e34947699c05a1ccd25bd150ea7f6b9326639efed663dbb80ca11b2c3f9da'
bridge=old.read_text().replace('_ASTRA_UB384_PARENT_B = _ipt_forward_b_ab','_ASTRA_UB384_PARENT_B = _ASTRA_CLUSTER_PARENT_B').replace('def _ipt_forward_b_ab(b, L, k, n):','def _astra_full384_forward_b(b, L, k, n):')
bridge='# HELD current-c149 composition. Original UB384 body unchanged; interpose below current cluster.\nassert _R320_RULE == {7: 384, 10: 320, 13: 384}\nassert _ASTRA_BCC_ON and _ASTRA_CLUSTER_ON\n_R320_RULE = {7: 384, 10: 384, 13: 384}\n\n'+bridge+'\n_ASTRA_CLUSTER_PARENT_B = _astra_full384_forward_b\n'
fragment=D/'fragment_full384_current_bridge_UNVALIDATED_astra_v1.py';put(fragment,bridge)
source=P.read_text()+'\n\n'+bridge
full=D/'candidate_full384_UNVALIDATED_astra_v1.py';put(full,source)
arm=(D/'fragment_recon_prefix_UNVALIDATED_astra_v1.py').read_text();assert arm.count('_ASTRA_RP_ON = True')==1
out=[]
for name,txt in [('off',arm.replace('_ASTRA_RP_ON = True','_ASTRA_RP_ON = False')),('on',arm)]:
 f=D/f'fragment_recon_prefix_{name}_UNVALIDATED_astra_v1.py';p=D/f'candidate_recon_prefix_{name}_UNVALIDATED_astra_v1.py';put(f,txt);put(p,source+'\n\n'+txt)
 out.append({'mode':name,'fragment':str(f),'fragment_sha256':sha(f),'candidate':str(p),'candidate_sha256':sha(p)})
for p in [fragment,full,*D.glob('fragment_recon_prefix*py'),*D.glob('candidate_recon_prefix*py')]:ast.parse(p.read_text())
r={'created_utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),'status':'HELD_SOURCE_ONLY_NO_IMPORT_NO_EVALUATION','base':{'path':str(P),'sha256':sha(P)},'full384':{'path':str(full),'sha256':sha(full),'fragment':str(fragment),'fragment_sha256':sha(fragment)},'reviewed_historical_UB384':{'path':str(old),'sha256':sha(old)},'arms':out,'fixed_physics':{'rank_map':{'7':384,'10':384,'13':384},'prefix_reconstruction_v_rank':320,'layers':[10,11,12,13],'full_read_rank':384,'common_rank':256,'state_and_tail_rank':384,'recent_L13_rank':384,'late_L14_L15_rank':384},'bridge_delta':'UB384 numerical/cache bodies exact; only B function name and captured parent changed to interpose below existing cluster, with final hook assignment. Current bank+quarter prefix is byte-exact. No ordinary alias deleted.','limits':['NO native or torch validation','Full384 current c149 bridge must be independently source-reviewed/import-smoked/qualified','OFF must match full384 all16 predictions, MSE, FLOPs/ops and new own census','No current320/current384 net or additive savings claim','Old joint and early-read failures immutable; independent fresh16 allocation required']}
p=D/'SOURCE_MANIFEST_astra_v1.json';put(p,json.dumps(r,indent=2)+'\n');print(p);print(sha(p))
