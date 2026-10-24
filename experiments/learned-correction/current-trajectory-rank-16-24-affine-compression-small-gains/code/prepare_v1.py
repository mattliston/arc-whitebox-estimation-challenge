"""Compose verified current-source edge8 and rank16 using the standing stack builder."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess
HERE=Path(__file__).resolve().parent;D=HERE.parent
K=D.parent/'fable_oct19_kc_v1';V=D.parent/'astra_oct20_vm_frontier_v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
utc=lambda:subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip()
def write(p,s):
    with p.open('x') as f:f.write(s)
def jwrite(p,x):write(p,json.dumps(x,indent=2,allow_nan=False)+'\n')
def defs(p):return {n.name:ast.dump(n) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
rankdir=D/'current_rank_distill_v1/native_v1';edge=D/'edge_hidden_nonlinear_v1/native_hidden8_v1'
rank=next(x for x in read(rankdir/'CANDIDATES_v1.json')['candidates'] if x['affine_rank']==16)
assert rank['append_tail_sha256']=='5f977d36f784269cd683a45a9cf1274a8ad1ec0532d0a9be90b9442ed373bb07'
rt=rankdir/Path(rank['append_tail']).name
et=edge/'tail_msg8_edge_hidden8_nonlinear_v1.py'
assert sha(rt)==rank['append_tail_sha256']
assert sha(et)==read(edge/'BUILD_v1.json')['tail_sha256']=='fb7e44209120a89b58da5e89133c903ebfd904ba00548102336c2663278a23c5'
suffixes=[]
for role,base,bank,pin in [
 ('science','candidate_rel_v4a_sci_v1.py','candidate_msg8_teacher_loaderfix_sci_v1.py','777a73b6d13ef0e1e51d70d4c0dc61f21b4c7287813e336c82c37ec7188ba510'),
 ('release','candidate_rel_v4a.py','candidate_msg8_teacher_loaderfix_rel_v1.py','e41095e464be4f5973cae13252a716bc1d85a56bd3ba9d0837e28509c3d08f05')]:
    bp=K/base;sp=V/'msg8_loaderfix_v1'/bank
    assert sha(sp)==pin and sp.read_text().startswith(bp.read_text())
    suffixes.append(sp.read_text()[len(bp.read_text()):])
assert suffixes[0]==suffixes[1]
banktail=HERE/'tail_exact337422_from_v4a_v1.py';write(banktail,suffixes[0])
name='astra_edge8_currentrank16_422_v1'
subprocess.run(['bash',str(K/'make_stack_v1.sh'),name,str(banktail),str(et),str(rt)],check=True)
science=K/('candidate_'+name+'_sci_v1.py');release=K/('candidate_rel_'+name+'.py')
functions=defs(science);rfunc=defs(rankdir/Path(rank['science']['path']).name)
efunc=defs(edge/'candidate_msg8_edge_hidden8_nonlinear_science_v1.py')
for fn in ('_gop2_gnn_step','_vmfr_current_lr_parameters','_vmfr_current_lr_linear'):assert functions[fn]==rfunc[fn]
for fn in ('_message_edge_source','_message_edge_parameters','_g5_message_edge_base','_ecut_raw'):assert functions[fn]==efunc[fn]
record=dict(utc=utc(),reference_submission=337422,kind='EXPLORATORY_CURRENT_RANK16_PLUS_EDGE8',
    make_stack_sha256=sha(K/'make_stack_v1.sh'),script_sha256=sha(__file__),
    components=[dict(path=str(t),sha256=sha(t)) for t in [banktail,et,rt]],
    component_rank_manifest_sha256=sha(rankdir/'CANDIDATES_v1.json'),
    component_edge_manifest_sha256=sha(edge/'BUILD_v1.json'),
    component_tiny_receipts=[dict(path=str(t),sha256=sha(t)) for t in [rankdir/'TINY_OFFICIAL_v1.json',edge/'TINY_EDGE_v1.json']],
    outputs={role:dict(path=str(path),sha256=sha(path)) for role,path in [('science',science),('release',release)]},
    no_added_functions_beyond_verified_components=True,rank_functions_exact=True,edge_functions_exact=True,
    no_claim_of_additive_gain=True,weak_rank_piece_requires_actual_stack_confirmation=True,
    no_auto_submission=True,no_new_fit=True)
jwrite(HERE/'BUILD_v1.json',record)
print(json.dumps(record,indent=2))

