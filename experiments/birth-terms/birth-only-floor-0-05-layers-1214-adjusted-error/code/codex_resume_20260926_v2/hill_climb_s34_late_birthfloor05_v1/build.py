"""One fixed L12-14 birth-only floor on S34; stdlib source preparation only."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import json

D = Path(__file__).resolve().parent
H = D.parent
P = H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T = H/'hill_climb_s34_terminal_full_star_v1/native_screen.json'

def pin(p):
    p=Path(p).resolve();b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))

def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')

def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    if isinstance(x,str):return x.replace(T.parent.name,D.name)
    return x

assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='be0a6a520dfe81bcc49c7fdcc5bc9baacf9ce90f762c5b4a89cbfa735680fd76'
before='            selected, births = select(_e2_gate_min, gate_floor)   # E2: g is bitwise response_gate (same operations on the same inputs)'
after='            selected, births = select(_e2_gate_min, .05 if l >= 12 else gate_floor)   # Fixed L12-14 birth-only pruning; terminal layer exits before this line.'
source=P.read_text();assert source.count(before)==1 and after not in source
candidate=source.replace(before,after,1);assert candidate.replace(after,before,1)==source
old_tree=ast.parse(source);new_tree=ast.parse(candidate)
assert len(old_tree.body)==len(new_tree.body)
changed=[(a,b) for a,b in zip(old_tree.body,new_tree.body) if ast.dump(a)!=ast.dump(b)]
assert len(changed)==1 and changed[0][0].name==changed[0][1].name=='_predict_binary'
old_assignment=ast.parse(before.strip()).body[0];new_assignment=ast.parse(after.strip()).body[0]
count=0
for a,b in zip(ast.walk(changed[0][0]),ast.walk(changed[0][1])):
    if isinstance(a,ast.Assign) and ast.dump(a)==ast.dump(old_assignment):
        assert isinstance(b,ast.Assign) and ast.dump(b)==ast.dump(new_assignment);count+=1
assert count==1
compile(candidate,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:
    f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P),tofile=str(D/'candidate.py'))))
protocol=dict(schema='s34-late-birthfloor05-v1',root_selected=True,parent=pin(P),
    change='One post-mean selected,births assignment: select(_e2_gate_min,.05 if l>=12 else gate_floor). The terminal branch exits before selection, so this affects L12-14 only.',
    rationale='Restrict newborn pairs in near-linear/dead response tails below5%; their modeled repeated-index plane remains in the recomputed residual star. Three late birth lifetimes may save more work than L14 alone.',
    birth_layers=[12,13,14],birth_floor=.05,original_floor=.01,parity_layers=list(range(13)),
    alive_and_response_selection_unchanged=True,means_computed_before_selection=True,
    selected_and_mask_remain_paired=True,born_and_bd_delta_share_new_mask=True,
    cp_diagonal_and_residual_star_recomputed=True,newborn_signed_pairs_and_cohort_width_match=True,
    compensation_limit='Current modeled K21 plane/diagonal compensation only. Rank32 star projection at future layers does not preserve the complete tensor or future predictions.',
    prior_evidence=[
        dict(source='fable_analyst_v1/RESULT.md:534',change='Birth floor .02 atL1-4',raw_pct=1.4,relationship='Earlier layers and smaller floor; related negative, not this endpoint.'),
        dict(source='fable_analyst_v1/RESULT.md:1312',change='Birth expansion .002 atL12-14',raw_pct=-.17,bill_forecast_pct=.4,relationship='Opposite direction; weak economics.'),
        dict(source='hill_climb_birth_plus32_v1/train16_result.json',change='Add32births atL1-10 onS32',adjusted_pct=1.046220,relationship='Distinct expansion failed; remains closed.')],
    distinction='This changes newborn columns after the mean, not response rows used to read K21.',
    no_additional_floor_or_window_sweep=True,scope='native2_only',roster=[35,100],
    setup_and_arena_unchanged=True,existing_native_gates_unchanged=True,
    broad_qualified=False,serving_qualified=False,submitted=False,model_executed=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),
    protocol=pin(D/'protocol.json'),diff=pin(D/'source_diff.patch'),edits=[dict(before=before,after=after)],
    source_inverse_exact=True,one_function_ast_changed=True,one_assignment_changed=True,
    arrays_imported=False,model_executed=False))
original=json.loads(T.read_text());c=rewrite(copy.deepcopy(original))
c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(13))
c['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'build.json',D/'protocol.json',D/'source_diff.patch',
    H/'official_bank_S34_v1.json',H/'hill_climb_birth_plus32_v1/history_review.json',
    H/'hill_climb_birth_plus32_v1/train16_result.json',H.parent/'fable_analyst_v1/RESULT.md')]
c['interpretation']='One fixed .05 birth-only floor atL12-14 onS34. Alive/read-response choices unchanged; modeled CP/star accounting uses same new selected/mask. L0-12 parity, original N2 gates/resources. No floor/window sweep or inherited qualification.'
assert c['arms']['base']==original['arms']['base'] and c['arms']['parity_parent']==original['arms']['parity_parent']
assert c['arms']['base']['source']==pin(P) and c['prospective_scope']=='native2_only'
assert all(c['stages'][s][k]==original['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
once(D/'native_screen.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),build=pin(D/'build.json'),protocol=pin(D/'protocol.json'))))
