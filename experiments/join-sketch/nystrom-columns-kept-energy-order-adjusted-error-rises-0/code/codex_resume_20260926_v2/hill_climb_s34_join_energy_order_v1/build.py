"""One active Nyström outer-sort removal on exact S34; no model execution."""
from pathlib import Path
import ast,copy,difflib,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=H/'hill_climb_s34_ynt_zonal3_v1/native_screen.json'
OLD=H.parent/'fable_engineer2_v1/uvsplit'
def pin(p):
    p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list):return [rewrite(v) for v in x]
    return x.replace(T.parent.name,D.name) if isinstance(x,str) else x
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='442bf11d03eebcb1ae3b6090d2280d57984386573551e665ae4a71f57d8c18a6'
source=P.read_text();tree=ast.parse(source)
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push'
    and any(isinstance(x,ast.Name) and x.id=='_need_core' for x in ast.walk(n))]
assert len(functions)==1;function=functions[0]
assignments=[x for x in ast.walk(function) if isinstance(x,ast.Assign)
    and any(isinstance(y,ast.Name) and y.id=='_jn_rows' for y in x.targets)]
assert len(assignments)==1;assignment=assignments[0]
assert isinstance(assignment.value,ast.Call) and ast.unparse(assignment.value.func)=='fnp.sort'
assert len(assignment.value.args)==1 and not assignment.value.keywords
before=ast.get_source_segment(source,assignment)
after='_jn_rows = '+ast.get_source_segment(source,assignment.value.args[0])
fn_before=ast.get_source_segment(source,function);assert fn_before.count(before)==1
fn_after=fn_before.replace(before,after,1);assert source.count(fn_before)==1
candidate=source.replace(fn_before,fn_after,1)
assert candidate.count(fn_after)==1 and candidate.replace(fn_after,fn_before,1)==source
newtree=ast.parse(candidate);assert len(newtree.body)==len(tree.body)
changed=[a.name for a,b in zip(tree.body,newtree.body) if ast.dump(a)!=ast.dump(b)]
assert changed==['_shared_old384_push']
newfn=next(n for n in newtree.body if isinstance(n,ast.FunctionDef) and n.lineno==function.lineno)
newassignment=next(x for x in ast.walk(newfn) if isinstance(x,ast.Assign)
    and any(isinstance(y,ast.Name) and y.id=='_jn_rows' for y in x.targets))
assert ast.dump(newassignment.value)==ast.dump(assignment.value.args[0])
assert after in (OLD/'uvp_fragment_v0.py').read_text()
assert before in (OLD/'uvp_fragment_v0c.py').read_text()
with (D/'candidate.py').open('x') as f:f.write(candidate)
with (D/'source_diff.patch').open('x') as f:f.write(''.join(difflib.unified_diff(source.splitlines(True),candidate.splitlines(True),fromfile=str(P),tofile=str(D/'candidate.py'))))
protocol=dict(schema='s34-join-energy-order-v1',root_selected=True,parent=pin(P),
    change='Remove only outer fnp.sort from active Nyström landmark assignment; keep the stable descending-energy argsort prefix as its column order.',
    join_layers=[7,10,13],selected_row_sets_unchanged=True,ranks_and_weights_unchanged=True,
    mechanism='Permuting Y columns preserves its complete exact-arithmetic span but changes leading QR coordinates. Current common256 consumers truncate those coordinates at L10–14 and15.',
    limits='Descending landmark energy does not prove optimal common-tensor approximation. Finite-precision QR/core/consumers can change; native endpoint gates decide.',
    old_core_and_all_other_operations_source_identical=True,parity_layers=list(range(7)),
    history=dict(v0=pin(OLD/'uvp_fragment_v0.py'),v0c=pin(OLD/'uvp_fragment_v0c.py'),v0b=pin(OLD/'uvp_fragment_v0b.py'),
        team_log=pin(H.parent/'fable_refit_v1/TEAM_LOG.md'),analyst_result=pin(H.parent/'fable_analyst_v1/RESULT.md'),
        interpretation='Mechanism previously executed in all-layer common256 UVP v0, raw+.41%/+1.43% with bill−1.343%/−1.613% on35/100. Pair-mean order v0b raw+.57%/+1.10%, ambiguous. v0c restored index-sorted joins. Older full-rank energy ordering had+.001% raw parity on net37 in float64. No isolated ordering-only current late-LCR ancestry study found in bounded review; not novel mathematics.'),
    cost='Removes three rank-length index sorts; no GEMM, shape, arena, setup or value-cache change. No material bill-saving claim.',
    setup_counts=dict(bound=682,lvfuse=649,psdb_fused=9),scope='native2_only',roster=[35,100],
    original_approximate_gates=True,no_order_or_window_sweep=True,source_inverse_exact=True,
    component_only=True,retention_requires_combined_validation=True,
    prospective_continuation='If N2 passes, combine once with retained PP+READ7+scan; require combined own N2, boundary and N8, comparing both S34 and saved cost-combination N8. No redundant individual energy N8. Original gates unchanged; final bank requires at least1% on VALID40.',
    model_executed=False,serving_qualified=False,broad_qualified=False,submitted=False)
once(D/'protocol.json',protocol)
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),protocol=pin(D/'protocol.json'),diff=pin(D/'source_diff.patch'),
    source_inverse_exact=True,changed_parent_functions=changed,original_line=assignment.lineno,
    old_statement=before,new_statement=after,only_outer_sort_removed_ast=True,model_executed=False))
base=json.loads(T.read_text());c=rewrite(copy.deepcopy(base));c['arms']['candidate']['source']=pin(D/'candidate.py');c['parity_layers']=list(range(7))
c['provenance']=[pin(p) for p in (P,T,Path(__file__),D/'build.json',D/'protocol.json',D/'source_diff.patch',OLD/'uvp_fragment_v0.py',OLD/'uvp_fragment_v0c.py',H/'official_bank_S34_v1.json')]
c['interpretation']='One active outer-sort removal; same landmarks/ranks/weights, energy-ordered QR nesting for current late common256 ancestry. Historical bundled adverse evidence disclosed; original N2/8 gates, L0–6 parity, no sweep or broader authority.'
assert c['arms']['base']==base['arms']['base'] and c['arms']['parity_parent']==base['arms']['parity_parent']
assert c['arms']['candidate']['setup']['counts']==protocol['setup_counts'] and c['prospective_scope']=='native2_only'
assert all(c['stages'][s][k]==base['stages'][s][k] for s in ('2','8') for k in ('gates','assessment_limits'))
once(D/'native_screen.json',c)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'),build=pin(D/'build.json'),protocol=pin(D/'protocol.json'))))
