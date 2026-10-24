
from pathlib import Path
import ast, hashlib, json
D=Path(__file__).resolve().parent; H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=H/'hill_climb_s34_late_c0_95_v1/native_screen.json'
def pin(p):
    p=Path(p).resolve(); b=p.read_bytes()
    return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
    with p.open('x') as f: json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def rewrite(x):
    if isinstance(x,dict): return {k:rewrite(v) for k,v in x.items()}
    if isinstance(x,list): return [rewrite(v) for v in x]
    if isinstance(x,str): return x.replace(T.parent.name,D.name)
    return x
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
assert pin(T)['sha256']=='25405d726b632f51dbc674d09f69946edfe262b051a91d5d85ff85f9e9fb4a36'
parent=P.read_text();frag=(D/'fragment.py').read_text();child=parent+'\n'+frag
old,new=ast.parse(parent).body,ast.parse(child).body
assert len(new)==len(old)+1 and all(ast.dump(a)==ast.dump(b) for a,b in zip(old,new))
scope={'_K31R_TABLE':{i:(.5341766552087044,.5026934507950427) for i in range(1,15)}}
scope['_K31R_TABLE_GN']={i:(.85,.5) for i in range(1,15)}
before={k:dict(v) for k,v in scope.items()}
exec(compile(frag,'<scalar-table>','exec'),scope)
assert abs(.5341766552087044*1.05-.5608854879691396)<2e-16
assert scope['_K31R_TABLE_GN']==before['_K31R_TABLE_GN']
assert all(scope['_K31R_TABLE'][i]==((.5608854879691396,v[1]) if i<=9 else v) for i,v in before['_K31R_TABLE'].items())
with (D/'candidate.py').open('x') as f:f.write(child)
once(D/'protocol.json',dict(baseline=pin(P),fixed_c0=.5608854879691396,nominal_factor=1.05,layers=list(range(1,10)),
    c1_unchanged=True,gn_unchanged=True,parity_layers=[0],roster=[35,100],scope='native2_only',sweep=False,
    hypothesis='Single early-layer increase after prior late-only reduction showed tiny gain; tests whether global fitted dial benefits from early/late differentiation. This is not an estimated native gradient.',
    old_late_failures_preserved=True,exact_duplicate_not_found_in_bounded_source_review=True,
    native2_gates=dict(adjusted_pct_at_most=-.1,wins_min=1,max_per_network_regression_pct=1),
    native8_gates=dict(adjusted_pct_at_most=-.1,wins_min=6,ci_upper_below=0,additional6_negative=True)))
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),fragment=pin(D/'fragment.py'),builder=pin(__file__),protocol=pin(D/'protocol.json'),exact_parent_prefix=True,original_ast_exact=True,scalar_table_check_passed=True))
cfg=rewrite(json.loads(T.read_text()));cfg['arms']['candidate']['source']=pin(D/'candidate.py');cfg['parity_layers']=[0]
cfg['provenance']=[pin(x) for x in (P,T,Path(__file__),D/'fragment.py',D/'build.json',D/'protocol.json',H/'official_bank_S34_v1.json')]
cfg['interpretation']='One fixed early L1-9 c0 increase, c1 and GN untouched; exact S34 prefix. Original TRAIN2/8 gates and resource limits; layer0 parity; no validation or boundary yet.'
once(D/'native_screen.json',cfg)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'))))

