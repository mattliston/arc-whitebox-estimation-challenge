from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;H=D.parent;P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py';T=H/'hill_climb_s34_terminal_full_star_v1/native_screen.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
s=P.read_text();frag=(D/'fragment_cached_v2.py').read_text();child=s+'\n'+frag;before=ast.parse(s).body;after=ast.parse(child).body
assert all(ast.dump(a)==ast.dump(b) for a,b in zip(before,after)) and len(after)==len(before)+3
compile(child,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(child)
write(D/'protocol.json',dict(parent=pin(P),change='Reuse exact existing covariance-channel T1/T4 in immediate GN consumer; cached view follows existing GN cache lifecycle.',
 prospective=True,exact_all16_byte_parity_required=True,forecast_gross_saved_flops=109051904,
 forecast_once_view_cost=2097152,scope='native2_only',no_model_executed=True,
 rationale_for_component_gate='This is a new arithmetic-reuse component requiring exact predictions, so preregister >=.02% cost/adjusted gain with every network improving, rather than approximate-model .1% gate. Final submission threshold remains fully validated >=1%.',
 no_weakened_post_result_gate=True,no_claim_of_global_bitwise_proof=True,
 alias_lifetime='Pair T45:60nn survives GN; GN writes only0:6nn, constants/external arrays. Existing GN state key checks arena/FT/PB and reset clears it.',
 native2_gates=dict(adjusted_pct_at_most=-.02,wins_min=2,max_regression_pct=0),
 native8_gates=dict(adjusted_pct_at_most=-.02,wins_min=8,max_regression_pct=0,ci_upper_below=0)))
write(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),fragment=pin(D/'fragment_cached_v2.py'),builder=pin(__file__),protocol=pin(D/'protocol.json'),exact_parent_prefix=True))
def rewrite(x):
 if isinstance(x,dict):return{k:rewrite(v) for k,v in x.items()}
 if isinstance(x,list):return[rewrite(v) for v in x]
 if isinstance(x,str):return x.replace(T.parent.name,D.name)
 return x
cfg=rewrite(json.loads(T.read_text()));cfg['arms']['candidate']['source']=pin(D/'candidate.py');cfg['parity_layers']=list(range(16))
cfg['provenance']=[pin(x) for x in (P,T,Path(__file__),D/'fragment_cached_v2.py',D/'protocol.json',D/'build.json',H/'official_bank_S34_v1.json')]
for n,st in cfg['stages'].items():
 st['gates']['adjusted_pct_at_most']=-.02;st['gates']['wins_min']=int(n);st['gates']['max_per_network_regression_pct']=0.
cfg['interpretation']='Prospective exact-prediction arithmetic-reuse gate; <=-.02%, all networks improve, no regressions, all16 byte parity. Final >=1% bank gate unchanged.'
write(D/'native_screen.json',cfg)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'))))

