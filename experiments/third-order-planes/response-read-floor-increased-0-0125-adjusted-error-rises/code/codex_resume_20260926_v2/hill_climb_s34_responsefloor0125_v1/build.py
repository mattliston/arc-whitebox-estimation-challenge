from pathlib import Path
import ast,hashlib,json
D=Path(__file__).resolve().parent;H=D.parent
P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py'
T=H/'hill_climb_s34_early_c0_95_v1/native_screen.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def once(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
parent=P.read_text();tree=ast.parse(parent)
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary' and n.lineno<3500]
node=nodes[-1];lines=parent.splitlines(keepends=True);old='            response_indices, response_mask = select(response_score, gate_floor)\n'
new='            response_indices, response_mask = select(response_score, .0125 if l >= 1 else gate_floor)\n'
idx=[i for i in range(node.lineno-1,node.end_lineno) if lines[i]==old];assert len(idx)==1
lines[idx[0]]=new;child=''.join(lines);assert child.replace(new,old,1)==parent
compile(child,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(child)
once(D/'protocol.json',dict(baseline=pin(P),roster=[35,100],scope='native2_only',
 hypothesis='Fixed modest increase of response-only floor .01 to .0125 on L1-14 prunes weak near-linear birth/plane rows while retaining original alive mask and omitted-radial compensation. Reduces work if raw-error penalty is small; no claim yet.',
 source_edit=dict(line=idx[0]+1,old=old,new=new),parity_layers=[0],sweep=False,
 history='Bounded searches of H/RESULTS.md and fable analyst/porter/physicist RESULT.md found no exact response-floor0125 test; not a claim of exhaustive novelty.',
 native2_gates=dict(adjusted_pct_at_most=-.1,wins_min=1,max_per_network_regression_pct=1),
 native8_gates=dict(adjusted_pct_at_most=-.1,wins_min=6,ci_upper_below=0,additional6_negative=True)))
once(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),builder=pin(__file__),protocol=pin(D/'protocol.json'),exact_one_line_change=True,compile_passed=True))
def rewrite(x):
 if isinstance(x,dict):return {k:rewrite(v) for k,v in x.items()}
 if isinstance(x,list):return [rewrite(v) for v in x]
 if isinstance(x,str):return x.replace(T.parent.name,D.name)
 return x
cfg=rewrite(json.loads(T.read_text()));cfg['arms']['candidate']['source']=pin(D/'candidate.py');cfg['parity_layers']=[0]
cfg['provenance']=[pin(x) for x in (P,T,Path(__file__),D/'build.json',D/'protocol.json',H/'official_bank_S34_v1.json')]
cfg['interpretation']='One response-only floor increase; all other S34 source bytes exact, alive selection unchanged. Existing N2/N8 gates, no validation or boundary yet.'
once(D/'native_screen.json',cfg)
print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'))))

