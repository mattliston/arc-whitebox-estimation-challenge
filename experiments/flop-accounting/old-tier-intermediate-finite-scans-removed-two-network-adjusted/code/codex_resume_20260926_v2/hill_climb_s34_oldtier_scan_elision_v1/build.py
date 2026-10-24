from pathlib import Path
import ast,json,hashlib
D=Path(__file__).resolve().parent;H=D.parent;P=H/'hill_climb_e1_ynt_psdb_d5_v1/candidate.py';T=H/'hill_climb_s34_gn_pair_reuse_v1/native_screen.json'
def pin(p):
 p=Path(p).resolve();b=p.read_bytes();return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
def write(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
assert pin(P)['sha256']=='42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
s=P.read_text();frag=(D/'fragment.py').read_text();child=s+'\n'+frag;a=ast.parse(s).body;b=ast.parse(child).body
assert len(b)==len(a)+1 and all(ast.dump(x)==ast.dump(y) for x,y in zip(a,b))
compile(child,str(D/'candidate.py'),'exec')
with (D/'candidate.py').open('x') as f:f.write(child)
write(D/'protocol.json',dict(parent=pin(P),prospective=True,scope='native2_only',exact_all16_byte_parity_required=True,
 change='Replace repeated whole-array isfinite/all scans of intermediate old-tier arrays by existing dtype check. Every numeric inference operation and result is unchanged whenever original checks pass.',
 unchanged='All shapes, overflow envelopes, dtypes, final output finite checks in validation, memory/time/swap guards, model parameters and arithmetic stay unchanged.',
 distinction='This changes early failure reporting for nonfinite intermediate values; it is not a claim that intermediate finiteness is mathematically guaranteed.',
 reason='Intermediate scans feed only an exception branch and do not enter predictions. Full fixed-panel output validation remains mandatory, including later serving/official validation before banking.',
 native2_gate='All16 output bytes equal S34; every network improves adjusted cost by pooled>=.02%; zero regressions.',
 final_submission_threshold_pct=1.,no_post_result_gate_change=True))
write(D/'build.json',dict(candidate=pin(D/'candidate.py'),parent=pin(P),fragment=pin(D/'fragment.py'),builder=pin(__file__),protocol=pin(D/'protocol.json'),exact_parent_prefix=True))
def rewrite(x):
 if isinstance(x,dict):return{k:rewrite(v) for k,v in x.items()}
 if isinstance(x,list):return[rewrite(v) for v in x]
 if isinstance(x,str):return x.replace(T.parent.name,D.name)
 return x
cfg=rewrite(json.loads(T.read_text()));cfg['arms']['candidate']['source']=pin(D/'candidate.py')
cfg['provenance']=[pin(x) for x in (P,T,Path(__file__),D/'fragment.py',D/'protocol.json',D/'build.json',H/'official_bank_S34_v1.json')]
cfg['interpretation']='Prospective exact-output cost component: defer intermediate finite scans; numerical and final-output integrity checks stay. Exact all16 byte parity and >=.02% gain/all wins required. Final >=1% bank threshold unchanged.'
write(D/'native_screen.json',cfg);print(json.dumps(dict(candidate=pin(D/'candidate.py'),config=pin(D/'native_screen.json'))))

