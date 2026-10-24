"""Assemble recovered, unshipped small changes without altering their sources."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
P2=HERE.parent
source=HERE/'candidate_r8_literals_v1.py'
src=source.read_text()
dials_path=P2/'fable_physicist_v1/refit6b/s6b_loopR8s_joint_3ep/dials_epoch1.json'
d=json.loads(dials_path.read_text())
new_values={
    '_BD_J1_PHYS': d['X1'],
    '_S19_BD_ADD': (d['X0'],0.0,d['X2'],d['X3']),
    '_S19_BSCALE': d['bscale'],
    '_S19_LAM': (d['s_carry'],d['s_new']),
    '_S19_K4BAND': (d['k4a'],d['k4b'],d['k4c'])}
lines=src.splitlines()
nodes=ast.parse(src).body
block_line=next(i+1 for i,line in enumerate(lines) if line.startswith('# PT-33 S19 dials'))
for name,value in new_values.items():
    found=[n for n in nodes if isinstance(n,ast.Assign) and n.lineno>block_line and len(n.targets)==1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id==name]
    assert len(found)==1 and found[0].lineno==found[0].end_lineno, name
    lines[found[0].lineno-1]=f'{name} = {value!r}'
found=[i for i,s in enumerate(lines) if s.startswith('_K31R_TABLE = {l: (')]
assert len(found)==1
lines[found[0]]=f"_K31R_TABLE = {{l: ({d['c0']!r}, {d['c1']!r}) for l in _K31R_TABLE}}   # dialed pair-map row factor"
out='\n'.join(lines)+'\n'
old_ast=ast.parse(src).body;new_ast=ast.parse(out).body
changes=[]
for a,b in zip(old_ast,new_ast):
    if ast.dump(a)!=ast.dump(b):
        assert isinstance(a,ast.Assign) and a.targets[0].id in {*new_values,'_K31R_TABLE'}
        changes.append(a.targets[0].id)
assert len(old_ast)==len(new_ast) and len(changes)==6
old=P2/'fable_engineer_v1/ksplit_fragment_v5w.py'
new=P2/'fable_engineer_v1/ksplit_fragment_v6_final.py'
assert out.count(old.read_text())==1
out=out.replace(old.read_text(),new.read_text())
treuse=P2/'fable_engineer_v1/treuse_fragment_v2.py'
p6=P2/'fable_engineer2_v1/depth7/p6_fragment_v2.py'
assert out.count(treuse.read_text())==1
out=out.replace(treuse.read_text(),p6.read_text()+treuse.read_text())
uv=P2/'fable_engineer2_v1/uvsplit/uvp_fragment_v0c.py'
out+=uv.read_text()
ast.parse(out)
target=HERE/'candidate_bundle_precore_v1.py'
with target.open('x') as f:f.write(out)
components=[source,dials_path,old,new,treuse,p6,uv]
manifest=dict(candidate=str(target),candidate_sha256=hashlib.sha256(out.encode()).hexdigest(),
    components={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in components},
    changes=['loopR8s epoch 1 literals and matched dials','k-split v6 replacing v5w in place','p6 v2 before transform reuse','L15-only u/v common truncation at 256'],
    status='screening only; no new prebind census or submission qualification yet')
with (HERE/'bundle_manifest_v1.json').open('x') as f:json.dump(manifest,f,indent=2)
print(json.dumps({k:v for k,v in manifest.items() if k!='components'}))
