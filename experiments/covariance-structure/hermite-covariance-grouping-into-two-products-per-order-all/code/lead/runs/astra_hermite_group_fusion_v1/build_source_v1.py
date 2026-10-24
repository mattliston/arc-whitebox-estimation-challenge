import ast, hashlib, json
from pathlib import Path
D=Path(__file__).resolve().parent
P=D.parent/'astra_rtt2_prefix_power_source_v1/candidate_prefix_power_QR_ON_HELD_v3.py'
s=P.read_text()
assert hashlib.sha256(P.read_bytes()).hexdigest()=='4d7739a5ad4444be8c20bdd5c5bc86f1cee5536a2c21e956d86dd95e6c83d8d9'
old="_e2_hold.append(fnp.multiply(_ps32(t4 * _r31 if _r31 is not None else t4)[:, None], rho, out=st['T31']))"
new="_astra_hf31_row = _ps32(t4 * _r31 if _r31 is not None else t4); _E2_LAYER['astra_hf31_row'] = _astra_hf31_row; _e2_hold.append(fnp.multiply(_astra_hf31_row[:, None], rho, out=st['T31']))"
assert s.count(old)==1
fragment=(D/'fragment_hermite_group_HELD_v1.py').read_text()
candidate=s.replace(old,new)+'\n\n'+fragment
t=ast.parse(candidate)
assert next(n for n in t.body if isinstance(n,ast.FunctionDef)and n.name=='_nonlinear_pair_batch').lineno==844
assert candidate[:len(s)].replace(new,old)==s
out=D/'candidate_hermite_group_QR_HELD_v1.py'
with out.open('x')as f:f.write(candidate)
r=dict(parent=str(P),parent_sha256=hashlib.sha256(P.read_bytes()).hexdigest(),candidate=str(out),candidate_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),fragment_sha256=hashlib.sha256((D/'fragment_hermite_group_HELD_v1.py').read_bytes()).hexdigest(),exact_parent_inverse=True,inline_preserves_line_count=True,nonlinear_definition_line=844,models=0)
with (D/'SOURCE_BUILD_v1.json').open('x')as f:json.dump(r,f,indent=2)
print(json.dumps(r))
