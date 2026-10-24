"""Verify exact parameter export; no estimator/model execution."""
from pathlib import Path
import ast, base64, hashlib, json, math, sys, zlib
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D))
from native_parameters import SOURCE,SOURCE_SHA,FROZEN_P,load_s34_arrays,LITERALS
def pin(p):
 p=Path(p).resolve();b=p.read_bytes()
 return dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b))
assert len(sys.argv)==3 and pin(D/'build.json')['sha256']==sys.argv[1]
b=json.loads((D/'build.json').read_text())
assert b['candidate']==pin(D/'candidate.py') and b['parent']==pin(SOURCE)
assert b['parent']['sha256']==SOURCE_SHA and b['source_inverse_exact']
assert b['optimizer_updates']==4 and b['frozen_parameter_bytes_exact']
old=SOURCE.read_text();new=(D/'candidate.py').read_text()
ot=ast.parse(old);nt=ast.parse(new)
assert len(ot.body)==len(nt.body)
names=set(LITERALS)|{'_BD_J1_PHYS','_S19_BD_ADD','_S19_BSCALE','_S19_LAM','_S19_K4BAND','_K31R_TABLE'}
def assignments(t):
 r={}
 for n in t.body:
  if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
   r[n.targets[0].id]=n
 return r
on,nn=assignments(ot),assignments(nt)
assert names<=on.keys() and names<=nn.keys()
allowed={id(on[k]) for k in names}
for x,y in zip(ot.body,nt.body):
 if id(x) not in allowed:assert ast.dump(x)==ast.dump(y)
lines=new.splitlines(keepends=True);ol=old.splitlines(keepends=True)
for name in sorted(names,key=lambda k:nn[k].lineno,reverse=True):
 a,z=on[name],nn[name]
 lines[z.lineno-1:z.end_lineno]=ol[a.lineno-1:a.end_lineno]
assert ''.join(lines)==old
assert ast.dump(on['_K31R_TABLE'].value.key)==ast.dump(nn['_K31R_TABLE'].value.key)
assert ast.dump(on['_K31R_TABLE'].value.generators[0])==ast.dump(nn['_K31R_TABLE'].value.generators[0])
import numpy as np
base,base_dials=load_s34_arrays(np)
assert b['endpoint']==pin(D/'endpoint.npz') and b['dials']==pin(D/'endpoint_dials.json')
with np.load(D/'endpoint.npz',allow_pickle=False) as z:values={k:z[k].copy() for k in z.files}
assert values.keys()==base.keys()
for k,v in values.items():assert v.shape==base[k].shape and v.dtype==np.float32 and np.isfinite(v).all()
for k in FROZEN_P+('msg_w','msg_b','enc0_w','enc0_b','enc2_w','enc2_b','enc4_w','enc4_b'):assert values['p.'+k].tobytes()==base['p.'+k].tobytes()
for k in ['p.head','p.head_b']:assert values[k][[0,15]].tobytes()==base[k][[0,15]].tobytes()
assert values['c'][:,[1,3]].tobytes()==base['c'][:,[1,3]].tobytes()
literals={k:ast.literal_eval(nn[k].value) for k in LITERALS}
decoded={}
for k,(shape,s) in literals['_GNN_LITERAL'].items():
 decoded['p.'+k]=np.frombuffer(zlib.decompress(base64.b64decode(s,validate=True)),dtype=np.float32).reshape(shape)
decoded['a']=np.asarray(literals['_BDNH_A_LITERAL'],dtype=np.float32)
decoded['c']=np.asarray(literals['_BD_COEFFICIENT_LITERAL'],dtype=np.float32)
for k,v in literals['_EDGE_LITERAL'].items():decoded['e.'+k]=np.asarray(v,dtype=np.float32)
for k,v in literals['_MESSAGE_EDGE_LITERAL'].items():decoded['e.context_'+k]=np.asarray(v,dtype=np.float32)
assert decoded.keys()==values.keys()
for k,v in values.items():assert v.shape==decoded[k].shape and v.tobytes()==decoded[k].tobytes()
d=json.loads((D/'endpoint_dials.json').read_text())
assert d.keys()==base_dials.keys() and all(math.isfinite(v) for v in d.values())
assert all(d[k]==base_dials[k] for k in ['X3','c1'])
expected=dict(_BD_J1_PHYS=d['X1'],_S19_BD_ADD=(d['X0'],0.,d['X2'],d['X3']),_S19_BSCALE=d['bscale'],_S19_LAM=(d['s_carry'],d['s_new']),_S19_K4BAND=(d['k4a'],d['k4b'],d['k4c']))
for k,v in expected.items():assert ast.literal_eval(nn[k].value)==v
assert ast.literal_eval(nn['_K31R_TABLE'].value.value)==(d['c0'],d['c1'])
r=dict(passed=True,candidate=b['candidate'],build=pin(D/'build.json'),probe=pin(__file__),parent=pin(SOURCE),endpoint=b['endpoint'],dials=b['dials'],arrays_roundtrip_exact=28,frozen_parameters_exact=True,frozen_trunk_exact=True,source_inverse_exact=True,model_executed=False,numerical_accuracy_claim=False)
with Path(sys.argv[2]).open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r))
