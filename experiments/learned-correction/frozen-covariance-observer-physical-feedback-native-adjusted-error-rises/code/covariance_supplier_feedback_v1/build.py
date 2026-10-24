from pathlib import Path
import ast,hashlib,json
import numpy as np

HERE=Path(__file__).resolve().parent;H2=HERE.parent
prior=H2/'recurrent_covariance_observer_v1'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def pin(p):return dict(path=str(p.resolve()),sha256=sha(p),bytes=p.stat().st_size)
parent=H2/'candidate_old_both_tail_trace_bounded_v6.py'
assert sha(parent)=='5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
model=prior/'model_reset.npz';norm=prior/'normalizers.npz'
assert sha(model)=='f407419c99d34a219acd8aa960cbdcf1b08be4fb44410e8eb7dcb51afec80052'
assert sha(norm)=='63ee82b3ebde58a99508386ace7a3f460e69052de384e6e4247a44237ecb9983'
payload={}
for p in (model,norm):
    with np.load(p,allow_pickle=False) as z:
        for key in z.files:
            assert key not in payload and z[key].dtype==np.float32
            assert np.isfinite(z[key]).all()
            payload[key]=z[key].tolist()
original=parent.read_text();needle='            means.append(m)\n            if l == depth - 1:'
tap='            C = _cf_feedback(l,m,C)\n'
assert original.count(needle)==1
body=original.replace(needle,tap+needle);assert body.replace(tap,'')==original
fragment=(HERE/'fragment.py').read_text()
source=body+'\n_CF_PAYLOAD = '+repr(payload)+'\n'+fragment
ast.parse(source)
candidate=HERE/'candidate.py'
with candidate.open('x') as f:f.write(source)
disabled=HERE/'disabled.py'
with disabled.open('x') as f:f.write(source+'\n_CF_ENABLED = False\n')
files=[HERE/f for f in ('PROTOCOL.md','fragment.py','build.py','candidate.py','disabled.py')]+[parent,model,norm]
with (HERE/'build.json').open('x') as f:json.dump(dict(source_inverse_exact=True,model_values_unchanged=True,
    parent=pin(parent),candidate=pin(candidate),disabled=pin(disabled),sources=[pin(p) for p in files]),f,indent=2)
print(json.dumps(dict(candidate=pin(candidate),model=pin(model),normalizers=pin(norm))),flush=True)
