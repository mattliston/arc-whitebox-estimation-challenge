import ast
from common import *
verify();guard_ok('fit')
for e in read(HERE/'native_seal.json')['sources']:assert pin(e['path'])==e
frozen=read(HERE/'model_frozen.json')
assert frozen['updates']==1680 and frozen['fresh_initialization']
assert not frozen['check_labels_used'] and not frozen['final_mean_targets_used']
payload={}
for name in ('model','normalizers'):
    entry=frozen[name];assert pin(entry['path'])==entry
    with np.load(entry['path']) as z:
        for key in z.files:
            assert key not in payload and z[key].dtype==np.float32 and np.isfinite(z[key]).all()
            payload[key]=z[key].tolist()
original=PARENT.read_text();needle='            means.append(m)\n            if l == depth - 1:'
tap='            C = _cf_feedback(l,m,C)\n'
assert original.count(needle)==1
body=original.replace(needle,tap+needle);assert body.replace(tap,'')==original
fragment=(HERE/'fragment.py').read_text();source=body+'\n_CF_PAYLOAD = '+repr(payload)+'\n'+fragment
ast.parse(source)
with (HERE/'candidate.py').open('x') as f:f.write(source)
with (HERE/'disabled.py').open('x') as f:f.write(source+'\n_CF_ENABLED = False\n')
files=[HERE/x for x in ('PROTOCOL.md','fragment.py','build.py','candidate.py','disabled.py','model.npz','normalizers.npz','model_frozen.json')]+[PARENT]
once(HERE/'build.json',dict(source_inverse_exact=True,model_values_unchanged=True,parent=pin(PARENT),
    candidate=pin(HERE/'candidate.py'),disabled=pin(HERE/'disabled.py'),sources=[pin(p) for p in files]))
once(HERE/'unit_seal.json',dict(sources=[pin(HERE/x) for x in ('unit.py','fit_result.json','build.json')]))
once(HERE/'endpoint_seal.json',dict(sources=[pin(HERE/x) for x in ('evaluate.py','assess.py','build.json')]))
print(json.dumps(dict(candidate=pin(HERE/'candidate.py'),model=frozen['model'],source_inverse_exact=True)),flush=True)
