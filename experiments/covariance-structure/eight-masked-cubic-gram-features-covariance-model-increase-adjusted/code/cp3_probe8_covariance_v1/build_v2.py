"""Embed one frozen CP8 supplier and probes into the pinned thin parent."""
import ast
from common import *

verify();guard_ok('fit')
for seal in ('fit_seal.json','native_seal_v2.json'):
    for e in read(HERE/seal)['sources']:assert pin(e['path'])==e
frozen=read(HERE/'model_frozen.json')
assert frozen['updates']==1680 and frozen['epochs']==40 and frozen['seed']==2026092767
assert frozen['fresh_initialization'] and not frozen['check_labels_used'] and not frozen['final_mean_targets_used']
assert frozen['feature_manifest']==pin(HERE/'feature/complete.json')
payload={}
for name in ('model','normalizers'):
    entry=frozen[name];assert pin(entry['path'])==entry
    with np.load(entry['path'],allow_pickle=False) as z:
        if name=='model':assert sum(z[k].size for k in z.files)==6209
        for key in z.files:
            assert key not in payload and z[key].dtype==np.float32 and np.isfinite(z[key]).all()
            payload[key]=z[key].tolist()
probes=checked_npz(pin(HERE/'probes.npz'))
assert set(probes)=={'Omega','Psi'}
for p in probes.values():
    assert p.shape==(1024,8) and p.dtype==np.float32 and np.all((p==-1)|(p==1))
assert sha(DEPLOY_PARENT)==DEPLOY_SHA
original=DEPLOY_PARENT.read_text()
needle='            means.append(m)\n            if l == depth - 1:'
tap='            C = _cp8_feedback(l,m,C)\n'
assert original.count(needle)==1 and tap not in original
body=original.replace(needle,tap+needle)
assert body.replace(tap,'')==original
fragment=(HERE/'fragment_v2.py').read_text()
suffix='\n_CP8_PAYLOAD = '+repr(payload)+'\n_CP8_PROBE_PAYLOAD = '+repr({k:v.tolist() for k,v in probes.items()})+'\n'+fragment
source=body+suffix
assert source[:-len(suffix)].replace(tap,'')==original
ast.parse(source)
with (HERE/'candidate.py').open('x') as f:f.write(source)
with (HERE/'disabled.py').open('x') as f:f.write(source+'\n_CP8_ENABLED = False\n')
files=[HERE/x for x in ('PROTOCOL.md','common.py','fragment_v2.py','build_v2.py','candidate.py','disabled.py',
    'model.npz','normalizers.npz','probes.npz','model_frozen.json','fit_result.json')]+[DEPLOY_PARENT]
once(HERE/'build.json',dict(source_inverse_exact=True,model_values_unchanged=True,probe_values_unchanged=True,
    parent=pin(DEPLOY_PARENT),training_parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),
    disabled=pin(HERE/'disabled.py'),model=frozen['model'],normalizers=frozen['normalizers'],
    probes=pin(HERE/'probes.npz'),sources=[pin(p) for p in files],
    products='ordinary FP32 fnp GEMMs; no dense RR Gram or D6/D7 supplier variant'))
once(HERE/'unit_seal.json',dict(sources=[pin(HERE/x) for x in
    ('unit_v2.py','fit_result.json','build.json','feature/complete.json')]))
print(json.dumps(dict(event='build_complete',candidate=pin(HERE/'candidate.py'),
    source_inverse_exact=True,parameters=6209)),flush=True)
