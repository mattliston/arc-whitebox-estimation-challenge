import ast
from common import *
verify();guard_ok('fit')
for e in read(HERE/'native_seal_mm.json')['sources']:assert pin(e['path'])==e
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
fragment=(HERE/'fragment_mm.py').read_text();source=body+'\n_CF_PAYLOAD = '+repr(payload)+'\n'+fragment
ast.parse(source)
with (HERE/'candidate.py').open('x') as f:f.write(source)
with (HERE/'disabled.py').open('x') as f:f.write(source+'\n_CF_ENABLED = False\n')
files=[HERE/x for x in ('PROTOCOL.md','fragment_mm.py','build_mm.py','unit_mm.py','launch_native_mm.py',
    'native_amendment_mm.json','native_seal_mm.json','fit_seal.json',
    'candidate.py','disabled.py','model.npz','normalizers.npz','model_frozen.json')]+[PARENT]
once(HERE/'build.json',dict(source_inverse_exact=True,model_values_unchanged=True,parent=pin(PARENT),
    candidate=pin(HERE/'candidate.py'),disabled=pin(HERE/'disabled.py'),sources=[pin(p) for p in files],
    native_amendment=pin(HERE/'native_amendment_mm.json'),native_source_seal=pin(HERE/'native_seal_mm.json'),
    fit_source_seal=pin(HERE/'fit_seal.json'),gram_route='fully metered bank _mm FP32 RRt',
    training_feature_unchanged=True,single_candidate=True))
once(HERE/'unit_seal.json',dict(sources=[pin(HERE/x) for x in ('unit_mm.py','fit_result.json','build.json','native_seal_mm.json')]))
once(HERE/'endpoint_seal.json',dict(sources=[pin(HERE/x) for x in ('evaluate.py','assess.py','build.json')]))
print(json.dumps(dict(candidate=pin(HERE/'candidate.py'),model=frozen['model'],source_inverse_exact=True)),flush=True)

bank=H2/'candidate_old_both_tail_trace_bounded_v6.py'
once(HERE/'candidates.json',dict(bank=pin(bank),component=pin(PARENT),protocol=pin(HERE/'PROTOCOL.md'),
    arms=[dict(name='global_pair_exact',candidate=pin(HERE/'candidate.py'))],scores_read=False))
