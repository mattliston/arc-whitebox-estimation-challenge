"""Export final fixed fit into current thin parent. Does not run an estimator."""
import ast,base64,zlib
import numpy as np
from common import *

verify();verify('fit_seal.json')
result=read(HERE/'fit_result.json')
assert result['completed'] and result['epochs']==40 and result['updates']==480
assert result['seed']==2026092766 and tuple(result['train'])==tuple(TRAIN)
assert read(HERE/'qualification.json')['passed']
guard=read(HERE/'fit_guard.json')
assert guard['completed'] and guard['whole_group_retired'] and guard['max_swap_growth_bytes']==0
payload={}
for name in ('model','normalizers'):
    rec=result[name];assert pin(rec['path'])==rec
    with np.load(rec['path'],allow_pickle=False) as z:
        for key in z.files:
            value=z[key]
            assert key not in payload and value.dtype==np.float32 and np.isfinite(value).all()
            payload[key]=(list(value.shape),base64.b64encode(zlib.compress(value.tobytes())).decode())
original=PARENT.read_text()
needle='            means.append(m)\n            if l == depth - 1:'
tap='            C = _tpc_feedback(l,m,C)\n'
assert original.count(needle)==1
body=original.replace(needle,tap+needle)
assert body.replace(tap,'')==original
fragment=(HERE/'fragment.py').read_text()
extra='\n_TPC_LITERAL = '+repr(payload)+'\n'+fragment
source=body+extra
ast.parse(source)
assert source[:-len(extra)].replace(tap,'')==original
for name,text in [('candidate.py',source),('disabled.py',source+'\n_TPC_ENABLED = False\n')]:
    with (HERE/name).open('x') as f:f.write(text)
once(HERE/'build.json',dict(parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),
    disabled=pin(HERE/'disabled.py'),fragment=pin(HERE/'fragment.py'),
    model=result['model'],normalizers=result['normalizers'],source_inverse_exact=True,
    scope='One post14 insertion after mean/recentering, before trace/birth/star assembly; original coefficients unchanged.'))
print(json.dumps(dict(candidate=pin(HERE/'candidate.py'),source_inverse_exact=True)),flush=True)
