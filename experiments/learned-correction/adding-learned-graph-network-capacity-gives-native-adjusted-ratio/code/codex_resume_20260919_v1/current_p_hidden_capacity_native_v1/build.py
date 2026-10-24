import ast
import base64
import zlib
from common import *
import exporter
fit=fit_success()
import numpy as np
assert np.__version__=='2.4.6'
source=ZERO.read_text();packed=exporter.literal(source)
def decode(value):
    shape,blob=value
    return np.frombuffer(zlib.decompress(base64.b64decode(blob,validate=True)),dtype=np.float32).reshape(shape).copy()
old=[]
for node in ast.parse(PARENT.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='_EDGE_LITERAL':
        old.append(ast.literal_eval(node.value))
assert len(old)==1
initial={k:np.asarray(v,dtype=np.float32) for k,v in old[0].items()}
for k in EKEYS:initial[k]=decode(packed[k])
assert initial['extra_hidden_w'].shape==(64,128) and initial['extra_hidden_b'].shape==(64,)
assert initial['extra_state_w'].shape==(64,64) and initial['extra_state_w'].tobytes()==bytes(64*64*4)
assert sum(initial[k].size for k in EKEYS)==12352
with np.load(fit['terminal_payload']['path'],allow_pickle=False) as z:terminal={k:z[k].copy() for k in z.files}
assert set(terminal)==set(initial)
for k,v in terminal.items():
    assert v.dtype==np.float32 and v.shape==initial[k].shape and np.isfinite(v).all()
    if k not in EKEYS:assert v.tobytes()==initial[k].tobytes()
    assert hashlib.sha256(v.tobytes()).hexdigest()==fit['terminal_edge_hashes'][k]
changed=[k for k in EKEYS if terminal[k].tobytes()!=initial[k].tobytes()]
assert changed
values={k:repr((packed[k][0],base64.b64encode(zlib.compress(terminal[k].tobytes(),9)).decode('ascii'))) for k in EKEYS}
new_source=exporter.replace_values(source,values)
decoded=exporter.literal(new_source)
for k in EKEYS:assert decode(decoded[k]).tobytes()==terminal[k].tobytes()
for k in set(packed)-set(EKEYS):assert decoded[k]==packed[k]
with (HERE/'candidate.py').open('x') as f:f.write(new_source)
with (HERE/'null_candidate.py').open('xb') as f:f.write(DISABLED.read_bytes())
assert (HERE/'null_candidate.py').read_bytes()==DISABLED.read_bytes()==PARENT.read_bytes()
write(HERE/'build.json',dict(completed=True,parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),
    null_candidate=pin(HERE/'null_candidate.py'),zero_source=pin(ZERO),disabled_source=pin(DISABLED),
    native_qualification=pin(QN/'qualification.json'),terminal_payload=fit['terminal_payload'],
    terminal_fit=pin(F/'fit_result.json'),fit_controller=pin(F/'controller_result.json'),
    whole_source_inverses_exact=True,only_three_hidden_values_changed=True,changed_hidden_keys=changed,
    hidden_trainable_values=12352,all_old_edge_arrays_frozen=True,
    fixed_literals_and_functions_unchanged=True,fp32_payload_bytes_exact=True,
    native_derivative_certified=False,score_measured=False))
print(dict(completed=True,changed=changed,candidate=pin(HERE/'candidate.py')),flush=True)
