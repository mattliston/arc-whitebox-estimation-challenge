from common import *
import exporter
fit=fit_success()
import numpy as np
assert np.__version__=='2.4.6'
source=PARENT.read_text();before=exporter.literal(source)
initial={k:np.asarray(v,dtype=np.float32) for k,v in before.items()}
with np.load(fit['terminal_payload']['path'],allow_pickle=False) as z:terminal={k:z[k].copy() for k in z.files}
assert set(terminal)==set(initial)
for k,v in terminal.items():
    assert v.dtype==np.float32 and v.shape==initial[k].shape and np.isfinite(v).all()
    if k not in EKEYS:assert v.tobytes()==initial[k].tobytes()
    assert hashlib.sha256(v.tobytes()).hexdigest()==fit['terminal_edge_hashes'][k]
changed=[k for k in EKEYS if terminal[k].tobytes()!=initial[k].tobytes()]
assert changed
outputs={}
for role,values in [('candidate',terminal),('null_candidate',initial)]:
    new_source=exporter.replace_values(source,{k:repr(values[k].tolist()) for k in EKEYS})
    decoded={k:np.asarray(v,dtype=np.float32) for k,v in exporter.literal(new_source).items()}
    assert set(decoded)==set(values)
    assert all(decoded[k].dtype==values[k].dtype and decoded[k].shape==values[k].shape and decoded[k].tobytes()==values[k].tobytes() for k in values)
    path=HERE/(role+'.py')
    with path.open('x') as f:f.write(new_source)
    outputs[role]=pin(path)
write(HERE/'build.json',dict(completed=True,parent=pin(PARENT),**outputs,terminal_payload=fit['terminal_payload'],
    terminal_fit=pin(F/'fit_result.json'),fit_controller=pin(F/'controller_result.json'),
    whole_source_inverses_exact=True,only_five_edge_values_changed=True,changed_edge_keys=changed,
    fixed_literals_and_functions_unchanged=True,fp32_payload_bytes_exact=True,
    native_derivative_certified=False,score_measured=False))
print(dict(completed=True,changed=changed,candidate=outputs['candidate']),flush=True)
