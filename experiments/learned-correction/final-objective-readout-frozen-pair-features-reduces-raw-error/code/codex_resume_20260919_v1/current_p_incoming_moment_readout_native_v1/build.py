"""Fold only terminal Theta into the qualified native output parameter."""
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
    return np.frombuffer(zlib.decompress(base64.b64decode(blob,validate=True)),dtype='<f4').reshape(shape).copy()
assert set(packed)==set(JKEYS)|{'output'}
zero_output=decode(packed['output'])
assert zero_output.shape==(32,3) and zero_output.tobytes()==bytes(96*4)
wanted=('_GNN_LITERAL','_EDGE_LITERAL','_MESSAGE_EDGE_LITERAL','_BDNH_A_LITERAL','_BD_COEFFICIENT_LITERAL')
literals={}
for node in ast.parse(PARENT.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name in wanted:
            assert name not in literals;literals[name]=ast.literal_eval(node.value)
assert set(literals)==set(wanted)
initial={k:np.asarray(v,dtype=np.float32) for k,v in literals['_EDGE_LITERAL'].items()}
for key in ('left','right'):initial['context_'+key]=np.asarray(literals['_MESSAGE_EDGE_LITERAL'][key],dtype=np.float32)
with np.load(SMALL/'fixed_zero_model.npz',allow_pickle=False) as z:fixed_model={k:z[k].copy() for k in z.files}
assert set(fixed_model)==set(packed)
for key in packed:assert fixed_model[key].dtype==np.float32 and fixed_model[key].tobytes()==decode(packed[key]).tobytes()
initial.update({'joint_'+k:fixed_model[k] for k in JKEYS})
initial['joint_theta']=np.zeros((32,3),dtype=np.float32)
initial['joint_output_scale']=np.load(SMALL/'output_column_scale.npy',allow_pickle=False)
assert initial['joint_output_scale'].shape==(3,) and initial['joint_output_scale'].dtype==np.float32
assert EKEYS==('joint_theta',) and EXPORT_KEYS==('output',)
digest=lambda v:hashlib.sha256(v.tobytes()).hexdigest()
assert {k:digest(decode(v)) for k,v in literals['_GNN_LITERAL'].items()}==fit['fixed_p_hashes']
assert [digest(np.asarray(literals[k],dtype=np.float32)) for k in ('_BDNH_A_LITERAL','_BD_COEFFICIENT_LITERAL')]==fit['fixed_ac_hashes']
assert {k:digest(v) for k,v in initial.items() if k not in EKEYS}==fit['fixed_edge_hashes']
with np.load(fit['terminal_payload']['path'],allow_pickle=False) as z:terminal={k:z[k].copy() for k in z.files}
assert set(terminal)==set(initial)
for key,value in terminal.items():
    assert value.dtype==np.float32 and value.shape==initial[key].shape and np.isfinite(value).all()
    if key not in EKEYS:assert value.tobytes()==initial[key].tobytes()
    assert digest(value)==fit['terminal_edge_hashes'][key]
assert terminal['joint_theta'].tobytes()!=bytes(96*4)
actual_output=(terminal['joint_theta']*terminal['joint_output_scale']).astype(np.float32)
assert actual_output.shape==(32,3) and np.isfinite(actual_output).all() and np.count_nonzero(actual_output)>0
values={'output':repr((packed['output'][0],base64.b64encode(zlib.compress(actual_output.tobytes(),9)).decode('ascii')))}
new_source=exporter.replace_values(source,values);decoded=exporter.literal(new_source)
assert decode(decoded['output']).tobytes()==actual_output.tobytes()
for key in set(packed)-{'output'}:assert decoded[key]==packed[key]
with (HERE/'candidate.py').open('x') as f:f.write(new_source)
with (HERE/'null_candidate.py').open('xb') as f:f.write(PARENT.read_bytes())
assert (HERE/'null_candidate.py').read_bytes()==PARENT.read_bytes()
write(HERE/'build.json',dict(completed=True,parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),
    null_candidate=pin(HERE/'null_candidate.py'),zero_source=pin(ZERO),disabled_source=pin(DISABLED),
    native_qualification=pin(QN/'qualification.json'),terminal_payload=fit['terminal_payload'],
    terminal_fit=pin(F/'fit_result.json'),fit_controller=pin(F/'controller_result.json'),
    whole_source_inverses_exact=True,only_joint_output_value_changed=True,changed_joint_keys=['output'],
    trainable_theta_keys=list(EKEYS),readout_trainable_values=96,
    actual_output_sha256=digest(actual_output),actual_output_shape=list(actual_output.shape),
    theta_sha256=digest(terminal['joint_theta']),column_scale_sha256=digest(terminal['joint_output_scale']),
    all_parent_and_hidden_values_and_normalizers_frozen=True,parameterization_folded_to_FP32=True,
    fixed_literals_and_functions_unchanged=True,fp32_payload_bytes_exact=True,
    null_is_byte_exact_current_bank=True,native_derivative_certified=False,score_measured=False))
print(dict(completed=True,changed_native_entry='output',candidate=pin(HERE/'candidate.py')),flush=True)

