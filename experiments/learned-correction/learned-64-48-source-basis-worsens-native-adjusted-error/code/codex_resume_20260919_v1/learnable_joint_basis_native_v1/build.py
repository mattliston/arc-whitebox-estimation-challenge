"""Export all five trained JR values while preserving every other source byte."""
import ast,base64,zlib
from common import *
import exporter
fit=fit_success()
import numpy as np
assert np.__version__=='2.4.6'
source=ZERO.read_text();packed=exporter.literal(source)
def decode(value):
    shape,blob=value
    return np.frombuffer(zlib.decompress(base64.b64decode(blob,validate=True)),dtype='<f4').reshape(shape).copy()
assert set(packed)==set(EXPORT_KEYS)==set(EXPORT_SHAPES)
initial_jr={key:decode(value) for key,value in packed.items()}
for key,value in initial_jr.items():assert value.shape==EXPORT_SHAPES[key] and value.dtype==np.float32 and np.isfinite(value).all()
for key in ('theta','phi','head','head_b'):assert initial_jr[key].tobytes()==bytes(initial_jr[key].nbytes)
assert np.array_equal(initial_jr['basis'],np.eye(64,48,dtype=np.float32))
wanted=('_GNN_LITERAL','_EDGE_LITERAL','_MESSAGE_EDGE_LITERAL','_BDNH_A_LITERAL','_BD_COEFFICIENT_LITERAL')
literals={}
for node in ast.parse(PARENT.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id in wanted:
        key=node.targets[0].id;assert key not in literals;literals[key]=ast.literal_eval(node.value)
assert set(literals)==set(wanted)
initial={k:np.asarray(v,dtype=np.float32) for k,v in literals['_EDGE_LITERAL'].items()}
for key in ('left','right'):initial['context_'+key]=np.asarray(literals['_MESSAGE_EDGE_LITERAL'][key],dtype=np.float32)
initial.update({'jr_'+k:v for k,v in initial_jr.items()})
assert set(EKEYS)=={'jr_'+k for k in EXPORT_KEYS} and sum(initial[k].size for k in EKEYS)==3169
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
changed=[key for key in EKEYS if terminal[key].tobytes()!=initial[key].tobytes()]
assert changed and changed==fit['changed_trainable_keys']
values={key:repr((packed[key][0],base64.b64encode(zlib.compress(terminal['jr_'+key].tobytes(),9)).decode('ascii'))) for key in EXPORT_KEYS}
new_source=exporter.replace_values(source,values);exported=exporter.literal(new_source)
for key in EXPORT_KEYS:
    actual=decode(exported[key]);assert actual.shape==EXPORT_SHAPES[key] and actual.tobytes()==terminal['jr_'+key].tobytes()
with (HERE/'candidate.py').open('x') as f:f.write(new_source)
with (HERE/'null_candidate.py').open('xb') as f:f.write(PARENT.read_bytes())
assert (HERE/'null_candidate.py').read_bytes()==PARENT.read_bytes()
write(HERE/'build.json',dict(completed=True,parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),
    null_candidate=pin(HERE/'null_candidate.py'),zero_source=pin(ZERO),disabled_source=pin(DISABLED),
    native_qualification=pin(QN/'qualification.json'),terminal_payload=fit['terminal_payload'],
    terminal_fit=pin(F/'fit_result.json'),fit_controller=pin(F/'controller_result.json'),
    whole_source_inverses_exact=True,only_jr_parameter_values_changed=True,allowed_jr_keys=list(EXPORT_KEYS),
    changed_jr_keys=[key.removeprefix('jr_') for key in changed],trainable_keys=list(EKEYS),trainable_values=3169,
    exported_values={key:dict(sha256=digest(terminal['jr_'+key]),shape=list(EXPORT_SHAPES[key])) for key in EXPORT_KEYS},
    all_original_bank_values_frozen=True,fixed_literals_and_functions_unchanged=True,fp32_payload_bytes_exact=True,
    null_is_byte_exact_current_bank=True,native_derivative_certified=False,score_measured=False))
print(dict(completed=True,exported_keys=list(EXPORT_KEYS),candidate=pin(HERE/'candidate.py')),flush=True)
