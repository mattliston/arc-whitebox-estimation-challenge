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
literals={}
for node in ast.parse(PARENT.read_text()).body:
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name in ('_EDGE_LITERAL','_MESSAGE_EDGE_LITERAL','_BDNH_A_LITERAL','_BD_COEFFICIENT_LITERAL'):
            assert name not in literals;literals[name]=ast.literal_eval(node.value)
assert len(literals)==4
initial={k:np.asarray(v,dtype=np.float32) for k,v in literals['_EDGE_LITERAL'].items()}
for key in ('left','right'):initial['context_'+key]=np.asarray(literals['_MESSAGE_EDGE_LITERAL'][key],dtype=np.float32)
initial['complement_enc_w']=decode(packed['complement_enc_w'])
assert initial['complement_enc_w'].shape==(128,288) and initial['complement_enc_w'].tobytes()==bytes(128*288*4)
assert EKEYS==('complement_enc_w',)
fixed={k:hashlib.sha256(decode(v).tobytes()).hexdigest() for k,v in packed.items() if k not in EKEYS}
assert fixed==fit['fixed_p_hashes']
assert [hashlib.sha256(np.asarray(literals[k],dtype=np.float32).tobytes()).hexdigest() for k in ('_BDNH_A_LITERAL','_BD_COEFFICIENT_LITERAL')]==fit['fixed_ac_hashes']
with np.load(fit['terminal_payload']['path'],allow_pickle=False) as z:terminal={k:z[k].copy() for k in z.files}
assert set(terminal)==set(initial)
for k,v in terminal.items():
    assert v.dtype==np.float32 and v.shape==initial[k].shape and np.isfinite(v).all()
    if k not in EKEYS:assert v.tobytes()==initial[k].tobytes()
    assert hashlib.sha256(v.tobytes()).hexdigest()==fit['terminal_edge_hashes'][k]
assert terminal['complement_enc_w'].tobytes()!=initial['complement_enc_w'].tobytes()
values={k:repr((packed[k][0],base64.b64encode(zlib.compress(terminal[k].tobytes(),9)).decode('ascii'))) for k in EKEYS}
new_source=exporter.replace_values(source,values);decoded=exporter.literal(new_source)
assert decode(decoded['complement_enc_w']).tobytes()==terminal['complement_enc_w'].tobytes()
for k in set(packed)-set(EKEYS):assert decoded[k]==packed[k]
with (HERE/'candidate.py').open('x') as stream:stream.write(new_source)
with (HERE/'null_candidate.py').open('xb') as stream:stream.write(DISABLED.read_bytes())
assert (HERE/'null_candidate.py').read_bytes()==DISABLED.read_bytes()==PARENT.read_bytes()
write(HERE/'build.json',dict(completed=True,parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),
    null_candidate=pin(HERE/'null_candidate.py'),zero_source=pin(ZERO),disabled_source=pin(DISABLED),
    native_qualification=pin(QN/'qualification.json'),terminal_payload=fit['terminal_payload'],
    terminal_fit=pin(F/'fit_result.json'),fit_controller=pin(F/'controller_result.json'),
    whole_source_inverses_exact=True,only_new_encoder_value_changed=True,changed_complement_keys=list(EKEYS),
    complement_trainable_values=36864,fixed_basis_values=9216,all_old_parameters_and_trained_context_frozen=True,
    fixed_literals_and_functions_unchanged=True,fp32_payload_bytes_exact=True,native_derivative_certified=False,score_measured=False))
print(dict(completed=True,changed=list(EKEYS),candidate=pin(HERE/'candidate.py')),flush=True)
