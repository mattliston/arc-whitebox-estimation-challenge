"""Embed authenticated frozen arrays; change only the selected mean callbacks."""
import ast
import json
import numpy as np
from common import HERE,PARENT,MODEL,OLD,read,pin,write,seal_check

seal_check()
assert read(OLD/'unit_fnp.json')['passed']
oldfit=read(OLD/'fit_result.json')
assert oldfit['completed'] and not oldfit['passed']
assert oldfit['optimizer_and_leaves_retired'] and oldfit['reloaded_nontrainable']
with np.load(MODEL,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files}
shapes=dict(W1=(102,32),b1=(32,),W2=(32,),b2=(),mu=(94,),sd=(94,),label_rms=())
assert set(p)==set(shapes)
for k,v in p.items():
    assert v.shape==shapes[k] and v.dtype==np.float32 and np.isfinite(v).all(),k
assert (p['sd']>0).all() and float(p['label_rms'])>0
literal={k:v.tolist() for k,v in p.items()}
for k,v in literal.items():assert np.asarray(v,dtype=np.float32).tobytes()==p[k].tobytes()
source=PARENT.read_text()
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_predict_learned_compact']
assert len(nodes)==1
lines=source.splitlines(keepends=True);n=nodes[0]
original=''.join(lines[n.lineno-1:n.end_lineno]);changed=original
edits=[
('        return fnp.asarray(delta,dtype=fnp.float64)',
 '        delta64 = fnp.asarray(delta,dtype=fnp.float64)\n        if _LMIO_ENABLED and 8 <= l <= 14:\n            delta64 = delta64 + _lmio_source(x,state,l,aux["sigma"])\n        return delta64'),
("        delta, _ = gnn_step(l, x, aux['sigma'], w, state, P)",
 "        delta, final_state = gnn_step(l, x, aux['sigma'], w, state, P)"),
('        return fnp.asarray(delta, dtype=fnp.float64)',
 "        delta64 = fnp.asarray(delta, dtype=fnp.float64)\n        if _LMIO_ENABLED:\n            delta64 = delta64 + _lmio_source(x,final_state,l,aux['sigma'])\n        return delta64")]
for old,new in edits:
    assert changed.count(old)==1
    changed=changed.replace(old,new,1)
restored=changed
for old,new in reversed(edits):
    assert restored.count(new)==1
    restored=restored.replace(new,old,1)
assert restored==original
prefix=''.join(lines[:n.lineno-1])+changed+''.join(lines[n.end_lineno:])
suffix='''

# Fixed frozen local mean source; native endpoint study, natural amplitude one.
_LMIO_ENABLED = True
_LMIO_PARAMETERS = None
_LMIO_LITERAL = '''+repr(literal)+'''

def _lmio_source(features, hidden, layer, sigma):
    global _LMIO_PARAMETERS
    if not 8 <= layer <= 15:
        raise ValueError('Frozen mean source layer')
    if features.shape[1] != 30 or hidden.shape[1] != 64:
        raise ValueError('Frozen mean source features')
    if _LMIO_PARAMETERS is None:
        _LMIO_PARAMETERS = {k:fnp.asarray(v,dtype=fnp.float32) for k,v in _LMIO_LITERAL.items()}
    p = _LMIO_PARAMETERS
    x = (fnp.concatenate([features,hidden],axis=1)-p['mu'])/p['sd']
    one = fnp.asarray([1.0 if j==layer-8 else 0.0 for j in range(8)],dtype=fnp.float32)
    embedding = fnp.ones((int(x.shape[0]),1),dtype=fnp.float32)*one[None,:]
    x = fnp.concatenate([x,embedding],axis=1)
    source = (fnp.maximum(x@p['W1']+p['b1'],0)@p['W2']+p['b2'])*p['label_rms']
    return source*sigma
'''
candidate=prefix+suffix
assert candidate[:-len(suffix)].replace(changed,original,1)==source
ast.parse(candidate)
path=HERE/'candidate.py'
with path.open('x') as f:f.write(candidate)
write(HERE/'build.json',dict(candidate=pin(path),parent=pin(PARENT),model=pin(MODEL),
    inverse_byte_exact=True,only_mean_callbacks_and_new_frozen_helper_changed=True,
    parameter_shapes={k:list(v.shape) for k,v in p.items()},parameter_scalars=sum(v.size for v in p.values()),
    label_rms=float(p['label_rms']),original_train_gate_passed=False,
    original_train_ratio=oldfit['train_ratio'],new_fit_steps=0,truth_arrays_accessed=False))
print(dict(candidate=pin(path),inverse_byte_exact=True,label_rms=float(p['label_rms'])),flush=True)
