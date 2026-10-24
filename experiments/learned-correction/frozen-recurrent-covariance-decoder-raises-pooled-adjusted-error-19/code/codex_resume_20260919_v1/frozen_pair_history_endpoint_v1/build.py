import ast,base64,resource,zlib
import numpy as np
from common import *
seal_check()
with np.load(MODEL,allow_pickle=False) as z:q={k:z[k].copy() for k in z.files}
with np.load(P0,allow_pickle=False) as z:p={k:z[k].copy() for k in z.files if k!='first'}
assert q['theta'].shape==(3,65,65) and q['feature_scale'].shape==(65,)
assert np.array_equal(q['theta'],q['theta'].transpose(0,2,1))
assert all(np.isfinite(v).all() for v in [*q.values(),*p.values()])
assert np.min(q['feature_scale'])>0 and float(q['sigma_scale'])>0 and float(q['cov_scale'])>0
assert all(v.dtype==np.float32 for v in p.values())
assert q['theta'].dtype==q['feature_scale'].dtype==np.float32
q={k:np.asarray(v,dtype=np.float32) for k,v in q.items()}
np.savez(HERE/'parameters.npz',**q)
def packed(a):return {k:(list(v.shape),base64.b64encode(zlib.compress(v.tobytes(order='C'))).decode()) for k,v in a.items()}
def audit_pack(raw,pack):
    for k,(shape,blob) in pack.items():
        a=np.frombuffer(zlib.decompress(base64.b64decode(blob,validate=True)),dtype=np.float32).reshape(shape)
        assert a.dtype==raw[k].dtype and a.tobytes()==raw[k].tobytes()
pl,ql=packed(p),packed(q);audit_pack(p,pl);audit_pack(q,ql)
source=PARENT.read_text();tree=ast.parse(source)
node=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='_predict_binary'][-1]
lines=source.splitlines(keepends=True);original=''.join(lines[node.lineno-1:node.end_lineno]);changed=original
edits=[('        shared_old = _shared_old384_state()\n','        shared_old = _shared_old384_state()\n        history_state = None\n'),
       ('            if _EDGE_ENABLED and 8 <= l <= 14:\n                C = C + _edge_source(l,m,K21,aux)\n',
        '            if _HISTORY_ENABLED:\n                history_state = _history_state_step(l,m,C,K21,aux,W[l],history_state)\n            if _EDGE_ENABLED and 8 <= l <= 14:\n                C = C + (_history_covariance(history_state,aux) if _HISTORY_ENABLED else _edge_source(l,m,K21,aux))\n')]
for a,b in edits:assert changed.count(a)==1;changed=changed.replace(a,b,1)
restored=changed
for a,b in reversed(edits):assert restored.count(b)==1;restored=restored.replace(b,a,1)
assert restored==original
prefix=''.join(lines[:node.lineno-1])+changed+''.join(lines[node.end_lineno:])
assert prefix.replace(changed,original,1)==source
extra='\n_HISTORY_P0_LITERAL='+repr(pl)+'\n_HISTORY_MODEL_LITERAL='+repr(ql)+'\n'+(HERE/'production.py').read_text()
candidate=prefix+extra;ast.parse(candidate)
with (HERE/'candidate.py').open('x') as f:f.write(candidate)
assert candidate[:-len(extra)].replace(changed,original,1)==source
write(HERE/'build.json',dict(completed=True,candidate=pin(HERE/'candidate.py'),parent=pin(PARENT),
    model=pin(MODEL),P0=pin(P0),parameters=pin(HERE/'parameters.npz'),edits=edits,
    exact_source_inverse=True,all_tensor_bytes_qualified=True,scalar_rounding='original Torch FP32 scalar law',
    trainable_parameters=0,model_not_historically_hash_sealed=True,
    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(built=True,candidate=pin(HERE/'candidate.py')),flush=True)
