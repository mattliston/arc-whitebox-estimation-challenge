import ast,base64,zlib
import numpy as np
from common import *
verify();result=read(HERE/'fit_result.json');assert result['passed']
g=read(HERE/'fit_guard.json');assert g['completed'] and g['whole_group_retired'] and g['max_swap_growth_bytes']==0
payload={}
for key in ('model','normalizers'):
    assert pin(result[key]['path'])==result[key]
    with np.load(result[key]['path']) as z:
        for k in z.files:assert k not in payload;payload[k]=z[k]
packed={}
for k,v in payload.items():
    assert v.dtype==np.float32 and np.isfinite(v).all()
    packed[k]=(list(v.shape),base64.b64encode(zlib.compress(v.tobytes())).decode())
s=PARENT.read_text();tree=ast.parse(s);changed=s;inverse=[];proof=[]
edits={
 '_predict_learned_compact':[(
    '    _bdnh_layer = -1\n    state=',
    '    _bdnh_layer = -1\n    _tp_pending = None\n    state='),(
    '        nonlocal state, _bdnh_pending, _bdnh_layer',
    '        nonlocal state, _bdnh_pending, _bdnh_layer, _tp_pending'),(
    '        delta,state=gnn_step(l,x,aux["sigma"],w,state,P)',
    '        delta,state=gnn_step(l,x,aux["sigma"],w,state,P)\n        if _TP_ENABLED and l == 14:\n            _tp_pending=(x,K21,aux["sigma"],state)'),(
    '    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook,',
    '    def terminal_covariance(C):\n        if _tp_pending is None:\n            raise RuntimeError("Missing terminal covariance inputs")\n        return _tp_covariance(C,*_tp_pending)\n    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook,terminal_covariance_hook=terminal_covariance,')],
 '_predict_binary':[(
    'coefficient_hook=None, covariance_hook=None):',
    'coefficient_hook=None, covariance_hook=None, terminal_covariance_hook=None):'),(
    '            if l == depth - 1 and final_hook is not None:\n                v, active_v',
    '            if l == depth - 1 and final_hook is not None:\n                if _TP_ENABLED:\n                    C=terminal_covariance_hook(C)\n                v, active_v')]
}
for name,pairs in edits.items():
    node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
    old=ast.get_source_segment(s,node);new=old
    for a,b in pairs:assert new.count(a)==1,(name,a);new=new.replace(a,b)
    back=new
    for a,b in reversed(pairs):assert back.count(b)==1;back=back.replace(b,a)
    assert back==old and changed.count(old)==1
    changed=changed.replace(old,new);inverse.append((old,new));proof.append(dict(function=name,edits=pairs,exact_inverse=True))
extra='\n_TP_LITERAL='+repr(packed)+'\n_TP_PARAMETERS=None\n_TP_ENABLED=True\n'+(HERE/'fragment.py').read_text()
candidate=changed+extra;ast.parse(candidate);back=candidate[:-len(extra)]
for old,new in reversed(inverse):back=back.replace(new,old)
assert back==s
for name,source in [('candidate.py',candidate),('disabled.py',candidate+'\n_TP_ENABLED=False\n')]:
    with (HERE/name).open('x') as f:f.write(source)
once(HERE/'build.json',dict(parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),disabled=pin(HERE/'disabled.py'),model=result['model'],normalizers=result['normalizers'],fragment=pin(HERE/'fragment.py'),exact_inverse=True,proof=proof))
print(json.dumps(dict(candidate=pin(HERE/'candidate.py'),source_inverse=True)),flush=True)
