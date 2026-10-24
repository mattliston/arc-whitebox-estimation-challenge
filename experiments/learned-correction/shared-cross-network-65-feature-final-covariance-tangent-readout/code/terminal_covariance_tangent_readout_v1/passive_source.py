import ast,hashlib
from common import SOURCE,PARENT_SHA,sha

def observed_source():
    s=SOURCE.read_text();assert sha(SOURCE)==PARENT_SHA
    tree=ast.parse(s);patches=[];proof=[]
    for name,old,new in [
        ('_predict_binary','                k3 = k3 + 3 * active_v * bpre',
         '                _TCI_TAP.update(k3base=k3,bpre=bpre,radial=radial,active_v=active_v)\n                k3 = k3 + 3 * active_v * bpre'),
        ('gnn_step','    msgs=fnp.concatenate([g*m1,g*m2,g*m3,fnp.log1p(fnp.maximum(e1,0)),fnp.log1p(fnp.maximum(e2,0)),fnp.log1p(fnp.maximum(e3,0))],axis=1)',
         '    if l == 15:\n        _TCI_TAP.update(m1=m1,m2=m2,m3=m3,e1=e1,e2=e2,e3=e3,normalized_x=x)\n    msgs=fnp.concatenate([g*m1,g*m2,g*m3,fnp.log1p(fnp.maximum(e1,0)),fnp.log1p(fnp.maximum(e2,0)),fnp.log1p(fnp.maximum(e3,0))],axis=1)')]:
        node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name][-1]
        original=ast.get_source_segment(s,node);assert original.count(old)==1
        changed=original.replace(old,new,1);assert changed.replace(new,old,1)==original
        ast.parse(changed);patches.append(changed)
        proof.append(dict(function=name,original_sha256=hashlib.sha256(original.encode()).hexdigest(),observed_sha256=hashlib.sha256(changed.encode()).hexdigest(),inverse_byte_exact=True))
    return '\n\n'.join(patches),proof
