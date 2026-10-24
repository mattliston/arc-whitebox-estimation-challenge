"""Only output-coefficient literals differ from the exact frozen parent."""
import ast,copy
import numpy as np
from common import HERE,PARENT,EPS,read,write,pin,seal_check
seal_check();source=PARENT.read_text();lines=source.splitlines(keepends=True)
node=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='_EDGE_LITERAL' for t in n.targets))
old=''.join(lines[node.lineno-1:node.end_lineno]);assert source.count(old)==1
literal=ast.literal_eval(node.value);parent={k:np.asarray(v,dtype=np.float32) for k,v in literal.items()}
assert parent['output'].shape==(16,) and np.isfinite(parent['output']).all()
(HERE/'variants').mkdir(exist_ok=False);rows=[]
settings=[(j,'plus',1) for j in range(16)]+[(j,label,factor) for j in [0,8] for label,factor in [('minus',-1),('double',2)]]
for j,label,factor in settings:
    new_literal=copy.deepcopy(literal);value=parent['output'].copy();value[j]=np.float32(value[j]+factor*EPS)
    new_literal['output']=value.tolist();new='_EDGE_LITERAL = '+repr(new_literal)+'\n'
    changed=source.replace(old,new,1);assert changed.replace(new,old,1)==source
    ast.parse(changed)
    for k,v in new_literal.items():
        a=np.asarray(v,np.float32);b=parent[k];assert a.shape==b.shape
        if k=='output':assert a.tobytes()==value.tobytes() and np.delete(a,j).tobytes()==np.delete(b,j).tobytes()
        else:assert a.tobytes()==b.tobytes()
    path=HERE/'variants'/f'head{j:02d}_{label}.py'
    with path.open('x') as f:f.write(changed)
    rows.append(dict(coordinate=j,label=label,factor=factor,epsilon=EPS,actual_delta=float(value[j])-float(parent['output'][j]),source=pin(path),inverse_source_exact=True,only_one_fp32_coefficient_changed=True))
write(HERE/'build.json',dict(parent=pin(PARENT),variants=rows,epsilon=EPS,base_head=parent['output'].tolist(),new_targets_accessed=False))
print(dict(variants=len(rows),all_source_inverses_exact=True,all_other_parameter_bytes_exact=True),flush=True)
