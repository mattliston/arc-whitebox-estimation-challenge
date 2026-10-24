import ast,resource
import numpy as np
from common import *
seal_check();old=read(FIT/'nonlinear_fit.json');assert old['completed'] and old['updates']==1680 and old['epochs']==40
verify(old['parameters']);verify(old['bank_parameters']);assert not read(FIT/'result.json')['passed']
with np.load(old['parameters']['path'],allow_pickle=False) as z:trained={k:z[k] for k in z.files}
with np.load(old['bank_parameters']['path'],allow_pickle=False) as z:bank={k:z[k] for k in z.files}
shapes={'left':(14,16),'right':(14,16),'edge':(3,16),'bias':(16,),'output':(16,)}
assert set(trained)==set(shapes) and sum(v.size for v in trained.values())==528
assert all(trained[k].shape==s and trained[k].dtype==np.float32 and np.isfinite(trained[k]).all() for k,s in shapes.items())
source=PARENT.read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(q,ast.Name) and q.id=='_EDGE_LITERAL' for q in n.targets))
original=ast.get_source_segment(source,node);literal=ast.literal_eval(node.value)
assert set(literal)==set(bank)
assert all(np.asarray(literal[k],np.float32).tobytes()==bank[k].tobytes() for k in bank)
p={k:trained.get(k,v).copy() for k,v in bank.items()};new_literal={k:v.tolist() for k,v in p.items()}
assert all(np.asarray(new_literal[k],np.float32).tobytes()==p[k].tobytes() for k in p)
assert all(p[k].tobytes()==bank[k].tobytes() for k in bank if k not in trained)
assert sum(v.size for v in p.values())==545
replacement='_EDGE_LITERAL='+repr(new_literal);assert source.count(original)==1
candidate=source.replace(original,replacement,1);ast.parse(candidate)
assert candidate.count(replacement)==1 and candidate.replace(replacement,original,1)==source
with (HERE/'candidate.py').open('x') as f:f.write(candidate)
with (HERE/'parameters.npz').open('xb') as f:np.savez(f,**p)
write(HERE/'bank_literal.json',literal)
write(HERE/'build.json',dict(parent=pin(PARENT),candidate=pin(HERE/'candidate.py'),parameters=pin(HERE/'parameters.npz'),
    bank_literal=pin(HERE/'bank_literal.json'),fitted_parameters=old['parameters'],fit_receipt=pin(FIT/'nonlinear_fit.json'),
    whole_source_inverse_exact=True,replaced_assignment=original,new_assignment=replacement,
    changed_parameter_scalars=528,total_edge_scalars=545,all_normalizers_bit_exact=True,
    old_proxy_gate_still_failed=True,fit_steps_added=0,targets_accessed=False,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
print(dict(candidate=pin(HERE/'candidate.py'),inverse_exact=True,changed_parameter_scalars=528),flush=True)
