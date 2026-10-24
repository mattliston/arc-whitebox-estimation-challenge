"""Correct insertion order without altering the frozen derivative-gate formula."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
D=HERE.parent
parent=D/'lazy_rotation_objective_v1/candidate.py'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=parent.read_text()
assert sha(parent)=='b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
kernel=(D/'gate_transport_v1/kernel.py').read_text()
assert sha(D/'gate_transport_v1/kernel.py')=='975e75b7db644179da421695c04aedb7a011550fd7a38cd27defea1ebcb05078'
bridge='''

def _igt_rebase_history(layer, aux, cp_pre, inherited):
    if layer == 0:
        return aux['gate'], inherited
    q = _gt_cp_gate(aux)
    return q, q[:, None] ** 2 * q[None, :] * cp_pre
'''
with (HERE/'kernel.py').open('x') as f:f.write(kernel+bridge)
tree=ast.parse(source);lines=source.splitlines()
node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'][-1]
old='\n'.join(lines[node.lineno-1:node.end_lineno])
edits=[('            cp_post = (inherited + born) * live',
        '            gcp, inherited_cp = _igt_rebase_history(l, aux, cp_pre, inherited)\n            cp_post = (inherited_cp + born) * live\n            del inherited_cp'),
       ('            p = g.astype(fnp.float32)[:, None] * pp',
        '            p = gcp.astype(fnp.float32)[:, None] * pp'),
       ('            _shared_old384_commit(l, g, alive, selected, shared_old)',
        '            _shared_old384_commit(l, gcp, alive, selected, shared_old)')]
new=old
for a,b in edits:
    assert new.count(a)==1
    new=new.replace(a,b,1)
inverse=new
for a,b in reversed(edits):
    assert inverse.count(b)==1
    inverse=inverse.replace(b,a,1)
assert inverse==old
old_cut=old.index(edits[0][0]);new_cut=new.index('            gcp, inherited_cp =')
assert old[:old_cut]==new[:new_cut]
assert old.index('inherited = g[:, None] ** 2 * g[None, :] * cp_pre')<old.index('innovation =')<old.index('born = born + _bd_plane_delta')<old_cut
extension='\n\n# Isolated history gate after original newborn construction.\n'+kernel+bridge+'\n'+new+'\n'
candidate=source+extension
assert candidate[:-len(extension)]==source
ast.parse(candidate)
with (HERE/'candidate.py').open('x') as f:f.write(candidate)
record=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),parent_sha256=sha(parent),
 candidate_sha256=sha(HERE/'candidate.py'),kernel_sha256=sha(HERE/'kernel.py'),
 build_source_sha256=sha(__file__),protocol_sha256=sha(HERE/'PROTOCOL.md'),
 parent_byte_prefix_exact=True,source_inverse_byte_exact=True,
 original_source_through_newborn_exact=True,gate_formula_byte_exact=True,
 changed_definitions=['_predict_binary'],historical_scope_error='old inherited replacement preceded innovation',
 old_gate_transport_remains_failed=True,full_model_executed=False,truth_accessed=False)
with (HERE/'build.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
