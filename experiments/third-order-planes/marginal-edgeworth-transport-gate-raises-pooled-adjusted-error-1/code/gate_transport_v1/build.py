"""Append only a coherent three-leg change to the historical CP inheritance."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
parent=HERE.parent/'weight5_reuse_v1/candidate.py'
source=parent.read_text()
sha=lambda b:hashlib.sha256(b).hexdigest()
expected='a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9'
assert sha(source.encode())==expected
tree=ast.parse(source)
node=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_predict_binary'][-1]
old=ast.get_source_segment(source,node)
replacements=[
    ("            g = aux['gate']", "            g = aux['gate']\n            gcp = _gt_cp_gate(aux)"),
    ('inherited = g[:, None] ** 2 * g[None, :] * cp_pre',
     'inherited = gcp[:, None] ** 2 * gcp[None, :] * cp_pre'),
    ('p = g.astype(fnp.float32)[:, None] * pp',
     'p = gcp.astype(fnp.float32)[:, None] * pp'),
    ('_shared_old384_commit(l, g, alive, selected, shared_old)',
     '_shared_old384_commit(l, gcp, alive, selected, shared_old)')]
new=old
for a,b in replacements:
    assert new.count(a)==1,a
    new=new.replace(a,b,1)
inverse=new
for a,b in reversed(replacements):
    assert inverse.count(b)==1,b
    inverse=inverse.replace(b,a,1)
assert inverse==old
extension='\n\n# Fixed Edgeworth CP-inheritance gate, 2026-09-19.\n'
extension+=(HERE/'kernel.py').read_text()+'\n\n'+new+'\n'
candidate=source+extension
ast.parse(candidate)
assert candidate[:-len(extension)]==source
with (HERE/'candidate.py').open('x') as f:f.write(candidate)
record=dict(parent_sha256=expected,candidate_sha256=sha(candidate.encode()),
            kernel_sha256=sha((HERE/'kernel.py').read_bytes()),
            builder_sha256=sha(Path(__file__).read_bytes()),
            protocol_sha256=sha((HERE/'PROTOCOL.md').read_bytes()),
            parent_byte_prefix_exact=True,source_inverse_exact=True,
            changed_definitions=['_predict_binary'],truth_accessed=False)
with (HERE/'build.json').open('x') as f:json.dump(record,f,indent=2)
print(json.dumps(record))
