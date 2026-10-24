"""Change only the seed of existing later shared-basis joins."""
import ast
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
parent=HERE/'cost_rule75_scalar.py'
source=parent.read_text()
parent_sha=hashlib.sha256(parent.read_bytes()).hexdigest()
assert parent_sha=='5f66f94afdd2909584527a299c3989377d374289b8f5e1f379047b4176c7521c'
nodes=[n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_shared_old384_push']
assert len(nodes)==1
node=nodes[0]
lines=source.splitlines(keepends=True)
old=''.join(lines[node.lineno-1:node.end_lineno])
line='        Omega = w32[previous, :rank]\n'
assert old.count(line)==1
new=old.replace(line,'        Omega = Qp if old_columns else w32[previous, :rank]\n')
changed=source.replace(old,new)
assert changed!=source and changed.replace(new,old)==source
ast.parse(changed)
out=HERE/'warm_basis_cost75.py'
with out.open('x') as f:f.write(changed)
rec=dict(parent_sha256=parent_sha,candidate_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
    source_inverse_exact=True,modified_function='_shared_old384_push',
    changed_rule='Omega=Qp on L6..L14; original physical-weight seed at L5',
    heads_changed=False,added_matrix_state=False,labels_used_for_selection=False)
with (HERE/'warm_basis_build.json').open('x') as f:json.dump(rec,f,indent=2)
print(json.dumps(rec),flush=True)
