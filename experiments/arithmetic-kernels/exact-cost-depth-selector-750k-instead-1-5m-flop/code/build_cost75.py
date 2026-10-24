"""One pinned shape-only efficiency change to the graded Cost150 parent."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
parent = HERE/'cost_rule150_scalar.py'
source = parent.read_text()
parent_sha = hashlib.sha256(parent.read_bytes()).hexdigest()
assert parent_sha == 'f06314111d7f41a7c1fbcb8d888a509d0e46222cd31b5ab227ec1e334e582a1f'
node, = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_rule_depth']
lines = source.splitlines(keepends=True)
old = ''.join(lines[node.lineno-1:node.end_lineno])
assert old.count('1500000') == 2
new = old.replace('1500000','750000')
changed = source.replace(old,new)
assert changed != source and changed.replace(new,old) == source
ast.parse(changed)
candidate = HERE/'cost_rule75_scalar.py'
with candidate.open('x') as f:
    f.write(changed)
rec = dict(parent_sha256=parent_sha, candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    source_inverse_exact=True, physics_payloads_changed=False, penalty_flops_per_request=750000,
    labels_used_for_selection=False, official_parent_submission=331584,
    official_parent_adjusted=4.015535762022727e-9, official_parent_raw=2.1687335731712666e-8)
with (HERE/'cost75_build.json').open('x') as f:
    json.dump(rec,f,indent=2)
print(json.dumps(rec),flush=True)
