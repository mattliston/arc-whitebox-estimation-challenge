"""One fixed newborn selector; retain every accepted parent byte."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = HERE.parent
parent = D / 'lazy_rotation_objective_v1/candidate.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(parent) == 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
source = parent.read_text()
tree = ast.parse(source)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary'][-1]
original = '\n'.join(source.splitlines()[node.lineno - 1:node.end_lineno])
old = '            selected, births = select(fnp.minimum(g, 1 - g), gate_floor)'
new = '            selected, births = _birth05_select(g)'
assert original.count(old) == 1
changed = original.replace(old, new, 1)
assert changed.replace(new, old, 1) == original
kernel = '''

def _birth05_select(gate):
    return select(fnp.minimum(gate, 1 - gate), 0.05)
'''
extension = '\n\n# Fixed current-Lazy birth-only objective experiment.\n' + kernel + '\n' + changed + '\n'
candidate = source + extension
ast.parse(candidate)
assert candidate[:-len(extension)] == source
assert changed.count('select(response_gate, gate_floor)') == 1
assert changed.count('select(response_score, gate_floor)') == 1
for name, content in [('candidate.py', candidate), ('kernel.py', kernel)]:
    with (HERE / name).open('x') as f:
        f.write(content)
record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
              parent_sha256=sha(parent), candidate_sha256=sha(HERE / 'candidate.py'),
              build_source_sha256=sha(__file__), kernel_sha256=sha(HERE / 'kernel.py'),
              protocol_sha256=sha(HERE / 'PROTOCOL.md'), parent_byte_prefix_exact=True,
              source_inverse_byte_exact=True, changed_definitions=['_predict_binary'],
              active_floor=.01, response_floor=.01, birth_floor=.05,
              learned_payloads_and_all_other_parent_source_unchanged=True,
              historical_birthonly_negative_unchanged=True,
              full_model_executed=False, truth_accessed=False)
with (HERE / 'build.json').open('x') as f:
    json.dump(record, f, indent=2)
print(json.dumps(record))
