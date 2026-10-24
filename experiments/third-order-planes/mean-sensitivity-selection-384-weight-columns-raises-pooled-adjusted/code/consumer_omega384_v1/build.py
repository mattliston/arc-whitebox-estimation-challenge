"""Two anchored last-definition edits, with a complete byte inverse."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'lazy_rotation_objective_v1/candidate.py'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
raw = PARENT.read_bytes()
assert sha(PARENT) == 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
tree = ast.parse(raw)
lines = raw.splitlines(keepends=True)
edits = []
replacements = {
    '_predict_binary': (
        b'''            omitted = alive & ~response_mask
            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
''',
        b'''            omitted = alive & ~response_mask
            if 5 <= l < 15:
                shared_old['_cs384_omega_indices'] = _cs384_indices(mu, vv)
            pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)
'''),
    '_shared_old384_push': (
        b'''        Omega = w32[previous, :rank]
''',
        b'''        omega_indices = state.pop('_cs384_omega_indices')
        Omega = _cs384_omega(w32, previous, omega_indices)
        del omega_indices
''')
}
for name, (before, after) in replacements.items():
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
    stop = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
    original = raw[start:stop]
    assert original.count(before) == 1
    changed = original.replace(before, after)
    assert changed.count(after) == 1 and changed.replace(after, before) == original
    edits.append((start, stop, original, changed, name))
candidate = raw
for start, stop, original, changed, name in sorted(edits, reverse=True):
    candidate = candidate[:start] + changed + candidate[stop:]
recovered = candidate
for start, stop, original, changed, name in sorted(edits):
    assert recovered[start:start+len(changed)] == changed
    recovered = recovered[:start] + original + recovered[start+len(changed):]
assert recovered == raw
fragment = (HERE / 'consumer_fnp.py').read_bytes()
assert all(isinstance(n, (ast.Expr, ast.FunctionDef)) for n in ast.parse(fragment).body)
delimiter = b'\n\n# BEGIN FIXED LOCAL CONSUMER OMEGA384 SELECTION\n'
candidate += delimiter + fragment
ast.parse(candidate)
with (HERE / 'candidate.py').open('xb') as stream:
    stream.write(candidate)
record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
              parent_sha256=sha(PARENT), candidate_sha256=sha(HERE/'candidate.py'),
              fragment_sha256=sha(HERE/'consumer_fnp.py'), build_source_sha256=sha(__file__),
              protocol_sha256=sha(HERE/'PROTOCOL.md'), whole_source_inverse_byte_exact=True,
              modified_last_definitions=[e[4] for e in sorted(edits)],
              rank=384, age=4, joins=list(range(5, 15)), selected_columns=384,
              terminal_selection_added=False, all_other_parent_source_bytes_unchanged=True,
              full_model_executed=False, truth_accessed=False)
with (HERE / 'build.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record))
