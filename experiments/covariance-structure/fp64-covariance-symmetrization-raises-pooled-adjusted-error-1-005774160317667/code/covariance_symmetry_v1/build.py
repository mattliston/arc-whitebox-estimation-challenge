"""Apply a single current-bank affine-covariance symmetry insertion."""
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
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary'][-1]
lines = raw.splitlines(keepends=True)
start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
stop = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
old = raw[start:stop]
anchor = b'            bpre = w.T @ b\n            vv = fnp.diag(Cpre)\n'
insert = (b'            Cpre = _covsym_full(Cpre)\n'
          b'            Bpre = Cpre if l == 0 else _covsym_full(Bpre)\n')
assert old.count(anchor) == 1 and b'_covsym_full' not in raw
changed = old.replace(anchor, insert + anchor)
fragment = (HERE / 'symmetry_fnp.py').read_bytes()
delimiter = b'\n\n# Fixed full covariance symmetry restoration.\n'
candidate = raw[:start] + changed + raw[stop:] + delimiter + fragment
recovered = candidate[:-len(delimiter)-len(fragment)]
recovered = recovered[:start] + changed.replace(insert, b'') + recovered[start+len(changed):]
assert recovered == raw
ast.parse(candidate)
with (HERE/'candidate.py').open('xb') as stream:
    stream.write(candidate)
proof = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
             parent_sha256=sha(PARENT), candidate_sha256=sha(HERE/'candidate.py'),
             fragment_sha256=sha(HERE/'symmetry_fnp.py'), build_source_sha256=sha(__file__),
             protocol_sha256=sha(HERE/'PROTOCOL.md'), exact_insertions=1,
             whole_source_inverse_byte_exact=True, rank=384, age=4,
             original_source_otherwise_byte_exact=True,
             final_diagonal_path_unchanged=True, third_plane_not_symmetrized=True,
             full_model_executed=False, truth_accessed=False)
with (HERE/'build.json').open('x') as stream:
    json.dump(proof,stream,indent=2)
    stream.write('\n')
print(json.dumps(proof))
