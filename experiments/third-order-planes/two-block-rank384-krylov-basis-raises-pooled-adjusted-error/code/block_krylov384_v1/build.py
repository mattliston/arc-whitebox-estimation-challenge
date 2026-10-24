"""One anchored last-definition edit with a complete byte inverse."""
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
node = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef)
        and n.name == '_shared_old384_push'][-1]
lines = raw.splitlines(keepends=True)
start = sum(map(len, lines[:node.lineno-1])) + node.col_offset
stop = sum(map(len, lines[:node.end_lineno-1])) + node.end_col_offset
original = raw[start:stop]
before = b'''        Omega = w32[previous, :rank]
        Y = _shared_old384_mm(J, weight[:, None] * _shared_old384_mm(J.T, Omega))
        if old_columns:
            Y = Y + _shared_old384_mm(
                Qp, _shared_old384_mm(state['core'], _shared_old384_mm(Qp.T, Omega)))
        Qn, unused_R = fnp.linalg.qr(Y, mode='reduced')
        del unused_R, Y, Omega
'''
after = b'''        Qn = _krylov384_basis(J, weight, Qp,
                              state['core'] if old_columns else None,
                              w32[previous, :192])
'''
assert original.count(before) == 1
changed = original.replace(before, after)
assert changed.count(after) == 1 and changed.replace(after, before) == original
fragment = (HERE/'krylov_fnp.py').read_bytes()
assert all(isinstance(n, (ast.Expr, ast.FunctionDef)) for n in ast.parse(fragment).body)
delimiter = b'\n\n# BEGIN FIXED TWO-BLOCK KRYLOV384 BASIS\n'
candidate = raw[:start] + changed + raw[stop:] + delimiter + fragment
recovered = candidate[:-len(delimiter)-len(fragment)]
recovered = recovered[:start] + original + recovered[start+len(changed):]
assert recovered == raw
ast.parse(candidate)
with (HERE/'candidate.py').open('xb') as stream:
    stream.write(candidate)
record = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
              parent_sha256=sha(PARENT), candidate_sha256=sha(HERE/'candidate.py'),
              fragment_sha256=sha(HERE/'krylov_fnp.py'), build_source_sha256=sha(__file__),
              protocol_sha256=sha(HERE/'PROTOCOL.md'), whole_source_inverse_byte_exact=True,
              last_push_basis_block_edits=1, rank=384, block_width=192, age=4,
              reorthogonalization_passes=2, final_joint_qr=True,
              all_other_parent_source_bytes_unchanged=True,
              full_model_executed=False, truth_accessed=False)
with (HERE/'build.json').open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print(json.dumps(record))
