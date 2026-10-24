"""Literal reversible insertion into the LAST submitted prediction owner."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'lazy_rotation_objective_v1/candidate.py'
sha = lambda b: hashlib.sha256(b).hexdigest()
raw = PARENT.read_bytes()
assert sha(raw) == 'b7c21ac7be5cafa45cd73ac0407605a144a724d62ffc7e2044e1432f5166377c'
source = raw.decode()
tree = ast.parse(source)
node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_predict_binary'][-1]
lines = source.splitlines(keepends=True)
old = ''.join(lines[node.lineno-1:node.end_lineno])
new = old
anchor = '            host = m.copy()\n'
assert new.count(anchor) == 1
capture = '            _cvr_host_v = fnp.diag(C).copy()\n'
new = new.replace(anchor, anchor + capture)
anchor2 = '            if collect_states:\n'
assert new.count(anchor2) == 1
action = ("            if l >= 1:\n"
          "                C, p, star, b, newborn, _cvr_scale = _cvr_apply(\n"
          "                    C, p, star, b, newborn, shared_old, _cvr_host_v, alive)\n"
          "                del _cvr_scale\n")
new = new.replace(anchor2, action + anchor2)
assert new.replace(action, '').replace(capture, '') == old
assert source.count(old) == 1
helper = (HERE / 'helper.py').read_text()
records = {}
for name, tail in [('candidate', helper), ('null', helper.replace('_CVR_RESTORE = True', '_CVR_RESTORE = False', 1))]:
    trailer = '\n\n# Fixed coherent variance restoration extension.\n' + tail
    built = source.replace(old, new) + trailer
    assert built.removesuffix(trailer).replace(new, old).encode() == raw
    ast.parse(built)
    out = HERE / (name + '.py')
    with out.open('x') as stream:
        stream.write(built)
    records[name + '_sha256'] = sha(out.read_bytes())
with (HERE / 'build.json').open('x') as stream:
    json.dump(dict(parent_sha256=sha(raw), helper_sha256=sha(helper.encode()),
                   whole_source_inverse_byte_exact=True, **records), stream, indent=2)
print(json.dumps(records))
