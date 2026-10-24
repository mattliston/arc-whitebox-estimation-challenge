"""Append the exact preserved lazy fragment and three current-parent definitions."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / 'weight5_reuse_v1/candidate.py'
ORIGIN = HERE.parent.parent / 'astra_phase2_reexamination_v1/current_comp_shared_lazy_rotation_v1/fnp.py'
sha = lambda b: hashlib.sha256(b).hexdigest()
parent = PARENT.read_bytes()
fragment = ORIGIN.read_bytes()
assert sha(parent) == 'a432b94336a1ec45f9140d2e0a7ed808a8a7826deb72034a7ca185d331c823b9'
assert sha(fragment) == 'e96ee6b660a688f767abd2b8300883e52920eace477e35e7b5edaae759ac6e92'
assert b'_slr_' not in parent and b'_SLR_' not in parent
text = parent.decode()
lines = text.splitlines(keepends=True)
tree = ast.parse(text)
changes = {
    '_shared_old384_state': [
        ('Q=None, F=None, core=None, pre_basis=None)',
         'Q=None, F=None, core=None, pre_basis=None, lazy=None)')],
    '_shared_old384_push': [
        ("            rotated = _shared_old384_mm(overlap, state['F'])\n"
         "            core = _shared_old384_mm(overlap, _shared_old384_mm(state['core'], overlap.T))\n"
         "            factors = fnp.concatenate([rotated, Fj], axis=1)\n"
         "            core = core + joined_core\n"
         "            del overlap, rotated",
         "            factors = _slr_join_factors(layer, overlap, state, Fj)\n"
         "            core = _shared_old384_mm(overlap, _shared_old384_mm(state['core'], overlap.T))\n"
         "            core = core + joined_core\n"
         "            del overlap"),
        ("        state['F'] = _shared_old384_finite(factors, 'old coefficient factors')",
         "        _slr_finish_factors(state, factors)"),
        ("        if tuple(state['F'].shape) != (rank, old_columns):\n"
         "            raise RuntimeError('Shared-old384 factor columns')",
         "        _slr_layout(state, old_columns)"),
        ("    old = _shared_old384_mm(U, state['F'])",
         "    old = _slr_reconstruct(U, state)")],
    '_shared_old384_read': [
        ("    weighted_factors = state['F'] * lam[:old_columns][None, :]\n"
         "    inner = _shared_old384_mm(old_left * old_left, weighted_factors.T)",
         "    inner = _slr_inner(old_left, lam[:old_columns], state)")]
}
definitions = []
proof = {}
for name, replacements in changes.items():
    node = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name][-1]
    original = ''.join(lines[node.lineno-1:node.end_lineno])
    changed = original
    for before, after in replacements:
        assert changed.count(before) == 1, (name, before)
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(replacements):
        assert inverse.count(after) == 1
        inverse = inverse.replace(after, before)
    assert inverse == original
    definitions.append(changed)
    proof[name] = dict(original_sha256=sha(original.encode()), changed_sha256=sha(changed.encode()),
                       replacements=len(replacements), inverse_exact=True)
candidate = parent + b'\n\n# Fixed Weight5 lazy-rotation port.\n' + fragment + b'\n\n' + '\n\n'.join(definitions).encode()
ast.parse(candidate)
assert candidate.startswith(parent)
for path, data in [(HERE/'kernel.py', fragment), (HERE/'candidate.py', candidate)]:
    with path.open('xb') as stream:
        stream.write(data)
record = dict(parent_sha256=sha(parent), candidate_sha256=sha(candidate), kernel_sha256=sha(fragment),
              builder_sha256=sha(Path(__file__).read_bytes()), protocol_sha256=sha((HERE/'PROTOCOL.md').read_bytes()),
              parent_byte_prefix_exact=True, source_inverse_exact=True, changed_definitions=proof,
              original_source=str(ORIGIN), truth_accessed=False)
with (HERE/'build.json').open('x') as stream:
    json.dump(record, stream, indent=2)
print(json.dumps(record, indent=2))
