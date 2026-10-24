"""Source-only, exclusive build of the fixed E-29 BD precision boundary."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
PARENT = H2 / 'hill_climb_star_projection32_v1/candidate.py'
PARENT_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'


def pin(path):
    data = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def main():
    assert pin(PARENT)['sha256'] == PARENT_SHA
    parent = PARENT.read_text()
    lines = parent.splitlines(keepends=True)
    tree = ast.parse(parent)
    active = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    edits = []

    def change_function(name, changes):
        node = active[name]
        old = ''.join(lines[node.lineno - 1:node.end_lineno])
        new = old
        for before, after in changes:
            assert new.count(before) == 1, (name, before, new.count(before))
            new = new.replace(before, after)
        assert new != old
        assert parent.count(old) == 1
        edits.append(dict(function=name, line=node.lineno, before=old, after=new,
                          replacements=[dict(before=a, after=b) for a, b in changes]))

    change_function('_e2_bd_state3', [
        ('D = fnp.empty((3, n, n), dtype=fnp.float64)',
         'D = fnp.empty((3, n, n), dtype=fnp.float32)   # E-29 BD-only: three current directions.')])
    change_function('_bd_update', [
        ('coefficient sum run once over the stack (same per-element float64 operations and order as the j-loop).',
         'coefficient sum run once over the stack; E-29 BD-only now computes this internal stack in float32.\n'
         '    Current three legs, coefficients, raw-leg additions and FP64 physical delta are preserved.'),
        ('c = coefficients.astype(fnp.float64)[:3]',
         'c = coefficients.astype(fnp.float32)[:3]'),
        ("_e2_hold.append(fnp.multiply(st['D12'], eta, out=st['D12']))",
         "_e2_hold.append(fnp.multiply(st['D12'], eta.astype(fnp.float32), out=st['D12']))"),
        ('projection = _e2_H(fnp.sum(_e2_H(bp * D), axis=1)) / denominator',
         'bp32 = bp.astype(fnp.float32)   # E-29: denominator is still formed in FP64, then rounded.\n'
         '    projection = _e2_H(fnp.sum(_e2_H(bp32 * D), axis=1)) / denominator.astype(fnp.float32)'),
        ("_j1m, _g5_j1tail, _g5_j1col = _g5_j1((1.0, 0.0, 1.0), _BD_J1_PHYS)   # G5: per-process constants",
         "_j1m, _g5_j1tail, _g5_j1col = _g5_j1((1.0, 0.0, 1.0), _BD_J1_PHYS)   # Original cached constants.\n"
         "        _j1m = _j1m.astype(fnp.float32)\n"
         "        _g5_j1tail = _g5_j1tail.astype(fnp.float32)\n"
         "        _g5_j1col = _g5_j1tail[:, None]   # Keep this internal coefficient branch in FP32."),
        ('_e2_hold.append(fnp.subtract(D, _g5_bcast3(bp, projection), out=D))',
         '_e2_hold.append(fnp.subtract(D, _g5_bcast3(bp32, projection), out=D))'),
        ('_e2_hold.append(fnp.multiply(bp, alive_column, out=old32))',
         '_e2_hold.append(fnp.multiply(bp32, alive_column, out=old32))'),
        ('_e2_hold.append(fnp.multiply(bp + change, alive_column, out=new32))',
         '_e2_hold.append(fnp.multiply(bp32 + change, alive_column, out=new32))')])

    candidate = parent
    for edit in edits:
        candidate = candidate.replace(edit['before'], edit['after'], 1)
    ast.parse(candidate)
    inverse = candidate
    for edit in reversed(edits):
        assert inverse.count(edit['after']) == 1
        inverse = inverse.replace(edit['after'], edit['before'], 1)
    assert inverse == parent
    # The physical repeated-plane correction, and every definition except these two, remain byte-identical.
    delta_marker = '    old = old32.astype(fnp.float64)\n'
    before_bd, after_bd = edits[1]['before'], edits[1]['after']
    assert before_bd[before_bd.index(delta_marker):] == after_bd[after_bd.index(delta_marker):]
    original_defs = {n.name: ast.get_source_segment(parent, n) for n in ast.parse(parent).body
                     if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    result_defs = {n.name: ast.get_source_segment(candidate, n) for n in ast.parse(candidate).body
                   if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert original_defs.keys() == result_defs.keys()
    assert {name for name in original_defs if original_defs[name] != result_defs[name]} == {
        '_e2_bd_state3', '_bd_update'}
    with (HERE / 'candidate.py').open('x') as f:
        f.write(candidate)
    report = dict(source_only=True, numerical_execution=False,
                  parent=pin(PARENT), candidate=pin(HERE / 'candidate.py'),
                  builder=pin(Path(__file__)), protocol=pin(HERE / 'PROTOCOL.md'),
                  precision_inventory=pin(HERE / 'PRECISION.md'),
                  reference_builder=pin(H2.parent / 'fable_engineer_v1/build_f32.py'),
                  reference_result=pin(H2.parent / 'fable_engineer_v1/RESULT.md'),
                  edits=edits, inverse_exact=True, physical_delta_tail_byte_exact=True,
                  modified_active_definitions=['_e2_bd_state3', '_bd_update'],
                  learned_literal_bytes_unchanged=True, source_added_bytes=len(candidate.encode())-len(parent.encode()),
                  expected_layer0_bytes_exact=True,
                  prediction_arithmetic='All new casts/products use metered FNP; no external numerical code.',
                  all_other_source_unchanged=True)
    with (HERE / 'build.json').open('x') as f:
        json.dump(report, f, indent=2)
        f.write('\n')
    print(json.dumps({k: report[k] for k in ['candidate', 'inverse_exact', 'physical_delta_tail_byte_exact',
                                          'modified_active_definitions', 'source_added_bytes']}))


if __name__ == '__main__':
    main()
