"""Source-only public-view savings; leaves all arithmetic and owners unchanged."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(parent, out, expected):
    assert sha(parent) == expected
    original = parent.read_text()
    tree = ast.parse(original)
    node = [x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == '_bd_update'][-1]
    before = ast.get_source_segment(original, node)
    after = before
    edits = [
        ('    c = coefficients.astype(fnp.float64)[:3]\n',
         "    c = (coefficients if coefficients.dtype == 'float64' else coefficients.astype(fnp.float64))[:3]\n"),
        ('_e2_append_inline((_e2_value_inline := _bl_raw[:, None]))',
         '_e2_append_inline((_e2_value_inline := _o18_bd_literal_column(_S19_BD_ADD[:3])))'),
        ('    _bl_projection = ',
         '    _o18_bd_rows = fnp.unstack(c32, axis=0)\n'
         '    _e2_H.extend(_o18_bd_rows)  # Retain all three public views until the existing hold drains.\n'
         '    _bl_projection = '),
        ('_e2_value_inline := c32[0]', '_e2_value_inline := _o18_bd_rows[0]'),
        ('_e2_value_inline := c32[2]', '_e2_value_inline := _o18_bd_rows[2]'),
    ]
    for old, new in edits:
        assert after.count(old) == 1, old
        after = after.replace(old, new)
    helper = '''# Immutable F64 literal column, owned and reset with the existing G5 cache.
def _o18_bd_literal_column(values):
    key = ('o18_bd_literal_column', tuple(float(x) for x in values))
    value = _G5_CONST.get(key)
    if value is None:
        value = _G5_CONST[key] = _g5_arr(values)[:, None]
    return value


'''
    assert original.count(before) == 1
    source = original.replace(before, helper + after)
    ast.parse(source)
    # All unrelated source is byte-identical; the following original context
    # capture therefore sees the replacement _bd_update normally.
    assert source.replace(helper + after, before) == original
    assert '_ASTRA_DENSE_CONTEXT = tuple((name, _bd_update if name == ' in source
    out.mkdir(parents=True, exist_ok=False)
    candidate = out / 'candidate_bd_view_only_HELD_v1.py'
    candidate.write_text(source)
    (out / 'fragment_v1.py').write_text(helper + after + '\n')
    receipt = {
        'status': 'HELD_SOURCE_ONLY_UNTESTED',
        'parent': str(parent), 'parent_sha256': expected,
        'candidate': str(candidate), 'sha256': sha(candidate),
        'changes': [
            'Skip same-dtype astype only for already-F64 coefficients; sliced coefficients are never mutated.',
            'Cache one immutable F64 literal column under its complete float tuple in existing G5 owner.',
            'Replace two c32 row getitems with one unstack and retain all three outputs in existing E2Hold.'
        ],
        'preserved': [
            'Every multiply/add/subtract/reduction and cast that changes dtype retains original order.',
            'D, old32, new32, delta and all persistent numeric owners are unchanged.',
            'No einsum rewrite, no signed-zero change, no setup bind addition, no new numeric allocation.',
            'Global _bd_update and following _ASTRA_DENSE_CONTEXT capture both use the changed function.'
        ],
        'request_bound': {
            'per_call_unstack_saving': 1,
            'per_2d_call_literal_column_saving_after_first': 1,
            'conditional_already_f64_saving_per_call': 1,
            'example_15_calls_all_2d_cold_saved': 29,
            'example_15_calls_all_2d_warm_saved': 30,
            'current_f32_coefficient_path_no_astype_saving_expected': True
        },
        'caveat': 'Public request reduction is a source bound, not measured instruction or score gain. Root must gate exact prediction bytes and runtime.'
    }
    (out / 'BUILD_v1.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('parent', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('expected_sha256')
    a = p.parse_args()
    build(a.parent, a.out, a.expected_sha256)
