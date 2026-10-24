"""Inert exclusive source generator for frozen-current birth directions.

Only --build performs source generation; --out-dir is then required and
must name a new directory.  It imports no numerical package
and never imports or executes the parent/candidate.  The supplied theta JSON
must contain a finite 15-by-4 list under "theta".  Coefficients are computed
offline as IEEE float32(math.tanh(theta64)) and embedded literally.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent.parent / 'unused_materializations_v1' / 'estimator.py'
PARENT_SHA = '409f903e3f01ed8f8530f16730dbb35ad84426bc7bcbb994d623b3d35b5bf31f'
CORE = HERE / 'core.py'
REFERENCE = HERE / 'reference_numpy.py'
PROTOCOL = HERE.parent / 'PROTOCOL.md'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), sha256=sha(raw), bytes=len(raw))


def last_function(source, name):
    nodes = [n for n in ast.parse(source).body
             if isinstance(n, ast.FunctionDef) and n.name == name]
    if not nodes:
        raise RuntimeError('Missing effective function: ' + name)
    node = nodes[-1]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1]))
    end = sum(map(len, lines[:node.end_lineno]))
    return start, end, source[start:end]


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('Unexpected effective-predict source fragment')
    return source.replace(before, after, 1)


def coefficient_literals(document):
    rows = document.get('theta')
    if not isinstance(rows, list) or len(rows) != 15:
        raise RuntimeError('theta must be a 15-by-4 JSON list')
    theta, coefficients = [], []
    for row in rows:
        if not isinstance(row, list) or len(row) != 4:
            raise RuntimeError('theta must be a 15-by-4 JSON list')
        if any(isinstance(x, bool) or not isinstance(x, (int, float))
               or not math.isfinite(float(x)) for x in row):
            raise RuntimeError('theta must contain finite real numbers')
        values = tuple(float(x) for x in row)
        theta.append(values)
        coefficients.append(tuple(struct.unpack('<f', struct.pack('<f', math.tanh(x)))[0]
                                  for x in values))
    return tuple(theta), tuple(coefficients)


def construct(parent, core, theta, coefficients):
    """Pure source transformation, with a full byte-inverse witness."""
    start, end, original = last_function(parent, '_predict_binary')
    changes = [
        ('    W = mlp.weights\n',
         '    W = mlp.weights\n    _bd_c = _bd_coefficients()\n'),
        ('        cp_post = (inherited + born) * live\n',
         '        _bd_base32, _bd_plane_delta = _bd_update(\n'
         '            bp, Cpre, K21, aux, d, alive, births, a, t, s, _bd_c[l])\n'
         '        born = born + _bd_plane_delta\n'
         '        del _bd_plane_delta\n'
         '        cp_post = (inherited + born) * live\n'),
        ('(bp[:, selected] * alive[:, None]).astype(fnp.float32)',
         '_bd_base32[:, selected]'),
        ('        p = g.astype(fnp.float32)[:, None] * pp\n',
         '        del _bd_base32\n'
         '        p = g.astype(fnp.float32)[:, None] * pp\n'),
    ]
    changed = original
    for before, after in changes:
        changed = replace_once(changed, before, after)
    inverse_body = changed
    for before, after in reversed(changes):
        inverse_body = replace_once(inverse_body, after, before)
    if inverse_body != original:
        raise RuntimeError('Effective function inverse failed')
    suffix = ('\n\n# Fixed learned birth directions; no import-time array arithmetic.\n'
              + '_BD_THETA_LITERAL = ' + repr(theta) + '\n'
              + '_BD_COEFFICIENT_LITERAL = ' + repr(coefficients) + '\n\n'
              + core)
    candidate = parent[:start] + changed + parent[end:] + suffix
    ast.parse(candidate)
    candidate_without_suffix = candidate[:-len(suffix)]
    inverse = (candidate_without_suffix[:start] + original
               + candidate_without_suffix[start + len(changed):])
    if inverse != parent:
        raise RuntimeError('Complete inverse parent bytes differ')
    # The suffix is data, a docstring, and two helpers only.  In particular
    # no candidate import, module-level call, or new numerical namespace.
    allowed_functions = {'_bd_coefficients', '_bd_update'}
    for node in ast.parse(core).body:
        if isinstance(node, ast.FunctionDef) and node.name in allowed_functions:
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            continue
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == '_BD_COEFFICIENTS'
                and isinstance(node.value, ast.Constant) and node.value.value is None):
            continue
        raise RuntimeError('Unexpected core top-level operation')
    return candidate, sha(original.encode()), sha(changed.encode())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--theta-json', type=Path)
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return
    if args.theta_json is None:
        parser.error('--build requires --theta-json')
    if args.out_dir is None:
        parser.error('--build requires --out-dir')
    theta_path = args.theta_json.resolve()
    output_dir = args.out_dir.resolve()
    if output_dir.exists():
        raise FileExistsError('Build --out-dir must be a new directory')
    sources = {name: binding(path) for name, path in
               [('parent', PARENT), ('core', CORE), ('reference', REFERENCE),
                ('protocol', PROTOCOL), ('builder', Path(__file__).resolve()),
                ('theta', theta_path)]}
    if sources['parent']['sha256'] != PARENT_SHA:
        raise RuntimeError('Frozen 409f parent hash mismatch')
    document = json.loads(theta_path.read_text())
    if not isinstance(document, dict):
        raise RuntimeError('theta JSON must be an object')
    theta, coefficients = coefficient_literals(document)
    candidate, old_body_sha, new_body_sha = construct(
        PARENT.read_bytes().decode(), CORE.read_bytes().decode(), theta, coefficients)
    for value in sources.values():
        if binding(Path(value['path'])) != value:
            raise RuntimeError('Source changed while constructing candidate')
    output = output_dir / 'estimator.py'
    receipt = output_dir / 'build.json'
    data = candidate.encode()
    record = dict(
        schema='birth_direction_fnp_source_v1', passed=True, executed=False,
        source_bindings=sources, parent_sha256=PARENT_SHA,
        candidate_sha256=sha(data), candidate=str(output), output_dir=str(output_dir),
        original_predict_sha256=old_body_sha, changed_predict_sha256=new_body_sha,
        inverse_source_byte_exact=True, all_other_parent_source_byte_exact=True,
        imports_and_learned_P_payloads_unchanged=True,
        edited_parent_functions=['_predict_binary'],
        added_functions=['_bd_coefficients', '_bd_update'],
        theta_shape=[15, 4], theta_literal=theta,
        coefficient_literal=coefficients,
        coefficient_rule='offline float32(math.tanh(float64(theta)))',
        initialization='one lazy FNP float32 coefficient array in first predict',
        no_numerical_execution=True, numerical_or_score_qualification=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    with output.open('xb') as stream:
        stream.write(data)
    with receipt.open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(record, sort_keys=True, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
