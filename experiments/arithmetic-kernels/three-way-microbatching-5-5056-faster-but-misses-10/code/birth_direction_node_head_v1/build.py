"""Inert stdlib-only zero/fixed-nonzero node-head source builder.

Only --build writes an exclusive new artifact directory.  No numerical
library or estimator/model is imported or executed.  Prototype A values
are literal IEEE FP32 zero or .01*sin((i+1)*(k+1)), computed offline with
stdlib math/struct.  This builder intentionally accepts no fitted payload.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import struct

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
BD = NEW / 'birth_direction_fit_v1'
PARENT = BD / 'build_terminal_v1/estimator.py'
PARENT_SHA = 'd50bb5c5ef1d0111227173fa1ec15d5dbf39842bd0338663635cbffe512fd451'
CORE = HERE / 'core.py'
PROTOCOL = HERE / 'PROTOCOL.md'
ANNEX = HERE / 'IMPLEMENTATION.md'
THETA = BD / 'training_v1/terminal_theta.json'
PINS = {
    PARENT: PARENT_SHA,
    BD / 'build_terminal_v1/build.json': '4889c6246e8e0dd420cdb8fbca3ccbce721a163befd226339ebe6d0023cc070f',
    BD / 'model.py': 'cea1ea473db821cea6b4b9d34aeabf06f43ded8c5d77ee4037dd3f3d269df613',
    BD / 'fnp/core_v2.py': 'c31670320840545051ff7dc1031d71442cec99313423cff1e86ac62444b296f0',
    BD / 'fnp/reference_numpy.py': '447f7234ae6810058f753bfc735ca132a877014e55d8b37c73a466cbbd8be493',
    BD / 'training_v1/result.json': '5fe2afd1ad2d204ac31625031c7d00c90de5749c24272b2d5516d27e72b828b8',
    THETA: 'de0eb2c2b3dbda9cef1e6d31146755f18f400aaa47f793460017fd9040753337',
    BD / 'terminal_score_v2/result.json': '884089500b2f7f31cc9f1fbe343c6396df3cf976daee63141915715f8064c29f',
    BD / 'terminal_score_v2/completion.json': 'c14e2e831060155ce9e1ea7c82d43eb9283cd6a2c8677635d29277ebffd59f3d',
    BD / 'panel_v2/score/result.json': '767cf92415a0391f7b7596c9681adede144bd10b3e5d36f354ec2dce7ce52c1b',
    BD / 'panel_v2/score/completion.json': '96e52e95d096b17c93375c5772622cb618fda6c6697e499be775aa564d2751ee',
    NEW / 'payload_response_rows_v1.npz': '6b55c0d6071b2350e53e243bcd4137651e2c80a2cd55d04467bf194d0ea54a1c',
    NEW / 'centered_bank_autodiff_v1.py': 'c70e7bc5de06e59d1c95a41658f406a4849820c28ecd8e5aa4edd48ea67ca3e2',
    NEW / 'centered_bank_autodiff_v3.py': '45c88f1e73d694f0781c4f0a01053511addbe465a6746ef11ac38378b093983c',
    NEW / 'full_carrier_autodiff_v1.py': 'd6480f443e807688dca380d19d34e7cfe5c1f7e625104aae87201cfba7d199f7',
    NEW / 'full_carrier_bank_autodiff_v1.py': 'fc6fb6f5bbe8451da88ba78bcc3dfaa0087f3c66d1b62c5e6ceed6362a0b5b1d',
    NEW / 'centered_torch_numerics_v1.py': '2b62c3bf08ec59f7d0f749fc4526417c0c2952ed274b8aa5cba67762ff7929cf',
    NEW / 'centered_torch_numerics_v3.py': '18df077006603d1b434216b2a4ffebc1b1d1e631983163fa0a9213eb0a11412e',
    NEW / 'current_transport_shadow_v1.py': '63e9302de34dbf078bcbedc088594b00c88531543a17c3ec032386b5c8fbb2a3',
    NEW / 'current_transport_shadow_v3.py': '2193ccf82b6fcb3dada1eeb3f2a75910d0b66342ccb8da0f7c55d4886d013326',
    NEW / 'joint_prefix_final_data_v1.py': '22fff8465a13e412c68c1e5c204d7bfa0ce3d7c1cb9b99e1e5378098a9fd8b6b',
    NEW / 'tensor_clamp_min_v1.py': 'c68768a7dc17a09c6e9105817a1b9d958b2409d076dab16f5ea427d5439e9c6b',
    NEW / 'gaussian_cdf_quadrature_v1.py': 'f6f96fbc8f47bd4ffab93ba9cafc805b2ea79435f5a2106649d94a3bc0754398',
    PROTOCOL: '52045c4d089092bc9728fa301a781b39934d8658ad3418ce9d459556554c8056',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def binding(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=sha(data), bytes=len(data))


def authenticate():
    """Authenticate sources and existing score JSONs, never read truth arrays."""
    result = {}
    for path, expected in PINS.items():
        item = binding(path)
        if item['sha256'] != expected:
            raise RuntimeError('Frozen input changed: ' + str(path))
        result[str(path.resolve())] = item
    for folder in (BD / 'terminal_score_v2', BD / 'panel_v2/score'):
        record = json.loads((folder / 'result.json').read_text())
        completion = json.loads((folder / 'completion.json').read_text())
        if not all(x.get('passed') and x.get('completed') for x in (record, completion)):
            raise RuntimeError('Completed passing parent qualification required')
        if completion['result_sha256'] != PINS[folder / 'result.json']:
            raise RuntimeError('Parent completion does not seal its result')
        field = ('retain_local_component' if folder.name == 'score'
                 else 'advances_to_independent14')
        if not record.get(field):
            raise RuntimeError('Parent did not pass its declared objective')
    old_build = json.loads((BD / 'build_terminal_v1/build.json').read_text())
    if not (old_build['passed'] and old_build['inverse_source_byte_exact']
            and old_build['candidate_sha256'] == PARENT_SHA):
        raise RuntimeError('Frozen parent source-build proof failed')
    for path in (CORE, HERE / 'model.py', Path(__file__).resolve(), ANNEX):
        result[str(path.resolve())] = binding(path)
    return result


def unchanged(record):
    if any(binding(item['path']) != item for item in record.values()):
        raise RuntimeError('A source changed during source generation')


def fp32(value):
    return struct.unpack('<f', struct.pack('<f', float(value)))[0]


def matrix_bytes(values, rows, columns):
    if not isinstance(values, (list, tuple)) or len(values) != rows:
        raise ValueError('Unexpected literal row count')
    output = bytearray()
    for row in values:
        if not isinstance(row, (list, tuple)) or len(row) != columns:
            raise ValueError('Unexpected literal column count')
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError('Numeric literal required')
            if not math.isfinite(float(value)) or fp32(value) != float(value):
                raise ValueError('Expected exact finite FP32 data literals')
            output.extend(struct.pack('<f', float(value)))
    return bytes(output)


def assignment_literal(source, name):
    matches = [node for node in ast.parse(source).body
               if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name
                       for target in node.targets)]
    if len(matches) != 1:
        raise RuntimeError('Expected one data literal: ' + name)
    return ast.literal_eval(matches[0].value)


def frozen_coefficients():
    """Copy stored coefficients, never recompute tanh(theta)."""
    if binding(THETA)['sha256'] != PINS[THETA]:
        raise RuntimeError('Frozen terminal data changed')
    document = json.loads(THETA.read_text())
    values = document['coefficients']
    encoded = matrix_bytes(values, 15, 4)
    parent = PARENT.read_text()
    if sha(parent.encode()) != PARENT_SHA:
        raise RuntimeError('Frozen participant changed')
    parent_values = assignment_literal(parent, '_BD_COEFFICIENT_LITERAL')
    build_values = json.loads((BD / 'build_terminal_v1/build.json').read_text())['coefficient_literal']
    if encoded != matrix_bytes(parent_values, 15, 4) or encoded != matrix_bytes(build_values, 15, 4):
        raise RuntimeError('Stored FP32 coefficient sources disagree')
    return tuple(tuple(float(value) for value in row) for row in values)


def prototype_matrix(fixture):
    if fixture == 'zero':
        return tuple((0.0, 0.0, 0.0, 0.0) for _ in range(64))
    if fixture == 'nonzero':
        return tuple(tuple(fp32(.01 * math.sin((i + 1) * (k + 1)))
                           for k in range(4)) for i in range(64))
    raise ValueError('Only the fixed zero/nonzero prototypes are permitted')


def last_function(source, name):
    nodes = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name == name]
    if not nodes:
        raise RuntimeError('Missing effective function: ' + name)
    node = nodes[-1]
    lines = source.splitlines(keepends=True)
    start = sum(map(len, lines[:node.lineno - 1]))
    end = sum(map(len, lines[:node.end_lineno]))
    return start, end, source[start:end]


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise RuntimeError('Unexpected frozen source span: ' + before)
    return source.replace(before, after, 1)


def construct(parent, core, matrix):
    """Change two effective function bodies, append data/helpers, prove inverse."""
    if sha(parent.encode()) != PARENT_SHA:
        raise RuntimeError('Only the qualified d50bb5c5 parent is supported')
    matrix_bytes(matrix, 64, 4)
    predict_changes = (
        ("gate_floor=0.01, final_hook=None):\n",
         "gate_floor=0.01, final_hook=None, coefficient_hook=None):\n"),
        ('        _bd_base32, _bd_plane_delta = _bd_update(\n'
         '            bp, Cpre, K21, aux, d, alive, births, a, t, s, _bd_c[l])\n',
         '        _bd_current_c = (_bd_c[l] if l == 0 or coefficient_hook is None\n'
         '                         else coefficient_hook(l))\n'
         '        _bd_base32, _bd_plane_delta = _bd_update(\n'
         '            bp, Cpre, K21, aux, d, alive, births, a, t, s, _bd_current_c)\n'
         '        del _bd_current_c\n'),
    )
    hook_changes = (
        ('    P=_GNN_PARAMS\n',
         '    P=_GNN_PARAMS\n'
         '    _bdnh_A = _bdnh_parameters()\n'
         '    _bdnh_cstar = _bd_coefficients()\n'
         '    _bdnh_pending = None\n'
         '    _bdnh_layer = -1\n'),
        ('        nonlocal state\n',
         '        nonlocal state, _bdnh_pending, _bdnh_layer\n'),
        ('        delta,state=gnn_step(l,x,aux["sigma"],w,state,P)\n',
         '        delta,state=gnn_step(l,x,aux["sigma"],w,state,P)\n'
         '        if 1 <= l <= 14:\n'
         '            _bdnh_pending = _bdnh_coefficients(state, _bdnh_A, _bdnh_cstar[l])\n'
         '            _bdnh_layer = l\n'
         '        else:\n'
         '            _bdnh_pending = None\n'
         '            _bdnh_layer = -1\n'),
        ('    return _predict_binary(mlp,hook=hook)\n',
         '    def coefficient_hook(l):\n'
         '        nonlocal _bdnh_pending, _bdnh_layer\n'
         '        if _bdnh_layer != l or _bdnh_pending is None:\n'
         '            raise RuntimeError("Missing current-call birth coefficients")\n'
         '        value = _bdnh_pending\n'
         '        _bdnh_pending, _bdnh_layer = None, -1\n'
         '        return value\n'
         '    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook)\n'),
    )
    edits = []
    for name, changes in (('_predict_binary', predict_changes),
                          ('_predict_learned_compact', hook_changes)):
        start, end, old = last_function(parent, name)
        new = old
        for before, after in changes:
            new = replace_once(new, before, after)
        inverse = new
        for before, after in reversed(changes):
            inverse = replace_once(inverse, after, before)
        if inverse != old:
            raise RuntimeError('Effective-function source inverse failed')
        edits.append((start, end, old, new, name))
    body = parent
    for start, end, old, new, name in sorted(edits, reverse=True):
        body = body[:start] + new + body[end:]
    allowed = {'_bdnh_parameters', '_bdnh_coefficients'}
    actual = set()
    for node in ast.parse(core).body:
        if isinstance(node, ast.FunctionDef) and node.name in allowed:
            actual.add(node.name)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
              and isinstance(node.value.value, str)):
            pass
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1
              and isinstance(node.targets[0], ast.Name)
              and node.targets[0].id == '_BDNH_PARAMETERS'
              and isinstance(node.value, ast.Constant) and node.value.value is None):
            pass
        else:
            raise RuntimeError('Core has an unapproved import-time operation')
    if actual != allowed:
        raise RuntimeError('Unexpected participant helper set')
    suffix = ('\n\n# Frozen node-conditioned birth-column prototype; no import-time arrays.\n'
              + '_BDNH_A_LITERAL = ' + repr(matrix) + '\n\n' + core)
    candidate = body + suffix
    ast.parse(candidate)
    inverse = candidate[:-len(suffix)]
    for start, end, old, new, name in edits:
        new_start, new_end, observed = last_function(inverse, name)
        if observed != new:
            raise RuntimeError('Candidate effective body does not match its splice')
        inverse = inverse[:new_start] + old + inverse[new_end:]
    if inverse.encode() != parent.encode():
        raise RuntimeError('Full source byte-inverse failed')
    old_imports = [ast.dump(node, include_attributes=False) for node in ast.parse(parent).body
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
    new_imports = [ast.dump(node, include_attributes=False) for node in ast.parse(candidate).body
                   if isinstance(node, (ast.Import, ast.ImportFrom))]
    if old_imports != new_imports:
        raise RuntimeError('Participant imports changed')
    if last_function(candidate, '_bd_update')[2] != last_function(parent, '_bd_update')[2]:
        raise RuntimeError('Stored-plane/basis function changed')
    audit = {name: dict(parent_sha256=sha(old.encode()), candidate_sha256=sha(new.encode()))
             for start, end, old, new, name in edits}
    return candidate, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--fixture', choices=('zero', 'nonzero'))
    parser.add_argument('--out-dir', type=Path)
    args = parser.parse_args()
    if not args.build:
        parser.print_help()
        return
    if args.fixture is None or args.out_dir is None:
        parser.error('--build requires --fixture and an exclusive --out-dir')
    out = args.out_dir.resolve()
    if out.exists():
        raise FileExistsError('Never overwrite a prototype directory')
    sources = authenticate()
    cstar = frozen_coefficients()
    matrix = prototype_matrix(args.fixture)
    candidate, audit = construct(PARENT.read_text(), CORE.read_text(), matrix)
    unchanged(sources)
    data = candidate.encode()
    fixture = dict(schema='birth_direction_node_head_fixed_A_v1', fixture=args.fixture,
                   shape=[64, 4], dtype='float32', A=matrix,
                   rule='zero' if args.fixture == 'zero' else 'float32(.01*sin((i+1)*(k+1)))',
                   A_bytes_sha256=sha(matrix_bytes(matrix, 64, 4)), learned=False)
    record = dict(schema='birth_direction_node_head_source_v1', passed=True,
        completed=True, executed=False, source_bindings=sources,
        candidate=str(out / 'estimator.py'), candidate_sha256=sha(data),
        parent_sha256=PARENT_SHA, fixture=args.fixture, A_literal=matrix,
        A_bytes_sha256=fixture['A_bytes_sha256'], frozen_cstar_literal=cstar,
        frozen_cstar_bytes_sha256=sha(matrix_bytes(cstar, 15, 4)),
        effective_functions=audit, edited_parent_functions=list(audit),
        added_functions=['_bdnh_parameters', '_bdnh_coefficients'],
        inverse_source_byte_exact=True, all_other_parent_source_byte_exact=True,
        imports_and_learned_P_payloads_unchanged=True,
        stored_birth_update_byte_exact=True, no_new_CP_columns=True,
        callback_scope='per invocation; current H is never global',
        layers=list(range(1, 15)), coefficient_shape='4-by-n, indexed by birth column',
        initialization='one lazy owned FNP 64-by-4 FP32 A; all head work paid even at zero',
        numerical_or_score_qualification=False, no_numerical_execution=True)
    out.mkdir(parents=True, exist_ok=False)
    try:
        with (out / 'estimator.py').open('xb') as stream:
            stream.write(data)
        with (out / 'A.json').open('x') as stream:
            json.dump(fixture, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        unchanged(sources)
        record['artifacts'] = {name: binding(out / name) for name in ('estimator.py', 'A.json')}
        with (out / 'build.json').open('x') as stream:
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        with (out / 'completion.json').open('x') as stream:
            json.dump(dict(passed=True, completed=True, source_only=True,
                           build_sha256=binding(out / 'build.json')['sha256']),
                      stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
    except BaseException as exc:
        with (out / 'failure.json').open('x') as stream:
            json.dump(dict(passed=False, completed=False, source_only=True,
                           error=repr(exc), source_bindings=sources), stream,
                      indent=2, sort_keys=True, allow_nan=False)
            stream.write('\n')
        raise
    print(json.dumps(dict(passed=True, source_only=True, candidate_sha256=sha(data),
                         out_dir=str(out)), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
