"""Source-only working port helper for the new receiver-direction experiment.

No build entrypoint, imports of estimators, numerical runtime, or writes.
The per-experiment builder will bind this source and the exact parent.
"""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE/'current_comp_receiver_directions_v1'
PARENT = HERE/'current_p1_node_composition_v1/build_v1/terminal_estimator.py'
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def function_span(source, name):
    nodes = [x for x in ast.parse(source).body
             if isinstance(x, ast.FunctionDef) and x.name == name]
    require(len(nodes) == 1, 'One original function owner: '+name)
    node = nodes[0]
    lines = source.splitlines(keepends=True)
    begin = sum(len(x) for x in lines[:node.lineno-1])
    end = sum(len(x) for x in lines[:node.end_lineno])
    return begin, end, source[begin:end]


def compose(receiver_literal):
    parent = PARENT.read_bytes()
    require(hashlib.sha256(parent).hexdigest() == PARENT_SHA, 'Frozen exact COMP parent')
    source = parent.decode('utf-8')
    changes = [
        ('    _bdnh_A = _bdnh_parameters()\n',
         '    _bdnh_A = _bdnh_parameters()\n    _bdr_A = _bdr_parameters()\n'),
        ('            _bdnh_pending = _bdnh_coefficients(state, _bdnh_A, _bdnh_cstar[l])\n',
         '            _bdnh_pending = (_bdnh_coefficients(state, _bdnh_A, _bdnh_cstar[l]),\n'
         '                             _bdr_modulation(state, _bdr_A))\n'),
    ]
    start, end, old_function = function_span(source, '_bd_update')
    new_function = old_function
    function_changes = [
        ('    n = int(bp.shape[0])\n',
         '    n = int(bp.shape[0])\n'
         '    if isinstance(coefficients, tuple):\n'
         '        coefficients, receiver = coefficients\n'
         '        if receiver.shape != (4, n) or str(receiver.dtype) != \'float32\':\n'
         '            raise ValueError(\'Typed receiver-row modulation\')\n'
         '    else:\n'
         '        receiver = None\n'),
        ('        fnp.fill_diagonal(direction, 0.0)\n',
         '        fnp.fill_diagonal(direction, 0.0)\n'
         '        if receiver is not None:\n'
         '            direction = direction * receiver[j].astype(fnp.float64)[:, None]\n'),
    ]
    for before, after in function_changes:
        require(new_function.count(before) == 1, 'Unique physical-function splice')
        new_function = new_function.replace(before, after)
    inverse_function = new_function
    for before, after in reversed(function_changes):
        require(inverse_function.count(after) == 1, 'Unique physical-function inverse')
        inverse_function = inverse_function.replace(after, before)
    require(inverse_function == old_function, 'Only receiver parsing and preprojection row weighting changed')
    changes.append((old_function, new_function))
    for before, after in changes:
        require(source.count(before) == 1, 'Unique full-source splice')
        source = source.replace(before, after)
    inverse = source
    for before, after in reversed(changes):
        require(inverse.count(after) == 1, 'Unique full-source inverse')
        inverse = inverse.replace(after, before)
    require(inverse.encode('utf-8') == parent, 'Whole original COMP source restored byte-exact')
    literal_source = repr(receiver_literal)
    require(ast.literal_eval(literal_source) == receiver_literal, 'Literal Python roundtrip')
    fragment = (EXPERIMENT/'fnp.py').read_text()
    candidate = source+'\n\n_BDR_LITERAL = '+literal_source+'\n\n'+fragment
    ast.parse(candidate)
    proof = dict(parent_sha256=PARENT_SHA,
        fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
        physical_function_original_sha256=hashlib.sha256(old_function.encode()).hexdigest(),
        physical_function_modified_sha256=hashlib.sha256(new_function.encode()).hexdigest(),
        physical_function_inverse_verified=True, full_source_inverse_verified=True,
        zero_parameters_not_bypassed=True, original_stored_pair_repair_once=True,
        original_P_A_cstar_literals_unchanged=True)
    return candidate, proof
