"""Inert source constructor for one independent covariance/mean action.

No model imports, numerical runtime, candidate writes or fitting. Callers
must bind this source, the original COMP source and the new FNP fragment.
"""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE/'current_comp_joint_p_covariance_v1'
PARENT = HERE/'current_p1_node_composition_v1/build_v1/terminal_estimator.py'
PARENT_SHA = '2e13d40a21e62691155918719f8ec088f581d495ae17800c5503c3b56504918a'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def last_function(source, name):
    matches = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(matches, 'Original function exists: '+name)
    node = matches[-1]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def value_span(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
        and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == name]
    require(len(nodes) == 1, 'Unique original literal owner: '+name)
    node = nodes[0].value; lines = source.splitlines(keepends=True)
    start = sum(len(line) for line in lines[:node.lineno-1])+node.col_offset
    end = sum(len(line) for line in lines[:node.end_lineno-1])+node.end_col_offset
    require(source.isascii() or all(ord(c) < 128 for c in lines[node.lineno-1][:node.col_offset]),
            'Unambiguous ASCII literal boundary')
    return start, end, source[start:end], ast.literal_eval(node)


def replace_one(source, before, after):
    require(source.count(before) == 1, 'Unique typed source insertion')
    return source.replace(before, after)


def compose(covariance_literal, mean_literal=None):
    original_bytes = PARENT.read_bytes()
    require(hashlib.sha256(original_bytes).hexdigest() == PARENT_SHA, 'Exact retained COMP parent')
    parent = original_bytes.decode('utf-8')
    prefix = last_function(parent, '_predict_learned_compact')
    binary = last_function(parent, '_predict_binary')
    prefix_changes = (
        ('    _bdnh_A = _bdnh_parameters()\n',
         '    _bdnh_A = _bdnh_parameters()\n    _pcc_Q = _pcc_parameters()\n'),
        ('    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook)\n',
         '    def covariance_hook(l, C):\n'
         '        if _bdnh_layer != l or _bdnh_pending is None:\n'
         '            raise RuntimeError("Missing current-call covariance hidden state")\n'
         '        return _pcc_covariance(l, C, state, _pcc_Q)\n'
         '    return _predict_binary(mlp,hook=hook,coefficient_hook=coefficient_hook,covariance_hook=covariance_hook)\n'),
    )
    binary_changes = (
        ('final_hook=None, coefficient_hook=None):\n',
         'final_hook=None, coefficient_hook=None, covariance_hook=None):\n'),
        ('        fnp.fill_diagonal(C, cv)\n',
         '        fnp.fill_diagonal(C, cv)\n'
         '        if covariance_hook is not None and 1 <= l <= 14:\n'
         '            C = covariance_hook(l, C)\n'),
    )
    edits = []
    for source, changes in ((prefix, prefix_changes), (binary, binary_changes)):
        new = source
        for before, after in changes:
            new = replace_one(new, before, after)
        inverse = new
        for before, after in reversed(changes):
            inverse = replace_one(inverse, after, before)
        require(inverse == source, 'Original typed function restored byte-exact')
        edits.append((source, new))
    changed = parent
    for before, after in edits:
        changed = replace_one(changed, before, after)
    inverse = changed
    for before, after in reversed(edits):
        inverse = replace_one(inverse, after, before)
    require(inverse == parent, 'Complete original COMP circuit byte inverse')
    mean_before = value_span(changed, '_GNN_LITERAL')[2]
    if mean_literal is not None:
        begin, end, _, _ = value_span(changed, '_GNN_LITERAL')
        changed = changed[:begin]+repr(mean_literal)+changed[end:]
        begin, end, _, _ = value_span(changed, '_GNN_LITERAL')
        restored = changed[:begin]+mean_before+changed[end:]
        for before, after in reversed(edits):
            restored = replace_one(restored, after, before)
        require(restored == parent, 'Only mean literal and declared covariance circuit may change')
    fragment = (EXPERIMENT/'fnp.py').read_text()
    candidate = changed+'\n\n_PCC_LITERAL = '+repr(covariance_literal)+'\n\n'+fragment
    ast.parse(candidate)
    for name in ('_BDNH_A_LITERAL', '_BD_COEFFICIENT_LITERAL'):
        require(value_span(candidate, name)[2] == value_span(parent, name)[2], 'Frozen NODE A/cstar source')
    proof = dict(parent_sha256=PARENT_SHA,
        fragment_sha256=hashlib.sha256(fragment.encode()).hexdigest(),
        prefix_original_sha256=hashlib.sha256(prefix.encode()).hexdigest(),
        prefix_modified_sha256=hashlib.sha256(edits[0][1].encode()).hexdigest(),
        binary_original_sha256=hashlib.sha256(binary.encode()).hexdigest(),
        binary_modified_sha256=hashlib.sha256(edits[1][1].encode()).hexdigest(),
        physical_function_inverse_verified=True, full_source_inverse_verified=True,
        original_NODE_A_cstar_literals_unchanged=True, original_stored_pair_repair_once=True,
        original_new_radial_unchanged=True, live_zero_covariance_action=True,
        current_pre_mean_hidden_state_used=True, covariance_after_active_diagonal_before_Bpost=True,
        mean_literal_supplied=mean_literal is not None)
    return candidate, proof
