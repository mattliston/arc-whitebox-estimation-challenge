"""Inert stdlib composer; no candidate writer, stage, estimator or array import.

authenticate(tick=None) returns flat source-only bindings, not run authority.
derive_source(tick=None) returns source text and a complete byte-inverse proof.
Neither function is called by import or direct invocation of this file.
The separate future harness must supply reviewed source/decision/receipt pins.
"""
import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
SHARED = NEW / 'current_comp_shared_old384_v1'
PARENT = NEW / 'current_comp_shared_gain_slice_v1/build_v1/estimator.py'
PARENT_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
FRAGMENT = HERE / 'metric_fnp.py'
FRAGMENT_SHA = '52731186b0ee2ecd932fb9b3a7a44bd23a305dd17d3afbf58f8949019df5c44e'
OWNER = SHARED / 'fnp.py'
PINS = {
    PARENT: PARENT_SHA,
    FRAGMENT: FRAGMENT_SHA,
    OWNER: '980853fe0b794efcf046b360ea03bbfaa671bd48383312e8e766c9b546021d81',
    P2 / 'data_gate/.venv/lib/python3.11/site-packages/flopscope/_dtype_billing.py':
        'd6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8',
    P2 / 'data_gate/.venv/lib/python3.11/site-packages/flopscope/data/default_weights.json':
        'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a',
}
HELPERS = ('_pair256_finite', '_pair256_coordinates', '_pair256_psd',
           '_pair256_metric', '_pair256_apply_kernel', '_pair256_nonzero_range')
DELIMITER = b'\n\n# BEGIN ENTRY-FROZEN PAIR256 METRIC-ONLY HELPERS\n'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path, tick=None):
    tick = tick or (lambda: None)
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
            tick()
    return digest.hexdigest()


def authenticate(tick=None):
    """Pin sources only. No positive scientific/prerequisite claim is made."""
    tick = tick or (lambda: None)
    result = {}
    for path, expected in PINS.items():
        require(path.is_file() and not path.is_symlink(), 'Direct source file: ' + str(path))
        actual = sha(path, tick)
        require(actual == expected, 'Frozen source hash: ' + str(path))
        result[str(path)] = actual
    for path in (Path(__file__).resolve(), HERE / 'DESIGN.md'):
        require(path.is_file() and not path.is_symlink(), 'Current source owner: ' + str(path))
        result[str(path)] = sha(path, tick)
    return result


def _function_span(raw, name):
    """AST owns the entire function; its columns are UTF8 byte offsets."""
    tree = ast.parse(raw.decode('utf-8'))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(selected) == 1, 'Unique AST function owner: ' + name)
    node = selected[0]
    require(not node.decorator_list, 'Undecorated function owner')
    offsets = [0]
    for line in raw.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    start = offsets[node.lineno - 1] + node.col_offset
    stop = offsets[node.end_lineno - 1] + node.end_col_offset
    return tree, start, stop


def _replace_once(source, before, after, edits):
    require(before != after and source.count(before) == 1, 'Unique contextual edit')
    position = source.index(before)
    edits.append((position, before, after))
    return source[:position] + after + source[position + len(before):]


def _inverse(source, edits):
    # Positions belong to each successive edit state. Reverse in the opposite
    # order, never by a basename/subsequence count which may collide later.
    for position, before, after in reversed(edits):
        require(source[position:position + len(after)] == after,
                'Exact positional inverse edit')
        source = source[:position] + before + source[position + len(after):]
    return source


def _inert_fragment(raw, parent_tree):
    tree = ast.parse(raw.decode('utf-8'))
    names = []
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)):
            continue
        require(isinstance(node, ast.FunctionDef) and not node.decorator_list
                and node.returns is None, 'Only inert undecorated helpers')
        arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
        require(all(argument.annotation is None for argument in arguments),
                'No evaluated argument annotations')
        require(all(isinstance(value, ast.Constant)
                    for value in (*node.args.defaults,
                                  *(value for value in node.args.kw_defaults if value is not None))),
                'Only literal default values')
        require(not any(isinstance(entry, (ast.Import, ast.ImportFrom))
                        for entry in ast.walk(node)), 'FNP fragment cannot import')
        names.append(node.name)
    require(tuple(names) == HELPERS, 'Exact six-helper roster')
    parent_names = {node.name for node in parent_tree.body
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    require(not parent_names.intersection(names), 'No parent implementation overwritten')


def derive_source(tick=None):
    tick = tick or (lambda: None)
    bindings = authenticate(tick)
    raw, fragment, owner = PARENT.read_bytes(), FRAGMENT.read_bytes(), OWNER.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_SHA
            and hashlib.sha256(fragment).hexdigest() == FRAGMENT_SHA
            and hashlib.sha256(owner).hexdigest() == PINS[OWNER], 'Fresh source bytes')
    tree, start, stop = _function_span(raw, '_shared_old384_push')
    _, owner_start, owner_stop = _function_span(owner, '_shared_old384_push')
    original = raw[start:stop]
    require(original == owner[owner_start:owner_stop], 'Current shared push equals its frozen owner')
    require(b'_pair256_' not in raw and DELIMITER not in raw, 'New namespace and unique delimiter')
    _inert_fragment(fragment, tree)
    tick()

    changed, edits = original, []
    substitutions = (
        (b'int(previous.shape[0]) < 384', b'int(previous.shape[0]) < 256'),
        (b'at least384 live rows', b'at least256 live rows'),
        (b'    rank = 384\n', b'    rank = 256\n'),
        (b'        # Frozen weighting rule: squared ACTUAL stored signed-cube coefficients.\n'
         b'        # Both equal-and-opposite pair members receive equal nonnegative weight.\n'
         b'        # No tensor-error, cancellation, spectrum, or endpoint certificate follows.\n'
         b'        coefficient = lam[start:stop]\n'
         b'        weight = coefficient * coefficient\n',
         b'        # Entry-frozen pair metric, separate from original physical x/y factors.\n'
         b'        # Norm-dependent blocks are NOT refreshed after gates or transport.\n'
         b'        coefficient = lam[start:stop]\n'
         b'        metric_basis, pair_kernel = _pair256_metric(J, coefficient)\n'),
        (b'        Y = _shared_old384_mm(J, weight[:, None] * _shared_old384_mm(J.T, Omega))\n',
         b'        Y = _shared_old384_mm(\n'
         b'            metric_basis, _pair256_apply_kernel(\n'
         b'                _shared_old384_mm(metric_basis.T, Omega), pair_kernel))\n'),
        (b'        Qn, unused_R = fnp.linalg.qr(Y, mode=\'reduced\')\n',
         b'        _pair256_nonzero_range(Y)\n'
         b'        Qn, unused_R = fnp.linalg.qr(Y, mode=\'reduced\')\n'),
        (b'        joined_core = _shared_old384_mm(Fj, weight[:, None] * Fj.T)\n',
         b'        # Fj itself stays in original x/y coordinates in the physical bank.\n'
         b'        metric_factors = _pair256_coordinates(Fj)\n'
         b'        joined_core = _shared_old384_mm(\n'
         b'            metric_factors, _pair256_apply_kernel(metric_factors.T, pair_kernel))\n'),
        (b'        del J, weight, coefficient, Fj, joined_core, factors, core\n',
         b'        del J, metric_basis, pair_kernel, metric_factors\n'
         b'        del coefficient, Fj, joined_core, factors, core\n'),
    )
    for before, after in substitutions:
        changed = _replace_once(changed, before, after, edits)
    inverse = _inverse(changed, edits)
    require(inverse == original, 'Entire shared push inverse')
    for statement in (
        b'        Fj = _shared_old384_mm(Qn.T, J)\n',
        b"            rotated = _shared_old384_mm(overlap, state['F'])\n",
        b'            factors = fnp.concatenate([rotated, Fj], axis=1)\n',
        b'            factors, core = Fj, joined_core\n',
    ):
        require(original.count(statement) == changed.count(statement) == 1,
                'Original physical factor coordinates and projection retained')

    suffix = DELIMITER + fragment
    derived = raw[:start] + changed + raw[stop:] + suffix
    recovered = derived[:-len(suffix)]
    recovered = recovered[:start] + inverse + recovered[start + len(changed):]
    require(recovered == raw, 'Entire exact c2 parent byte inverse')
    ast.parse(derived.decode('utf-8'))
    tick()
    proof = dict(
        schema='current_comp_shared_pair256_source_proof_v1',
        parent_sha256=PARENT_SHA, fragment_sha256=FRAGMENT_SHA,
        composed_source_sha256=hashlib.sha256(derived).hexdigest(),
        original_push_sha256=hashlib.sha256(original).hexdigest(),
        changed_push_sha256=hashlib.sha256(changed).hexdigest(),
        source_bindings=bindings, exact_contextual_edits=len(edits),
        full_source_inverse_verified=True, function_inverse_verified=True,
        original_physical_factor_coordinates=True, original_Fj_projection_unchanged=True,
        original_learned_literals_and_heads_byte_exact=True,
        original_scalar_gain_slice_byte_exact=True, original_CP_reads_byte_exact=True,
        original_star_BD_radial_diagonal_consumers_byte_exact=True,
        parent_arena_and_dispatchers_byte_exact=True, original_full_bank_still_present=True,
        rank=256, age=4, first_join_layer=5, last_join_layer=14, joins=10,
        no_terminal_join=True, metric_entry_frozen=True,
        block_norm_dtype='float64', matrix_dtype='float32',
        metric='per_pair_mode1_Gram_midpoint_difference_entry_frozen',
        metric_core_reassociation_is_not_bit_exact=True,
        full_signed_tensor_Gram_claimed=False, current_tensor_Gram_after_gates_claimed=False,
        extra_incumbent_plane_repair=False, no_new_persistent_numerical_cache=True,
        no_import_time_helper_array_work=True, numerical_estimator_executed=False,
        candidate_written=False, target_accessed=False, fitted=False,
        actual_cost_measured=False, source_only_not_run_authority=True,
    )
    return derived.decode('utf-8'), proof


if __name__ == '__main__':
    raise SystemExit('Source-only helper. No candidate writer or numerical stage is available.')
