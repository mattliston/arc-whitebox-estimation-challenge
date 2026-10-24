"""Inert stdlib source composer. No stage/output directory or estimator import.

API: authenticate(tick=None) -> flat absolute source/metadata pins;
derive_source(tick=None) -> (complete source text, exact inverse proof).
The separate reviewed root harness may compile the result only after admission.
"""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEW = HERE.parent
P2 = NEW.parent
APUSH = NEW/'current_comp_aligned_push_d6_v1'
PUBLIC = NEW/'public_504aldo_20260912_v1/repository'
PARENT = APUSH/'build_v1/estimator.py'
PARENT_SHA = '33879ab58f3a5e4e1ad41deb5db21b73ba703438374225d30fd0ebfc7511b89a'
FRAGMENT = HERE/'fnp.py'
PINS = {
    PARENT: PARENT_SHA,
    FRAGMENT: '980853fe0b794efcf046b360ea03bbfaa671bd48383312e8e766c9b546021d81',
    HERE/'PROTOCOL.md': 'ef2331d9adfe20c8bdea5c173d9f2d19ec523b85bc0cde4a653c93ead4351607',
    HERE/'PRICE.json': '04bf11c7a3f8148daa0ad9371d725f5cc8ed4c17426796ef14c6dafebecf61fd',
    HERE/'price_metadata.py': '5538dba5a46169009eafe0b3d51cfa80e6b7cb1f2eb8e51c70dadfbc247fa530',
    APUSH/'build.py': 'ef008756a35e3e15e3f0286b6c846ea8d47c63a838d575bb01e54661cbe60f2b',
    APUSH/'DECISION.md': 'ff026a7659875606a06206a89f047331a77d86af7da030266be74130bbf784f0',
    APUSH/'build_v1/result.json': 'f841d13473244938a24de14822529960455a2f344306d2899c6656fed4e44f0a',
    APUSH/'build_v1/completion.json': 'e3d2ac39f6a2b0b489eb9a83beab987d2b36623e561d0611fe523fcdaacb0707',
    APUSH/'build_v1/finalization.json': '9bac7e303cad3ec1a83045e16f18fc6874c40b6c2444372e1d890eb898502ffe',
    APUSH/'native40_terminal_v1/result.json': '23b88f12449442cad00eb352e2aca1d90d277120b7ed348f800a9bcff64684a7',
    APUSH/'native40_terminal_v1/completion.json': '9cba07e8d9e7531a7b8bd0c9842e6f44c2d258f089bdd5c3253242a6a5a3ba42',
    APUSH/'native40_terminal_v1/finalization.json': 'dee0a6eb80aba2b5bb56b71752dfd83e23f64f7e8b9a8985cbd9f27dce5b307e',
    APUSH/'native40_watch_v1/result.json': '8665b4705c07d14f7f315ee1e1229154177fc86e3345f12d7029fe31ed8b526b',
    APUSH/'native40_watch_v1/completion.json': '669bf57b4e3efa6cafac6b48d8592e4a02e3907ee6f818c0fe5d101bfe3c479a',
    APUSH/'native40_watch_v1/finalization.json': 'd68ac6dcd781d28490a4271e8480a98e240056b7ce1e9bbb1f7bd62d5f1a91d9',
    APUSH/'score40_v1/result.json': 'b9667c17e474f99e394481169f9081d5e3320ca74255e2e3e411b7eaf9995279',
    APUSH/'score40_v1/completion.json': '0d81736f3f104eb8efeb8c64b1f1ff481e573b85d04ac59469e4c6dbef478c81',
    APUSH/'score40_v1/finalization.json': 'a4c9386b43bb5fc73a3a8ddaf0b18a175cefd46c584ebe4a8691cfbeff18ad67',
    APUSH/'score40_watch_v1/result.json': '577c84e81fe81e0f06a199463ffe0ccd68f215f50166a9cd84b2321ef7d9cf14',
    APUSH/'score40_watch_v1/completion.json': '7d6eea910c417aeb6b588c3708c5ef6f2a7ee75918be1d5b1c50fc0b665ac19f',
    APUSH/'score40_watch_v1/finalization.json': '875aa37234f3e45bb2ec365e1d688424a7357dd96b59318b99249534627f0655',
    PUBLIC/'estimators/estimator_v29.py': '86d9ca9b28e6fe2b6c74750a0b6ae4bba4c14f742ddc3f5f56bc7fb4ec9d8e27',
    PUBLIC/'docs/findings_log.md': '7524c1159b72e53400d01e69e21c2f7042330ad454a28331d1dd22becde0c572',
    PUBLIC/'docs/derivation_code_map.md': 'f1a916e79cf8762bf58dddf7c11fc83f9e020218c041c402ca559f83cf3ee68c',
    PUBLIC/'docs/audit_v29_ns_d0.log': '3ecd107e5ff9e7687867624a791dcc6ae90ad4fe5aa079bb8921d330b7ddd1bd',
    NEW/'DIFFERENCE_LEG_CUTS_V1_CLOSED.md': '509d034e7d65eb7deb623d9a93bf8223f9941953d4ec3749f3589a3004f8129c',
    NEW/'HYBRID_BIRTH_CORE_SCOPE_20260909_V2.md': 'f9f4ffc5ab838b79e50bd40130f09a8c5d37124985bba9ce571c52fc2adb4686',
    P2/'public_g20_emission_retrofit_v1/estimator_retro.py': '5b66b2e8e42c546ff53eeb267435aa8de990c78c49901f55ec57b69ade575eb9',
    P2/'public_g20_emission_retrofit_v1/RETRO_SCREEN_g30_lr_P256.json': 'a11c09d8520e00033e31500ba21d73ca8b03d57a2f594f01c6e41bf0e4b8668b',
    P2/'public_g20_emission_retrofit_v1/RETRO_SCREEN_g30_lr_C.json': '805cb7ca8349d689d826d4a5da6b12bb06e5c6766697569668e9f42b44877fbd',
    P2/'data_gate/.venv/lib/python3.11/site-packages/flopscope/_dtype_billing.py': 'd6f4df8b2fcf162f41afde2dae23a66d894eba62f173e57d89051e5873569cb8',
    P2/'data_gate/.venv/lib/python3.11/site-packages/flopscope/numpy/linalg/_decompositions.py': '0cd6f08ac3fe3cef97d506be6713d0aa85f6ced74e8fd35016a420bfe6d8d942',
    P2/'data_gate/.venv/lib/python3.11/site-packages/flopscope/data/default_weights.json': 'f729bc47f07c080b8c7b9d7b045e5663f0af313669b4c2a962b5a9044a94cb7a',
}
DELIMITER = '\n\n# BEGIN FIXED SHARED-OLD384 INERT SOURCE FRAGMENT\n'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _sealed(folder):
    result, end, final = (json.loads((folder/name).read_text())
                          for name in ('result.json', 'completion.json', 'finalization.json'))
    require(all(z.get('passed') is True and z.get('completed') is True
                for z in (result, end, final)), 'Actual positive three-seal boundary: ' + str(folder))
    digest, end_digest = PINS[folder/'result.json'], PINS[folder/'completion.json']
    require(end['result_sha256'] == final['result_sha256'] == digest
            and final['completion_sha256'] == end_digest, 'Actual seal links: ' + str(folder))
    return result


def authenticate(tick=None):
    tick = tick or (lambda: None)
    pins = {str(Path(__file__).resolve()): sha(Path(__file__).resolve())}
    for path, digest in PINS.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == digest,
                'Frozen direct source/metadata pin: ' + str(path))
        pins[str(path)] = digest
        tick()
    build = _sealed(APUSH/'build_v1')
    native = _sealed(APUSH/'native40_terminal_v1')
    watch = _sealed(APUSH/'native40_watch_v1')
    _sealed(APUSH/'score40_v1')
    _sealed(APUSH/'score40_watch_v1')
    require(build['candidate_sha256'] == native['candidate_sha256'] == PARENT_SHA,
            'Actual APUSH candidate boundary')
    require(native['full_model_calls'] == native['retired_call_count'] == 41
            and native['all16_domains_qualified'] and native['all47_selectors_qualified']
            and native['own_complete_bills'] and native['initialization_flops'] == 1944970298,
            'Actual complete parent native40 acquisition')
    require(watch['whole_owned_tree_retired'] and watch['cleanup_passed']
            and watch['observational_groups_retired']
            and watch['child_result_sha256'] == PINS[APUSH/'native40_terminal_v1/result.json']
            and watch['child_completion_sha256'] == PINS[APUSH/'native40_terminal_v1/completion.json']
            and watch['child_finalization_sha256'] == PINS[APUSH/'native40_terminal_v1/finalization.json']
            and watch['maximum_combined_rss_bytes'] < 8000000000
            and watch['maximum_swap_growth_bytes'] == 0, 'Parent native40 retirement/resources')
    plan = json.loads((HERE/'PRICE.json').read_text())
    require(plan['parent_sha256'] == PARENT_SHA and len(plan['rows']) == 40,
            'Frozen geometry price source')
    for row, packet, trial in zip(plan['rows'], native['diagnostic_records'][:40], native['trials'][:40]):
        path = Path(packet['path'])
        require(path == APUSH/'native40_terminal_v1'/f"call_{row['call']:02d}_diagnostics.json",
                'Only named target-free diagnostic JSON')
        require(row['diagnostic_sha256'] == packet['sha256'] == sha(path)
                and row['rank'] == trial['rank'] and row['call'] == trial['call']
                and row['actual_parent_flops'] == trial['flops'], 'Exact geometry/bill binding')
        pins[str(path)] = packet['sha256']
        tick()
    return pins


def _byte_spans(raw, tree):
    # AST columns are UTF8 BYTE offsets. Cache source LF offsets once; no repeated
    # ast.get_source_segment/_splitlines scan through the huge learned literals.
    offsets, total = [0], 0
    for line in raw.split(b'\n'):
        total += len(line) + 1
        offsets.append(total)
    return {id(node): (offsets[node.lineno - 1] + node.col_offset,
                       offsets[node.end_lineno - 1] + node.end_col_offset)
            for node in tree.body}


def derive_source(tick=None):
    tick = tick or (lambda: None)
    pins = authenticate(tick)
    raw = PARENT.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == PARENT_SHA, 'Fresh parent source bytes')
    tree = ast.parse(raw.decode('utf-8'))
    spans = _byte_spans(raw, tree)
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == '_predict_binary']
    require(len(definitions) == 4, 'Exact final parent physical function owner')
    node = definitions[-1]
    start, stop = spans[id(node)]
    original = raw[start:stop].decode('utf-8')
    edits = (
        ('    newborn = None\n',
         '    newborn = None\n    shared_old = _shared_old384_state()\n'),
        ('        pp = _push_bank_shared_newborn(w32, p, previous, newborn)\n',
         '        pp = _shared_old384_push(l, w32, p, previous, newborn, lam, shared_old)\n'),
        ('            response_plane = _mm(left * left, (right * lam).T).astype(fnp.float64)\n',
         '            response_plane = _shared_old384_read(left, right, lam, live_indices, shared_old).astype(fnp.float64)\n'),
        ('        p = alive.astype(fnp.float32)[:, None] * p\n',
         '        p = alive.astype(fnp.float32)[:, None] * p\n'
         '        _shared_old384_commit(l, g, alive, selected, shared_old)\n'),
    )
    changed = original
    for before, after in edits:
        require(changed.count(before) == 1 and after not in changed, 'Unique contextual CP-state splice')
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique inverse CP-state splice')
        inverse = inverse.replace(after, before)
    require(inverse == original, 'Whole physical-function inverse')
    tick()
    fragment = FRAGMENT.read_text()
    require(hashlib.sha256(fragment.encode()).hexdigest() == PINS[FRAGMENT], 'Fresh inert fragment bytes')
    fragment_tree = ast.parse(fragment)
    functions = []
    for entry in fragment_tree.body:
        if isinstance(entry, ast.Expr) and isinstance(entry.value, ast.Constant) and isinstance(entry.value.value, str):
            continue
        require(isinstance(entry, ast.FunctionDef) and not entry.decorator_list
                and not entry.args.defaults and not any(entry.args.kw_defaults),
                'Fragment may contain only inert nondecorated function definitions')
        require(entry.name.startswith('_shared_old384_'), 'Namespaced new helper')
        functions.append(entry.name)
    require(len(functions) == len(set(functions)) == 6, 'Exact helper roster')
    parent_names = {entry.name for entry in tree.body if isinstance(entry, (ast.FunctionDef, ast.ClassDef))}
    require(not parent_names.intersection(functions), 'No parent name or implementation replaced')
    require(DELIMITER.encode() not in raw, 'Unique fragment boundary')
    suffix = (DELIMITER + fragment).encode('utf-8')
    terminal = raw[:start] + changed.encode('utf-8') + raw[stop:] + suffix
    recovered = terminal[:-len(suffix)]
    recovered = recovered[:start] + inverse.encode('utf-8') + recovered[start + len(changed.encode('utf-8')):]
    require(recovered == raw, 'Entire parent source byte-exact inverse')
    tick()
    text = terminal.decode('utf-8')
    compile(text, str(HERE/'composed_estimator.py'), 'exec')
    tick()
    untouched = {}
    for index, entry in enumerate(tree.body):
        if entry is node:
            continue
        lo, hi = spans[id(entry)]
        untouched[str(index)] = hashlib.sha256(raw[lo:hi]).hexdigest()
    proof = dict(schema='current_comp_shared_old384_source_proof_v1',
        parent_sha256=PARENT_SHA, composed_source_sha256=hashlib.sha256(terminal).hexdigest(),
        full_source_inverse_verified=True, physical_function_inverse_verified=True,
        exact_cp_state_splices=4, appended_inert_helpers=functions,
        all_unrelated_parent_bytes_unchanged=True, learned_literals_and_imports_byte_exact=True,
        original_full_p_gating_still_paid=True, original_all_row_diagonal_source_unchanged=True,
        original_omitted_radial_source_unchanged=True, original_star_assembly_unchanged=True,
        original_BD_stored_pair_repair_once=True, parent_arena_and_dispatchers_unchanged=True,
        no_new_persistent_numerical_cache=True, no_import_time_helper_array_work=True,
        rank=384, first_join_layer=5, last_join_layer=14, weight_rule='actual_FP32_lambda_squared',
        extra_incumbent_plane_repair=False, inherited_plane_is_approximated=True,
        original_predict_sha256=hashlib.sha256(original.encode()).hexdigest(),
        changed_predict_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        untouched_top_level_source_spans=untouched, source_bindings=pins,
        numerical_estimator_executed=False, truth_accessed=False, fitted=False,
        actual_candidate_bill_measured=False, price_is_conditional_on_parent_geometry=True)
    return text, proof
