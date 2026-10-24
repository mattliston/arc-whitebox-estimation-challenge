"""Inert adapter for the exact two-call native/OpenBLAS gain-slice qualification."""
import ast
import hashlib
import math
from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
SHARED = HERE.parent / 'current_comp_shared_old384_v1'
OWNER = SHARED / 'backend_probe.py'
OWNER_SHA = '36d9068c8dead7f32b53da9c5c5382f9004a8a45c0c47c29d7e1f4b2435fdcc8'
OLD_NATIVE = (
    '86bb44e7722f1316d4cb60df76f56dced930b135f743cd9f3a8594c8dbc0e53d',
    'e384355ce3cf266181ca4cdff2edc4cc72804cffb2d678e82d2f98496338a943',
    '85e511890536dc6d77776b1d14e6daa6280a33ac8f7636326ef54cbc087d742d')
OLD_WATCH = (
    '25c1ae665625c574175a6323cc4c209b83e87e8a683c5a23f542cf6d6693ade6',
    'e45d16cfbdaf091b183c2be07b0af1a6b101a3cee6a413fdfbc75a4e42bfe9d3',
    'b73a63a72e0ca0c875ab6b9ee598c742675e90f47ac71beba758c60ab7ecbb49')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


require(hashlib.sha256(OWNER.read_bytes()).hexdigest() == OWNER_SHA, 'Exact original same-OpenBLAS native owner')
_base = runpy.run_path(str(OWNER), run_name='_gain_slice_original_backend_metadata')
sha, read, bind = (_base[k] for k in ('sha', 'read', 'bind'))
PINS = {**_base['_BP_PINS'], OWNER: OWNER_SHA,
        SHARED / 'BACKEND_PROBE_PROTOCOL.md': '2d7a773171100f0234e652d68bb606d1d641fcdb2c896477515a6c6ab019a507'}


def command_args(source_sha, watch_sha, protocol_sha, watch_spec_sha, builder_sha,
                 build_protocol_sha, build_sha, candidate_sha):
    return ['--run', '--source-sha256', source_sha, '--watch-sha256', watch_sha,
            '--protocol-sha256', protocol_sha, '--watch-spec-sha256', watch_spec_sha,
            '--builder-sha256', builder_sha, '--build-protocol-sha256', build_protocol_sha,
            '--build-sha256', build_sha, '--candidate-sha256', candidate_sha]


def gain_source_owners(args):
    mapping = {HERE / 'native.py': args.source_sha256, HERE / 'watch.py': args.watch_sha256,
        HERE / 'NATIVE_PROTOCOL.md': args.protocol_sha256, HERE / 'PAIR_WATCH_SPEC.md': args.watch_spec_sha256,
        HERE / 'build.py': args.builder_sha256, HERE / 'BUILD_PROTOCOL.md': args.build_protocol_sha256,
        _base['LEGACY']: _base['LEGACY_SHA'], _base['OBSERVER']: _base['OBSERVER_SHA'], **PINS}
    require(all(isinstance(v, str) and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for v in mapping.values()), 'Every current native source owner is explicit')
    return {str(p): d for p, d in mapping.items()}


def gain_candidate(result, text, args, builder, tick):
    admitted = builder.verify_build(args.build_sha256, args.candidate_sha256,
                                   args.builder_sha256, args.build_protocol_sha256, tick)
    require(result['candidate_sha256'] == args.candidate_sha256
            and Path(admitted['candidate']).read_bytes() == text.encode(), 'Actual built one-span candidate')
    result['source_bindings'].update(admitted['source_bindings'])
    result['input_bindings'].update(admitted['input_bindings'])
    result.update(candidate=admitted['candidate'], build_sha256=args.build_sha256,
        build_proof=admitted['proof'], native_source_proof=NATIVE_SOURCE_PROOF,
        parent_sha256=_base['_BP_CANDIDATE_SHA'], candidate_has_only_gain_slice_change=True,
        backend_selection='explicit production OpenBLAS site prepend', old_fidelity_still_failed=True,
        production_pair_fidelity_pending=True, score_measured=False, raw_error_measured=False,
        adjusted_score_measured=False, automatic_next_stage=False, uploaded=False, submission_authorized=False)


def gain_controls(inputs, tick=None):
    """The negative old stage is only a saved native-output reference, never a PASS."""
    chains = []
    for folder, expected in (('backend_probe_v1', OLD_NATIVE), ('backend_probe_watch_v1', OLD_WATCH)):
        paths = [SHARED / folder / n for n in ('result.json', 'completion.json', 'finalization.json')]
        ds = tuple(sha(p, tick) for p in paths); r, e, f = map(read, paths)
        require(ds == expected and all(x.get('passed') is False and x.get('completed') is False for x in (r, e, f))
                and e['result_sha256'] == f['result_sha256'] == ds[0] and f['completion_sha256'] == ds[1],
                'Exact old negative three-seal evidence remains negative')
        inputs.update({str(p): d for p, d in zip(paths, ds)}); chains.append(r)
    old, watch = chains
    require(old['candidate_sha256'] == _base['_BP_CANDIDATE_SHA'] and old['ranks'] == watch['ranks'] == [35, 35]
            and old['full_model_calls'] == old['retired_call_count'] == watch['retired_calls'] == 2
            and len(old['trials']) == 2
            and len(old['artifacts']) == 14 and watch['cleanup_passed'] and watch['whole_owned_tree_retired']
            and watch['observational_groups_retired'] and watch['maximum_swap_growth_bytes'] == 0
            and watch['maximum_interval_s'] <= 1 and watch['maximum_combined_rss_bytes'] < 8_000_000_000,
            'Two saved, actually retired old native contexts; no overall fidelity authority')
    bind(old['artifacts'], tick); inputs.update(old['artifacts'])
    for index, trial in enumerate(old['trials']):
        marker = read(SHARED / 'backend_probe_v1' / f'call_{index:02d}_complete.json')
        life = read(trial['lifecycle_receipt'])
        require(trial['call'] == index and trial['rank'] == 35
                and trial['flops'] == (403400296797, 401455326499)[index]
                and marker['passed'] and marker['completed'] and marker['context_retired']
                and marker['output_sha256'] == trial['output_sha256']
                and life['complete'] and life['context_released'] and life['closed_context_bill_unchanged_by_reset']
                and read(trial['input_bill'])['weight_file_sha256'] == _base['FIXTURE'][_base['ROW'] / 'weights.npy'],
                'Saved same-physical35 native input/output/lifecycle')
    production, production_seals = _base['backend_controls'](inputs, tick)
    return old['trials'], dict(old_native=list(OLD_NATIVE), old_watch=list(OLD_WATCH), old_overall_passed=False,
        old_production_controls=production, old_production_seals=production_seals)


def gain_compare(np, result, tick):
    require(result['full_model_calls'] == result['retired_call_count'] == 2 and len(result['artifacts']) == 14,
            'Both new native contexts and all14 artifacts exist before target-free comparison')
    bind(result['artifacts'], tick)
    _base['backend_origins'](result, tick)
    checks = []
    for call, (trial, control) in enumerate(zip(result['trials'], result['controls'], strict=True)):
        old = np.load(control['output'], allow_pickle=False)
        new = np.load(trial['output'], allow_pickle=False)
        require(old.dtype == new.dtype == np.float32 and old.shape == new.shape == (16, 1024)
                and np.isfinite(old).all() and np.isfinite(new).all() and (old >= 0).all() and (new >= 0).all(),
                'Same all16 finite nonnegative saved-output contract')
        difference = new.astype(np.float64) - old.astype(np.float64)
        maxima = np.max(np.abs(difference), axis=1).tolist()
        old_d = read(SHARED / 'backend_probe_v1' / f'call_{call:02d}_diagnostics.json')
        new_d = read(result['diagnostic_records'][call]['path'])
        checks.append(dict(call=call, rank=35, layer_maxabs=maxima, all16_maxabs=max(maxima),
            tolerance=1e-5, within_original_vector_tolerance=max(maxima) <= 1e-5,
            output_byte_exact=old.tobytes() == new.tobytes(), domains_and_selectors_exact=old_d['domains'] == new_d['domains'],
            dispatches_exact=old_d['dispatches'] == new_d['dispatches'],
            own_complete_flops=trial['flops'], old_native_complete_flops=control['flops'],
            complete_flops_delta=trial['flops'] - control['flops'], operation_count=trial['operation_count']))
        del old, new, difference
        tick()
    result.update(old_native_checks=checks, old_native_comparison_completed=True,
        old_native_output_byte_exact=all(x['output_byte_exact'] for x in checks),
        old_native_complete_flops_equal=all(x['complete_flops_delta'] == 0 for x in checks),
        old_native_all16_within_tolerance=all(x['within_original_vector_tolerance'] for x in checks),
        all_native_outputs_sealed_before_comparison=True, actual_production_backend_qualified=True)


def verify_native(record, tick=None):
    """Metadata-only post-retirement check; root separately validates its three seals."""
    require(record['schema'] == 'current_comp_shared_gain_slice_native_v1'
            and record['passed'] is True and record['completed'] is True and record['ranks'] == [35, 35]
            and record['full_model_calls'] == record['retired_call_count'] == 2 and len(record['artifacts']) == 14
            and len(record['trials']) == len(record['diagnostic_records']) == len(record['input_preparation']) == 2
            and record['initialization_flops'] == 1944970298
            and all(record[k] is True for k in ('all16_domains_qualified', 'all47_selectors_qualified', 'repeat_byte_exact',
                'own_complete_bills', 'old_native_comparison_completed', 'all_native_outputs_sealed_before_comparison',
                'actual_production_backend_qualified', 'candidate_has_only_gain_slice_change', 'old_fidelity_still_failed',
                'hashes_unchanged', 'sources_unchanged'))
            and all(record[k] is False for k in ('truth_accessed', 'score_measured', 'raw_error_measured',
                'adjusted_score_measured', 'automatic_next_stage', 'uploaded', 'submission_authorized')),
            'Actual two-call native qualification is not borrowed candidate fidelity or score')
    require(record['candidate_sha256'] == record['build_proof']['candidate_sha256']
            and record['build_proof']['parent_sha256'] == _base['_BP_CANDIDATE_SHA']
            and record['build_proof']['one_span_gain_slice'] and record['build_proof']['whole_source_inverse_verified']
            and record['native_source_proof'] == NATIVE_SOURCE_PROOF, 'Current candidate and exact paid owner inverse')
    a = record['backend_attestation']
    require(a['site'] == str(_base['_BP_SITE']) and a['actual_origins_qualified'] and a['production_blas_qualified']
            and a['official_prefix'] == str(_base['OFFPY'].parent.parent)
            and a['numpy_version'] == '2.4.6' and a['flopscope_version'] == '0.12.0'
            and a['whestbench_version'] == '0.16.0' and a['scipy_loaded'] is a['torch_loaded'] is False,
            'Actual unchanged OpenBLAS numerical origins')
    require({'numpy','numpy._core._multiarray_umath','numpy.linalg._linalg',
        'numpy.linalg._umath_linalg','flopscope','flopscope.numpy','threadpoolctl',
        'whestbench.domain','whestbench.scoring','whestbench.seeds'} <= set(a['modules']),
        'Every original required numerical/official owner actually loaded')
    pools = [p for p in a['threadpools'] if p.get('user_api') == 'blas']
    require(len(pools) == 1 and pools[0]['internal_api'] == 'openblas' and pools[0]['num_threads'] == 1
            and pools[0]['version'] == '0.3.31.188.0' and pools[0]['threading_layer'] == 'pthreads'
            and Path(pools[0]['filepath']).resolve() ==
                (_base['_BP_SITE'] / 'numpy/.dylibs/libscipy_openblas64_.dylib').resolve(),
            'Exactly one production BLAS pool')
    for name, item in a['modules'].items():
        root = _base['_BP_OFFSITE'] if name.split('.')[0] == 'whestbench' else _base['_BP_SITE']
        require(Path(item['path']).is_relative_to(root) and record['source_bindings'][item['path']] == item['sha256'],
                'Actual module belongs to declared pinned installation')
    inputs = {}; controls, seals = gain_controls(inputs, tick)
    require(record['controls'] == controls and record['control_seals'] == seals
            and all(record['input_bindings'].get(p) == d for p, d in inputs.items()), 'Exact saved negative baseline scope')
    require(len(record['old_native_checks']) == 2, 'Both saved native comparisons')
    for call, (trial, check) in enumerate(zip(record['trials'], record['old_native_checks'], strict=True)):
        require(trial['call'] == check['call'] == call and trial['rank'] == check['rank'] == 35
                and trial['includes_initialization'] is (call == 0) and 0 < trial['flops'] <= 2**41
                and 0 <= trial['context_wall_s'] < 120 and trial['peak_rss_bytes'] < 8_000_000_000
                and check['own_complete_flops'] == trial['flops'] and len(check['layer_maxabs']) == 16
                and all(math.isfinite(x) and x >= 0 for x in check['layer_maxabs'])
                and check['all16_maxabs'] == max(check['layer_maxabs']) and check['tolerance'] == 1e-5
                and check['within_original_vector_tolerance'] is (check['all16_maxabs'] <= 1e-5)
                and check['old_native_complete_flops'] == controls[call]['flops']
                and check['complete_flops_delta'] == trial['flops'] - controls[call]['flops'], 'Measured, not imposed, old-native comparisons')
    return True


def derive_source():
    original, prior = _base['derive_source']()
    tree = ast.parse(original)
    owners = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'source_owners')
    owners_text = ''.join(original.splitlines(keepends=True)[owners.lineno - 1:owners.end_lineno])
    edits = (
        ("KEYS = ('source','watch','protocol','watch_spec','compose','fnp','design')",
         "KEYS = ('source','watch','protocol','watch_spec','builder','build_protocol','build','candidate')"),
        (owners_text, 'def source_owners(args):\n    return gain_source_owners(args)\n'),
        ("args.out = HERE/'backend_probe_v1'", "args.out = HERE/'native_v1'"),
        ("'current_comp_shared_old384_backend_probe_v1'", "'current_comp_shared_gain_slice_native_v1'"),
        ("composition = load(HERE/'compose.py','shared384_composition',args.compose_sha256)",
         "composition = load(HERE/'build.py','gain_slice_builder',args.builder_sha256)"),
        ('backend_candidate(result,text,tick)', 'gain_candidate(result,text,args,composition,tick)'),
        ('return backend_controls(inputs,tick)', 'return gain_controls(inputs,tick)'),
        ('backend_compare(np,result,tick)', 'gain_compare(np,result,tick)'),
    )
    changed = original
    for before, after in edits:
        require(changed.count(before) == 1, 'Unique native source/authority splice: ' + before[:80])
        changed = changed.replace(before, after)
    inverse = changed
    for before, after in reversed(edits):
        require(inverse.count(after) == 1, 'Unique native source inverse')
        inverse = inverse.replace(after, before)
    require(inverse == original, 'Full same-OpenBLAS native body inverse')
    return changed, dict(owner_sha256=OWNER_SHA, prior=prior, whole_source_inverse_verified=True,
        derived_source_sha256=hashlib.sha256(changed.encode()).hexdigest(), source_and_authority_splices=8,
        original_paid_loop_and_observers_unchanged=True, exact_OpenBLAS_selection_unchanged=True,
        all16_all47_and_input_lifecycle_unchanged=True, original_negative_stage_preserved=True)


_source, NATIVE_SOURCE_PROOF = derive_source()
_api = dict(_base, __file__=str(Path(__file__).resolve()), __name__='_gain_slice_native_body',
    gain_source_owners=gain_source_owners, gain_candidate=gain_candidate,
    gain_controls=gain_controls, gain_compare=gain_compare)
exec(compile(_source, str(Path(__file__).resolve()) + ':exact-native-body', 'exec'), _api)
main = _api['main']
paid_loop_source = _api['paid_loop_source']
del _source


if __name__ == '__main__':
    raise SystemExit(main())
