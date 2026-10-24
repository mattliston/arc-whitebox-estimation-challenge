"""Unchanged COMP/partial-D6 native40, after the completed CHECK4 endpoint."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE/'native.py'
_PARENT_SHA = 'f5d0e710ce97e292cd3ecce8f7ebcded3fbdc8194a54fbf8e41d5695e61d483a'
_CANDIDATE_SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
_RANKS40 = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
    204,219,224,227,233,234)
_CHECK4_SOURCES = {
    'native.py': _PARENT_SHA,
    'NATIVE_PROTOCOL.md': '58c28a340e169ccf31650e4244d0a04e718fe7890025008c0fd57801bc25a813',
    'watch_native.py': '9eb726e3387c3d4bf5f4890ddea6aeb83d6b81bb9277954558dbac39f759e08a',
    'NATIVE_WATCH_SPEC.md': '45857510d1e4d7d4e391ce9bdb00fc1f19fafd82d7d642927828479d057ca057',
    'score.py': '900d99f3dd4b195922cec27cefcfa23c002bd09e086d71c6b3eeeb7685337b03',
    'SCORE_PROTOCOL.md': 'b59c3b0c125859cbfc64d1c614f155069d08bc8160c05ac407326af4200c0dc3',
    'score_v2.py': '0c1e2d8a1e41b0502a186eb53b65a62416d7111e34665a08b54f5badd09a7acd',
    'SCORE_PROTOCOL_V2.md': '4eefd4e9213d6f0879049fd0acd3423c719c030215c31847d5c1b6f2b305b338',
    'watch_score.py': '6e2aa9b1b700e850c9099bfc885fd994f40e96e91bdd122db43402f02a5ec49f',
    'SCORE_WATCH_SPEC.md': '68fe4181d94f3727be4f788bfb8f0b424b517b3a97896d33cd6c5ea1a2a4c969',
    'watch_score_v2.py': 'c2e464345e0a77da272fef90f661286fe1205da24153a122126067ad4d90ca90',
    'SCORE_WATCH_SPEC_V2.md': 'a7f819bafdeadba7d72935a60de4f5f66e7cb733c3151a60c0b02f3d965e325b',
    'NATIVE40_DECISION.md': 'fcff29cbd256bee54ec619d27979c971dbe91c85c0bd37ec63dd9b9a79f04926',
    'CHECK4_RESULT_20260910.md': '4bcba88152d583035f27916a14c23a92d26c0bbc3c2f472e4e0e7bccd24dd3b5',
}
_CHECK4_RECEIPTS = {
    'score_v2/result.json': 'f13553b12399fc92b4953dbc4021de0d0ee22333d17b6e9995b3e7d7de57c582',
    'score_v2/completion.json': 'bc47ebae6496951845a3942a11c0e5696269c9ac97326e0eea4373681cb3881c',
    'score_v2/finalization.json': 'cf7df6869213d2439f06b1cd39bbec419b1bc99a9e10863164b86197024c7fd8',
    'score_watch_v2/result.json': 'ab1a66159649870ff38031368f0d0825c4da33a12ca55730fb43ff11f52a093e',
    'score_watch_v2/completion.json': '8364c636fef1156e4816da457b8a05826e48040dbd545ee03205eaf1a0a72c12',
    'score_watch_v2/finalization.json': '24741039e719c364fbfd7802c46ec3d9357ca6eb38827696e0d7964f286a61fa',
}


def _require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _function(source, name):
    found = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    _require(len(found) == 1, 'Unique source function '+name)
    n = found[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def _adapt(source, changes):
    changed = source
    for before, after, count in changes:
        _require(changed.count(before) == count, 'Exact native40 source span '+before[:100])
        changed = changed.replace(before, after)
    inverse = changed
    for before, after, count in reversed(changes):
        _require(inverse.count(after) == count, 'Exact native40 inverse span '+after[:100])
        inverse = inverse.replace(after, before)
    _require(inverse == source, 'Complete source inverse')
    ast.parse(changed)
    return changed


def _native40_prerequisites(g, args, sources, inputs, guard):
    """Reverify the passed CHECK4 metadata/bytes only; never invoke score/main."""
    require, here = g['require'], g['HERE']
    require(args.check4_score_sha256 == _CHECK4_RECEIPTS['score_v2/result.json']
        and args.check4_watch_sha256 == _CHECK4_RECEIPTS['score_watch_v2/result.json']
        and args.candidate_sha256 == _CANDIDATE_SHA, 'Explicit selected completed CHECK4 and unchanged candidate')
    g['bind']({str(here/p): h for p, h in {**_CHECK4_SOURCES, **_CHECK4_RECEIPTS}.items()})
    scorer = g['module'](here/'score_v2.py', 'native40_completed_check4_metadata_owner')
    sg = scorer.SCORER_NAMESPACE
    result = dict(source_bindings={}, input_bindings={})
    admit = sg['Admission'](result, guard)
    for path, digest in sg['PINS'].items():
        admit.pin(path, digest)
    for path, digest in {**_CHECK4_SOURCES, **_CHECK4_RECEIPTS}.items():
        admit.pin(here/path, digest)
    score, _, ss = admit.chain(here/'score_v2', args.check4_score_sha256, limit=120)
    watch, _, ws = admit.chain(here/'score_watch_v2', args.check4_watch_sha256, watch=True, limit=120)
    require(score['schema'] == 'current_comp_partial_d6_score_v2' and score['integrity_passed']
        and score['hashes_unchanged'] and score['sources_unchanged'] and score['advances_to_native40']
        and score['partial_D6_component'] and score['exact_v1_records_summary_comparisons_replay']
        and score['no_tuning_or_new_scientific_choice'] and score['prior_external_watch_passed'] is False
        and score['comparisons']['terminal_vs_COMP']['mean_adjusted_ratio'] < 1
        and score['comparisons']['terminal_vs_COMP']['adjusted_wins'] >= 3
        and score['ranks'] == [37,39,42,48] and len(score['records']) == 4,
        'Genuinely passed v2 CHECK4 small-gain gate, no rescue of failed v1 watch')
    require(watch['schema'] == 'current_comp_partial_d6_score_watch_v2' and watch['completed']
        and watch['cpu_slot_clear_attested'] and watch['numerical_imports'] is False
        and watch['full_model_calls'] == watch['retired_calls'] == 0 and watch['whole_cap_s'] == 120
        and 0 < watch['maximum_combined_rss_bytes'] < 1_000_000_000
        and 0 < watch['maximum_interval_s'] <= 1
        and all(watch['child_'+key] == value for key, value in ss.items()),
        'Passed independent score supervisor and every actual child seal')
    requested = score['requested_seals']
    require(requested['source_sha256'] == _CHECK4_SOURCES['score_v2.py']
        and requested['protocol_sha256'] == _CHECK4_SOURCES['SCORE_PROTOCOL_V2.md']
        and requested['builder_sha256'] == args.builder_sha256
        and requested['build_protocol_sha256'] == args.build_protocol_sha256
        and requested['terminal_build_sha256'] == args.build_sha256
        and requested['terminal_candidate_sha256'] == args.candidate_sha256,
        'Same selected build and candidate, not a refit/rebuild')
    for name in ('samples.jsonl', 'worker.log', 'started.json'):
        admit.pin(here/'score_watch_v2'/name, g['sha'](here/'score_watch_v2'/name))
    samples = [json.loads(line) for line in (here/'score_watch_v2/samples.jsonl').read_text().splitlines()]
    require(samples and len(samples) == watch['observation_samples']
        and g['math'].isfinite(watch['initial_swap_bytes']) and watch['initial_swap_bytes'] >= 0
        and max(s['combined_rss_bytes'] for s in samples) == watch['maximum_combined_rss_bytes']
        and all(0 <= s['swap_bytes'] <= watch['initial_swap_bytes'] and 0 < s['combined_rss_bytes'] < 1_000_000_000
            and 0 <= s['elapsed_s'] < 120 and s['retired_calls'] == 0 for s in samples),
        'Exact zero-added-swap source-bound score supervision samples')
    score_args = sg['SimpleNamespace'](**requested)
    sg['v2_prior_admission'](sg, score_args, admit)
    env, _ = sg['helpers'](admit)
    folder, baseline, _, refs = sg['baseline'](admit)
    sg['original_control_checks'](admit, env, folder, baseline, refs)
    builder = g['module'](here/'build.py', 'native40_check4_source_build_verifier')
    _, native, lineage = sg['arm_admission'](score_args, 'terminal', admit, env, refs, builder)
    require(score['lineage']['terminal'] == lineage and native['build_watch_sha256'] == args.build_watch_sha256
        and native['full_model_calls'] == native['retired_call_count'] == 5,
        'All five original calls/outputs/bills/domains and retired source/native/build/watch seals reverified')
    packet = {}
    for call in range(4):
        trial, prepared = native['trials'][call], native['input_preparation'][call]
        diagnostic = g['read'](native['diagnostic_records'][call]['path'])
        packet[str(call)] = dict(rank=trial['rank'],
            trial={k: trial[k] for k in ('flops','operation_count','multiplier','includes_initialization',
                'output_bytes_sha256','output_sha256')}, input_flops=prepared['flops'],
            weight_file_sha256=prepared['weight_file_sha256'], domains=diagnostic['domains'],
            dispatches=diagnostic['dispatches'])
    require(not any(k in g['sys'].modules for k in ('numpy','torch','flopscope','threadpoolctl')),
        'CHECK4 expansion admission remained target-free stdlib only')
    for destination, additions in ((sources,result['source_bindings']), (inputs,result['input_bindings'])):
        for path, digest in additions.items():
            require(path not in destination or destination[path] == digest, 'Consistent expansion source/input ownership')
            destination[path] = digest
    guard()
    return dict(prerequisites=dict(score_sha256=args.check4_score_sha256, score_watch_sha256=args.check4_watch_sha256,
        score_seals=ss, score_watch_seals=ws, native_lineage=lineage,
        all_five_prior_calls_reverified=True, strict_CHECK4_gain_gate_passed=True), by_call=packet)


def _native40_replay(trial, prepared, diagnostic, expected):
    require(trial['call'] == prepared['call'] == diagnostic['call'] < 4
        and trial['rank'] == prepared['rank'] == diagnostic['rank'] == expected['rank']
        and all(trial[key] == value for key, value in expected['trial'].items())
        and prepared['flops'] == expected['input_flops']
        and prepared['weight_file_sha256'] == expected['weight_file_sha256']
        and diagnostic['domains'] == expected['domains'] and diagnostic['dispatches'] == expected['dispatches'],
        'First4 exact CHECK4 output bytes, deterministic complete cost, input F, domains and dispatches; no timing/RSS equality')


def _native40_metadata():
    original, parent_proof = _PARENT_NAMESPACE['metadata_sources']()
    parts = {name: _function(original, name) for name in
        ('validate_diagnostic','after_bill','finish_native','verify_acquisition')}
    after = _adapt(parts['after_bill'], [
        ("    result['artifacts'][str(path)] = item['sha256']; result['diagnostic_records'].append(item)\n",
         "    result['artifacts'][str(path)] = item['sha256']; result['diagnostic_records'].append(item)\n"
         "    if trial['call'] < 4:\n"
         "        _native40_replay(trial, result['input_preparation'][-1], diagnostic, result['check4_by_call'][str(trial['call'])])\n", 1)])
    finish = _adapt(parts['finish_native'], [
        ("len(result['diagnostic_records']) == 5", "len(result['diagnostic_records']) == 41", 1),
        ('Complete four primaries and repeat37', 'Complete40 primaries and repeat37', 1),
        ('records[:4]', 'records[:40]', 1),
        ('full_model_calls=5,retired_call_count=5', 'full_model_calls=41,retired_call_count=41', 1),
        ('    for call,record in enumerate(records): validate_diagnostic(record,call)\n',
         '    for call,record in enumerate(records): validate_diagnostic(record,call)\n'
         "    for call in range(4):\n"
         "        _native40_replay(result['trials'][call], result['input_preparation'][call], records[call], result['check4_by_call'][str(call)])\n", 1),
        ('        full_model_calls=41,retired_call_count=41,COMP_parameter_initialization_owned=True,\n',
         '        full_model_calls=41,retired_call_count=41,COMP_parameter_initialization_owned=True,\n'
         '        check4_first4_output_byte_exact=True,check4_first4_deterministic_bills_exact=True,\n'
         '        check4_first4_domains_exact=True,check4_first4_dispatches_exact=True,\n', 1)])
    verify = _adapt(parts['verify_acquisition'], [
        ("record['full_model_calls'] == record['retired_call_count'] == 5", "record['full_model_calls'] == record['retired_call_count'] == 41", 1),
        ("len(record['diagnostic_records']) == 5", "len(record['diagnostic_records']) == 41", 1),
        ('Exact five own prediction/input/diagnostic records', 'Exact41 own prediction/input/diagnostic records', 1),
        ("len(paths) == 30, 'Only original25 artifacts plus five small diagnostics'",
         "len(paths) == 246, 'Only original205 artifacts plus41 small diagnostics'", 1),
        ("    for group in ('source_bindings','input_bindings','artifacts'): bind(record.get(group,{}),tick)\n",
         "    require(record['check4_prerequisites'] == context['check4_prerequisites']\n"
         "            and record['check4_by_call'] == context['check4_by_call']\n"
         "            and all(record.get(k) is True for k in ('check4_expansion_prerequisite_passed',\n"
         "                'check4_first4_output_byte_exact','check4_first4_deterministic_bills_exact',\n"
         "                'check4_first4_domains_exact','check4_first4_dispatches_exact')), 'Passed CHECK4 lineage and first4 deterministic replay')\n"
         "    for group in ('source_bindings','input_bindings','artifacts'): bind(record.get(group,{}),tick)\n", 1)])
    changed = dict(validate_diagnostic=parts['validate_diagnostic'], after_bill=after,
        finish_native=finish, verify_acquisition=verify)
    proof = {name: dict(original_sha256=hashlib.sha256(parts[name].encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(), inverse_verified=True)
        for name, source in changed.items()}
    proof['CHECK4_parent_metadata'] = parent_proof
    return '\n'.join(changed.values()), proof


def _native40_lifecycle():
    original, parent_proof = _PARENT_NAMESPACE['lifecycle_source']()
    before_admit, before_run = _function(original,'admit'), _function(original,'run')
    shared = [("HERE/'NATIVE_PROTOCOL.md'", "HERE/'NATIVE40_PROTOCOL.md'", 1),
        ("HERE/'watch_native.py'", "HERE/'watch_native40.py'", 1)]
    admitted = _adapt(before_admit, shared + [
        ("HERE/'NATIVE_WATCH_SPEC.md'", "HERE/'NATIVE40_WATCH_SPEC.md'", 2),
        ("HERE/f'native_{args.role}_admission_v1'", "HERE/f'native40_{args.role}_admission_v1'", 1),
        ('        admitted_inputs(n, refs, sources, inputs)\n',
         '        admitted_inputs(n, refs, sources, inputs)\n'
         '        check4 = _native40_prerequisites(globals(), args, sources, inputs, guard)\n', 1),
        ('            source_bindings=sources, input_bindings=inputs, reference_by_rank=refs, proof=proof,\n',
         '            source_bindings=sources, input_bindings=inputs, reference_by_rank=refs, proof=proof,\n'
         "            check4_prerequisites=check4['prerequisites'],check4_by_call=check4['by_call'],\n", 1)])
    running = _adapt(before_run, shared + [
        ("HERE/f'native_{args.role}_v1'", "HERE/f'native40_{args.role}_v1'", 1),
        ("HERE/f'native_{args.role}_admission_v1'", "HERE/f'native40_{args.role}_admission_v1'", 1),
        ("        build_watch_sha256=context['build_watch_sha256'], versions=versions,",
         "        build_watch_sha256=context['build_watch_sha256'], versions=versions,\n"
         "        check4_prerequisites=context['check4_prerequisites'],check4_by_call=context['check4_by_call'],\n"
         "        check4_expansion_prerequisite_passed=True,", 1),
        ('elapsed < 600', 'elapsed < 1900', 1)])
    source = admitted+'\n'+running
    proof = dict(parent_source_sha256=_PARENT_SHA, CHECK4_parent_lifecycle=parent_proof,
        admission_inverse_verified=True, run_inverse_verified=True,
        original_admit_sha256=hashlib.sha256(before_admit.encode()).hexdigest(),
        original_run_sha256=hashlib.sha256(before_run.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(), unchanged_paid_loop=True,
        only_new_scope='passed CHECK4 metadata prerequisite,41 fixed calls,1900s and versioned paths')
    return source, proof


_require(_sha(_PARENT) == _PARENT_SHA, 'Exact frozen CHECK4 native source owner')
_PARENT_NAMESPACE = runpy.run_path(str(_PARENT), run_name='inert_original_CD_CHECK4_native')
_original = _PARENT.read_text()
_pd = _PARENT_NAMESPACE['PD']
_require(_sha(_pd) == _PARENT_NAMESPACE['OWNERS'][_pd], 'Original native40 roster owner')
_rosters = [n for n in ast.parse(_pd.read_text()).body if isinstance(n,ast.Assign)
    and len(n.targets) == 1 and isinstance(n.targets[0],ast.Name) and n.targets[0].id == 'VALID']
_require(len(_rosters) == 1 and ast.literal_eval(_rosters[0].value) == _RANKS40, 'Exact original40 order, not a new panel')
_extra_pins = ''.join('    HERE/'+repr(path)+': '+repr(digest)+',\n' for path,digest in _CHECK4_SOURCES.items())
_changes = (
    ('RANKS = VALID = (37, 39, 42, 48)', 'RANKS = VALID = '+repr(_RANKS40), 1),
    ("OUTPUT = HERE/'native_terminal_v1'", "OUTPUT = HERE/'native40_terminal_v1'", 1),
    ("SCHEMA = NATIVE_SCHEMA = 'current_comp_partial_d6_native_v1'", "SCHEMA = NATIVE_SCHEMA = 'current_comp_partial_d6_native40_v1'", 1),
    ('OWNERS = {\n', 'OWNERS = {\n'+_extra_pins, 1),
    ("[refs[r]['call'] for r in RANKS] == [0, 1, 2, 3]", "[refs[r]['call'] for r in RANKS] == list(range(40))", 1),
    ('Same cold37/warm39/42/48 reference frame', 'Same complete40 COMP call history', 1),
    (_function(_original,'metadata_sources'), 'def metadata_sources():\n    return _native40_metadata()\n', 1),
    (_function(_original,'lifecycle_source'), 'def lifecycle_source():\n    return _native40_lifecycle()\n', 1),
    ('        candidate_source_changes_owned_only_by_builder=True)',
     '        candidate_source_changes_owned_only_by_builder=True, native40_adapter=dict(_NATIVE40_SOURCE_PROOF))', 1),
    ("'build-protocol', 'build', 'candidate', 'build-watch'):",
     "'build-protocol', 'build', 'candidate', 'build-watch', 'check4-score', 'check4-watch'):", 1),
    ("('build-protocol', 'build', 'candidate', 'build-watch') if args.admit",
     "('build-protocol', 'build', 'candidate', 'build-watch', 'check4-score', 'check4-watch') if args.admit", 1),
)
_derived = _adapt(_original, _changes)
_NATIVE40_SOURCE_PROOF = dict(parent=str(_PARENT), parent_sha256=_PARENT_SHA,
    derived_sha256=hashlib.sha256(_derived.encode()).hexdigest(), full_source_inverse_verified=True,
    paid_main_unchanged=True, all_three_observer_sources_unchanged=True,
    candidate_sha256=_CANDIDATE_SHA, no_fit_or_rebuild=True, no_new_array_operations=True,
    exact_original40_roster=True, first4_replay_metadata_only=True)
exec(compile(_derived, str(_PARENT)+':fixed-native40-expansion', 'exec'), globals())
