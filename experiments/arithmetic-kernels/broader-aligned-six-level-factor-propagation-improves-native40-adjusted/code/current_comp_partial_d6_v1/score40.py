"""Fixed native40 saved-output score; no estimator or numerical work on import."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE/'score_v2.py'
_PARENT_SHA = '0c1e2d8a1e41b0502a186eb53b65a62416d7111e34665a08b54f5badd09a7acd'
_CD_SCORE_SHA = '900d99f3dd4b195922cec27cefcfa23c002bd09e086d71c6b3eeeb7685337b03'
_RANKS40 = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,
    204,219,224,227,233,234)
_BUILD_SHA = 'bf17b06afb5e8b7e5b526768c550a7c59eb6ec9808d060cb79b8e1dd083a18af'
_CANDIDATE_SHA = 'ca33a2a4600e7adb10071f6f904936eefe13915278513a8e6291fb5d3e6b562d'
_CHECK4_SCORE_SHA = 'f13553b12399fc92b4953dbc4021de0d0ee22333d17b6e9995b3e7d7de57c582'
_CHECK4_WATCH_SHA = 'ab1a66159649870ff38031368f0d0825c4da33a12ca55730fb43ff11f52a093e'
_FROZEN = {
    'score_v2.py': _PARENT_SHA,
    'SCORE_PROTOCOL_V2.md': '4eefd4e9213d6f0879049fd0acd3423c719c030215c31847d5c1b6f2b305b338',
    'native40.py': '2d7ed4d25338b199b22919fbcce2ee86140a286d88ab18b27cf6f5ecafadff83',
    'NATIVE40_PROTOCOL.md': '1d894b8517ba8a1e7062595a25437332f69dd98d40e54311e6e7743d6b20072d',
    'watch_native40.py': '547a21000879c0f1a7e26e25b5dec0eb4d38a05bc149de7b490b7e3943e4c6d2',
    'NATIVE40_WATCH_SPEC.md': '25a2591062b9f5b020c766dfbac9ec849ea077db75c3186fdbd0a680ff46ba0f',
    'build.py': '6ecb36ab4e855e79ea56f66e86940526f44467be9ea240517f34805499d8e7fc',
    'BUILD_PROTOCOL.md': 'bbc2688be5c65d76492a019bf1aeab89ec3d060e03458a8b9078756f044c69d5',
    'NATIVE40_DECISION.md': 'fcff29cbd256bee54ec619d27979c971dbe91c85c0bd37ec63dd9b9a79f04926',
    'CHECK4_RESULT_20260910.md': '4bcba88152d583035f27916a14c23a92d26c0bbc3c2f472e4e0e7bccd24dd3b5',
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
    nodes = [n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name]
    _require(len(nodes) == 1, 'Unique frozen scorer function '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def _adapt(source, changes):
    changed = source
    for before, after, count in changes:
        _require(changed.count(before) == count, 'Exact score40 source span '+before[:110])
        changed = changed.replace(before, after)
    inverse = changed
    for before, after, count in reversed(changes):
        _require(inverse.count(after) == count, 'Exact score40 inverse span '+after[:110])
        inverse = inverse.replace(after, before)
    _require(inverse == source, 'Whole resolved source inverse')
    ast.parse(changed)
    return changed


def _score40_prior_admission(g, args, admit):
    require, here = g['require'], g['HERE']
    fields = {'native_source': 'native40.py', 'native_protocol': 'NATIVE40_PROTOCOL.md',
        'native_watch_source': 'watch_native40.py', 'native_watch_spec': 'NATIVE40_WATCH_SPEC.md',
        'builder': 'build.py', 'build_protocol': 'BUILD_PROTOCOL.md'}
    require(all(getattr(args, name+'_sha256') == _FROZEN[path] for name, path in fields.items())
        and args.terminal_build_sha256 == _BUILD_SHA and args.terminal_candidate_sha256 == _CANDIDATE_SHA
        and args.check4_score_sha256 == _CHECK4_SCORE_SHA and args.check4_watch_sha256 == _CHECK4_WATCH_SHA,
        'Exactly the reviewed unchanged candidate/native40 and completed CHECK4 authority')
    for path, digest in _FROZEN.items():
        admit.pin(here/path, digest)
    admit.result['native40_score_source_inverse_proof'] = g['SCORE40_SOURCE_PROOF']


def _score40_check4(g, args, admit, owner, context, native):
    """Reconstruct the original target-free five-call authority, never run its scorer."""
    g['require'](owner.NATIVE_SCHEMA == 'current_comp_partial_d6_native40_v1'
        and owner.RANKS == _RANKS40 and owner._NATIVE40_SOURCE_PROOF == native['proof']['native40_adapter'],
        'Exact expanded native metadata source/roster inverse')
    sources, inputs = {}, {}
    check_args = g['SimpleNamespace'](check4_score_sha256=args.check4_score_sha256,
        check4_watch_sha256=args.check4_watch_sha256, candidate_sha256=args.terminal_candidate_sha256,
        builder_sha256=args.builder_sha256, build_protocol_sha256=args.build_protocol_sha256,
        build_sha256=args.terminal_build_sha256, build_watch_sha256=context['build_watch_sha256'])
    prior = owner._native40_prerequisites(vars(owner), check_args, sources, inputs, admit.guard)
    admit.bind(sources); admit.bind(inputs)
    g['require'](context['check4_prerequisites'] == native['check4_prerequisites'] == prior['prerequisites']
        and context['check4_by_call'] == native['check4_by_call'] == prior['by_call']
        and all(context[group].get(path) == digest for group, mapping in
            (('source_bindings', sources), ('input_bindings', inputs)) for path, digest in mapping.items()),
        'Native40 was admitted against exactly the reverified passed CHECK4 and its first-four packets')
    admit.result.update(check4_expansion_prerequisite_passed=True,
        check4_first4_native_packets_reverified=True, prior_external_watch_passed=False,
        prior_scoring_worker_passed=True,
        completed_CHECK4_score_sha256=args.check4_score_sha256,
        completed_CHECK4_watch_sha256=args.check4_watch_sha256)


def _score40_first4_replay(g, result, records, summary, comparisons):
    old = g['read'](g['HERE']/'score_v2/result.json')
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')
    actual, prior = canonical(records[:4]), canonical(old['records'])
    g['require'](len(records) == 40 and actual == prior
        and {path: result['truth_bindings'].get(path) for path in old['truth_bindings']} == old['truth_bindings'],
        'Exact first-four CHECK4 records and original truth bindings; full40 summary is independently new')
    result.update(exact_CHECK4_first4_score_replay=True,
        CHECK4_first4_records_sha256=hashlib.sha256(prior).hexdigest(),
        native40_first4_records_sha256=hashlib.sha256(actual).hexdigest(),
        no_tuning_or_candidate_change=True, no_tuning_or_new_scientific_choice=True)


_require(_sha(_PARENT) == _PARENT_SHA and _sha(_HERE/'score.py') == _CD_SCORE_SHA,
    'Frozen successful CHECK4 scorer source lineage')
_parent_namespace = runpy.run_path(str(_PARENT), run_name='inert_completed_CD_CHECK4_scorer')
_original = _parent_namespace['_source']
_admission_original = _function((_HERE/'score.py').read_text(), '_composition_admission')
_admission_changes = (
    ("here/'native_terminal_v1'", "here/'native40_terminal_v1'", 1),
    ("here/'native_terminal_watch_v1'", "here/'native40_terminal_watch_v1'", 1),
    ("here/'native_terminal_admission_v1'", "here/'native40_terminal_admission_v1'", 1),
    ("admit.chain(nf, get('native'))", "admit.chain(nf, get('native'), limit=1900)", 1),
    ("admit.chain(wf, get('watch'), watch=True)", "admit.chain(wf, get('watch'), watch=True, limit=1900)", 1),
    ("native['full_model_calls'] == native['retired_call_count'] == 5",
     "native['full_model_calls'] == native['retired_call_count'] == 41", 1),
    ('Same retired full admission and five-call candidate lifetime', 'Same retired full admission and41-call candidate lifetime', 1),
    ("here/'native.py'", "here/'native40.py'", 3),
    ("here/'watch_native.py'", "here/'watch_native40.py'", 1),
    ("here/'NATIVE_PROTOCOL.md'", "here/'NATIVE40_PROTOCOL.md'", 1),
    ("here/'NATIVE_WATCH_SPEC.md'", "here/'NATIVE40_WATCH_SPEC.md'", 1),
    ("'current_comp_partial_d6_native_watch_v1'", "'current_comp_partial_d6_native40_watch_v1'", 1),
    ("watch['retired_calls'] == 5 and watch['whole_cap_s'] == 600",
     "watch['retired_calls'] == 41 and watch['whole_cap_s'] == 1900", 1),
    ("('full_source_inverse_verified', 'resource_loop_and_finalization_unchanged', 'only_fixed_terminal5')",
     "('full_source_inverse_verified', 'process_and_resource_loop_unchanged',\n"
     "             'whole1900s_original_native40_envelope', 'per_call120s_and8GB_unchanged',\n"
     "             'separately_retired_source_admission_not_retrospectively_observed', 'same_candidate_only_roster_expansion')", 1),
    ("samples[-1]['retired_calls'] == 5", "samples[-1]['retired_calls'] == 41", 1),
    ("0 <= x['elapsed_s'] < 600 and 0 <= x['retired_calls'] <= 5",
     "0 <= x['elapsed_s'] < 1900 and 0 <= x['retired_calls'] <= 41", 1),
    ("    metadata_source, metadata_proof = owner.metadata_sources()\n",
     "    score40_check4(g, args, admit, owner, context, native)\n"
     "    metadata_source, metadata_proof = owner.metadata_sources()\n", 1),
    ("PRODUCT_FLOPS=owner.PRODUCT_FLOPS, SHAPE=owner.SHAPE, sha=hash_tick, bind=bind_tick,",
     "PRODUCT_FLOPS=owner.PRODUCT_FLOPS, SHAPE=owner.SHAPE, sha=hash_tick, bind=bind_tick,\n"
     "        _native40_replay=owner._native40_replay,", 1),
    ('all_five_domains_and_selectors_reverified=True', 'all41_domains_and_selectors_reverified=True', 1),
    ('native=native_seals, watch=watch_seals, context_sha256=context_sha)',
     "native=native_seals, watch=watch_seals, context_sha256=context_sha, check4=context['check4_prerequisites'])", 1),
)
_admission_source = _adapt(_admission_original, _admission_changes)
_admission_env = dict(ast=ast, json=json, score40_check4=_score40_check4)
exec(compile(_admission_source, str(_HERE/'score.py')+':native40-metadata-only', 'exec'), _admission_env)
_pin_lines = ''.join('    HERE/'+repr(path)+': '+repr(digest)+',\n' for path, digest in _FROZEN.items())
_changes = (
    ("RANKS, ARMS, BUDGET = (37, 39, 42, 48), ('terminal',), 2**41",
     "RANKS, ARMS, BUDGET = "+repr(_RANKS40)+", ('terminal',), 2**41", 1),
    ("SCHEMA = 'current_comp_partial_d6_score_v2'", "SCHEMA = 'current_comp_partial_d6_score40_v1'", 1),
    ('PINS = {\n', 'PINS = {\n'+_pin_lines, 1),
    ("HERE/'SCORE_PROTOCOL_V2.md': args.protocol_sha256", "HERE/'SCORE40_PROTOCOL.md': args.protocol_sha256", 1),
    ("HERE/'native.py': args.native_source_sha256", "HERE/'native40.py': args.native_source_sha256", 1),
    ("HERE/'watch_native.py': args.native_watch_source_sha256", "HERE/'watch_native40.py': args.native_watch_source_sha256", 1),
    ("HERE/'NATIVE_PROTOCOL.md': args.native_protocol_sha256", "HERE/'NATIVE40_PROTOCOL.md': args.native_protocol_sha256", 1),
    ("HERE/'NATIVE_WATCH_SPEC.md': args.native_watch_spec_sha256", "HERE/'NATIVE40_WATCH_SPEC.md': args.native_watch_spec_sha256", 1),
    ("out = HERE/'score_v2'", "out = HERE/'score40_v1'", 1),
    ("native['trials'][:4]", "native['trials'][:40]", 1),
    ('    selected = (0, 1, 2, 3, 40)', '    selected = tuple(range(41))', 1),
    ('for i in (0, 1, 2, 3, 40)', 'for i in range(41)', 1),
    ('for i in range(4)', 'for i in range(40)', 1),
    ('all_five_new_outputs_verified=True', 'all41_new_outputs_verified=True', 1),
    ('Exact four-row original COMP score replay', 'Exact40-row original COMP score replay', 1),
    ("sum(r['per_role'][role][key] for r in records)/4", "sum(r['per_role'][role][key] for r in records)/40", 1),
    ('at_least_three_wins=wins >= 3, gate_passed=improved and wins >= 3',
     'at_least_24_wins=wins >= 24, gate_passed=improved and wins >= 24', 1),
    ("        advances_to_native40=comparisons['terminal_vs_COMP']['gate_passed'],",
     "        advances_to_independent14=comparisons['terminal_vs_COMP']['gate_passed'],", 1),
    ("'partial_D6_component', 'advances_to_native40'", "'partial_D6_component', 'advances_to_independent14'", 1),
    ('Mean of four per-net final raw*max(.1,own completeF/2^41); repeat excluded',
     'Mean of40 per-net final raw*max(.1,own completeF/2^41); repeat excluded', 1),
    ("'builder', 'build-protocol', 'native-watch-spec']",
     "'builder', 'build-protocol', 'native-watch-spec', 'check4-score', 'check4-watch']", 1),
    ('    v2_prior_admission(globals(), args, admit)', '    score40_prior_admission(globals(), args, admit)', 1),
    ('    v2_replay_prior(globals(), result, records, summary, comparisons)',
     '    score40_first4_replay(globals(), result, records, summary, comparisons)', 1),
)
_source = _adapt(_original, _changes)
_proof = dict(parent=str(_PARENT), parent_sha256=_PARENT_SHA,
    resolved_parent_sha256=hashlib.sha256(_original.encode()).hexdigest(),
    derived_source_sha256=hashlib.sha256(_source.encode()).hexdigest(), full_source_inverse_verified=True,
    admission_parent_sha256=_CD_SCORE_SHA, admission_full_source_inverse_verified=True,
    admission_derived_sha256=hashlib.sha256(_admission_source.encode()).hexdigest(),
    original_FP64_per_network_arithmetic_unchanged=True, original_Pareto_definition_unchanged=True,
    original120s_1GB_and_finalization_unchanged=True, same_candidate_only_roster_expansion=True,
    original40_order_and_COMP_comparator=True, first4_scientific_replay_required=True,
    no_retrospective_rescue_of_failed_v1_watch=True, no_estimator_or_fit=True)
SCORER_NAMESPACE = {'__name__': __name__, '__file__': __file__,
    'composition_admission': _admission_env['_composition_admission'],
    'score40_prior_admission': _score40_prior_admission, 'score40_first4_replay': _score40_first4_replay,
    'ENDPOINT_SOURCE_PROOF': _parent_namespace['_parent_namespace']['_proof'],
    'V2_SOURCE_PROOF': _parent_namespace['_proof'], 'SCORE40_SOURCE_PROOF': _proof}
exec(compile(_source, str(_PARENT)+':fixed-native40-score', 'exec'), SCORER_NAMESPACE)
