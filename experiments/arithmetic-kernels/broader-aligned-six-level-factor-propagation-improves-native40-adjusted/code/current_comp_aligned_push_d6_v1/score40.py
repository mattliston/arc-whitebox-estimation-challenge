"""Fixed original40 APUSH saved-output score; inert without explicit --run."""
import ast
import hashlib
import json
from pathlib import Path
import runpy

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE/'score.py'
_PARENT_SHA = '4804a14f9442d431234d724d9de3ac15fb483aa08ed39b7df339286dbf781074'
_RANKS40 = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)
_FROZEN = {
    'score.py': _PARENT_SHA,
    'SCORE_PROTOCOL.md': '5517899018c68710c7dd728d241592ff1976cd4a9f84b136d59da368e15ab212',
    'native40.py': 'cae10c7f225c9ea34e60e439508582e93a933cac4ee3b9b4c0abb822436c2ee9',
    'NATIVE40_PROTOCOL.md': 'e261df146c48ce7b68e93f7b5800a842e0125a36e953635e33905d576d81743d',
    'watch_native40.py': '429b9e3de389030e3a6d3326cc76a3abdef26432776d5b88c6056e194b5d277b',
    'NATIVE40_WATCH_SPEC.md': '4a74b80e92691b414093b7a4133b4ccfebb351ae75118d40986b6a3e857d3f72',
    'NATIVE40_DECISION.md': 'be6bc210be17bb4e325e555fb484bf8ec7b23c6506a027cf8d7c2269aa198f0a',
}


def _require(ok, message):
    if not ok: raise RuntimeError(message)


def _source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _function(source, name):
    nodes = [node for node in ast.parse(source).body if isinstance(node,ast.FunctionDef) and node.name == name]
    _require(len(nodes) == 1, 'Unique frozen scorer function '+name)
    node = nodes[0]
    return ''.join(source.splitlines(keepends=True)[node.lineno-1:node.end_lineno])


def _adapt(source, changes):
    changed = source
    for before,after,count in changes:
        _require(changed.count(before) == count, 'Exact score40 source span '+before[:110])
        changed = changed.replace(before,after)
    inverse = changed
    for before,after,count in reversed(changes):
        _require(inverse.count(after) == count, 'Exact score40 inverse span '+after[:110])
        inverse = inverse.replace(after,before)
    _require(inverse == source, 'Whole frozen APUSH scorer source inverse')
    ast.parse(changed)
    return changed


def _score40_source_args(args):
    for field,name in (('native_source','native40.py'),('native_protocol','NATIVE40_PROTOCOL.md'),
        ('native_watch_source','watch_native40.py'),('native_watch_spec','NATIVE40_WATCH_SPEC.md')):
        require(getattr(args,field+'_sha256') == _FROZEN[name], 'Exactly the prospectively reviewed native40 owner')


def _score40_check4(admit, owner, context, native, args):
    """Passed CHECK4 is a direct decision boundary, not a stage to rerun."""
    require(owner.NATIVE_SCHEMA == 'current_comp_aligned_push_d6_native40_v1'
        and owner.RANKS == _RANKS40 and owner._NATIVE40_SOURCE_PROOF == native['proof']['native40_adapter']
        and len(native['artifacts']) == 246, 'Exact frozen native40 source/roster and complete own artifact packet')
    for name,digest in owner._CHECK4_SOURCES.items():
        admit.pin(HERE/name,digest)
    chains = owner._CHECK4_CHAINS
    score,_,ss = admit.chain(HERE/'score_v2',chains['score_v2'][0],limit=120,
        expected=chains['score_v2'],inventory=True)
    watch,_,ws = admit.chain(HERE/'score_watch_v1',chains['score_watch_v1'][0],limit=120,
        expected=chains['score_watch_v1'],inventory=True)
    require(score['schema'] == 'current_comp_aligned_push_d6_score_v2'
        and score['candidate_sha256'] == args.candidate_sha256
        and [row['rank'] for row in score['records']] == [37,39,42,48]
        and all(score.get(k) is True for k in ('integrity_passed','hashes_unchanged','sources_unchanged',
            'advances_to_native40','aligned_push_component','all_predictions_verified_before_truth',
            'exact_CD_score_replay','complete_APUSH_acquisition_reverified','saved_external_watch_revalidated',
            'original_external_watch_still_failed'))
        and score['numerical_acquisition_repeated'] is False
        and score['comparisons']['terminal_vs_CD']['mean_adjusted_ratio'] < 1
        and score['comparisons']['terminal_vs_CD']['adjusted_wins'] >= 3,
        'Actual passed CHECK4 decision, not an invented future prerequisite')
    require(watch['schema'] == 'current_comp_aligned_push_d6_score_watch_v1'
        and watch['candidate_sha256'] == args.candidate_sha256
        and all(watch.get(k) is True for k in ('hashes_unchanged','sources_unchanged','scientific_gate_passed',
            'whole_owned_tree_retired','cleanup_passed','observational_groups_retired','actual_completed_samples'))
        and all(watch['child_'+key] == value for key,value in ss.items())
        and watch['elapsed_s'] < 120 and 0 < watch['maximum_combined_rss_bytes'] < 1_000_000_000
        and 0 < watch['maximum_interval_s'] <= 1 and watch['maximum_swap_growth_bytes'] == 0,
        'Positive CHECK4 score-watch three seals and actual saved resource authority')
    prerequisite = context['check4_prerequisites']
    require(prerequisite == native['check4_prerequisites']
        and prerequisite['score_seals'] == ss and prerequisite['score_watch_seals'] == ws
        and prerequisite['score_sha256'] == ss['result_sha256']
        and prerequisite['score_watch_sha256'] == ws['result_sha256']
        and prerequisite['native_lineage'] == score['lineage']['terminal']
        and all(prerequisite.get(k) is True for k in ('all_five_prior_calls_reverified','strict_CHECK4_gain_gate_passed',
            'original_native_watch_still_failed','completed_score_v2_trust_boundary','no_score_or_metadata_repair_reexecution'))
        and context['check4_by_call'] == native['check4_by_call']
        and context['check4_expansion_prerequisite_passed'] is True,
        'New native40 was admitted against the exact completed CHECK4 boundary and replay packet')
    # Preserve the three exact failed historical seals without running their repair again.
    negative = []
    for name,digest in zip(('result.json','completion.json','finalization.json'),chains['native_watch_v1']):
        path = HERE/'native_watch_v1'/name; admit.pin(path,digest); negative.append(read(path))
    require(all(row.get('passed') is False and row.get('completed') is False for row in negative)
        and negative[1]['result_sha256'] == negative[2]['result_sha256'] == chains['native_watch_v1'][0]
        and negative[2]['completion_sha256'] == chains['native_watch_v1'][1], 'Original native5 watch remains FAILED')
    flags = ('check4_expansion_prerequisite_passed','check4_first4_output_byte_exact',
        'check4_first4_deterministic_bills_exact','check4_first4_domains_exact','check4_first4_dispatches_exact')
    require(all(native.get(k) is True for k in flags), 'Complete new first4 byte/cost/domain/dispatch replay')
    admit.result.update(check4_expansion_prerequisite_passed=True,check4_first4_native_packets_reverified=True,
        original_external_watch_still_failed=True,completed_CHECK4_score_sha256=ss['result_sha256'],
        completed_CHECK4_watch_sha256=ws['result_sha256'],check4_seals=dict(score=ss,watch=ws),
        completed_CHECK4_boundary_not_reexecuted=True,native40_score_source_inverse_proof=dict(_SCORE40_SOURCE_PROOF))


def _score40_first4_replay(result, records, summary):
    prior = read(HERE/'score_v2/result.json')
    canonical = lambda value: json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    before,after = canonical(prior['records']),canonical(records[:4])
    require(len(records) == 40 and before == after
        and {path:result['truth_bindings'].get(path) for path in prior['truth_bindings']} == prior['truth_bindings'],
        'Exact original CHECK4 four score rows, including CD/APUSH raw, ownF, layerMSE, ratios and original truth bindings')
    require(summary['CD'] == read(CD/'score40_v1/result.json')['summary']['terminal'],
        'Exact all40 CD original-precision summary replay')
    result.update(exact_CHECK4_first4_score_replay=True,exact_CD_all40_summary_replay=True,
        CHECK4_first4_records_sha256=hashlib.sha256(before).hexdigest(),
        native40_first4_records_sha256=hashlib.sha256(after).hexdigest(),no_tuning_or_candidate_change=True)


def _score40_arithmetic():
    original,parent_proof = _PARENT_NAMESPACE['arithmetic_source']()
    changed = _adapt(original, [
        ('for i in (0, 1, 2, 3, 40)','for i in range(41)',1),
        ('for i in range(4)','for i in range(40)',1),
        ('all_five_new_outputs_verified=True','all41_new_outputs_verified=True',1),
        ('Exact four-row original CD native40 score replay','Exact40-row original CD native40 score replay',1),
        ("sum(r['per_role'][role][key] for r in records)/4","sum(r['per_role'][role][key] for r in records)/40",1),
        ('at_least_three_wins=wins >= 3, gate_passed=improved and wins >= 3',
         'at_least_30_wins=wins >= 30, gate_passed=improved and wins >= 30',1),
        ("        advances_to_native40=comparisons['terminal_vs_CD']['gate_passed'],",
         "        advances_to_independent14=comparisons['terminal_vs_CD']['gate_passed'],",1),
        ('Mean of four per-net final raw*max(.1,own completeF/2^41); repeat excluded',
         'Mean of40 per-net final raw*max(.1,own completeF/2^41); repeat excluded',1),
        ("    for path, digest in {**admit.pins, **result['truth_bindings'], **result['artifacts']}.items():\n",
         "    _score40_first4_replay(result,records,summary)\n"
         "    for path, digest in {**admit.pins, **result['truth_bindings'], **result['artifacts']}.items():\n",1)])
    return changed,dict(parent_CHECK4_arithmetic=parent_proof,original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_function_sha256=hashlib.sha256(changed.encode()).hexdigest(),full_function_inverse_verified=True,
        original_FP64_per_network_arithmetic_unchanged=True,original_mean_product_objective_unchanged=True,
        exact_original40_roster=True,prospective_strict_mean_and30wins=True,exact_CHECK4_first4_replay_required=True)


_require(_source_sha(_PARENT) == _PARENT_SHA, 'Frozen original APUSH CHECK4 arithmetic/scorer owner')
_PARENT_NAMESPACE = runpy.run_path(str(_PARENT),run_name='inert_original_APUSH_CHECK4_scorer')
_original = _PARENT.read_text()
_pin_lines = ''.join('    HERE/'+repr(name)+': '+repr(digest)+',\n' for name,digest in _FROZEN.items())
_changes = (
    ("RANKS, ARMS, BUDGET = (37, 39, 42, 48), ('terminal',), 2**41",
     "RANKS, ARMS, BUDGET = "+repr(_RANKS40)+", ('terminal',), 2**41",1),
    ("SCHEMA = 'current_comp_aligned_push_d6_score_v1'","SCHEMA = 'current_comp_aligned_push_d6_score40_v1'",1),
    ('PINS = {\n','PINS = {\n'+_pin_lines,1),
    (_function(_original,'arithmetic_source'),'def arithmetic_source():\n    return _score40_arithmetic()\n',1),
    ('def acquisition_admission(args, out, result, guard):\n',
     'def acquisition_admission(args, out, result, guard):\n    _score40_source_args(args)\n',1),
    ("('Archived own COMP', 'Archived own CD', 1)])",
     "('Archived own COMP', 'Archived own CD', 1),\n"
     "        ('    selected = (0, 1, 2, 3, 40)', '    selected = tuple(range(41))', 1)])",1),
    ("native['initialization_flops'] == INITIALIZATION and native['ranks'][:4] == list(RANKS)",
     "native['initialization_flops'] == INITIALIZATION and native['ranks'] == list(RANKS)",1),
    ("native['trials'][:4]","native['trials'][:40]",1),
    ('Qualified completed CD native40 boundary, original first-four histories',
     'Qualified completed CD native40 boundary, all40 original histories',1),
    ('One canonical weight fixture per CHECK rank','One canonical weight fixture per original VALID rank',1),
    ('CD native first-four exact canonical weight SHA','CD native40 exact canonical weight SHA',1),
    ("HERE/'SCORE_PROTOCOL.md': args.protocol_sha256","HERE/'SCORE40_PROTOCOL.md': args.protocol_sha256",1),
    ("HERE/'watch_score.py': args.watcher_sha256","HERE/'watch_score40.py': args.watcher_sha256",1),
    ("HERE/'SCORE_WATCH_SPEC.md': args.watch_spec_sha256","HERE/'SCORE40_WATCH_SPEC.md': args.watch_spec_sha256",1),
    ("HERE/'native.py': args.native_source_sha256","HERE/'native40.py': args.native_source_sha256",1),
    ("spec_from_file_location('apush_saved_metadata_verifier', HERE/'native.py')",
     "spec_from_file_location('apush_saved_metadata_verifier', HERE/'native40.py')",1),
    ("HERE/'NATIVE_PROTOCOL.md': args.native_protocol_sha256","HERE/'NATIVE40_PROTOCOL.md': args.native_protocol_sha256",1),
    ("HERE/'watch_native.py': args.native_watch_source_sha256","HERE/'watch_native40.py': args.native_watch_source_sha256",1),
    ("HERE/'NATIVE_WATCH_SPEC.md': args.native_watch_spec_sha256","HERE/'NATIVE40_WATCH_SPEC.md': args.native_watch_spec_sha256",1),
    ("('native_terminal_v1', 'native_terminal_admission_v1', 'native_watch_v1')",
     "('native40_terminal_v1', 'native40_terminal_admission_v1', 'native40_watch_v1')",1),
    ('limit=600, inventory=True','limit=1900, inventory=True',2),
    ("'current_comp_aligned_push_d6_native_admission_v1'","'current_comp_aligned_push_d6_native40_admission_v1'",1),
    ("'current_comp_aligned_push_d6_native_context_v1'","'current_comp_aligned_push_d6_native40_context_v1'",1),
    ("'current_comp_aligned_push_d6_native_v1'","'current_comp_aligned_push_d6_native40_v1'",1),
    ("native['full_model_calls'] == native['retired_call_count'] == len(native['trials']) == 5\n"
     "        and native['initialization_flops'] == INITIALIZATION and native['truth_accessed'] is False",
     "native['full_model_calls'] == native['retired_call_count'] == len(native['trials']) == 41\n"
     "        and native['initialization_flops'] == INITIALIZATION and native['truth_accessed'] is False",1),
    ('Complete APUSH five-call own bills, all16/47, initialization, ordered generic dispatch qualification',
     'Complete APUSH41 own bills, all16/47, initialization, ordered generic dispatch qualification',1),
    ("'current_comp_aligned_push_d6_native_watch_v1'","'current_comp_aligned_push_d6_native40_watch_v1'",1),
    ("watch['full_model_calls'] == watch['retired_call_count'] == 5",
     "watch['full_model_calls'] == watch['retired_call_count'] == 41",1),
    ("watch['elapsed_s'] < 600","watch['elapsed_s'] < 1900",1),
    ("watch['all5_complete'] is True","watch['all41_complete'] is True",1),
    ("('native.py', 'NATIVE_PROTOCOL.md', 'watch_native.py', 'NATIVE_WATCH_SPEC.md')",
     "('native40.py', 'NATIVE40_PROTOCOL.md', 'watch_native40.py', 'NATIVE40_WATCH_SPEC.md')",1),
    ('APUSH was admitted against the exact retained CD first-four packets',
     'APUSH was admitted against the exact retained CD all40 packets',1),
    ('    owner.verify_acquisition(native, end, args.native_sha256, na, context, guard)\n',
     '    owner.verify_acquisition(native, end, args.native_sha256, na, context, guard)\n'
     '    _score40_check4(admit,owner,context,native,args)\n',1),
    ('all_five_new_outputs_verified=True','all41_new_outputs_verified=True',1),
    ('all_five_domains_and_selectors_reverified=True','all41_domains_and_selectors_reverified=True',1),
    ("watch=ws, context_sha256=context_sha)","watch=ws, context_sha256=context_sha, check4=context['check4_prerequisites'])",1),
    ("out = HERE/'score_v1'","out = HERE/'score40_v1'",1),
    ("str(JOINT)+':fixed-APUSH-CHECK4'","str(JOINT)+':fixed-APUSH-original40'",1),
    ("'advances_to_native40', 'error'","'advances_to_independent14', 'error'",1),
)
_derived = _adapt(_original,_changes)
_SCORE40_SOURCE_PROOF = dict(parent=str(_PARENT),parent_sha256=_PARENT_SHA,
    derived_source_sha256=hashlib.sha256(_derived.encode()).hexdigest(),full_source_inverse_verified=True,
    same_fourteen_explicit_SHA_arguments=True,original120s_1GB_and_three_seals_unchanged=True,
    original_FP64_per_network_arithmetic_unchanged=True,original_mean_product_objective_unchanged=True,
    original40_order_and_CD_reference=True,first4_CHECK4_scientific_replay_required=True,
    prospective_strict_mean_and30wins=True,positive_native40_watch_required=True,
    completed_CHECK4_boundary_not_reexecuted=True,original_failed_native5_watch_preserved=True,
    no_estimator_or_fit=True,no_future_receipt_hashes_invented=True)
exec(compile(_derived,str(_PARENT)+':fixed-native40-saved-score','exec'),globals())
