"""Inert fixed APUSH native40 expansion after the completed CHECK4 gate."""
import ast
import hashlib
from pathlib import Path
import runpy
from types import SimpleNamespace

_HERE = Path(__file__).resolve().parent
_PARENT = _HERE/'native.py'
_PARENT_SHA = '751bc155506b2825a8c0d27553e8ced8d7db66d68434247bb158c22fad3c3ad6'
_RANKS40 = (37,39,42,48,59,63,66,67,71,73,75,79,93,101,122,130,132,135,
    139,145,146,149,150,151,155,159,163,169,173,175,181,192,198,199,204,219,224,227,233,234)
_CHECK4_SOURCES = {
    'native.py': _PARENT_SHA,
    'NATIVE_PROTOCOL.md': '96e1da8241eee46e18fcf9adb75194d22269d2894ad3f8e1e1dee3d2c6ef4c38',
    'watch_native.py': 'dca03cd8830f02faee4757383e541e759bc0170cb9299cc60d1283326af31064',
    'NATIVE_WATCH_SPEC.md': 'f6f48bd29779e9b61663250cde281937e6cc0df249d3f278871331686cf9e19b',
    'score.py': '4804a14f9442d431234d724d9de3ac15fb483aa08ed39b7df339286dbf781074',
    'SCORE_PROTOCOL.md': '5517899018c68710c7dd728d241592ff1976cd4a9f84b136d59da368e15ab212',
    'score_v2.py': '5108fdbffaa581abecfc846863a75dca7b2095ca8c0878f31bd838eede34c60e',
    'SCORE_PROTOCOL_V2.md': 'e9a8f3e4a86fdefe2a501afacd38e50071df3afdf78e3b0e2a0182b50e372ece',
    'watch_score.py': '09277db8bf2fd12c881fa6399d81ee103b6012f0832c2061c35e5f4ee77a1c6b',
    'SCORE_WATCH_SPEC.md': '031fe1bad20197896801a309bde6cf8411ef5b94a77092f44e9582c4971494ca',
}
_CHECK4_CHAINS = {
    'score_v2': ('73b1db635ae9ec4369a7a767b7ed175ba2b0c5e96eb3b921d67b680d7a463a67',
        '0a27c7494c1a602410fff5b40008dd763e0bb2140cc2fb4e5c60fbe65f5a4f87',
        '635aea06e6500e875f41e84baa659910149db7d977c9880a92f584e66e0ca112'),
    'score_watch_v1': ('56601e0862eff783e103d6fbf7a7bc71cc3874127320d06b3762bc6e17996d9f',
        '7c8e0ff73e97dacc4e9e3a627e47cf2208ab6452ec58f20bb7dde9489c5776c5',
        'f1deff31731846e5e08574e1cfe53f4763a4bc8eff06ea10f510d617f100a5e8'),
    'native_terminal_v1': ('b9372a024b1257ead34fe8e36fcd85014735f73e0b2a6bbb83bfaca2b8f3c1e8',
        'ed9222dc1cfdc50409746642a8102c426b314e447e920d3b2e1a0e57e62ebba1',
        'b6ccb318ef0763fb2fb09f5266238de7ad633cb711c237e13b07230b135be5c3'),
    'native_terminal_admission_v1': ('b001350aa7bd9ff495923dca6bd8d99c16c5ec76336fe771119163608cc0c052',
        'd07ca6cb175d7f56d245b25e084f9015d8c0e09bfaa5c309366df56a03b36239',
        '3b58771d8230b1150c72f0d3e3924d39bb3ae03ddc8442b84a03297556727fb1'),
    'native_watch_v1': ('94b6e66e317cb458ebb6000370f4451fac9052f15264aae2d3ad1d49a8da388d',
        '595c8d3a2cf1da3902bba456bbdc3b667ad1880b1bab582c3f2124c74a400804',
        'b13a7bff299a26d87589ef1cddb0337a53494e882b07bb5d171fa55aef87be69'),
}
_CHECK4_CONTEXT_SHA = '33ac0536452e6440e0931df26a6d4ce48af9b6ab8cd8e3f7a177c2cda3a4e810'


def _require(ok, message):
    if not ok: raise RuntimeError(message)


def _source_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _function(source, name):
    found = [n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name == name]
    _require(len(found) == 1, 'Unique native40 source function '+name)
    n = found[0]
    return ''.join(source.splitlines(keepends=True)[n.lineno-1:n.end_lineno])


def _adapt(source, changes):
    changed = source
    for before,after,count in changes:
        _require(changed.count(before) == count, 'Exact native40 source span '+before[:100])
        changed = changed.replace(before,after)
    inverse = changed
    for before,after,count in reversed(changes):
        _require(inverse.count(after) == count, 'Exact native40 inverse span '+after[:100])
        inverse = inverse.replace(after,before)
    _require(inverse == source, 'Complete frozen APUSH source inverse')
    ast.parse(changed)
    return changed


def _merge(destination, additions):
    for path,digest in additions.items():
        require(path not in destination or destination[path] == digest, 'Consistent direct prerequisite binding')
        destination[path] = digest


def _native40_reviewed_owners(g, args):
    owners = {**g['OWNERS'], **{_HERE/name:digest for name,digest in _CHECK4_SOURCES.items()},
        Path(__file__).resolve():args.source_sha256,
        _HERE/'NATIVE40_PROTOCOL.md':args.protocol_sha256,
        _HERE/'watch_native40.py':args.watcher_sha256,
        _HERE/'NATIVE40_WATCH_SPEC.md':args.watch_spec_sha256}
    return {str(path):digest for path,digest in owners.items()}


def _native40_prerequisites(args, sources, inputs, tick):
    """Trust the passed CHECK4 decision; independently reverify its native5 packet."""
    require(args.check4_score_sha256 == _CHECK4_CHAINS['score_v2'][0]
        and args.check4_watch_sha256 == _CHECK4_CHAINS['score_watch_v1'][0],
        'Explicit exact completed CHECK4 score and positive saved-evidence watch')
    records = {}
    for name,digests in _CHECK4_CHAINS.items():
        folder = HERE/name
        if name == 'native_watch_v1':
            rows = []
            for filename,digest in zip(('result.json','completion.json','finalization.json'),digests):
                path = folder/filename
                require(sha(path,tick) == digest, 'Preserved exact negative original native-watch seal')
                inputs[str(path)] = digest; rows.append(read(path))
            require(all(row.get('passed') is False and row.get('completed') is False for row in rows)
                and rows[1]['result_sha256'] == rows[2]['result_sha256'] == digests[0]
                and rows[2]['completion_sha256'] == digests[1], 'Original watch remains FAILED, never promoted')
            records[name] = rows
        else:
            records[name] = seal3(folder,digests,inputs,tick=tick)
    score,_,sf = records['score_v2']; watch,_,wf = records['score_watch_v1']
    flags = ('integrity_passed','hashes_unchanged','sources_unchanged','advances_to_native40',
        'aligned_push_component','all_predictions_verified_before_truth','exact_CD_score_replay',
        'complete_APUSH_acquisition_reverified','all_five_new_outputs_verified',
        'all_five_domains_and_selectors_reverified','own_complete_bills','runtime_prerequisite_qualified',
        'no_model_imported','saved_external_watch_revalidated','original_external_watch_still_failed',
        'repaired_metadata_only')
    comparison = score['comparisons']['terminal_vs_CD']
    require(score['schema'] == 'current_comp_aligned_push_d6_score_v2' and score['stage'] == 'score'
        and score['candidate_sha256'] == CANDIDATE_SHA and all(score.get(k) is True for k in flags)
        and score['full_model_calls'] == score['parameter_updates'] == 0
        and score['numerical_acquisition_repeated'] is False and score['original_observation_samples_verified'] == 644
        and [row['rank'] for row in score['records']] == [37,39,42,48]
        and comparison['gate_passed'] is True and comparison['mean_adjusted_ratio'] < 1
        and comparison['adjusted_wins'] >= 3 and 0 <= score['elapsed_s'] <= sf['elapsed_s'] < 120
        and 0 < score['peak_rss_bytes'] < 1_000_000_000, 'Actual strict own-cost CHECK4 decision, not a new score')
    watch_flags = ('hashes_unchanged','sources_unchanged','child_execution_completed','whole_owned_tree_retired',
        'cleanup_passed','observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked',
        'scientific_gate_passed')
    require(watch['schema'] == 'current_comp_aligned_push_d6_score_watch_v1' and watch['stage'] == 'score'
        and watch['candidate_sha256'] == CANDIDATE_SHA and all(watch.get(k) is True for k in watch_flags)
        and tuple(watch[k] for k in ('child_result_sha256','child_completion_sha256','child_finalization_sha256'))
            == _CHECK4_CHAINS['score_v2']
        and watch['full_model_calls'] == watch['parameter_updates'] == 0
        and watch['numerical_imports'] is False and watch['truth_arrays_decoded'] is False
        and 0 <= watch['elapsed_s'] <= wf['elapsed_s'] < 120
        and 0 < watch['maximum_combined_rss_bytes'] < 1_000_000_000
        and 0 < watch['maximum_interval_s'] <= 1 and watch['maximum_swap_growth_bytes'] == 0
        and 0 <= wf['terminal_publication_interval_s'] <= 1,
        'Actual independent positive score-watch resource and retirement boundary')
    for record in (score,watch):
        require(record['uploaded'] is False and record['automatic_next_stage'] is False, 'No inherited upload/automatic authority')
        # truth_bindings are sealed metadata, deliberately never traversed or copied.
        for group,destination in (('source_bindings',sources),('input_bindings',inputs),('artifacts',inputs)):
            bind(record[group],tick); _merge(destination,record[group])
    requested = score['requested_seals']; lineage = score['lineage']['terminal']
    require(all(requested[key] == value for key,value in {
        'source_sha256':_CHECK4_SOURCES['score_v2.py'], 'protocol_sha256':_CHECK4_SOURCES['SCORE_PROTOCOL_V2.md'],
        'watcher_sha256':_CHECK4_SOURCES['watch_score.py'], 'watch_spec_sha256':_CHECK4_SOURCES['SCORE_WATCH_SPEC.md'],
        'native_source_sha256':_PARENT_SHA, 'native_protocol_sha256':_CHECK4_SOURCES['NATIVE_PROTOCOL.md'],
        'native_watch_source_sha256':_CHECK4_SOURCES['watch_native.py'],
        'native_watch_spec_sha256':_CHECK4_SOURCES['NATIVE_WATCH_SPEC.md'],
        'native_sha256':_CHECK4_CHAINS['native_terminal_v1'][0],
        'native_watch_sha256':_CHECK4_CHAINS['native_watch_v1'][0],
        'admission_sha256':_CHECK4_CHAINS['native_terminal_admission_v1'][0],
        'runtime_sha256':RUNTIME_SHA,'build_sha256':BUILD_SHA,'candidate_sha256':CANDIDATE_SHA}.items()),
        'Exact passed score subject, original native5 and unchanged runtime/build')
    seal_keys = ('result_sha256','completion_sha256','finalization_sha256')
    expected_lineage = {key:dict(zip(seal_keys,_CHECK4_CHAINS[name])) for key,name in (
        ('native','native_terminal_v1'),('admission','native_terminal_admission_v1'),('watch','native_watch_v1'))}
    expected_lineage.update(context_sha256=_CHECK4_CONTEXT_SHA,
        build=dict(zip(seal_keys,CHAINS[HERE/'build_v1'])),runtime=dict(zip(seal_keys,CHAINS[HERE/'runtime_v1'])))
    require(lineage == expected_lineage, 'Direct exact native5 lineage; no ancestor experiment reconstruction')
    old_args = SimpleNamespace(role='terminal',source_sha256=_PARENT_SHA,
        protocol_sha256=_CHECK4_SOURCES['NATIVE_PROTOCOL.md'],watcher_sha256=_CHECK4_SOURCES['watch_native.py'],
        watch_spec_sha256=_CHECK4_SOURCES['NATIVE_WATCH_SPEC.md'],context_sha256=_CHECK4_CONTEXT_SHA,
        builder_sha256=BUILDER_SHA)
    old_context = _PARENT_NAMESPACE['read_context'](old_args,_CHECK4_CONTEXT_SHA,tick=tick)
    native,end,_ = records['native_terminal_v1']
    _PARENT_NAMESPACE['verify_native'](native,end,_CHECK4_CHAINS['native_terminal_v1'][0],old_args,old_context,tick=tick)
    require(native['full_model_calls'] == native['retired_call_count'] == 5 and len(native['artifacts']) == 30,
        'All five original input/prediction/domain/lifecycle packets independently reverified')
    for record in (old_context,native):
        for group,destination in (('source_bindings',sources),('input_bindings',inputs),('artifacts',inputs)):
            additions = record.get(group,{})
            bind(additions,tick); _merge(destination,additions)
    inputs[str(HERE/'native_terminal_admission_v1/context.json')] = _CHECK4_CONTEXT_SHA
    packet = {}
    for call in range(4):
        trial,prepared = native['trials'][call],native['input_preparation'][call]
        diagnostic = read(native['diagnostic_records'][call]['path'])
        packet[str(call)] = dict(rank=trial['rank'],
            trial={k:trial[k] for k in ('flops','operation_count','multiplier','includes_initialization',
                'output_bytes_sha256','output_sha256')},input_flops=prepared['flops'],
            weight_file_sha256=prepared['weight_file_sha256'],domains=diagnostic['domains'],dispatches=diagnostic['dispatches'])
    require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules), 'Target-free stdlib CHECK4 admission')
    tick()
    return dict(prerequisites=dict(score_sha256=args.check4_score_sha256,score_watch_sha256=args.check4_watch_sha256,
        score_seals=dict(zip(seal_keys,_CHECK4_CHAINS['score_v2'])),
        score_watch_seals=dict(zip(seal_keys,_CHECK4_CHAINS['score_watch_v1'])),native_lineage=lineage,
        all_five_prior_calls_reverified=True,strict_CHECK4_gain_gate_passed=True,
        original_native_watch_still_failed=True,completed_score_v2_trust_boundary=True,
        no_score_or_metadata_repair_reexecution=True),by_call=packet)


def _native40_replay(trial, prepared, diagnostic, expected):
    require(trial['call'] == prepared['call'] == diagnostic['call'] < 4
        and trial['rank'] == prepared['rank'] == diagnostic['rank'] == expected['rank']
        and all(trial[key] == value for key,value in expected['trial'].items())
        and prepared['flops'] == expected['input_flops']
        and prepared['weight_file_sha256'] == expected['weight_file_sha256']
        and diagnostic['domains'] == expected['domains'] and diagnostic['dispatches'] == expected['dispatches'],
        'First4 exact CHECK4 bytes, complete F/opcounts, official inputs, all16/47 and full dispatch packet; timing excluded')


def _native40_context(context):
    prerequisite = context['check4_prerequisites']
    require(prerequisite['score_sha256'] == _CHECK4_CHAINS['score_v2'][0]
        and prerequisite['score_watch_sha256'] == _CHECK4_CHAINS['score_watch_v1'][0]
        and all(prerequisite.get(k) is True for k in ('all_five_prior_calls_reverified','strict_CHECK4_gain_gate_passed',
            'original_native_watch_still_failed','completed_score_v2_trust_boundary','no_score_or_metadata_repair_reexecution'))
        and context['check4_expansion_prerequisite_passed'] is True
        and set(context['check4_by_call']) == {'0','1','2','3'}
        and [context['check4_by_call'][str(i)]['rank'] for i in range(4)] == [37,39,42,48],
        'Source-bound fixed CHECK4 expansion prerequisite and exact first4 packet')


def _native40_metadata():
    original,parent_proof = _PARENT_NAMESPACE['metadata_sources']()
    parts = {name:_function(original,name) for name in
        ('validate_diagnostic','after_bill','finish_native','verify_acquisition')}
    after = _adapt(parts['after_bill'], [
        ("    result['artifacts'][str(path)] = item['sha256']; result['diagnostic_records'].append(item)\n",
         "    result['artifacts'][str(path)] = item['sha256']; result['diagnostic_records'].append(item)\n"
         "    if trial['call'] < 4:\n"
         "        _native40_replay(trial,result['input_preparation'][-1],diagnostic,result['check4_by_call'][str(trial['call'])])\n",1)])
    finish = _adapt(parts['finish_native'], [
        ("len(result['diagnostic_records']) == 5","len(result['diagnostic_records']) == 41",1),
        ('Complete four primaries and repeat37','Complete40 primaries and repeat37',1),
        ('records[:4]','records[:40]',1),
        ('full_model_calls=5,retired_call_count=5','full_model_calls=41,retired_call_count=41',1),
        ('    for call,record in enumerate(records): validate_diagnostic(record,call)\n',
         '    for call,record in enumerate(records): validate_diagnostic(record,call)\n'
         "    for call in range(4):\n"
         "        _native40_replay(result['trials'][call],result['input_preparation'][call],records[call],result['check4_by_call'][str(call)])\n",1),
        ('        full_model_calls=41,retired_call_count=41,COMP_parameter_initialization_owned=True,\n',
         '        full_model_calls=41,retired_call_count=41,COMP_parameter_initialization_owned=True,\n'
         '        check4_first4_output_byte_exact=True,check4_first4_deterministic_bills_exact=True,\n'
         '        check4_first4_domains_exact=True,check4_first4_dispatches_exact=True,\n',1)])
    verify = _adapt(parts['verify_acquisition'], [
        ("record['full_model_calls'] == record['retired_call_count'] == 5",
         "record['full_model_calls'] == record['retired_call_count'] == 41",1),
        ("len(record['diagnostic_records']) == 5","len(record['diagnostic_records']) == 41",1),
        ('Exact five own prediction/input/diagnostic records','Exact41 own prediction/input/diagnostic records',1),
        ("len(paths) == 30, 'Only original25 artifacts plus five small diagnostics'",
         "len(paths) == 246, 'Only original205 artifacts plus41 small diagnostics'",1),
        ("    for group in ('source_bindings','input_bindings','artifacts'): bind(record.get(group,{}),tick)\n",
         "    require(record['check4_prerequisites'] == context['check4_prerequisites']\n"
         "        and record['check4_by_call'] == context['check4_by_call']\n"
         "        and all(record.get(k) is True for k in ('check4_expansion_prerequisite_passed',\n"
         "            'check4_first4_output_byte_exact','check4_first4_deterministic_bills_exact',\n"
         "            'check4_first4_domains_exact','check4_first4_dispatches_exact')), 'Passed CHECK4 and first4 deterministic replay')\n"
         "    for group in ('source_bindings','input_bindings','artifacts'): bind(record.get(group,{}),tick)\n",1)])
    changed = dict(validate_diagnostic=parts['validate_diagnostic'],after_bill=after,finish_native=finish,verify_acquisition=verify)
    proof = {name:dict(original_sha256=hashlib.sha256(parts[name].encode()).hexdigest(),
        derived_sha256=hashlib.sha256(source.encode()).hexdigest(),inverse_verified=True) for name,source in changed.items()}
    return '\n'.join(changed.values()),dict(parent_CHECK4_metadata=parent_proof,functions=proof)


def _native40_worker_source():
    original,parent_proof = _PARENT_NAMESPACE['worker_source']()
    changed = _adapt(original, [
        ("HERE/f'native_{args.role}_v1'","HERE/f'native40_{args.role}_v1'",1),
        ("HERE/f'native_{args.role}_admission_v1'","HERE/f'native40_{args.role}_admission_v1'",1),
        ("HERE/'NATIVE_PROTOCOL.md'","HERE/'NATIVE40_PROTOCOL.md'",1),
        ("HERE/'watch_native.py'","HERE/'watch_native40.py'",1),
        ("HERE/'NATIVE_WATCH_SPEC.md'","HERE/'NATIVE40_WATCH_SPEC.md'",1),
        ("        runtime_sha256=context['runtime_sha256'], versions=versions,",
         "        runtime_sha256=context['runtime_sha256'], versions=versions,\n"
         "        check4_prerequisites=context['check4_prerequisites'],check4_by_call=context['check4_by_call'],\n"
         "        check4_expansion_prerequisite_passed=True,",1),
        ('elapsed < 600','elapsed < 1900',1)])
    return changed,dict(parent_CHECK4_worker=parent_proof,original_sha256=hashlib.sha256(original.encode()).hexdigest(),
        derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),full_worker_inverse_verified=True,
        unchanged_paid_loop=True,only_paths_CHECK4_header_and_whole_cap_changed=True)


_require(_source_sha(_PARENT) == _PARENT_SHA, 'Exact frozen APUSH CHECK4 native source owner')
_PARENT_NAMESPACE = runpy.run_path(str(_PARENT),run_name='inert_original_APUSH_CHECK4_native')
_original = _PARENT.read_text()
_pd = _PARENT_NAMESPACE['PD']
_require(_source_sha(_pd) == _PARENT_NAMESPACE['OWNERS'][_pd], 'Frozen original VALID40 source owner')
_rosters = [node for node in ast.parse(_pd.read_text()).body if isinstance(node,ast.Assign)
    and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id == 'VALID']
_require(len(_rosters) == 1 and ast.literal_eval(_rosters[0].value) == _RANKS40, 'Exact original40 order')
_changes = (
    ('RANKS = VALID = (37,39,42,48)','RANKS = VALID = '+repr(_RANKS40),1),
    ("OUTPUT, ADMISSION = HERE/'native_terminal_v1', HERE/'native_terminal_admission_v1'",
     "OUTPUT, ADMISSION = HERE/'native40_terminal_v1', HERE/'native40_terminal_admission_v1'",1),
    ("SCHEMA = NATIVE_SCHEMA = 'current_comp_aligned_push_d6_native_v1'",
     "SCHEMA = NATIVE_SCHEMA = 'current_comp_aligned_push_d6_native40_v1'",1),
    ("CONTEXT_SCHEMA = 'current_comp_aligned_push_d6_native_context_v1'",
     "CONTEXT_SCHEMA = 'current_comp_aligned_push_d6_native40_context_v1'",1),
    ("ADMIT_SCHEMA = 'current_comp_aligned_push_d6_native_admission_v1'",
     "ADMIT_SCHEMA = 'current_comp_aligned_push_d6_native40_admission_v1'",1),
    (_function(_original,'reviewed_owners'),'def reviewed_owners(args):\n    return _native40_reviewed_owners(globals(),args)\n',1),
    (_function(_original,'metadata_sources'),'def metadata_sources():\n    return _native40_metadata()\n',1),
    (_function(_original,'worker_source'),'def worker_source():\n    return _native40_worker_source()\n',1),
    ("and 0 <= final['elapsed_s'] < 600","and 0 <= final['elapsed_s'] < 1900",1),
    ("'Exact same first-four cold history'","'Exact same original40 CD cold history'",1),
    ('    refs = references(sources,inputs,tick)\n',
     '    refs = references(sources,inputs,tick)\n    check4 = _native40_prerequisites(args,sources,inputs,tick)\n',1),
    ('        reference_by_rank=refs,proof=proof,build_proof=build_proof,admission_pid=os.getpid(),\n',
     '        reference_by_rank=refs,proof=proof,build_proof=build_proof,admission_pid=os.getpid(),\n'
     "        check4_prerequisites=check4['prerequisites'],check4_by_call=check4['by_call'],\n"
     '        check4_expansion_prerequisite_passed=True,\n',1),
    ("    context['reference_by_rank'] = {int(k):v for k,v in context['reference_by_rank'].items()}\n",
     "    _native40_context(context)\n    context['reference_by_rank'] = {int(k):v for k,v in context['reference_by_rank'].items()}\n",1),
    ('no_estimator_source_or_parameter_changes=True,kernel_tariff_owner_sha256=KERNEL_SHA)',
     'no_estimator_source_or_parameter_changes=True,kernel_tariff_owner_sha256=KERNEL_SHA,\n'
     '        native40_adapter=dict(_NATIVE40_SOURCE_PROOF))',1),
    ("('source','protocol','watcher','watch-spec','runtime','build','candidate','context'):",
     "('source','protocol','watcher','watch-spec','runtime','build','candidate','context','check4-score','check4-watch'):",1),
    ("(('runtime','build','candidate') if args.admit else ('context',))",
     "(('runtime','build','candidate','check4-score','check4-watch') if args.admit else ('context',))",1),
    ("all(getattr(args,k+'_sha256') is None for k in ('runtime','build','candidate'))",
     "all(getattr(args,k.replace('-','_')+'_sha256') is None for k in ('runtime','build','candidate','check4-score','check4-watch'))",1),
)
_derived = _adapt(_original,_changes)
_NATIVE40_SOURCE_PROOF = dict(parent=str(_PARENT),parent_sha256=_PARENT_SHA,
    derived_sha256=hashlib.sha256(_derived.encode()).hexdigest(),full_source_inverse_verified=True,
    exact_original40_roster=True,all_three_observers_byte_unchanged=True,
    generic_policy_and_tariffs_unchanged=True,paid_main_byte_unchanged=True,
    no_estimator_or_parameter_changes=True,no_new_participant_array_operations=True,
    direct_completed_CHECK4_boundary=True,original_failed_watch_preserved=True,first4_replay_metadata_only=True)
exec(compile(_derived,str(_PARENT)+':fixed-native40-expansion','exec'),globals())
