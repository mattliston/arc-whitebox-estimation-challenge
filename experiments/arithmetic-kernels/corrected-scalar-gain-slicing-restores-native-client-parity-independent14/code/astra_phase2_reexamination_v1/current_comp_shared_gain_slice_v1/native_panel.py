"""Fixed target-free gain-slice independent14 on production OpenBLAS."""
import ast
import hashlib
import math
from pathlib import Path
import runpy

_GP_HERE = Path(__file__).resolve().parent
_GP_SHARED = _GP_HERE.parent / 'current_comp_shared_old384_v1'
_GP_OWNER = _GP_SHARED / 'native_panel_openblas.py'
_GP_OWNER_SHA = 'e7864f38057fe460c6314d301b9555256ff9002f861c3c28a343bfb1c09098ba'
if hashlib.sha256(_GP_OWNER.read_bytes()).hexdigest() != _GP_OWNER_SHA:
    raise RuntimeError('Exact parked OpenBLAS panel source owner')
_GP_PARENT = runpy.run_path(str(_GP_OWNER), run_name='_inert_gain_panel_parent')
for _gp_key, _gp_value in _GP_PARENT.items():
    if not _gp_key.startswith('__') and not _gp_key.startswith('_GP'):
        globals()[_gp_key] = _gp_value
HERE, NEW, P2 = _GP_HERE, _GP_HERE.parent, _GP_HERE.parent.parent
_GP_HISTORY = _GP_PARENT['_OB_PARENT']
COMPOSED_SHA = 'c2c2ffada6feb16c08eb820c9401be68d5e8e7fe12d4034a20eed15af849ccea'
PANEL_SCHEMA = 'current_comp_shared_gain_slice_native_panel_v1'
PANEL_WATCH_SCHEMA = 'current_comp_shared_gain_slice_native_panel_watch_v1'
_GP_PINS = {
    _GP_OWNER: _GP_OWNER_SHA,
    _GP_SHARED / 'watch_native_panel_openblas.py': 'a497a5f181515173d97be4a50071ae68964f67a02645d879854ed8b4ac57dfbf',
    _GP_SHARED / 'NATIVE_PANEL_OPENBLAS_PROTOCOL.md': 'a4a72d6ad2eafb775997a1ad42449653e1663c2d2eafd08d70514a58843074ed',
    _GP_SHARED / 'NATIVE_PANEL_OPENBLAS_WATCH_SPEC.md': '3561933670b8ba5e437b367685ffaea90354e312502ab9661a1b4e4bf098b7b7',
    HERE / 'build.py': 'd417e12be774fae3cf33c2fd5eb17f098a43827efc556920100b4ee2bd72dd15',
    HERE / 'BUILD_PROTOCOL.md': '7598dd79a817cfb68a792be005fd2c7e9ff60c7b29bbb7dd1e1a9dd4753cef11',
    HERE / 'native.py': '90017490bd1f1d5efabe1f32167789ecce948ee339b27c0ea0555f28d08b0587',
    HERE / 'NATIVE_PROTOCOL.md': '01f8eecb0a42d68c4e36cef7f7029c03803c66283fba8f831d5c49d878848b62',
    HERE / 'client.py': 'dbdd9d28b0bd3c15bb83e1fe8669e593aba925b576b27b8f80bb650143f72fad',
    HERE / 'watch.py': '53e797e07d44a058e0b30dad77ad2dd8456c69be2680bdea37eeda18e2b1950b',
    HERE / 'pair_checks.py': '341ab3e95afa891d902d1a68372103f039e8a2fb8aa5f563add3c84b7f07040b',
    HERE / 'PAIR_WATCH_SPEC.md': '2a655a495aaf19dcf94361bccec4770ee3e434c6e94bbf1759ea1eb8ed8b1973',
    HERE / 'remote_watch_v2.py': '4a55a8a97e7d96d7abdb6427e06a3e73c8091ab30a9c98a50d1aafd897c7b8b0',
    HERE / 'REMOTE_PAIR_V2_SPEC.md': 'b6e844ead8ae6cf8c701793bd98e05d7c925e6b25953a50fc500b0b0cf0d1a92',
    HERE / 'client_v2.py': '2b5818ae5013c9a5e460437fff7770a9059f1139732f627c01253e3cba484e40',
}
EXTRA_SOURCES = {**EXTRA_SOURCES, **_GP_PINS}
PANEL_LINEAGE_KEYS = (
    'native40_prerequisite_seals','control_seals','APUSH_panel_seals','panel_rows',
    'fixture_bindings','production_runtime_seals','production_overlap','pair_prerequisite_seals',
    'gain_slice_pair_qualified','historical_native40_selection_only','current_candidate_native40_qualified',
    'historical_native40_candidate_sha256','APUSH_comparator_backend_matched','old_fidelity_still_failed',
    'current_candidate_production_five_qualified','historical_production_five_reused')
_GP_PAIR_API = _GP_BUILDER_API = None


def pair_api():
    global _GP_PAIR_API
    if _GP_PAIR_API is None:
        _GP_PAIR_API = load(HERE / 'native.py', 'gain_panel_pair_owner', _GP_PINS[HERE / 'native.py'])
    return _GP_PAIR_API


def builder_api():
    global _GP_BUILDER_API
    if _GP_BUILDER_API is None:
        _GP_BUILDER_API = load(HERE / 'build.py', 'gain_panel_builder_owner', _GP_PINS[HERE / 'build.py'])
    return _GP_BUILDER_API


def gain_panel_source_owners(args):
    require(args.builder_sha256 == _GP_PINS[HERE / 'build.py']
            and args.build_protocol_sha256 == _GP_PINS[HERE / 'BUILD_PROTOCOL.md'], 'Exact frozen c2 builder owners')
    owners = {HERE / 'native_panel.py': args.source_sha256, HERE / 'watch_native_panel.py': args.watch_sha256,
        HERE / 'NATIVE_PANEL_PROTOCOL.md': args.protocol_sha256,
        HERE / 'NATIVE_PANEL_WATCH_SPEC.md': args.watch_spec_sha256,
        LEGACY: LEGACY_SHA, OBSERVER: OBSERVER_SHA, **EXTRA_SOURCES}
    require(all(re.fullmatch('[0-9a-f]{64}', digest or '') for digest in owners.values()), 'Explicit source authorities')
    return {str(path): digest for path, digest in owners.items()}


def pair_prerequisite(args, inputs, tick=None):
    """New positive actual pair only; never make a parked old failure eligible."""
    w, ws, wf = _panel_chain(HERE / 'pair_watch_v2', args.pair_sha256, inputs, 240, tick)
    require(w['schema'] == 'current_comp_shared_gain_slice_pair_watch_v2'
        and w['candidate_sha256'] == args.candidate_sha256 == COMPOSED_SHA
        and w['full_model_calls'] == 2 and w['fresh_native_calls'] == 0
        and w['joined_native_remote_calls'] == 4 and w['native_markers'] == []
        and len(w['prior_native_observed_markers']) == len(w['remote_records']) == 2
        and all(w.get(k) is True for k in ('pair_vector_gate_passed','pair_timing_gate_passed','own_complete_bills',
            'whole_owned_tree_retired','cleanup_passed','observational_groups_retired','actual_completed_samples',
            'foreign_numerical_workers_checked','original_negative_fidelity_preserved','fresh_objective_required',
            'current_pair_qualified','previous_pair_watch_still_failed','prior_native_phase_reused',
            'prior_native_phase_resource_qualified'))
        and all(w.get(k) is False for k in ('numerical_imports','truth_accessed','raw_error_measured',
            'adjusted_score_measured','full_five_network_fidelity_qualified','submission_authorized'))
        and w['maximum_swap_growth_bytes'] == 0 and 0 < w['maximum_combined_rss_bytes'] < RSS
        and 0 < w['maximum_interval_s'] <= 1
        and max(w['completed_sample_intervals_s']) == w['maximum_interval_s']
        and w['server_max_array_bytes'] == 4294967296
        and 0 <= w['elapsed_s'] <= wf['elapsed_s'] < 240, 'Actual joined pair/resource gate, not expected success')
    n, ns, nf = _panel_chain(HERE / 'native_v1', w['native_seals']['result_sha256'], inputs, 300, tick)
    b, bs, bf = _panel_chain(HERE / 'build_v1', args.build_sha256, inputs, 60, tick)
    require(ns == w['native_seals'] and bs == w['build_seals']
        and n['candidate_sha256'] == b['candidate_sha256'] == COMPOSED_SHA
        and 0 <= n['elapsed_s'] <= nf['elapsed_s'] < 300 and 0 < n['peak_rss_bytes'] < RSS
        and 0 <= b['elapsed_s'] <= bf['elapsed_s'] < 60 and 0 < b['peak_rss_bytes'] < 2_000_000_000,
        'Actual paired native/build three seals and exact new source')
    require(pair_api().verify_native(n, tick), 'Original new-pair native metadata contract')
    built = builder_api().verify_build(args.build_sha256, args.candidate_sha256,
        args.builder_sha256, args.build_protocol_sha256, tick)
    require(built['proof'] == n['build_proof'] and built['proof']['one_span_gain_slice']
        and built['proof']['whole_source_inverse_verified'], 'Exact only permitted source correction')
    repair = load(HERE / 'remote_watch_v2.py', 'gain_panel_joined_pair_owner', _GP_PINS[HERE / 'remote_watch_v2.py'])
    unused, repair_proof = repair.derive_source()
    require(w['repair_source_proof'] == repair_proof, 'Exact restored documented server-limit boundary')
    del unused
    negative_paths = [HERE / 'pair_watch_v1' / name for name in ('result.json','completion.json','finalization.json')]
    negative_hashes = [sha(path,tick) for path in negative_paths]
    old, old_end, old_final = map(read,negative_paths)
    require(negative_hashes == list(repair.PREVIOUS['pair_watch_v1'])
        and all(x.get('passed') is False and x.get('completed') is False for x in (old,old_end,old_final))
        and old_end['result_sha256'] == old_final['result_sha256'] == negative_hashes[0]
        and old_final['completion_sha256'] == negative_hashes[1]
        and old['native_seals'] == ns and old['build_seals'] == bs
        and old['native_markers'] == w['prior_native_observed_markers'] and old['remote_records'] == []
        and all(old[k] for k in ('cleanup_passed','whole_owned_tree_retired','observational_groups_retired'))
        and old['maximum_swap_growth_bytes'] == 0 and old['maximum_interval_s'] <= 1
        and old['maximum_combined_rss_bytes'] < RSS, 'Actual earlier native resource receipt stays overall FALSE')
    require(all(w['input_bindings'].get(str(path)) == digest for path,digest in zip(negative_paths,negative_hashes)),
        'Joined positive result explicitly owns the negative prior chain')
    for record in (w, n, b):
        for group in ('source_bindings','input_bindings','artifacts'):
            require(record[group], 'Complete direct new prerequisite inventories')
            _panel_merge(inputs, record[group])
    for path, digest in _GP_PINS.items():
        if path.parent == HERE:
            require(w['source_bindings'].get(str(path)) == digest, 'Exact actually executed pair owner')
    checks = w['comparison']['checks']
    require(len(checks) == 2 and w['comparison']['all16_vector_gate_passed']
        and w['comparison']['native_repeat_byte_exact'] and w['comparison']['production_repeat_byte_exact']
        and w['comparison']['native_initialization_flops'] ==
            w['comparison']['production_initialization_flops'] == 1944970298, 'Both actual complete pair modes')
    for call, (check, trial, remote) in enumerate(zip(checks, n['trials'], w['remote_records'], strict=True)):
        require(check['call'] == trial['call'] == remote['index'] == call
            and check['rank'] == trial['rank'] == remote['rank'] == 35
            and len(check['layer_max_abs']) == 16
            and all(math.isfinite(v) and 0 <= v <= 1e-5 for v in check['layer_max_abs'])
            and check['all16_max_abs'] == max(check['layer_max_abs'])
            and check['original_all16_vector_gate_passed']
            and check['native_flops'] == trial['flops'] and check['production_flops'] == remote['flops'],
            'Positive saved all16 pair comparison with each own actual bill')
    bind(inputs, tick)
    return n, w, dict(pair_watch=ws, native=ns, build=bs, prior_negative_pair=dict(zip(
        ('result_sha256','completion_sha256','finalization_sha256'),negative_hashes)))


def panel_admission(args, inputs, tick=None):
    # This exact helper still validates all four historical chains and the .97
    # gate against its own 4d98/Accelerate globals. It is not a c2 full40 gate.
    history = _GP_HISTORY['_panel_advance'](args, inputs, tick)
    prior, pair, pair_seals = pair_prerequisite(args, inputs, tick)
    controls, seals, rows, fixtures, old_seals = _GP_HISTORY['_panel_controls'](inputs, tick)
    require(prior['input_preparation'][0]['weight_file_sha256'] == fixtures['weights_35']['sha256'],
        'Current pair and original panel have exactly the same physical35')
    overlap = []
    for prod, call in ((0,0),(1,14)):
        item = pair['remote_records'][prod]
        path = NEW / f'current_comp_shared_gain_slice_pair_v2_{prod}_r35.f32'
        require(pair['artifacts'][str(path)] == item['output_sha256'], 'Actual new production35 output authority')
        overlap.append(dict(production_call=prod,native_call=call,rank=35,
            weights_path=fixtures['weights_35']['path'],weight_file_sha256=fixtures['weights_35']['sha256'],
            production_output=str(path),production_output_sha256=item['output_sha256'],
            production_flops=item['flops'],production_timing=item['timing'],
            operation_count_equality_required=False,fidelity_checked=False))
    _panel_merge(inputs, {str(p): h for p,h in EXTRA_SOURCES.items()})
    bind(inputs, tick)
    return dict(native40_prerequisite_seals=history, controls=controls, control_seals=seals,
        APUSH_panel_seals=old_seals,panel_rows=rows,fixture_bindings=fixtures,
        production_runtime_seals=pair_seals['pair_watch'],production_overlap=overlap,
        pair_prerequisite_seals=pair_seals,gain_slice_pair_qualified=True,
        historical_native40_selection_only=True,current_candidate_native40_qualified=False,
        historical_native40_candidate_sha256='4d98fef2c11c571efe1eca945018bb9fe9ece1454d16b4abde2eb52d71e755c3',
        APUSH_comparator_backend_matched=False,old_fidelity_still_failed=True,
        current_candidate_production_five_qualified=False,historical_production_five_reused=False)


def gain_panel_candidate(result, text, args, builder, tick):
    built = builder.verify_build(args.build_sha256, args.candidate_sha256,
        args.builder_sha256, args.build_protocol_sha256, tick)
    require(result['candidate_sha256'] == COMPOSED_SHA == args.candidate_sha256
        and Path(built['candidate']).read_bytes() == text.encode(), 'Actual built c2 gain-slice source only')
    result['source_bindings'].update(built['source_bindings'])
    _panel_merge(result['input_bindings'], built['input_bindings'])
    result.update(candidate=built['candidate'],build_sha256=args.build_sha256,build_proof=built['proof'],
        candidate_has_only_gain_slice_change=True,old_fidelity_still_failed=True,
        current_candidate_native40_qualified=False,production_five_qualified=False,
        native_fidelity_qualified=False,fresh_same_backend_objective_required_before_bank=True,
        automatic_next_stage=False,uploaded=False,submission_authorized=False)


def panel_final_checks(result, tick=None):
    checked = _GP_HISTORY['panel_final_checks'](result, tick)
    require(result['gain_slice_pair_qualified'] and result['historical_native40_selection_only']
        and result['current_candidate_native40_qualified'] is False
        and result['APUSH_comparator_backend_matched'] is False
        and result['unaffected_prefix_checks'] == []
        and result['cross_backend_APUSH_prefix_equality_required'] is False
        and _GP_PARENT['_openblas_attestation'](result, tick), 'Explicit c2/OpenBLAS qualification boundary')
    expected = result['pair_prerequisite_seals']['native']
    paths = [HERE / 'native_v1' / name for name in ('result.json','completion.json','finalization.json')]
    require([sha(p,tick) for p in paths] == [expected[k] for k in
        ('result_sha256','completion_sha256','finalization_sha256')], 'Exact new pair native reference')
    prior = read(paths[0]); replay = []
    for call, old_call in ((0,0),(14,1)):
        a,b = result['trials'][call],prior['trials'][old_call]
        da,db = (read(r['diagnostic_records'][i]['path']) for r,i in ((result,call),(prior,old_call)))
        require(a['rank'] == b['rank'] == 35 and all(a[k] == b[k] for k in
            ('flops','multiplier','includes_initialization','output_sha256','output_bytes_sha256'))
            and Path(a['output']).read_bytes() == Path(b['output']).read_bytes()
            and da['domains'] == db['domains'] and da['startup'] == db['startup'],
            'Exact current-source same-backend35 fullF/output/domains/startup replay')
        physical = lambda xs: [{k:v for k,v in x.items() if k != 'operation_count'} for x in xs]
        require(physical(da['dispatches']) == physical(db['dispatches']), 'Same physical35 dispatch and complete tariff')
        if call == 0:
            require(a['operation_count'] == b['operation_count'] and da['dispatches'] == db['dispatches'],
                'Actual cold history operation count; passive ledger does not change billing')
        replay.append(dict(call=call,pair_native_call=old_call,rank=35,output_byte_exact=True,
            complete_flops_exact=True,domains_exact=True,startup_exact=True,physical_dispatch_exact=True,
            operation_count_equality_required=(call == 0),
            prior_operation_count=b['operation_count'],current_operation_count=a['operation_count']))
        if tick is not None:
            tick()
    future = []
    for prod, call, rank in PRODUCTION_MAPPING:
        trial, prep = result['trials'][call],result['input_preparation'][call]
        require(trial['rank'] == prep['rank'] == rank, 'Original five-network future control mapping')
        future.append(dict(production_call=prod,native_call=call,rank=rank,native_output=trial['output'],
            native_output_sha256=trial['output_sha256'],native_output_bytes_sha256=trial['output_bytes_sha256'],
            weight_file_sha256=prep['weight_file_sha256'],native_complete_flops=trial['flops'],
            production_measured=False,operation_count_equality_required=False))
    checked.update(gain_pair35_replay=replay,gain_pair35_cold_exact=True,gain_pair35_repeat_physics_exact=True,
        same_production_backend_qualified=True,backend_actual_origins_qualified=True,
        historical_native40_selection_only=True,current_candidate_native40_qualified=False,
        current_candidate_production_five_qualified=False,production_five_native_controls=future,
        old_fidelity_still_failed=True,fresh_same_backend_objective_required_before_bank=True,
        native_fidelity_qualified=False,automatic_next_stage=False,submission_authorized=False,uploaded=False)
    return checked


def verify_native(record, tick=None):
    """Saved-output metadata API; callers separately bind actual worker/watch three seals."""
    require(record['schema'] == PANEL_SCHEMA and record['passed'] is True and record['completed'] is True
        and record['candidate_sha256'] == COMPOSED_SHA and record['ranks'] == list(SEQUENCE)
        and record['full_model_calls'] == record['retired_call_count'] == 15 and len(record['artifacts']) == 105
        and record['panel_source_inverse_proof'] == PANEL_SOURCE_INVERSE_PROOF
        and all(record.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'repeat_byte_exact','own_complete_bills','hashes_unchanged','sources_unchanged'))
        and record['truth_accessed'] is False and record['score_measured'] is False,
        'Actual positive current-source fixed15 acquisition, no borrowed score')
    for group in ('source_bindings','input_bindings','artifacts'):
        bind(record[group], tick)
    for call, (trial, prep, diag) in enumerate(zip(record['trials'],record['input_preparation'],
                                                 record['diagnostic_records'],strict=True)):
        rank = SEQUENCE[call]; folder = HERE / 'native_panel_v1'
        marker = read(folder / f'call_{call:02d}_complete.json')
        require(trial['call'] == prep['call'] == marker['call'] == call
            and trial['rank'] == prep['rank'] == marker['rank'] == rank
            and read(folder / f'call_{call:02d}_bill.json') == trial and read(trial['input_bill']) == prep
            and prep['weight_file_sha256'] == record['fixture_bindings']['weights_'+str(rank)]['sha256']
            and prep['flops'] == 16777216 and trial['multiplier'] == max(.1,trial['flops']/BUDGET)
            and marker['passed'] and marker['completed'] and marker['context_retired']
            and marker['bill_sha256'] == sha(folder / f'call_{call:02d}_bill.json',tick)
            and marker['diagnostic_sha256'] == diag['sha256']
            and marker['output_sha256'] == trial['output_sha256']
            and record['artifacts'][trial['output']] == trial['output_sha256'], 'Every own paid call/retired output')
    checked = panel_final_checks(record, tick)
    require(all(record[k] == v for k,v in checked.items()), 'All saved ledger/backend/replay checks match exactly')
    return checked


def derive_source():
    original = _GP_PARENT['_OB_DERIVED']
    node = next(n for n in ast.parse(original).body if isinstance(n,ast.FunctionDef) and n.name == 'source_owners')
    owners = ''.join(original.splitlines(keepends=True)[node.lineno-1:node.end_lineno])
    edits = [
        (owners, 'def source_owners(args):\n    return gain_panel_source_owners(args)\n'),
        ("KEYS = ('source','watch','protocol','watch_spec','compose','fnp','design')+GATE_KEYS+BACKEND_GATE_KEYS",
         "KEYS = ('source','watch','protocol','watch_spec','builder','build_protocol','build','candidate','pair')+GATE_KEYS"),
        ("HERE/'native_panel_openblas_v1'", "HERE/'native_panel_v1'"),
        ("'current_comp_shared_old384_native_panel_openblas_v1'", "'current_comp_shared_gain_slice_native_panel_v1'"),
        ("composition = load(HERE/'compose.py','shared384_composition',args.compose_sha256)",
         "composition = load(HERE/'build.py','gain_panel_builder',args.builder_sha256)"),
        ('backend_api().backend_candidate(result,text,tick)', 'gain_panel_candidate(result,text,args,composition,tick)'),
        ("result.update(panel,native40_expansion_prerequisite_passed=True,APUSH_panel_comparator_qualified=True)",
         "result.update(panel,historical_native40_selection_only=True,current_candidate_native40_qualified=False,APUSH_panel_comparator_qualified=True)"),
    ]
    changed, spans = original, []
    for before, after in edits:
        require(changed.count(before) == 1, 'Unique gain-panel authority span: '+before[:90])
        pos = changed.index(before); spans.append((pos,before,after))
        changed = changed[:pos]+after+changed[pos+len(before):]
    inverse = changed
    for pos,before,after in reversed(spans):
        require(inverse[pos:pos+len(after)] == after, 'Positional gain-panel source inverse')
        inverse = inverse[:pos]+before+inverse[pos+len(after):]
    require(inverse == original, 'Full parked OpenBLAS panel effective source restored')
    return changed, dict(owner=str(_GP_OWNER),owner_sha256=_GP_OWNER_SHA,
        whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(changed.encode()).hexdigest(),
        source_splices=len(spans),candidate_paid_loop_observers_and_closed_ledger_unchanged=True,
        fixed_roster=list(SEQUENCE),whole_cap_s=1200.,current_positive_gain_pair_required=True,
        historical_four_chain_selection_unchanged=True,current_candidate_native40_qualified=False,
        source_change_limited_to_previously_built_gain_slice=True,no_production_five_evidence_borrowed=True)


_GP_DERIVED, PANEL_SOURCE_INVERSE_PROOF = derive_source()
exec(compile(_GP_DERIVED,str(_GP_OWNER)+':gain-slice-panel-authority','exec'), globals())
