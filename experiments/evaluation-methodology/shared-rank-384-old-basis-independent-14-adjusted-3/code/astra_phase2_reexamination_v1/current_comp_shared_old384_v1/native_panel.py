"""Conditional unchanged independent14; direct sealed controls, no target access."""
import ast
import hashlib
from pathlib import Path
import runpy

_NP_HERE = Path(__file__).resolve().parent
_NP_OWNER = _NP_HERE/'native40.py'
_NP_OWNER_SHA = '0ac42cceec2bcba30813536deb33906a30cc2873ae99e5799d74c4c5e2a3472a'
if hashlib.sha256(_NP_OWNER.read_bytes()).hexdigest() != _NP_OWNER_SHA:
    raise RuntimeError('Frozen shared native40 paid-loop owner')
_NP_PARENT = runpy.run_path(str(_NP_OWNER),run_name='inert_native_panel_parent')
for _np_key,_np_value in _NP_PARENT.items():
    if not _np_key.startswith('__') and not _np_key.startswith('_NP'):
        globals()[_np_key] = _np_value
PUBLIC = (35,100,150,200,250,300,400,500)
VIRGIN = (3016500,3016501,3016502,3016503,3016504,3016505)
RANKS,SEQUENCE = PUBLIC+VIRGIN,PUBLIC+VIRGIN+(35,)
GATE_KEYS = ('native40','native40_watch','score40','score40_watch')
PANEL_SCHEMA = 'current_comp_shared_old384_native_panel_v1'
PANEL_WATCH_SCHEMA = 'current_comp_shared_old384_native_panel_watch_v1'
_NP_SOURCES = {
    _NP_OWNER:_NP_OWNER_SHA,
    HERE/'watch_native40.py':'c4536d3c237f393efe5cef180057b4e72650c31e2d8589411ea3bb022fabd98f',
    HERE/'NATIVE40_PROTOCOL.md':'8dbf45273f4835e7dc60d942b59b9fa95143420b7e8ef752f4e91e2966734599',
    HERE/'NATIVE40_WATCH_SPEC.md':'fcf0104e3d9abfdffd9fa68ca85e5a25fbf1bc3f3ac88291b355bd0c4fa4beea',
    HERE/'score40.py':'b99c9fef43da28a5f84ba6769d4d0f7b378480862752d0784ceb28bd8378fde7',
    HERE/'watch_score40.py':'5c110483447bcd1b5d8720429fa88038effe9ad929e14db73d238bb2d491a896',
    HERE/'SCORE40_PROTOCOL.md':'8a2c13dd68b50fcd962a9c917bb96d7e7c2c0182f92c1725ceeb5308b2022068',
    APUSH/'panel_common.py':'b8ae22e9586f76cafd92763562d2a631efae6a8ea81d2417e6cad5ce6828129a',
    APUSH/'panel_admit.py':'c6d94d71669fc2c03a1c594d96d7f7ecd42342cc5300195a44175b36e8fd0fd5',
    APUSH/'panel_native.py':'a30a4a634e729df078497bd003e1fec4b95220af30ee049ae8680968d682b499',
    APUSH/'panel_watch.py':'b406f813f293f4c55ba251881d3315fe864bb5849ff663d3efd678137b05aa52',
    APUSH/'PANEL_PROTOCOL.md':'9ff7c4591af0d7931dda4dba084d43db4a898ad5d2ef82b40a2255e28b236c47',
    APUSH/'PANEL_WATCH_SPEC.md':'305e7f07528ec2edfdfdff91fdbeac39e531d14f5cb54e9395d76cde6e3297c3',
    HERE/'runtime_watch.py':'1c4383e63b25337ddf1e051008174430aeee189ae1c5d9babe99e5c2b4ce5674',
    HERE/'client.py':'cee32e84ca797e5708aced91d3d0c76f5da2f8c436d65d340790782cdf50deae',
    HERE/'runner.py':'c6f684c7fcd7bb51d5ef52240b777da3787f466d4522b61d01ee877dd7e34af3',
    HERE/'RUNTIME_PROTOCOL.md':'d8df84bbd0cc5817ae61ce150263114b6284c817b0363675d9cb65f67d3d1de3',
}
EXTRA_SOURCES = {**EXTRA_SOURCES,**_NP_SOURCES}
APUSH_PANEL_SEALS = {
    'admission':('37f32dfe69ece8355a1d2658e2229b63be13be8da10f0c66e35e35b80999dd30',
        'e48401fc616c46ddb197be96b0e2d847611bbec3395cee6be7f5d6306f51a7f8',
        '8536f304440e9f14d724c1b911d88eb923088599c8756bf436bf917bd13f600e'),
    'terminal':('8be856a31f70a7878286e42b1abf5c6bb71979b5a29853f870c7e00d8e96c733',
        '20df13b47f827313a41d9f3248690eadd80ec0a7f9f1000f102b8c23b3c3e2fc',
        'ce99bdd2e22a6bc91593df77f03a0ec0680c623105b95589fcd106b4724daacf'),
    'terminal_watch':('e2d8f13623d507b0369331f079df50524b53369df3ffe527caee2009ec632828',
        '8059755827c4fad48301ba61657acbb9acd0f46d17794a9b217fd73b089204a1',
        '359d025eb95a32bc7e1be2ab12df68153916699a92f79df451ce5b77edf00cc5'),
}
APUSH_PANEL_CONTEXT_SHA = '1c7cf80de79664bd4454ca4059a8601476084a4040214221566f66ec1112a4ff'
RUNTIME_SEALS = ('c8d325e35229396bf67f63fd65e92f09f75ed8305ef1e169388d975507318af0',
    'b37e54981a24af7433fd65b8ca90b38fb84d092a910f91c7655542b658169f6d',
    '40b6bde559cbee4960cd8a5202bd12c08e241cfa3d5dcea38d460b43a67bcc48')
PRODUCTION_MAPPING = ((0,0,35),(1,1,100),(2,7,500),(3,8,3016500),(4,14,35))


def _panel_merge(destination, mapping):
    require(isinstance(mapping,dict), 'Flat direct metadata inventory')
    for path,digest in mapping.items():
        require(isinstance(path,str) and re.fullmatch('[0-9a-f]{64}',digest or '')
            and (path not in destination or destination[path] == digest), 'Consistent direct file authority')
        destination[path] = digest


def _panel_chain(folder, expected, inputs, limit, tick=None):
    """Read actual three seals, without importing/reexecuting their producer."""
    require(re.fullmatch('[0-9a-f]{64}',expected or ''), 'Explicit actual completed result hash')
    paths = [folder/n for n in ('result.json','completion.json','finalization.json')]
    digests = [sha(p,tick) for p in paths]; r,e,f = map(read,paths)
    require(digests[0] == expected and all(x.get('passed') is True and x.get('completed') is True for x in (r,e,f))
        and e['result_sha256'] == f['result_sha256'] == expected and f['completion_sha256'] == digests[1]
        and r['hashes_unchanged'] and r['sources_unchanged'] and 0 <= f['elapsed_s'] < limit,
        'Actual positive three-seal boundary inside its original whole scope')
    _panel_merge(inputs,{str(p):h for p,h in zip(paths,digests)})
    return r,dict(zip(('result_sha256','completion_sha256','finalization_sha256'),digests)),f


def _panel_watch_gate(record, child, limit, rss, *, old_panel=False):
    require(all(record.get(k) is True for k in ('whole_owned_tree_retired','observational_groups_retired',
        'actual_completed_samples','foreign_numerical_workers_checked','child_execution_completed'))
        and all(record['child_'+k] == v for k,v in child.items()) and record['numerical_imports'] is False
        and record['maximum_swap_growth_bytes'] == 0 and 0 <= record['elapsed_s'] < limit,
        'Actual independently retired whole-tree watch and exact child seals')
    memory = record['maximum_sampled_rss_bytes' if old_panel else 'maximum_combined_rss_bytes']
    interval = record['maximum_sample_interval_s' if old_panel else 'maximum_interval_s']
    require(0 < memory < rss and 0 < interval <= 1 and record['completed_sample_intervals_s']
        and max(record['completed_sample_intervals_s']) == interval, 'Original actual resource/sample scope')
    require(record['experiment_groups_retired'] if old_panel else record['cleanup_passed'],
        'Actual historical cleanup field, not an invented receipt')


def _panel_advance(args, inputs, tick=None):
    records,seals = {},{}
    for key,folder,limit,rss in (('native40','native40_v1',3000,8_000_000_000),
            ('native40_watch','native40_watch_v1',3000,8_000_000_000),
            ('score40','score40_v1',120,1_000_000_000),('score40_watch','score40_watch_v1',120,1_000_000_000)):
        r,s,f = _panel_chain(HERE/folder,getattr(args,key+'_sha256'),inputs,limit,tick)
        require(0 <= r['elapsed_s'] <= f['elapsed_s'], 'Actual current-stage complete clock')
        if not key.endswith('_watch'):
            require(0 < r['peak_rss_bytes'] < rss, 'Actual worker peak in original scope')
        for group in ('source_bindings','input_bindings','artifacts'):
            require(r[group], 'Complete direct current-stage inventory')
            _panel_merge(inputs,r[group])
        # In particular, do not traverse the separate truth_bindings map.
        records[key],seals[key] = r,s
    n,w,s,sw = (records[k] for k in GATE_KEYS)
    _panel_watch_gate(w,seals['native40'],3000,8_000_000_000)
    _panel_watch_gate(sw,seals['score40'],120,1_000_000_000)
    require(n['schema'] == 'current_comp_shared_old384_native40_v1'
        and w['schema'] == 'current_comp_shared_old384_native40_watch_v1'
        and n['candidate_sha256'] == w['candidate_sha256'] == COMPOSED_SHA
        and n['ranks'] == w['ranks'] == list(_NP_PARENT['SEQUENCE'])
        and n['full_model_calls'] == n['retired_call_count'] == w['retired_calls'] == 41
        and len(n['artifacts']) == 287 and len(n['trials']) == len(n['input_preparation']) == len(n['diagnostic_records']) == 41
        and n['initialization_flops'] == 1944970298 and n['truth_accessed'] is False and n['score_measured'] is False
        and all(n.get(k) is True for k in ('check4_prerequisite_qualified','check4_first4_exact',
            'check4_repeat37_physics_exact','all16_domains_qualified','all47_selectors_qualified','repeat_byte_exact',
            'own_complete_bills','operation_ledgers_verified','repeat_operation_counts_history_aware'))
        and _v2_validate_ledgers(n), 'Passed unchanged own-cost full40 acquisition and positive CHECK4 boundary')
    require(s['schema'] == 'current_comp_shared_old384_score40_v1'
        and sw['schema'] == 'current_comp_shared_old384_score40_watch_v1'
        and s['ranks'] == list(_NP_PARENT['RANKS']) and len(s['records']) == 40
        and s['lineage']['probe'] == seals['native40'] and s['lineage']['probe_watch'] == seals['native40_watch']
        and s['full_model_calls'] == s['parameter_updates'] == 0 and s['repeat_excluded'] and s['no_model_imported']
        and all(s.get(k) is True for k in ('integrity_passed','information_promising','scientific_gate_passed',
            'complete_probe_acquisition_reverified','all_predictions_verified_before_truth','exact_APUSH_score_replay',
            'own_complete_bills')) and sw['information_promising'] is True and s['comparison']['adjusted_ratio'] <= .97,
        'Actual prospective full40 own-adjusted .97 gate, not an anticipated result')
    for path,digest in _NP_SOURCES.items():
        if path.parent == HERE and path.name in ('native40.py','watch_native40.py','NATIVE40_PROTOCOL.md','NATIVE40_WATCH_SPEC.md'):
            require(n['source_bindings'].get(str(path)) == w['source_bindings'].get(str(path)) == digest,
                'Exact current native40 source owners')
        if path.parent == HERE and path.name in ('score40.py','watch_score40.py','SCORE40_PROTOCOL.md'):
            require(s['source_bindings'].get(str(path)) == digest, 'Exact reviewed score40 owner')
    return seals


def _panel_controls(inputs, tick=None):
    """Trust direct positive APUSH boundaries; recheck all90 used control artifacts."""
    old,seals,finals = {},{},{}
    for name,expected in APUSH_PANEL_SEALS.items():
        old[name],seals[name],finals[name] = _panel_chain(APUSH/'panel_v1'/name,expected[0],inputs,
            60 if name == 'admission' else 1200,tick)
        require(tuple(seals[name].values()) == expected, 'Exact actual original three seals')
    a,n,w = (old[k] for k in ('admission','terminal','terminal_watch'))
    _panel_watch_gate(w,seals['terminal'],1200,8_000_000_000,old_panel=True)
    path = APUSH/'panel_v1/admission/context.json'
    require(sha(path,tick) == APUSH_PANEL_CONTEXT_SHA, 'Exact original panel physical fixture context')
    context = read(path); inputs[str(path)] = APUSH_PANEL_CONTEXT_SHA
    require(a['schema'] == 'current_comp_aligned_push_d6_independent14_admission_v1'
        and context['schema'] == 'current_comp_aligned_push_d6_independent14_context_v1'
        and a['context_sha256'] == a['artifacts'][str(path)] == APUSH_PANEL_CONTEXT_SHA
        and a['reviewed_owners'] == context['reviewed_owners'] and a['full_frozen_admission_passed']
        and a['estimator_executed'] is False and a['numerical_imports'] is False and a['truth_accessed'] is False
        and 0 <= a['elapsed_s'] <= finals['admission']['elapsed_s'] < 60 and 0 < a['peak_rss_bytes'] < 2_000_000_000,
        'Actual old retired source admission, without rerunning it')
    require(n['schema'] == 'current_comp_aligned_push_d6_independent14_native_v1'
        and w['schema'] == 'current_comp_aligned_push_d6_independent14_watch_v1'
        and n['candidate_sha256'] == w['candidate_sha256'] == context['candidate_sha256'] == PARENT_SHA
        and n['ranks'] == context['ranks'] == list(RANKS)
        and [t['rank'] for t in n['trials']] == list(SEQUENCE)
        and n['full_model_calls'] == n['retired_call_count'] == w['completed_bill_count'] == 15
        and len(n['artifacts']) == 90 and len(n['diagnostic_records']) == len(n['input_preparation']) == 15
        and n['initialization_flops'] == 1944970298 and n['fixture_bindings'] == context['fixtures']
        and n['admission_context_sha256'] == w['admission_context_sha256'] == APUSH_PANEL_CONTEXT_SHA
        and w['admission_process_reaped_before_native'] and w['all15_output_bill_domain_qualified']
        and all(w['admission_'+k] == v for k,v in seals['admission'].items())
        and all(n.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified','all90_artifacts_qualified',
            'own_complete_bills','repeat_output_domain_byte_exact','same_existing_arena_qualified',
            'all_eligible_push_dispatches_qualified','paid_arithmetic_and_lifecycle_unchanged')),
        'Exact complete positive APUSH15 and independent whole-tree authority')
    require([x['phase'] for x in w['phase_records']] == ['admission','native']
        and all(x['exit_code'] == 0 and x['whole_owned_tree_retired'] for x in w['phase_records']),
        'Both old producer groups actually reaped')
    for path,digest in _NP_SOURCES.items():
        if path.parent == APUSH:
            require(n['source_bindings'].get(str(path)) == context['reviewed_owners'].get(str(path)) == digest,
                'Direct original panel source authority')
    rows,fixtures = context['rows'],context['fixtures']
    require([x['rank'] for x in rows] == list(RANKS), 'Original public8/virgin6 fixed ordered roster')
    for row in rows:
        rank = row['rank']; item = fixtures['weights_'+str(rank)]
        expected = STORE/f'rows/observer_fit/{rank:04d}/weights.npy' if rank in PUBLIC else P2/'public_g19_attribution_v1'/f'fresh_{rank}_weights.npy'
        require(row['weights_path'] == item['path'] == str(expected), 'Original canonical physical fixture path')
        if rank in PUBLIC:
            manifest = fixtures['row_'+str(rank)]
            require(manifest['path'] == str(expected.parent/'row_manifest.json')
                and sha(manifest['path'],tick) == manifest['sha256']
                and read(manifest['path'])['mlp_seed'] == row['mlp_seed'], 'Exact original public MLP seed')
        else:
            require(row['mlp_seed'] == rank, 'Original virgin physical seed, no reseeding')
    _panel_merge(inputs,{v['path']:v['sha256'] for v in fixtures.values()})
    expected_artifacts = set()
    for call,(trial,prep,diag) in enumerate(zip(n['trials'],n['input_preparation'],n['diagnostic_records'])):
        rank = SEQUENCE[call]
        require(trial['call'] == prep['call'] == call and trial['rank'] == prep['rank'] == rank
            and prep['weight_file_sha256'] == fixtures['weights_'+str(rank)]['sha256']
            and prep['flops'] == 16777216 and prep['operation_count'] == 16
            and trial['includes_initialization'] is (call == 0)
            and trial['multiplier'] == max(.1,trial['flops']/BUDGET), 'Actual own control bills and physical input')
        bill = APUSH/'panel_v1/terminal'/f'call_{call:02d}_bill.json'
        paths = (trial['output'],str(bill),trial['input_bill'],trial['input_reset_receipt'],trial['lifecycle_receipt'],diag['path'])
        for p in paths:
            require(sha(p,tick) == n['artifacts'][p], 'Every selected actual control artifact unchanged')
            inputs[p] = n['artifacts'][p]; expected_artifacts.add(p)
        require(read(bill) == trial and read(trial['input_bill']) == prep
            and n['artifacts'][trial['output']] == trial['output_sha256']
            and n['artifacts'][diag['path']] == diag['sha256'], 'Stored control rows exactly match their bills')
        for p in (trial['input_reset_receipt'],trial['lifecycle_receipt']):
            life = read(p)
            require(life['call'] == call and life['rank'] == rank and life['context_released'] and life['complete']
                and life['after_context_exit'] and life['closed_context_bill_unchanged_by_reset']
                and life['candidate_state_reset'] is False and life['reset_api'] == 'flopscope.budget_reset',
                'Original actual post-context reset and full retirement')
    require(expected_artifacts == set(n['artifacts']), 'All90 actual control artifacts, no missing consumer')
    return n['trials'],dict(native=list(seals['terminal'].values()),watch=list(seals['terminal_watch'].values())),rows,fixtures,seals


def _panel_runtime(inputs, fixtures, tick=None):
    r,seals,f = _panel_chain(HERE/'runtime_v1',RUNTIME_SEALS[0],inputs,600,tick)
    require(tuple(seals.values()) == RUNTIME_SEALS and r['schema'] == 'current_comp_shared_old384_runtime_v1'
        and r['candidate_sha256'] == COMPOSED_SHA and r['ranks'] == [x[2] for x in PRODUCTION_MAPPING]
        and r['full_model_calls'] == 5 and len(r['observed_records']) == 5 and r['initialization_flops'] == 1944970298
        and all(r.get(k) is True for k in ('all5_production_timing_passed','all16_outputs_finite','repeat_byte_exact',
            'one_persistent_numpy_free_client_and_server','own_complete_bills','whole_owned_tree_retired',
            'cleanup_passed','observational_groups_retired','actual_completed_samples','foreign_numerical_workers_checked'))
        and r['native_fidelity_pending'] is True and r['native_fidelity_qualified'] is False
        and r['production_five_qualified'] is False and r['numerical_imports'] is False
        and r['truth_arrays_accessed'] is False and r['maximum_swap_growth_bytes'] == 0
        and 0 < r['maximum_combined_rss_bytes'] < RSS and 0 < r['maximum_interval_s'] <= 1
        and 0 <= r['elapsed_s'] <= f['elapsed_s'] < 600,
        'Actual completed timing-only production5; fidelity is not retrospectively claimed')
    for p,h in _NP_SOURCES.items():
        if p.name in ('runtime_watch.py','client.py','runner.py','RUNTIME_PROTOCOL.md'):
            require(r['source_bindings'].get(str(p)) == h, 'Exact executed production adapter owner')
    _panel_merge(inputs,r['artifacts'])
    overlap = []
    for prod,native,rank in PRODUCTION_MAPPING:
        item = r['observed_records'][prod]; weight = fixtures['weights_'+str(rank)]
        require(item['index'] == prod and item['rank'] == rank and item['output_dtype'] == 'float32'
            and item['output_shape'] == [16,1024] and r['fixture_bindings'][str(rank)] ==
                dict(path=weight['path'],sha256=weight['sha256'])
            and 0 < item['flops'] <= BUDGET and item['util'] == item['flops']/BUDGET
            and 0 <= item['timing']['wall_time_s'] < 60 and 0 <= item['timing']['residual_wall_time_s'] < .4,
            'Same actual physical weights, own complete production F and original strict time gates')
        output = NEW/f"{r['tag']}_{prod}_r{rank}.f32"
        require(str(output) in r['artifacts'] and r['artifacts'][str(output)] == item['output_sha256']
            and output.stat().st_size == 16*1024*4, 'Exact existing all16 production payload')
        overlap.append(dict(production_call=prod,native_call=native,rank=rank,
            weights_path=weight['path'],weight_file_sha256=weight['sha256'],
            production_output=str(output),production_output_sha256=item['output_sha256'],
            production_flops=item['flops'],production_timing=item['timing'],
            operation_count_equality_required=False,fidelity_checked=False))
    return overlap,seals


def panel_admission(args, inputs, tick=None):
    """Same API in worker/supervisor; no numerical producer or truth is imported."""
    advance = _panel_advance(args,inputs,tick)
    controls,seals,rows,fixtures,old_seals = _panel_controls(inputs,tick)
    overlap,runtime = _panel_runtime(inputs,fixtures,tick)
    _panel_merge(inputs,{str(p):h for p,h in _NP_SOURCES.items()})
    bind(inputs,tick)
    return dict(native40_prerequisite_seals=advance,controls=controls,control_seals=seals,
        APUSH_panel_seals=old_seals,panel_rows=rows,fixture_bindings=fixtures,
        production_runtime_seals=runtime,production_overlap=overlap)


def panel_final_checks(result, tick=None):
    """Stdlib saved-file parity/ledger checks, reused after independent retirement."""
    require(result['ranks'] == list(SEQUENCE) and len(result['trials']) == len(result['input_preparation']) ==
        len(result['diagnostic_records']) == 15 and len(result['artifacts']) == 105
        and _v2_validate_ledgers(result), 'All15 full own acquisitions and105 artifacts')
    a,b = result['trials'][0],result['trials'][14]
    da,db = (read(result['diagnostic_records'][i]['path']) for i in (0,14))
    require(a['rank'] == b['rank'] == 35 and a['flops']-b['flops'] == 1944970298
        and a['output_sha256'] == b['output_sha256'] and a['output_bytes_sha256'] == b['output_bytes_sha256']
        and Path(a['output']).read_bytes() == Path(b['output']).read_bytes() and da['domains'] == db['domains'],
        'Own cold/repeat35 full initialization and all16/all47 repeat parity')
    physical = lambda xs:[{k:v for k,v in x.items() if k != 'operation_count'} for x in xs]
    require(physical(da['dispatches']) == physical(db['dispatches']), 'History-aware repeat preserves physical dispatch/F')
    overlap = []
    for item in result['production_overlap']:
        t = result['trials'][item['native_call']]; p = result['input_preparation'][item['native_call']]
        require(t['rank'] == item['rank'] and p['weight_file_sha256'] == item['weight_file_sha256'],
            'Native production-control correspondence uses the exact physical input')
        overlap.append(dict(item,native_output=t['output'],native_output_sha256=t['output_sha256'],
            native_output_bytes_sha256=t['output_bytes_sha256'],native_complete_flops=t['flops'],
            native_input_preparation_flops=p['flops'],native_operation_count=t['operation_count']))
        if tick is not None:
            tick()
    return dict(operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,
        repeat_operation_counts_history_aware=True,panel_repeat35_physics_exact=True,
        repeat_operation_count_history=dict(cold=a['operation_count'],warm=b['operation_count'],equality_required=False),
        production_native_controls=overlap,production_same_input_controls_sealed=True,
        production_fidelity_checked=False,production_native_cost_equality_required=False,
        production_five_qualified=False,production_timing_borrowed=False)


_NP_ORIGINAL = _NP_PARENT['_N40_DERIVED']
_NP_EDITS = [
    ('BUDGET, CAP, RSS = 2**41, 3000., 8_000_000_000','BUDGET, CAP, RSS = 2**41, 1200., 8_000_000_000'),
    ("HERE/'native40.py'","HERE/'native_panel.py'"),
    ("HERE/'watch_native40.py'","HERE/'watch_native_panel.py'"),
    ("HERE/'NATIVE40_PROTOCOL.md'","HERE/'NATIVE_PANEL_PROTOCOL.md'"),
    ("HERE/'NATIVE40_WATCH_SPEC.md'","HERE/'NATIVE_PANEL_WATCH_SPEC.md'"),
    ("HERE/'native40_v1'","HERE/'native_panel_v1'"),
    ('current_comp_shared_old384_native40_v1','current_comp_shared_old384_native_panel_v1'),
    ('native40_source_inverse_proof=NATIVE40_SOURCE_INVERSE_PROOF','panel_source_inverse_proof=PANEL_SOURCE_INVERSE_PROOF'),
    ('Whole native stage3000s/process8GB','Whole native stage1200s/process8GB'),
    ("        prior_check4,result['check4_prerequisite_seals'] = _n40_prerequisites(args,result['input_bindings'],tick)\n"
     "        result['check4_prerequisite_qualified'] = True",
     "        panel = panel_admission(args,result['input_bindings'],tick)\n"
     "        result.update(panel,native40_expansion_prerequisite_passed=True,APUSH_panel_comparator_qualified=True)"),
    ("        for rank,(row_sha,weight_sha) in _n40_store_rows(result['input_bindings'],tick).items():\n"
     "            FIXTURE[STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json'] = row_sha\n"
     "            FIXTURE[STORE/f'rows/observer_fit/{rank:04d}/weights.npy'] = weight_sha\n"
     "        result['input_bindings'].update({str(p):v for p,v in FIXTURE.items()})\n"
     "        result['controls'],result['control_seals'] = controls(result['input_bindings'],tick)",
     "        # Direct positive panel_admission already bound all14 canonical physical fixtures."),
    ("        rows = {rank:dict(weights=str(STORE/f'rows/observer_fit/{rank:04d}/weights.npy'),\n"
     "            mlp_seed=int(read(STORE/f'rows/observer_fit/{rank:04d}/row_manifest.json')['mlp_seed'])) for rank in RANKS}",
     "        rows = {x['rank']:dict(weights=x['weights_path'],mlp_seed=x['mlp_seed']) for x in panel['panel_rows']}"),
    ('if call == 40:  # Only the final37 is a repeated network.',
     'if call == 14:  # Only the final35 is a repeated network.'),
    ("require(len(result['trials']) == len(result['diagnostic_records']) == 41, 'All41 complete fixed calls')",
     "require(len(result['trials']) == len(result['diagnostic_records']) == 15, 'All15 complete fixed panel calls')"),
    ("        result['check4_parity_checks'] = _n40_parity(result,prior_check4)\n"
     "        result.update(check4_first4_exact=True,check4_repeat37_physics_exact=True,\n"
     "            operation_ledgers_verified=True,operation_counts_reported_not_rebased=True,\n"
     "            repeat_operation_counts_history_aware=True)",
     "        result.update(panel_final_checks(result,tick))"),
    ("result['trials'][0]['flops']-result['trials'][40]['flops']","result['trials'][0]['flops']-result['trials'][14]['flops']"),
    ("result['full_model_calls'] = result['retired_call_count'] = 41","result['full_model_calls'] = result['retired_call_count'] = 15"),
]
_NP_DERIVED = _NP_ORIGINAL
for _np_before,_np_after in _NP_EDITS:
    if _NP_DERIVED.count(_np_before) != 1:
        raise RuntimeError('Unique panel metadata span '+_np_before[:100])
    _NP_DERIVED = _NP_DERIVED.replace(_np_before,_np_after)
_NP_INVERSE = _NP_DERIVED
for _np_before,_np_after in reversed(_NP_EDITS):
    if _NP_INVERSE.count(_np_after) != 1:
        raise RuntimeError('Unique panel inverse span '+_np_after[:100])
    _NP_INVERSE = _NP_INVERSE.replace(_np_after,_np_before)
if _NP_INVERSE != _NP_ORIGINAL:
    raise RuntimeError('Entire frozen native40 resolved source restored byte-exact')
ast.parse(_NP_DERIVED)
PANEL_SOURCE_INVERSE_PROOF = dict(owner=str(_NP_OWNER),owner_sha256=_NP_OWNER_SHA,
    whole_source_inverse_verified=True,derived_sha256=hashlib.sha256(_NP_DERIVED.encode()).hexdigest(),
    candidate_paid_loop_observers_and_closed_ledger_unchanged=True,fixed_roster=list(SEQUENCE),
    whole_cap_s=1200.,direct_positive_native40_objective_required=True,
    same_original_APUSH_panel_physical_fixtures=True,production_overlap_metadata_only=True,
    no_ancestor_admission_or_numerical_reexecution=True)
exec(compile(_NP_DERIVED,str(_NP_OWNER)+':conditional-panel-metadata','exec'),globals())
