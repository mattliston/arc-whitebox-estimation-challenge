"""Direct completed APUSH40/CD14 boundaries; no ancestor admission or score."""
import json
import resource
import signal
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
import panel_common as c


def qualified_native40(args,guard):
    sources,inputs = {},{}
    records = {}
    for name,digests in c.APUSH_CHAINS.items():
        records[name] = c.direct_chain(c.HERE/name,digests,inputs,guard,
            60 if name.endswith('admission_v1') else 1900 if 'native40' in name else 120)
    score = records['score40_v1'][0]; watch = records['score40_watch_v1'][0]
    c.require((args.score40_sha256,args.score40_watch_sha256) == (c.SCORE_SHA,c.SCORE_WATCH_SHA)
        and score['schema'] == 'current_comp_aligned_push_d6_score40_v1'
        and score['candidate_sha256'] == c.CANDIDATE_SHA and len(score['records']) == 40
        and all(score.get(k) is True for k in ('integrity_passed','advances_to_independent14',
            'aligned_push_component','all41_domains_and_selectors_reverified','all41_new_outputs_verified',
            'all_predictions_verified_before_truth','complete_APUSH_acquisition_reverified',
            'check4_expansion_prerequisite_passed','exact_CHECK4_first4_score_replay','exact_CD_score_replay',
            'exact_CD_all40_summary_replay','no_model_imported','own_complete_bills','runtime_prerequisite_qualified',
            'original_external_watch_still_failed','completed_CHECK4_boundary_not_reexecuted','no_tuning_or_candidate_change'))
        and score['full_model_calls'] == score['parameter_updates'] == 0,
        'Direct completed score40, unchanged candidate and explicit old watch failure')
    comparison = score['comparisons']['terminal_vs_CD']
    c.require(comparison['gate_passed'] and comparison['mean_adjusted_ratio'] < 1
        and comparison['adjusted_wins'] >= 30 and comparison['at_least_30_wins'],
        'Actual prospective strict own-cost native40 gate')
    c.watch_boundary(watch,c.APUSH_CHAINS['score40_v1'],120,1_000_000_000)
    c.require(watch['schema'] == 'current_comp_aligned_push_d6_score40_watch_v1'
        and watch['stage'] == 'score' and watch['full_model_calls'] == 0
        and watch['numerical_imports'] is False,'Actual separate CPU-only positive score watch')
    request = score['requested_seals']
    expected = dict(source_sha256=c.SCORE40_SOURCES['score40.py'],
        protocol_sha256=c.SCORE40_SOURCES['SCORE40_PROTOCOL.md'],
        watcher_sha256=c.SCORE40_SOURCES['watch_score40.py'],watch_spec_sha256=c.SCORE40_SOURCES['SCORE40_WATCH_SPEC.md'],
        native_source_sha256=c.SCORE40_SOURCES['native40.py'],native_protocol_sha256=c.SCORE40_SOURCES['NATIVE40_PROTOCOL.md'],
        native_watch_source_sha256=c.SCORE40_SOURCES['watch_native40.py'],
        native_watch_spec_sha256=c.SCORE40_SOURCES['NATIVE40_WATCH_SPEC.md'],
        native_sha256=c.APUSH_CHAINS['native40_terminal_v1'][0],
        native_watch_sha256=c.APUSH_CHAINS['native40_watch_v1'][0],
        admission_sha256=c.APUSH_CHAINS['native40_terminal_admission_v1'][0],
        build_sha256=c.BUILD_SHA,runtime_sha256=c.RUNTIME_SHA,candidate_sha256=c.CANDIDATE_SHA)
    c.require(all(request[k] == value for k,value in expected.items()),'Actual score request and all current owner/receipt joins')
    native,end,_ = records['native40_terminal_v1']; nw = records['native40_watch_v1'][0]
    c.watch_boundary(nw,c.APUSH_CHAINS['native40_terminal_v1'],1900,c.RSS_LIMIT)
    c.require(nw['schema'] == 'current_comp_aligned_push_d6_native40_watch_v1'
        and nw['all41_complete'] and nw['full_model_calls'] == nw['retired_call_count'] == 41
        and nw['admission_producer_reaped'] and nw['context_sha256'] == c.NATIVE40_CONTEXT_SHA
        and nw['admission_seals'] == c.seal_map(c.APUSH_CHAINS['native40_terminal_admission_v1'])
        and [p['phase'] for p in nw['phases']] == ['admission','worker']
        and all(p['exited'] for p in nw['phases'])
        and nw['candidate_sha256'] == c.CANDIDATE_SHA and nw['initialization_flops'] == c.INIT
        and nw['maximum_admission_rss_bytes'] < 2_000_000_000
        and nw['numerical_imports'] is False and nw['truth_arrays_accessed'] is False,
        'Actual two retired bounded native40 phases and original context')
    owner = c.load(c.HERE/'native40.py','apush_panel_completed40_verifier',c.SCORE40_SOURCES['native40.py'])
    owner.sha = c.sha
    old_args = SimpleNamespace(role='terminal',source_sha256=c.SCORE40_SOURCES['native40.py'],
        protocol_sha256=c.SCORE40_SOURCES['NATIVE40_PROTOCOL.md'],
        watcher_sha256=c.SCORE40_SOURCES['watch_native40.py'],
        watch_spec_sha256=c.SCORE40_SOURCES['NATIVE40_WATCH_SPEC.md'],context_sha256=c.NATIVE40_CONTEXT_SHA)
    context = owner.read_context(old_args,c.NATIVE40_CONTEXT_SHA,tick=guard)
    owner.verify_native(native,end,c.APUSH_CHAINS['native40_terminal_v1'][0],old_args,context,tick=guard)
    c.require([r['rank'] for r in score['records']] == native['ranks'] == list(owner.RANKS)
        and native['full_model_calls'] == native['retired_call_count'] == 41 and len(native['artifacts']) == 246
        and nw['retired_calls'] == [dict(call=i,rank=r) for i,r in enumerate(owner.RANKS+(37,))],
        'All41 saved native packets reverified, original40 order and exact retired roster')
    lineage = score['lineage']['terminal']
    for key,folder in (('native','native40_terminal_v1'),('watch','native40_watch_v1'),
                       ('admission','native40_terminal_admission_v1')):
        c.require(lineage[key] == c.seal_map(c.APUSH_CHAINS[folder]),'Completed score/native40 lineage '+key)
    for key in ('build','runtime'):
        c.require(lineage[key] == c.seal_map(c.d.CHAINS[c.HERE/(key+'_v1')]),'Same actual '+key+' prerequisite')
    c.require(lineage['context_sha256'] == c.NATIVE40_CONTEXT_SHA
        and lineage['check4'] == context['check4_prerequisites'],
        'Completed CHECK4 boundary retained as metadata, not reexecuted')
    c.inventory(context,inputs,guard)
    inputs[str(c.HERE/'native40_terminal_admission_v1/context.json')] = c.NATIVE40_CONTEXT_SHA
    for record,_,_ in records.values(): sources.update(record['source_bindings'])
    sources.update(context['source_bindings'])
    c.d.runtime_admission(SimpleNamespace(runtime_sha256=c.RUNTIME_SHA,build_sha256=c.BUILD_SHA,
        candidate_sha256=c.CANDIDATE_SHA),sources,inputs,guard)
    c.require(not {'numpy','torch','flopscope','threadpoolctl'} & set(sys.modules),
        'Only frozen stdlib saved-acquisition verifiers; no score/native/ancestor stages')
    return dict(source_bindings=sources,input_bindings=inputs),native,dict(
        score=c.seal_map(c.APUSH_CHAINS['score40_v1']),score_watch=c.seal_map(c.APUSH_CHAINS['score40_watch_v1']),
        native_lineage=lineage,all41_saved_packets_reverified=True,
        original_external_watch_still_failed=True,no_old_repair_or_score_reexecution=True)


def qualified_cd(guard,inputs):
    """Direct qualified CD panel; no reconstruction of its training ancestry."""
    records = {name:c.direct_chain(c.CD/'panel_v2'/name,digests,inputs,guard,
        60 if name == 'admission' else 120 if name in ('score','score_watch') else 1200)
        for name,digests in c.CD_CHAINS.items()}
    ad = records['admission'][0]; native,end,_ = records['terminal']
    watch,score,sw = (records[k][0] for k in ('terminal_watch','score','score_watch'))
    path = c.CD/'panel_v2/admission/context.json'
    c.pin_to(inputs,guard)(path,c.CD_CONTEXT_SHA); context = c.read(path,c.CONTEXT_LIMIT)
    retired_path = c.CONTROL_WATCH/'admission_retired.json'
    c.pin_to(inputs,guard)(retired_path,c.CD_RETIRED_SHA); retired = c.read(retired_path,c.CONTEXT_LIMIT)
    c.require(context['schema'] == 'current_comp_partial_d6_independent14_context_v2'
        and ad['schema'] == 'current_comp_partial_d6_independent14_admission_v2'
        and ad['context_sha256'] == ad['artifacts'][str(path)] == c.CD_CONTEXT_SHA
        and context['reviewed_owners'] == ad['reviewed_owners'] == retired['reviewed_owners']
        and context['source_bindings'] == ad['source_bindings'] and context['input_bindings'] == ad['input_bindings']
        and context['proof'] == ad['proof'] == native['proof']
        and ad['full_frozen_admission_passed'] and context['full_frozen_admission_passed']
        and ad['estimator_executed'] is False and ad['numerical_imports'] is False and ad['truth_accessed'] is False
        and 0 <= ad['elapsed_s'] < 60 and 0 < ad['peak_rss_bytes'] < 2_000_000_000
        and context['candidate_sha256'] == native['candidate_sha256'] == c.d.CD_SHA
        and context['ranks'] == native['ranks'] == list(c.RANKS),
        'Exact direct CD source admission, current candidate and14 fixtures')
    c.require(retired['schema'] == 'current_comp_partial_d6_independent14_admission_watch_v2'
        and all(retired.get(k) is True for k in ('passed','completed','whole_owned_tree_retired',
            'observational_groups_retired','foreign_numerical_workers_checked','actual_completed_samples'))
        and retired['context_sha256'] == c.CD_CONTEXT_SHA and retired['admission_elapsed_s'] < 60
        and retired['whole_elapsed_s'] < 120 and retired['maximum_sampled_rss_bytes'] < c.RSS_LIMIT
        and retired['maximum_swap_growth_bytes'] == 0 and retired['maximum_sample_interval_s'] <= 1
        and retired['monitor_mock_result_sha256'] == c.MONITOR_TEST_RESULT_SHA,'Actual retired CD admission')
    c.watch_boundary(watch,c.CD_CHAINS['terminal'],1200,c.RSS_LIMIT,cd=True)
    c.watch_boundary(sw,c.CD_CHAINS['score'],120,2_000_000_000,cd=True)
    c.require(watch['schema'] == 'current_comp_partial_d6_independent14_watch_v2'
        and watch['stage'] == 'native' and watch['completed_bill_count'] == 15
        and watch['admission_process_reaped_before_native'] and watch['all15_output_bill_domain_qualified']
        and watch['admission_context_sha256'] == c.CD_CONTEXT_SHA and watch['admission_watch_sha256'] == c.CD_RETIRED_SHA
        and tuple(watch['admission_'+k] for k in ('result_sha256','completion_sha256','finalization_sha256')) == c.CD_CHAINS['admission']
        and sw['schema'] == 'current_comp_partial_d6_independent14_score_watch_v1' and sw['stage'] == 'score'
        and sw['acquisition_sha256'] == c.CONTROL_SHA and sw['acquisition_watch_sha256'] == c.CONTROL_WATCH_SHA,
        'Both actual CD supervisors and complete child/admission chains')
    c.require(score['schema'] == 'current_comp_partial_d6_independent14_score_v1' and score['mode'] == 'score'
        and score['candidate_sha256'] == c.d.CD_SHA and score['acquisition_sha256'] == c.CONTROL_SHA
        and score['acquisition_watch_sha256'] == c.CONTROL_WATCH_SHA
        and score['acquisition_finalization_sha256'] == c.CD_CHAINS['terminal'][2]
        and all(score.get(k) is True for k in ('integrity_passed','retain_local_component','all90_artifacts_qualified',
            'all15_domains_and_selectors_reverified','all15_new_outputs_verified','all_predictions_verified_before_truth',
            'baseline_recomputed_exact','exact_COMP14_rows_and_three_summaries_replay','own_complete_bills'))
        and score['group_strict_gain_gate'] == dict(public8=True,virgin6=True,combined14=True)
        and all(score['adjusted_candidate_over_parent'][k] < 1 for k in ('public8','virgin6','combined14'))
        and [r['rank'] for r in score['per_role']['terminal']] == list(c.RANKS),
        'Completed positive BOTH-group CD score; truth metadata only, never decode')
    h,p,v = c.panel_modules(); fixtures,rows = p.fixtures(h)
    c.require(fixtures == context['fixtures'] == native['fixture_bindings'] == score['fixture_bindings']
        and rows == context['rows'] and tuple(r['rank'] for r in rows) == c.RANKS,
        'Exact original independent14 physical fixture/seed ownership')
    for record in (context,native,watch,score,sw):
        c.require_fresh_union(record); c.inventory(record,inputs,guard)
    context['reference_by_rank'] = {int(k):v for k,v in context['reference_by_rank'].items()}
    # Original CD record includes this retired context/handoff union in sources.
    context['input_bindings'].update({str(path):c.CD_CONTEXT_SHA,str(retired_path):c.CD_RETIRED_SHA})
    for name in ('result.json','completion.json','finalization.json'):
        context['input_bindings'][str(path.parent/name)] = c.sha(path.parent/name,guard)
    old_args = SimpleNamespace(context_sha256=c.CD_CONTEXT_SHA,admission_watch_sha256=c.CD_RETIRED_SHA)
    c.cd_callbacks(context)['verify_acquisition'](native,end,c.CONTROL_SHA,old_args,context,guard)
    core = dict(native,artifacts={p:h for p,h in native['artifacts'].items() if not p.endswith('_diagnostics.json')})
    outputs = c.pc.five_artifact_verifier(context,c.CD/'build_v1/estimator.py',c.d.CD_SHA,c.INIT)(
        core,c.CONTROL,c.RANKS,c.pin_to(inputs,guard))
    c.require(len(outputs) == native['full_model_calls'] == native['retired_call_count'] == 15
        and len(native['artifacts']) == 90 and native['initialization_flops'] == c.INIT
        and native['layer0_byte_exact'] and native['all90_artifacts_qualified']
        and native['admission_watch_sha256'] == c.CD_RETIRED_SHA,
        'All90 actual CD output/input/bill/domain/dispatch/reset artifacts independently reverified')
    refs = {}
    for trial,prepared,diagnostic in zip(native['trials'],native['input_preparation'],native['diagnostic_records']):
        refs.setdefault(trial['rank'],dict(trial,trial=trial,input_preparation=prepared,
            weight_file_sha256=prepared['weight_file_sha256'],diagnostic=diagnostic,diagnostics=diagnostic))
    c.require(set(refs) == set(c.RANKS),'All14 actual CD comparator rows')
    return fixtures,rows,refs


def full_admission(args,guard):
    reviewed = c.owners(args,guard)
    admitted,native40,lineage = qualified_native40(args,guard)
    sources,inputs = {**admitted['source_bindings'],**reviewed},dict(admitted['input_bindings'])
    inputs.update(c.transport_admission(guard))
    fixtures,rows,refs = qualified_cd(guard,inputs)
    for item in fixtures.values(): c.pin_to(inputs,guard)(item['path'],item['sha256'])
    n = c.d.base(); legacy = n.legacy()
    sources.update({str(path):digest for path,digest in legacy.PINS.items()})
    _observer,_source,proof = c.paid_source(n)
    c.require(proof['observer'] == native40['proof']['observer'] == c.d.observer_sources()[1]
        and proof['parent_CD']['paid_loop'] == native40['proof']['parent_CD']['paid_loop']
        and proof['metadata'] == native40['proof']['metadata']['parent_CHECK4_metadata']
        and proof['panel_fixture_inverse_verified'] and proof['no_CHECK4_replay_on_panel_inputs'],
        'Same broad APUSH observer/paid loop/metadata; only original panel roster and fixtures')
    fresh = c.final_bind((sources,inputs),guard); c.environment(); guard()
    return dict(schema=c.CONTEXT_SCHEMA,role='terminal',ranks=list(c.RANKS),reviewed_owners=reviewed,
        candidate_sha256=c.CANDIDATE_SHA,build_sha256=c.BUILD_SHA,builder_sha256=c.BUILDER_SHA,runtime_sha256=c.RUNTIME_SHA,
        fixtures=fixtures,rows=rows,reference_by_rank=refs,source_bindings=sources,input_bindings=inputs,
        proof=proof,prerequisites=c.prerequisites(),native40_lineage=lineage,helper_source_proof=c.source_proof(),
        native40_expansion_prerequisite_passed=True,CD_panel_comparator_qualified=True,runtime_prerequisite_qualified=True,
        full_frozen_admission_passed=True,estimator_executed=False,numerical_imports=False,truth_accessed=False,**fresh)


c.install('admit',('main',),globals())

if __name__ == '__main__':
    raise SystemExit(main())
