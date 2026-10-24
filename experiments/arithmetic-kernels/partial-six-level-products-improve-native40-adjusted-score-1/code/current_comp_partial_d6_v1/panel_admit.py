"""Short-lived full panel admission; no estimator, targets or numerical imports."""
import json
import resource
import signal
import sys
import time
import traceback
from types import SimpleNamespace
import panel_common as c


def qualified_native40(args,guard):
    """Call only the already qualified target-free scorer admission functions."""
    result = dict(source_bindings={},input_bindings={})
    scorer = c.load(c.HERE/'score40.py','cd_panel_completed40_metadata',c.SCORE40_SOURCES['score40.py'])
    g = scorer.SCORER_NAMESPACE
    admit = g['Admission'](result,guard)
    for path,digest in g['PINS'].items(): admit.pin(path,digest)
    for path,digest in {**c.SCORE40_SOURCES,**c.SCORE40_SEALS}.items(): admit.pin(c.HERE/path,digest)
    score,_,ss = admit.chain(c.HERE/'score40_v1',args.score40_sha256,limit=120)
    watch,_,ws = admit.chain(c.HERE/'score40_watch_v1',args.score40_watch_sha256,watch=True,limit=120)
    c.require(score['schema'] == 'current_comp_partial_d6_score40_v1'
        and all(score.get(k) is True for k in ('passed','completed','integrity_passed','hashes_unchanged',
            'sources_unchanged','advances_to_independent14','partial_D6_component','all41_domains_and_selectors_reverified',
            'check4_expansion_prerequisite_passed','check4_first4_native_packets_reverified',
            'exact_CHECK4_first4_score_replay','all_predictions_verified_before_truth','no_tuning_or_new_scientific_choice'))
        and score['comparisons']['terminal_vs_COMP']['gate_passed']
        and score['comparisons']['terminal_vs_COMP']['mean_adjusted_ratio'] < 1
        and score['comparisons']['terminal_vs_COMP']['adjusted_wins'] >= 24
        and len(score['ranks']) == len(score['records']) == 40,'Completed prospective native40 scientific gate')
    c.require(watch['schema'] == 'current_comp_partial_d6_score40_watch_v1'
        and watch['completed'] and watch['cpu_slot_clear_attested'] and watch['numerical_imports'] is False
        and watch['full_model_calls'] == watch['retired_calls'] == 0 and watch['whole_cap_s'] == 120
        and 0 < watch['maximum_combined_rss_bytes'] < 1_000_000_000
        and 0 < watch['maximum_interval_s'] <= 1
        and all(watch['child_'+name] == digest for name,digest in ss.items()),
        'Actual passed external score40 supervisor and three child seals')
    for name in ('samples.jsonl','worker.log','started.json'):
        admit.pin(c.HERE/'score40_watch_v1'/name,c.sha(c.HERE/'score40_watch_v1'/name))
    samples = [json.loads(line) for line in (c.HERE/'score40_watch_v1/samples.jsonl').read_text().splitlines()]
    c.require(samples and len(samples) == watch['observation_samples']
        and c.math.isfinite(watch['initial_swap_bytes']) and watch['initial_swap_bytes'] >= 0
        and max(s['combined_rss_bytes'] for s in samples) == watch['maximum_combined_rss_bytes']
        and all(0 <= s['swap_bytes'] <= watch['initial_swap_bytes'] and 0 < s['combined_rss_bytes'] < 1_000_000_000
            and 0 <= s['elapsed_s'] < 120 and s['retired_calls'] == 0 for s in samples),
        'Complete zero-added-swap score40 resource history')
    request = score['requested_seals']; prior_args = SimpleNamespace(**request)
    c.require(request['source_sha256'] == c.SCORE40_SOURCES['score40.py']
        and request['protocol_sha256'] == c.SCORE40_SOURCES['SCORE40_PROTOCOL.md']
        and request['terminal_candidate_sha256'] == c.CANDIDATE_SHA
        and request['terminal_build_sha256'] == c.BUILD_SHA
        and request['builder_sha256'] == c.BUILDER_SHA and request['build_protocol_sha256'] == c.BUILD_PROTOCOL_SHA,
        'Exactly the same source build and candidate, not a fit or rebuild')
    g['score40_prior_admission'](g,prior_args,admit)
    env,_ = g['helpers'](admit)
    folder,baseline,_,refs = g['baseline'](admit)
    g['original_control_checks'](admit,env,folder,baseline,refs)
    builder = c.load(c.HERE/'build.py','cd_panel_unchanged_build_verifier',c.BUILDER_SHA)
    _,native,lineage = g['arm_admission'](prior_args,'terminal',admit,env,refs,builder)
    c.require(lineage == score['lineage']['terminal']
        and native['full_model_calls'] == native['retired_call_count'] == 41
        and native['build_watch_sha256'] == c.BUILD_WATCH_SHA,
        'Every native40 output/bill/domain/dispatch, retired producer and CHECK4 prerequisite reverified')
    c.require(not any(name in sys.modules for name in ('numpy','torch','flopscope','threadpoolctl')),
        'No scoring/estimator/numerical import in selected-source admission')
    return result,native,dict(score=ss,score_watch=ws,native_lineage=lineage)


def qualified_comp(guard,inputs):
    """Authenticate the actual historical TWO-seal independent14 comparator."""
    pc,oa = c.comp_helper(),c.old_args()
    for name,digest in c.COMP_SEALS.items(): c.pin_to(inputs,guard)(c.COMP/'panel_v2'/name,digest)
    context = pc.read_context(oa,oa.context_sha256,guard)
    pc.retired_admission(oa,context,guard)
    native,end = c.sealed(c.CONTROL,c.CONTROL_SHA,inputs,c.CONTROL_END_SHA,guard)
    pc.verify_native(native,end,c.CONTROL_SHA,oa,context,inputs)
    native_watch = pc.verify_watch(oa,context,inputs,guard)
    c.require(native['candidate_sha256'] == pc.CANDIDATE_SHA
        and native['ranks'] == list(c.RANKS) and len(native['trials']) == 15
        and native_watch['prerequisites'] == native['prerequisites'],
        'Current COMP panel, not its old P1 comparator')
    score,_ = c.sealed(c.CONTROL_SCORE,c.CONTROL_SCORE_SHA,inputs,tick=guard)
    watch,_ = c.sealed(c.COMP/'panel_v2/score_watch',c.CONTROL_SCORE_WATCH_SHA,inputs,tick=guard)
    c.require(score['schema'] == pc.SCORE_SCHEMA and score['mode'] == 'score'
        and score['integrity_passed'] and score['retain_local_component'] and score['baseline_recomputed_exact']
        and score['candidate_sha256'] == pc.CANDIDATE_SHA and score['acquisition_sha256'] == c.CONTROL_SHA
        and score['acquisition_watch_sha256'] == c.CONTROL_WATCH_SHA
        and [row['rank'] for row in score['per_role']['terminal']] == list(c.RANKS),
        'Same completed COMP independent4M scoring frame, truth arrays unopened')
    c.require(watch['schema'] == pc.WATCH_SCHEMA and watch['stage'] == 'score'
        and watch['sources_unchanged'] and watch['whole_owned_tree_retired'] and watch['child_execution_completed']
        and watch['child_result_sha256'] == c.CONTROL_SCORE_SHA
        and watch['child_completion_sha256'] == c.COMP_SEALS['score/completion.json']
        and watch['acquisition_sha256'] == c.CONTROL_SHA and watch['acquisition_watch_sha256'] == c.CONTROL_WATCH_SHA
        and watch['elapsed_s'] < 120 and watch['maximum_sampled_rss_bytes'] < 2_000_000_000
        and watch['maximum_swap_growth_bytes'] == 0 and 0 < watch['maximum_sample_interval_s'] <= 1
        and all(watch[k] for k in ('actual_completed_samples','observational_groups_retired','foreign_numerical_workers_checked'))
        and watch['monitor_mock_result_sha256'] == c.MONITOR_TEST_RESULT_SHA,
        'Passed historical independent score watch; no invented finalization')
    for record in (context,native,native_watch,score,watch): c.inventory(record,inputs,guard)
    h,p,_ = c.panel_modules(); fixtures,rows = p.fixtures(h)
    c.require(fixtures == native['fixture_bindings'] and rows == context['rows']
        and tuple(row['rank'] for row in rows) == c.RANKS,'Exact original14 physical fixtures and seeds')
    refs = {}
    for trial,prepared,diagnostic in zip(native['trials'],native['input_preparation'],native['diagnostic_records']):
        refs.setdefault(trial['rank'],dict(trial,weight_file_sha256=prepared['weight_file_sha256'],diagnostic=diagnostic))
    c.require(set(refs) == set(c.RANKS),'All14 current COMP canonical references')
    return fixtures,rows,refs


def full_admission(args,guard):
    reviewed = c.owners(args,guard)
    admitted,native40,lineage = qualified_native40(args,guard)
    sources,inputs = {**admitted['source_bindings'],**reviewed},dict(admitted['input_bindings'])
    inputs.update(c.transport_admission(guard))
    fixtures,rows,refs = qualified_comp(guard,inputs)
    for item in fixtures.values(): c.pin_to(inputs,guard)(item['path'],item['sha256'])
    n = c.e.old(); legacy = n.legacy()
    sources.update({str(path):digest for path,digest in legacy.PINS.items()})
    _observer,_source,proof = c.paid_source(n)
    for key,value in c.d.observer_sources()[1].items():
        c.require(proof[key] == native40['proof'][key] == value,'Unchanged qualified ten-code CD observer')
    c.require(proof['paid_loop'] == native40['proof']['paid_loop']
        and proof['panel_fixture_inverse_verified'] and proof['no_CHECK4_replay_on_panel_inputs'],
        'Same paid loop and participant observation, only original panel fixtures/roster')
    c.bind(sources,guard); c.bind(inputs,guard); c.environment(); guard()
    return dict(schema=c.CONTEXT_SCHEMA,role='terminal',ranks=list(c.RANKS),reviewed_owners=reviewed,
        candidate_sha256=c.CANDIDATE_SHA,build_sha256=c.BUILD_SHA,builder_sha256=c.BUILDER_SHA,
        build_protocol_sha256=c.BUILD_PROTOCOL_SHA,build_watch_sha256=c.BUILD_WATCH_SHA,
        fixtures=fixtures,rows=rows,reference_by_rank=refs,source_bindings=sources,input_bindings=inputs,
        proof=proof,prerequisites=c.prerequisites(),native40_lineage=lineage,
        native40_expansion_prerequisite_passed=True,COMP_panel_comparator_qualified=True,
        full_frozen_admission_passed=True,estimator_executed=False,numerical_imports=False,truth_accessed=False)


def main():
    args = c.arguments('admitter')
    if args is None: return 0
    began = time.monotonic(); c.environment()
    c.require(not c.ADMISSION.exists() and not c.ROOT.is_symlink(),'Exclusive bounded panel admission')
    c.ROOT.mkdir(exist_ok=True); c.ADMISSION.mkdir(exist_ok=False)
    result = dict(schema=c.ADMIT_SCHEMA,role='terminal',passed=False,completed=False,hashes_unchanged=False,
        sources_unchanged=False,reviewed_owners={},source_bindings={},input_bindings={},artifacts={},
        estimator_executed=False,numerical_imports=False,truth_accessed=False)
    def guard(*unused):
        c.require(time.monotonic()-began < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000,
            'Full stdlib admission60s/2GB, within native campaign120s progress')
    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        context = full_admission(args,guard)
        c.require(len((json.dumps(context,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()) <= c.CONTEXT_LIMIT,
            'Only bounded metadata is handed to the fresh worker')
        c.write(c.ADMISSION/'context.json',context)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            reviewed_owners=context['reviewed_owners'],source_bindings=context['source_bindings'],
            input_bindings=context['input_bindings'],context_sha256=c.sha(c.ADMISSION/'context.json'),
            full_frozen_admission_passed=True,proof=context['proof'])
        result['artifacts'][str(c.ADMISSION/'context.json')] = result['context_sha256']; guard()
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        result.update(elapsed_s=time.monotonic()-began,peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        if result['elapsed_s'] >= 60 or result['peak_rss_bytes'] >= 2_000_000_000:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False)
        c.write(c.ADMISSION/'result.json',result)
        result_sha = c.sha(c.ADMISSION/'result.json')
        c.write(c.ADMISSION/'completion.json',dict(passed=result['passed'],completed=result['completed'],result_sha256=result_sha))
        completion_sha = c.sha(c.ADMISSION/'completion.json'); elapsed = time.monotonic()-began
        passed = bool(result['passed'] and elapsed < 60 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 2_000_000_000)
        c.write(c.ADMISSION/'finalization.json',dict(passed=passed,completed=passed,result_sha256=result_sha,
            completion_sha256=completion_sha,elapsed_s=elapsed))
    return 0 if passed and time.monotonic()-began < 60 else 1


if __name__ == '__main__':
    raise SystemExit(main())
