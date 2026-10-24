"""Short-lived full source admission; no estimator, numerical imports or truths."""
import json
import signal
import time
import traceback
from types import SimpleNamespace
import panel_common as c


def full_admission(args,guard):
    reviewed = c.owners(args,guard)
    n = c.e.old()
    admitted = n.authenticate('terminal',n.BUILD_SHA,c.CANDIDATE_SHA,n.BUILDER_SHA,
        c.e.OLD_NATIVE_SHA,c.e.OLD_WATCH_SHA,c.e.ZERO_SHA,c.e.ZERO_WATCH_SHA)
    sources,inputs = {**admitted['source_bindings'],**reviewed},dict(admitted['input_bindings'])
    inputs.update(c.transport_admission(guard))
    native,end = c.sealed(c.e.OUTPUT,c.NATIVE40_SHA,inputs,c.NATIVE40_END_SHA)
    watch,_ = c.sealed(c.e.WATCH_OUT,c.NATIVE40_WATCH_SHA,inputs)
    old_args = SimpleNamespace(watcher_sha256=c.WATCH_OWNER_SHA,worker_sha256=c.NATIVE_OWNER_SHA,
        admitter_sha256=c.ADMIT_OWNER_SHA,helper_sha256=c.ENDPOINT_OWNER_SHA,endpoint_sha256=c.e.ENDPOINT_SHA,
        context_sha256=watch['admission_context_sha256'],admission_watch_sha256=watch['admission_watch_sha256'])
    old_context = c.e.read_context(old_args,old_args.context_sha256,guard)
    c.e.retired_admission(old_args,old_context)
    c.e.verify_native(native,end,c.NATIVE40_SHA,old_args,old_context)
    c.require(watch['schema'] == c.e.WATCH_SCHEMA and watch['role'] == 'terminal' and
        all(watch.get(k) is True for k in ('sources_unchanged','child_execution_completed',
            'whole_owned_tree_retired','admission_process_reaped_before_native','first22_output_bill_domain_byte_exact'))
        and watch['child_result_sha256'] == c.NATIVE40_SHA and
        watch['child_completion_sha256'] == c.NATIVE40_END_SHA and watch['completed_bill_count'] == 41 and
        watch['first22_exact_checks'] == 22 and watch['maximum_sampled_rss_bytes'] < 8_000_000_000 and
        watch['maximum_swap_growth_bytes'] == 0 and watch['maximum_sample_interval_s'] <= 1 and
        watch['elapsed_s'] < 1900, 'Complete qualified NODE/P1 native40, original failure retained')
    for record in (native,watch):
        c.inventory(record,inputs,guard)
    score,_ = c.sealed(c.HERE/'score_v3',c.SCORE_SHA,inputs,c.SCORE_END_SHA)
    c.require(score['schema'] == 'current_p1_node_composition_score_v3' and
        all(score.get(k) is True for k in ('integrity_passed','sources_unchanged','advances_to_independent14',
            'objective_improved','all_predictions_verified_before_truth','first22_output_bill_domain_byte_exact')) and
        score['candidate_sha256'] == c.CANDIDATE_SHA and score['baseline_sha256'] == c.P1_SHA and
        len(score['records']) == 40 and score['adjusted_wins'] >= 24 and score['mean_adjusted_ratio'] < 1 and
        score['source_bindings'][str(c.HERE/'score_v3.py')] == c.SCORE_OWNER_SHA,
        'Single native-objective-selected component, not a refit or partial score')
    c.inventory(score,inputs,guard)
    seal_path = c.HERE/'score_v3/predictions_and_bills_sealed.json'
    c.pin_to(inputs,guard)(seal_path,score['predictions_and_bills_seal_sha256'])
    seal = c.read(seal_path)
    c.require(seal['truth_accessed'] is False and seal['terminal_sha256'] == c.NATIVE40_SHA and
              seal['terminal_watch_sha256'] == c.NATIVE40_WATCH_SHA, 'Native40 pretruth output/bill ownership')
    h,p,v = c.panel_modules()
    fixtures,rows = p.fixtures(h)  # Original public metadata and explicit virgin fixtures; hash only.
    c.require(tuple(r['rank'] for r in rows) == c.RANKS, 'Original fixed fixture order')
    control,_ = c.sealed(c.CONTROL,c.CONTROL_SHA,inputs,c.CONTROL_END_SHA)
    control_watch,_ = c.sealed(c.P1/'panel_v1/terminal_watch',c.CONTROL_WATCH_SHA,inputs)
    c.require(control['candidate_sha256'] == c.P1_SHA and control['truth_accessed'] is False and
        control['truth_files_accessed'] is False and control['fixture_bindings'] == fixtures and
        all(control.get(k) is True for k in ('all16_domains_qualified','all47_selectors_qualified',
            'own_complete_bills','sources_unchanged','layer0_byte_exact')) and
        control['source_bindings'][str(c.P1/'panel.py')] == c.P1_PANEL_SHA and
        control_watch['schema'] == 'current_d50_native_objective_p1_independent14_watch_v1' and
        control_watch['stage'] == 'native' and control_watch['child_result_sha256'] == c.CONTROL_SHA and
        control_watch['child_completion_sha256'] == c.CONTROL_END_SHA and
        control_watch['child_execution_completed'] and control_watch['whole_owned_tree_retired'] and
        control_watch['completed_bill_count'] == 15 and
        control_watch['source_bindings'][str(c.P1/'watch_panel.py')] == c.P1_WATCH_SHA,
        'Actual sealed P1 same-sequence independent14 comparator and resource watch')
    for record in (control,control_watch):
        c.inventory(record,inputs,guard)
    fixture_context = dict(fixtures=fixtures,rows=rows)
    c.five_artifact_verifier(fixture_context,c.P1_CANDIDATE,c.P1_SHA,1383913786)(
        control,c.CONTROL,c.RANKS,c.pin_to(inputs,guard))
    control_score,_ = c.sealed(c.CONTROL_SCORE,c.CONTROL_SCORE_SHA,inputs)
    c.require(control_score['schema'] == 'current_d50_native_objective_p1_independent14_v1' and
        control_score['mode'] == 'score' and control_score['integrity_passed'] and
        control_score['retain_local_component'] and control_score['baseline_recomputed_exact'] and
        control_score['candidate_sha256'] == c.P1_SHA and control_score['acquisition_sha256'] == c.CONTROL_SHA and
        control_score['acquisition_watch_sha256'] == c.CONTROL_WATCH_SHA and
        [r['rank'] for r in control_score['per_role']['terminal']] == list(c.RANKS),
        'Same original independent4M scored P1 frame; truth arrays remain unopened')
    c.inventory(control_score,inputs,guard)
    refs = {}
    for trial,prepared in zip(control['trials'],control['input_preparation']):
        refs.setdefault(trial['rank'],dict(trial,weight_file_sha256=prepared['weight_file_sha256']))
    c.require(set(refs) == set(c.RANKS), 'All14 canonical P1 reference inputs')
    for item in fixtures.values():
        c.pin_to(inputs,guard)(item['path'],item['sha256'])
    _observer,_paid,proof = c.paid_source(n)
    c.require(all(proof[k] == native['proof'][k] for k in native['proof']) and
              proof['interested_code_count'] == 8, 'Exact qualified NODE observer and paid-loop parent')
    sources.update({str(n.BD/'native_v2.py'):n.NATIVE_SHA,str(n.CENSUS):n.CENSUS_SHA,
                    str(n.TL/'native.py'):n.OBSERVER_SHA,str(n.TL/'common.py'):n.COMMON_SHA})
    c.bind(sources,guard); c.bind(inputs,guard); c.environment(); guard()
    return dict(schema=c.CONTEXT_SCHEMA,reviewed_owners=reviewed,ranks=list(c.RANKS),
        candidate_sha256=c.CANDIDATE_SHA,build_sha256=c.BUILD_SHA,fixtures=fixtures,rows=rows,
        reference_by_rank=refs,source_bindings=sources,input_bindings=inputs,proof=proof,
        prerequisites=dict(native40_score_sha256=c.SCORE_SHA,native40_sha256=c.NATIVE40_SHA,
            native40_watch_sha256=c.NATIVE40_WATCH_SHA,control_sha256=c.CONTROL_SHA,
            control_watch_sha256=c.CONTROL_WATCH_SHA,control_score_sha256=c.CONTROL_SCORE_SHA),
        full_frozen_admission_passed=True,estimator_executed=False,numerical_imports=False,truth_accessed=False)


def main():
    args = c.arguments('admitter')
    if args is None: return 0
    began = time.monotonic(); c.environment()
    c.require(not c.ADMISSION.exists() and not c.ROOT.is_symlink(), 'Exclusive source-admission output')
    c.ROOT.mkdir(exist_ok=True); c.ADMISSION.mkdir(exist_ok=False)
    result = dict(schema=c.ADMIT_SCHEMA,passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,
        reviewed_owners=c.owners(args),source_bindings={},input_bindings={},artifacts={},
        estimator_executed=False,numerical_imports=False,truth_accessed=False)
    def guard(*unused):
        c.require(time.monotonic()-began < 60, 'Complete source admission60s, inside campaign120s progress')
    previous = signal.signal(signal.SIGALRM,guard); signal.setitimer(signal.ITIMER_REAL,.2,.2)
    try:
        context = full_admission(args,guard)
        encoded = json.dumps(context,sort_keys=True,indent=2,allow_nan=False)+'\n'
        c.require(len(encoded.encode()) <= c.CONTEXT_LIMIT, 'Bounded metadata, no literals/arrays/modules')
        c.write(c.ADMISSION/'context.json',context)
        result.update(passed=True,completed=True,hashes_unchanged=True,sources_unchanged=True,
            source_bindings=context['source_bindings'],input_bindings=context['input_bindings'],
            context_sha256=c.sha(c.ADMISSION/'context.json'),full_frozen_admission_passed=True,proof=context['proof'])
        result['artifacts'][str(c.ADMISSION/'context.json')] = result['context_sha256']; guard()
    except BaseException:
        result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False,error=traceback.format_exc())
    finally:
        signal.setitimer(signal.ITIMER_REAL,0); signal.signal(signal.SIGALRM,previous)
        result['elapsed_s'] = time.monotonic()-began
        if result['elapsed_s'] >= 60:
            result.update(passed=False,completed=False,hashes_unchanged=False,sources_unchanged=False)
        c.write(c.ADMISSION/'result.json',result)
        c.write(c.ADMISSION/'completion.json',dict(passed=result['passed'],completed=result['completed'],
            result_sha256=c.sha(c.ADMISSION/'result.json')))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
