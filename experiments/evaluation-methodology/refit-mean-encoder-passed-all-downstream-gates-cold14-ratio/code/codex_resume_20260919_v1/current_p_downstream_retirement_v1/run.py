"""Explicit one-shot downstream launch, only after the existing pipeline ends."""
from datetime import datetime,timezone
import os
import time
from common import *


def admit():
    verify()
    controller = read(ENDPOINT/'controller_result.json')
    assert controller['completed'] and controller['scientific_passed']
    assert controller['completed_new_native_calls'] == 41 and controller['reused_qualified_calls'] == 44 and not controller['fit_or_candidate_changed']
    assert controller['no_automatic_downstream'] and controller['no_external_actions']
    check(controller['endpoint'])
    endpoint = read(ENDPOINT/'endpoint_complete.json')
    assert endpoint['completed'] and endpoint['passed'] and endpoint['native_records'] == 85 and endpoint['new_native_calls'] == 41 and endpoint['reused_qualified_calls'] == 44
    assert endpoint['no_automatic_downstream_launch'] and endpoint['original_fit_and_candidate_unchanged']
    assert endpoint['closed_context_lifecycle_qualified'] and not endpoint['official_changed']
    assert endpoint['all_new_retirements_qualified'] and endpoint['historical_trained_prefix_exact_calls'] == 30
    check(endpoint['score'])
    score = read(ENDPOINT/'native40_score.json')
    assert score['completed'] and score['integrity_passed'] and score['passed'] and score['native_score_measured']
    assert score['adjusted_ratio'] < 1 and score['adjusted_wins'] >= 24
    assert score['prediction_seal'] == pin(ENDPOINT/'native40_predictions_sealed.json')
    fit = read(FIT/'fit_result.json')
    assert fit['completed'] and fit['passed'] and fit['training_graphs'] == 160 and fit['updates'] == 40 and fit['epoch'] == 1
    assert fit['fixed_parameters_unchanged'] and fit['inactive_parameters_unchanged']
    assert not fit['validation_targets_accessed'] and not fit['checkpoint_selection']
    check(fit['terminal_payload'])
    for name in ['fit','build']:
        clean_watch(FIT/f'{name}_watch.json')
    for name in ['native_null','qualify_null','qualify_null_lifecycle','native_parent','qualify_parent_lifecycle',
                 'native_trained','qualify_trained_lifecycle','seal_predictions','score_native40']:
        clean_watch(ENDPOINT/f'{name}_watch.json')
    assert read(ENDPOINT/'native_null_qualification.json')['passed']
    assert read(FIT/'native_null_qualification.json')['passed']
    for role,count in [('null',3),('parent',12),('trained',30)]:
        q = read(ENDPOINT/f'{role}_lifecycle_qualified.json')
        assert q['passed'] and q['historical_prefix_exact_calls'] == count
        assert q['closed_contexts_released'] and q['saved_bills_unchanged'] and not q['truth_accessed']
    trained = read(ENDPOINT/'trained_lifecycle_qualified.json')
    assert trained['all_retirements_qualified'] and len(trained['retirement_pins']) == 41
    check(trained['allocator_qualification'])
    allocator_dir = D/'current_p_native_allocator_qualification_v1'
    allocator = read(allocator_dir/'controller_result.json')
    assert allocator['completed'] and allocator['passed'] and allocator['completed_calls'] == 8
    check(allocator['qualification'])
    for name in ['native_baseline','qualify_baseline','native_relief','qualify_comparison']:
        clean_watch(allocator_dir/f'{name}_watch.json')
    copies = read(ENDPOINT/'source_copies.json')
    assert copies['inverse_source_byte_exact'] and copies['paid_body_AST_exact']
    assert copies['original_scoring_and_prediction_sealer_byte_exact'] and copies['reused_qualified_calls'] == 44
    for pair in copies['copies']:
        check(pair['original']);check(pair['copy'])
        assert pair['byte_exact'] and pair['original']['sha256'] == pair['copy']['sha256']
    lifecycle = read(ENDPOINT/'lifecycle_receipts_sealed.json')
    assert lifecycle['completed'] and lifecycle['before_validation_targets'] and lifecycle['all_85_closed_context_calls_qualified']
    for path in [FIT/'sources_sealed.json',FIT/'endpoint_sources_sealed.json',ENDPOINT/'sources_sealed.json',
                 ENDPOINT/'endpoint_sources_sealed.json',ENDPOINT/'native40_predictions_sealed.json',
                 ENDPOINT/'lifecycle_receipts_sealed.json']:
        for p in read(path)['pins']:
            check(p)
    build = read(FIT/'build.json')
    assert build['passed'] and build['only_gnn_literal_changed']
    assert build['fixed_arrays_byte_exact'] and build['inactive_coordinates_byte_exact']
    assert build['parent'] == pin(BANK) and build['candidate'] == score['candidate']
    assert build['candidate']['sha256'] == '8260ddb1f4966cc6acc4c318a53835981fde1b839d946ac10604b18325d2a040'
    check(build['candidate'])
    pins = [pin(FIT/n) for n in ['fit_result.json','fit_watch.json','fit_complete.json','build.json',
        'build_watch.json','native_null_qualification.json','native_parent_watch.json']]
    pins += [pin(ENDPOINT/n) for n in ['controller_result.json','endpoint_complete.json','sources_sealed.json',
        'endpoint_sources_sealed.json','native_null_qualification.json','qualify_null_watch.json',
        'null_lifecycle_qualified.json','parent_lifecycle_qualified.json','trained_lifecycle_qualified.json',
        'native40_predictions_sealed.json','lifecycle_receipts_sealed.json','seal_predictions_watch.json',
        'native40_score.json','score_native40_watch.json']]
    pins += [fit['terminal_payload'],build['parent'],build['candidate']]
    write(HERE/'admission.json',dict(passed=True,sole_terminal_fit=True,bank=build['parent'],candidate=build['candidate'],
        pins=pins,native40_artifacts_verified=True,previous_fit_and_native_workers_retired=True,
        lifecycle_native_endpoint=pin(ENDPOINT/'endpoint_complete.json'),original_failed_endpoint_preserved=True,
        scientific_rosters_gates_and_candidate_unchanged=True))


def main():
    verify()
    assert not (HERE/'run_claim.json').exists() and not (HERE/'admission.json').exists()
    processes = subprocess.check_output(['/bin/ps','-axo','pid=,pgid=,command='],text=True,timeout=3)
    workers = []
    for line in processes.splitlines():
        a = line.strip().split(None,2)
        if len(a) == 3 and (a[2].startswith('/Users/matt/anaconda3/bin/python') or '/openblas_runtime_env_v1/bin/python -B ' in a[2]):
            workers.append(line)
    assert not workers, workers
    admit()
    write(HERE/'run_claim.json',dict(utc=datetime.now(timezone.utc).isoformat(),pid=os.getpid(),pgid=os.getpgrp(),
        admission=pin(HERE/'admission.json'),no_other_numerical_worker=True,maximum_new_calls=55))
    started = time.monotonic()
    result = dict(completed=False,passed=False,completed_stages=[],error=None,official_changed=False,
        no_automatic_release_or_upload=True,no_retry=True)
    try:
        guarded('control',480,'native.py','control','parent')
        guarded('qualify_control',180,'qualify_control.py')
        for stage,ranks in [('panel',PANEL),('confirmation',CONFIRM),('independent',INDEPENDENT)]:
            if stage == 'independent':
                for role in ['parent','trained']:
                    guarded(f'independent_{role}',2400,'native.py',stage,role)
            else:
                for rank in ranks:
                    guarded(f'{stage}_{rank}',240,'native.py',stage,'trained','--rank',rank)
            guarded(f'seal_{stage}',180,'seal_stage.py',stage)
            guarded(f'score_{stage}',180,'score_stage.py',stage)
            score = read(HERE/f'{stage}_score.json')
            assert score['completed'] and score['integrity_passed']
            result['completed_stages'].append(dict(stage=stage,passed=score['passed'],score=pin(HERE/f'{stage}_score.json')))
            if not score['passed']:
                result.update(completed=True,scientific_stop_stage=stage)
                break
        else:
            result.update(completed=True,passed=True)
    except BaseException as error:
        result['error'] = repr(error)
    finally:
        result['wall_s'] = time.monotonic()-started
        result['admission'] = pin(HERE/'admission.json')
        write(HERE/'result.json',result)
        print(result,flush=True)
    return 0 if result['completed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
