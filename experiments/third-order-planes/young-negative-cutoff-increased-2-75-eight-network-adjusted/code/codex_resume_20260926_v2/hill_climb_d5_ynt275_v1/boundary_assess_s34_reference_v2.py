"""One explicitly admitted residual-reference refresh; stdlib and saved receipts only."""
from pathlib import Path
import copy
import hashlib
import importlib.util
import json
import math
import sys

D = Path(__file__).resolve().parent
OLD_SHA = 'f8a474c88e61e27179b54b9374d558b319ece544a62ef91429d9a2bbe339fa74'
BANK_SHA = '8b2cba1ddc5381b4ce70948107511bb888fe86fb3f8766d80abac4417f37d0e2'


def pin(p):
    p = Path(p).resolve(); b = p.read_bytes()
    return dict(path=str(p), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))


def assess(path, digest):
    cp = pin(path); assert cp['sha256'] == digest
    c = json.loads(path.read_text())
    assert c['schema'] == 's34-residual-reference-v1' and c['prospective_authorization'] is True
    assert c['engine'] == pin(__file__) and c['original_assessor']['sha256'] == OLD_SHA
    assert c['original_assessor'] == pin(c['original_assessor']['path'])
    spec = importlib.util.spec_from_file_location('first_boundary_assessor', c['original_assessor']['path'])
    old_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_module)
    _, q = old_module.helpers()
    q.checked(cp)
    for record in c['provenance']: q.checked(record)
    root_selection = json.loads(q.checked(c['root_selection']))
    assert c['root_selection']['sha256'] == '001ee5b05c07cb60cc1bcd9f2935d219f06839f68f3b4471c477fc05036b1949'
    assert root_selection['root_admitted'] is True and root_selection['explicit_post_outcome'] is True
    assert root_selection['new_external_evidence'] == c['s34_bank']
    assert root_selection['original_failure'] == c['original_failure']
    assert root_selection['parent_first_measurement'] == c['parent_boundary']
    assert root_selection['candidate'] == c['candidate']
    assert root_selection['multiplier'] == 4 and root_selection['strict_cap_s'] == .4
    assert root_selection['threshold_changed'] is False and root_selection['model_execution'] is False
    assert root_selection['timing_rerun'] is False and root_selection['original_failure_preserved'] is True
    original_config = json.loads(q.checked(c['original_config']))
    original = json.loads(q.checked(c['original_failure']))
    rebuilt = old_module.assess(Path(c['original_config']['path']), c['original_config']['sha256'])
    assert rebuilt == original  # Revalidate the first failure without writing or rerunning a model.
    assert original['receipt_checks_passed'] is True and original['passed'] is False
    assert original['gates'] == dict(setup=True, residual_projection=False, runtime_projection=True, cpu_agreement=True)
    q.guard(c['original_assessment_guard'], [str(q.LIGHT), '-B', c['original_assessor']['path'],
        c['original_config']['path'], c['original_config']['sha256']], 120)
    assert original['candidate'] == c['candidate'] and original['assessor'] == c['original_assessor']
    assert original_config['limits'] == old_module.LIMITS
    assert original['residual']['multiplier'] == 4 and original['residual']['gate_s'] == .4
    assert original['residual']['strict_upper_bound'] is True

    admission_pin = pin(D/'boundary_s34_reference_v2_admission.json')
    admission = json.loads(q.checked(admission_pin))
    assert admission['root_admitted'] is True and admission['source_changes'] is False
    assert admission['serving_qualified'] is False and admission['no_model_or_timing_rerun'] is True
    assert admission['config'] == cp and admission['assessor'] == c['engine']
    assert admission['candidate'] == c['candidate'] and admission['original_failure'] == c['original_failure']
    assert admission['native_config'] == pin(c['native_config_path'])
    nc = json.loads(q.checked(admission['native_config']))
    prior_nc = json.loads(q.checked(original['native_config']))
    assert nc['prospective_authorization'] is True
    for key in ('schema', 'engine', 'arms', 'parity_layers', 'future_only', 'raw_error_veto'):
        assert nc[key] == prior_nc[key]
    for stage in ('2', '8'):
        for key in ('gates', 'assessment_limits'):
            assert nc['stages'][stage][key] == prior_nc['stages'][stage][key]
    for record in nc['provenance']: q.checked(record)
    stage = nc['stages']['2']
    assert admission['native2'] == pin(stage['result']) and admission['native2_guard'] == pin(stage['assessment_guard'])
    result2 = json.loads(q.checked(admission['native2']))
    prior2 = json.loads(q.checked(original['native2']))
    assert result2['config'] == admission['native_config'] and result2['engine'] == nc['engine']
    excluded = {'config', 'evidence'}
    assert {k: v for k, v in result2.items() if k not in excluded} == {k: v for k, v in prior2.items() if k not in excluded}
    for record in result2['evidence']: q.checked(record)
    q.guard(admission['native2_guard'], [str(q.NATIVE), '-B', nc['engine']['path'],
        admission['native_config']['path'], admission['native_config']['sha256'], '2'], 120)
    req = nc['boundary_before8']
    assert req['assessor'] == c['engine'] and req['receipt'] == str(D/'boundary_s34_reference_v2_assessment.json')
    assert req['assessment_guard'] == str(D/'boundary_s34_reference_v2_assess_guard.json')
    assert req['assessment_limits'] == dict(seconds=120, rss_bytes=7500000000)
    assert req['assessment_command'] == [str(q.LIGHT), '-B', str(Path(__file__).resolve()), str(path), digest]

    bank = json.loads(q.checked(c['s34_bank']))
    assert c['s34_bank']['sha256'] == BANK_SHA and bank['submission_id'] == 333050
    assert bank['candidate'] == c['parent'] and bank['source_sha256'] == c['parent']['sha256']
    assert bank['candidate'] == nc['arms']['base']['source'] == nc['arms']['parity_parent']['source']
    assert c['parent']['sha256'] == '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
    q.checked(c['candidate']); q.checked(c['parent'])
    assert bank['authenticated_poll']['grading_status_cd'] == 'graded'
    assert bank['evaluation_receipt'] == c['s34_evaluation']
    evaluation = json.loads(q.checked(c['s34_evaluation']))
    assert evaluation['submission_id'] == 333050 and evaluation['grading_status_cd'] is None
    ev = evaluation['evaluation']
    assert ev['state'] == 'completed' and ev['error'] is None
    assert ev['submission_id'] == '333050' and ev['progress']['mlps_completed'] == 100 and ev['progress']['mlps_failed'] == 0
    telemetry = [row['telemetry'] for row in ev['results']['per_mlp']]; assert len(telemetry) == 100
    assert all(math.isfinite(t['residual_wall_time_s']) and t['residual_wall_time_s'] >= 0 for t in telemetry)
    maximum = max(t['residual_wall_time_s'] for t in telemetry)
    assert maximum == bank['telemetry']['max_residual_wall_s']
    parent_receipt = json.loads(q.checked(c['parent_boundary']))
    assert parent_receipt['receipt_checks_passed'] is True and parent_receipt['candidate'] == c['parent']
    assert parent_receipt['setup']['setup_wall_s'] <= 4.1
    tag = Path(c['parent']['path']).parent.name+'_boundary_feasibility_v1'
    parent_rows, parent_guard = q.boundary(parent_receipt['current_census'], parent_receipt['launcher'],
        parent_receipt['guard'], c['parent'], tag)
    candidate_rows, _ = q.boundary(original['current_census'], original['launcher'], original['guard'],
        c['candidate'], original_config['tag'])
    native_parent_rows = [json.loads(s) for s in q.checked(result2['blocks'][0]['arms']['base']['rows']).decode().splitlines()]
    assert [r['rank'] for r in parent_rows] == [r['rank'] for r in candidate_rows] == [35, 100]
    assert [(r['rank'], r['flops']) for r in parent_rows] == [(r['rank'], r['flops']) for r in native_parent_rows]
    comparisons = [dict(rank=x['rank'], candidate_residual_s=x['residual_s'], parent_residual_s=y['residual_s'],
        residual_delta_s=x['residual_s']-y['residual_s']) for x, y in zip(candidate_rows, parent_rows)]
    projected = maximum + 4*max(0., *(v['residual_delta_s'] for v in comparisons))
    assert [v['residual_delta_s'] for v in comparisons] == root_selection['paired_deltas_s']
    assert projected == root_selection['expected_projection_s']
    gates = dict(original['gates']); gates['residual_projection'] = projected < .4
    for record in original['evidence']: q.checked(record)
    result = copy.deepcopy(original)
    result.update(passed=all(gates.values()), gates=gates, stage='CURRENT_WINNER_RESIDUAL_REFERENCE_REFRESH',
        assessor=c['engine'], config=cp, admission=admission_pin, native2=admission['native2'], native_config=admission['native_config'],
        original_assessor=c['original_assessor'], original_failure=c['original_failure'], original_failure_passed=False,
        root_selection=c['root_selection'],
        original_failure_preserved=True, original_projected_residual_s=original['residual']['projected_s'],
        residual=dict(official_s34_max_s=maximum, multiplier=4, projected_s=projected, gate_s=.4, strict_upper_bound=True),
        residual_reference=dict(source=c['parent'], official_bank=c['s34_bank'], official_evaluation=c['s34_evaluation'],
            parent_boundary=c['parent_boundary'], parent_census=parent_receipt['current_census'],
            parent_guard=parent_receipt['guard'], candidate_census=original['current_census'], comparisons=comparisons),
        no_model_or_timing_rerun=True, selection_is_outcome_informed=True, wall_and_cpu_reference_unchanged=True,
        maximum_boundary_rss_bytes=max(original['maximum_boundary_rss_bytes'], parent_guard['max_rss_bytes']),
        evidence=list(q.EVIDENCE.values()))
    assert result['runtime'] == original['runtime'] and result['runtime_reference'] == original['runtime_reference']
    for record in list(q.EVIDENCE.values()): q.checked(record)
    return result


def main():
    assert len(sys.argv) == 3
    path = Path(sys.argv[1]).resolve(); target = D/'boundary_s34_reference_v2_assessment.json'
    assert not target.exists()
    result = assess(path, sys.argv[2])
    with target.open('x') as f: json.dump(result, f, indent=2, allow_nan=False); f.write('\n')
    print(json.dumps({k: result[k] for k in ('passed', 'receipt_checks_passed', 'gates', 'residual')}))


if __name__ == '__main__': main()
