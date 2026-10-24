"""Assess fixed native8 under an explicit research-only residual-limit extension.

NATIVE -B native8.py ADMISSION_PATH ADMISSION_SHA. No model or target access.
The original failed boundary is verified and preserved, never marked passed.
"""
from pathlib import Path
import importlib.util
import json
import math
import os
import sys

HERE = Path(__file__).resolve().parent
H = HERE.parent
ENGINE = H / 'native_screen_engine_v1/screen.py'
spec = importlib.util.spec_from_file_location('original_native_screen', ENGINE)
q = importlib.util.module_from_spec(spec)
spec.loader.exec_module(q)
STUDIES = ('hill_climb_e1_ynt_psdb_d5_v1', 's33_two_update_warm_proposal_v1')

def assess(a, admission_pin):
    assert a['schema'] == 'official-limit-native8-research-admission-v1' and a['root_admitted'] is True
    q.checked(q.pin(__file__)); policy = json.loads(q.checked(a['policy']))
    assert policy['root_admitted'] is True and policy['explicit_post_outcome'] is True
    assert policy['candidates'] == list(STUDIES) and policy['required_projected_residual_below'] == .4
    assert policy['original_boundary_failure_preserved'] is True and policy['progression_is_not_release_qualification'] is True
    study = Path(a['candidate']['path']).parent
    assert study.parent == H and study.name in STUDIES
    assert a['native_config']['path'] == str(study / 'native_screen.json')
    cp, c = q.configure(Path(a['native_config']['path']), a['native_config']['sha256'])
    assert cp == a['native_config'] and a['candidate'] == c['arms']['candidate']['source']
    assert c['engine']['sha256'] == '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
    assert a['result'] == str(study / 'native8_official_limit_extension.json')
    assert a['assessment_guard'] == str(study / 'native8_official_limit_extension_guard.json')
    assert c['parity_layers'] == [0] and c['arms']['parity_parent'] == c['arms']['base']
    stage = c['stages']['2']; prior_pin = q.pin(stage['result'])
    assert prior_pin == a['native2'] and q.pin(stage['assessment_guard']) == a['native2_assess_guard']
    prior = json.loads(q.checked(prior_pin))
    assert prior['passed_integrity'] is True and prior['admitted'] is True and all(prior['gates'].values())
    assert prior['n'] == 2 and prior['ranks'] == q.ROSTERS[2]
    assert prior['config'] == cp and prior['engine'] == c['engine']
    assert prior['candidate'] == c['arms']['candidate']['source'] and prior['comparator'] == c['arms']['base']['source']
    q.guard(stage['assessment_guard'], stage['assessment_limits'],
            [str(q.NATIVE), '-B', str(ENGINE), cp['path'], cp['sha256'], '2'])
    for record in prior['evidence']:
        q.checked(record)
    req = c['boundary_before8']
    assert a['original_boundary'] == q.pin(req['receipt'])
    assert a['original_boundary_assess_guard'] == q.pin(req['assessment_guard'])
    boundary = json.loads(q.checked(a['original_boundary']))
    assert boundary['guard'] == a['original_boundary_guard']; q.checked(a['original_boundary_guard'])
    assert boundary['passed'] is False and boundary['receipt_checks_passed'] is True
    assert boundary['gates'] == dict(setup=True, residual_projection=False, runtime_projection=True, cpu_agreement=True)
    assert boundary['candidate'] == prior['candidate'] and boundary['native2'] == prior_pin
    assert boundary['native_config'] == cp and boundary['assessor'] == req['assessor']
    residual = boundary['residual']['projected_s']
    assert a['original_boundary_stays_failed'] is True and a['all_other_original_boundary_gates_passed'] is True
    assert a['projected_residual_s'] == residual and a['projected_residual_cap_s'] == .4
    assert boundary['residual']['gate_s'] == .35 and math.isfinite(residual) and .35 < residual < .4
    assert boundary['runtime']['gate_s'] == 105 and boundary['runtime']['projected_worst_wall_s'] <= 105
    assert 0 < boundary['maximum_boundary_rss_bytes'] < 7500000000
    q.guard(req['assessment_guard'], req['assessment_limits'], req['assessment_command'])
    for record in boundary['evidence']:
        q.checked(record)
    block = c['arms']['candidate']['blocks']['6']
    assert a['fixed_additional6'] == q.ROSTERS[6] and a['native6_guard'] == block['guard']
    assert a['native6_limits'] == block['limits'] == dict(seconds=300, rss_bytes=7500000000)
    assert a['native6_command'] == [str(q.NATIVE)] + (['-B'] if block['no_bytecode'] else []) + [
        block['driver']['path'], a['candidate']['path'], ','.join(map(str, q.ROSTERS[6])), block['tag']]
    rows = {role: [] for role in c['arms']}; blocks = []
    for size in (2, 6):
        outputs = {}; evidence = {}
        for role, arm in c['arms'].items():
            rr, aa, records = q.panel(arm, size, role)
            rows[role].extend(rr); outputs[role] = aa; evidence[role] = records
        for rank in q.ROSTERS[size]:
            assert outputs['candidate'][rank][0].tobytes() == outputs['parity_parent'][rank][0].tobytes(), rank
        blocks.append(dict(size=size, arms=evidence))
    assert blocks[0] == prior['blocks'][0]
    result = q.metrics(rows['base'], rows['candidate'])
    limits = c['stages']['8']['gates']
    assert limits == a['pooled8_gates'] and a['no_score_gate_relaxation'] is True and a['no_timing_retry'] is True
    assert limits == dict(adjusted_pct_at_most=-.1, strict_adjusted=False, wins_min=6,
                         ci_upper_below=0, max_per_network_regression_pct=None, additional6_gain_required=True)
    additional = q.metrics([r for r in rows['base'] if r['rank'] in q.ROSTERS[6]],
                           [r for r in rows['candidate'] if r['rank'] in q.ROSTERS[6]])
    gates = dict(adjusted_gain=result['adjusted_pct'] <= -.1, wins=result['adjusted_wins'] >= 6,
                 ci_upper_negative=result['adjusted_ci95_pct'][1] < 0,
                 additional6_gain=additional['adjusted_pct'] < 0)
    if study.name == STUDIES[0]:
        assert a['additional6_gate'] == dict(adjusted_pct_at_most=-.5, strict=False)
        assert c['additional6_gate']['adjusted_pct_at_most'] == -.5
        assert c['additional6_gate']['required_before_any_broad_or_retention'] is True
        gates['additional6_gain_at_least_half_percent'] = additional['adjusted_pct'] <= -.5
    else:
        assert a['additional6_gate'] == dict(adjusted_pct_at_most=0, strict=True)
    for record in list(q.EVIDENCE.values()):
        q.checked(record)
    return dict(passed_integrity=True, admitted=all(gates.values()), **result, gates=gates,
        candidate=c['arms']['candidate']['source'], comparator=c['arms']['base']['source'],
        config=cp, engine=q.pin(__file__), reused_original_engine=c['engine'], extension_admission=admission_pin,
        extension_policy=a['policy'], prior2=prior_pin, blocks=blocks, evidence=list(q.EVIDENCE.values()),
        parity_parent=c['arms']['parity_parent']['source'], parity_layers=[0],
        boundary=a['original_boundary'], original_boundary_passed=False, boundary_failure_preserved=True,
        research_extension=dict(eligible=True, projected_residual_s=residual,
                                original_cutoff_s=.35, official_limit_s=.4, strict_below_official_limit=True),
        all16_predictions_finite=True, configured_parent_bytes_exact=True, all16_parent_bytes_exact=False,
        additional6=additional, bootstrap=dict(resamples=50000, seed=20260926), raw_error_veto=False,
        future_only=True, targets_read=False, model_executed=False, research_only=True,
        release_qualified=False, submitted=False)

def main():
    assert len(sys.argv) == 3
    assert Path(sys.executable).absolute() == q.NATIVE and q.np.__version__ == '2.4.6'
    assert all(os.environ.get(k) == '1' for k in q.THREADS)
    admission_pin = q.pin(Path(sys.argv[1]).resolve())
    assert admission_pin['sha256'] == sys.argv[2]
    a = json.loads(q.checked(admission_pin))
    target = Path(a['result'])
    study = Path(a['candidate']['path']).parent
    assert study.parent == H and study.name in STUDIES and target == study / 'native8_official_limit_extension.json'
    assert not target.exists()
    try:
        result = assess(a, admission_pin)
    except Exception as exc:
        q.once(target, dict(passed_integrity=False, admitted=False, n=8,
            extension_admission=admission_pin, engine=q.pin(__file__), reused_original_engine=q.pin(ENGINE),
            error=dict(type=type(exc).__name__, message=str(exc)), research_only=True,
            original_boundary_passed=False, release_qualified=False, submitted=False, model_executed=False))
        raise
    q.once(target, result)
    print(json.dumps({k: result[k] for k in ('passed_integrity', 'admitted', 'n', 'adjusted_pct', 'adjusted_wins', 'gates')}))

if __name__ == '__main__':
    main()
