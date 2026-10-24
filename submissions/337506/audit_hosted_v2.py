#!/usr/bin/env python3
"""Read-only independent hosted-receipt audit; no polling, imports, or bank mutation.

Checks all100 runtime rows. Public score rows cover only the public50 and are
rounded; exact page score is not reconstructed by silently treating them as exact.
Historical same-name timing comparisons are descriptive across hosted executions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess

HERE = Path(__file__).resolve().parent
H = HERE.parent
O = H.parents[3]
LA = O / 'lead/runs/fable_oct19_kc_v1/agent_astra_joint_v1/lead_audit_2323_v1'
REFERENCE_INPUTS = {
    337466: (LA / 'public_337466_v2/raw_20261021T045554Z.json',
             '5d5cb926fca7e645bcd92d58f81db6ee42848d3e1c889227f2f6bdb20bc39b02'),
    337473: (LA / 'public_337473_v2/raw_20261021T055632Z.json',
             'd915362785b3fb513e78b110b74a3f0e83e2a84fdce2c68d1e06c0c3038e5520'),
}
FROZEN_ARTIFACTS = {
    H / 'candidate_first128_paid_nocensus_release_v1.py':
        'f9381d0982f5d24e76d38884188ba7c998548e4a3f306825e7e80fb9fa0fc3b6',
    O / 'release/runs/astra_first128_paid_nocensus_f9381d09_v1/submission.tar.gz':
        'c63a799cc9402f3fb9aecf9b3fb5bf059db65362b9c8dc63803a83d7176f3d1d',
    H / 'root_release_v1/INDEPENDENT_RELEASE_ARTIFACT_AUDIT_v1.json':
        '2807390fbe6bd1fa8c340610f79eb1e1743fbd84462e77d7c77805d5d1d44a02',
}
RESIDUAL_CAP_S = .400
SETUP_CAP_S = 5.0
J2W_LAST_VERIFIED = 1.416013781559781e-9
BEST466_EXACT = 1.4167660830412778e-9
FLAGS = ('budget_exhausted', 'time_exhausted',
         'residual_wall_time_exhausted', 'combined_budget_exhausted')
METRICS = ('participant_predict_wall_s', 'flopscope_server_kernel_wall_s',
           'flopscope_server_overhead_wall_s', 'residual_wall_time_s',
           'participant_setup_wall_s', 'flops_used', 'effective_compute')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def numeric(x):
    return isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)


def stats(values):
    a = sorted(values)
    if not a:
        return {'n': 0}
    return {'n': len(a), 'mean': statistics.mean(a), 'median': statistics.median(a),
            'min': a[0], 'max': a[-1], 'p95_nearest_rank': a[math.ceil(.95*len(a))-1]}


def evaluate(raw, expected_id):
    e = raw['evaluation']
    results = e['results']
    rows = results['per_mlp']
    checks = []

    def check(name, passed, detail=None):
        checks.append({'name': name, 'passed': bool(passed), 'detail': detail})

    check('submission_identity', str(raw['submission_id']) == str(expected_id) == str(e['submission_id']))
    check('final_scoring_state', raw['submission_status'] == 'GRADED' and
          e['state'] == 'completed' and e['phase'] == 'scoring' and bool(e['completed_at']))
    check('no_global_error_or_truncation', e['error'] is None and e['truncation'] is None)
    config = e['config_snapshot']
    check('expected_geometry', (config['n_mlps'], config['depth'], config['width']) == (100,16,1024))
    check('full_progress', e['progress'] == {
        'total_mlps': 100, 'mlps_completed': 100, 'mlps_in_flight': 0,
        'mlps_failed': 0, 'pct_complete': 100.0}, e['progress'])
    check('all100_unique_indices', len(rows) == 100 and
          sorted(r['mlp_index'] for r in rows) == list(range(100)))
    check('all100_unique_named_cases', len({r['mlp_name'] for r in rows}) == 100 and
          all(isinstance(r['mlp_name'], str) and r['mlp_name'] for r in rows))
    smoke = e['smoke_test_result']
    check('smoke_success', smoke['passed'] is True and smoke['worker_failures'] == 0 and
          smoke['worker_passes'] == len(results['setup_executions']) >= 5 and smoke['error_code'] is None, smoke)
    check('all_workers_retired_idle', len(e['workers']) == 5 and
          all(w['state'] == 'idle' and w['current_mlp'] is None for w in e['workers']))
    failures, row_checks = [], []
    for r in rows:
        t = r['telemetry']
        bad = []
        if r['state'] != 'completed' or r['error_code'] not in (None, 'SUCCESS'):
            bad.append('state_or_error')
        for k in FLAGS:
            if t.get(k) is not False:
                bad.append(k)
        for k in METRICS:
            if k == 'participant_setup_wall_s' and t.get(k) is None:
                continue  # Warm predictions do not repeat an initial setup time.
            if not numeric(t.get(k)) or t[k] < 0:
                bad.append('invalid_' + k)
        if not bad:
            if t['residual_wall_time_s'] > RESIDUAL_CAP_S:
                bad.append('residual_above_cap')
            if t['participant_setup_wall_s'] is not None and t['participant_setup_wall_s'] > SETUP_CAP_S:
                bad.append('setup_above_cap')
            if t['flops_used'] != t['effective_compute']:
                bad.append('effective_compute_differs_from_flops')
            if t['flops_used'] > config['flop_budget']:
                bad.append('flops_above_budget')
            if not float(t['flops_used']).is_integer():
                bad.append('nonintegral_flop_bill')
            accounted = sum(t[k] for k in ('flopscope_server_kernel_wall_s',
                            'flopscope_server_overhead_wall_s', 'residual_wall_time_s'))
            if not math.isclose(accounted, t['participant_predict_wall_s'], rel_tol=0, abs_tol=1e-8):
                bad.append('predict_time_partition')
            sr = t['flopscope_self_report']
            if not math.isclose(sr['residual_s'], t['residual_wall_time_s'], rel_tol=0, abs_tol=1e-10):
                bad.append('self_report_residual')
            if not math.isclose(sr['backend_s'], t['flopscope_server_kernel_wall_s'], rel_tol=0, abs_tol=1e-10):
                bad.append('self_report_backend')
        rc = {'index': r['mlp_index'], 'name': r['mlp_name'], 'state': r['state'],
              'error_code': r['error_code'], 'bad_checks': bad, 'telemetry': t,
              'headroom_ms': 1000*(RESIDUAL_CAP_S-t['residual_wall_time_s'])}
        row_checks.append(rc)
        if bad:
            failures.append(rc)
    check('all_runtime_rows_clean', not failures, [{'index': r['index'], 'bad_checks': r['bad_checks']} for r in failures])
    setups = results['setup_executions']
    keys = [(r['worker_id'], r['generation']) for r in setups]
    check('setup_worker_generation_valid', bool(setups) and all(w in {f'w-{i}' for i in range(5)} and isinstance(g, int) and g >= 0 for w,g in keys))
    check('setups_all_success_within_5s', all(r['error_code'] == 'SUCCESS' and
          numeric(r['wall_s']) and 0 <= r['wall_s'] <= SETUP_CAP_S and
          isinstance(r['flops'], int) and r['flops'] >= 0 for r in setups), setups)
    check('all5_initial_setups_present', {(w, g) for w, g in keys if g == 0} ==
          {(f'w-{i}', 0) for i in range(5)})
    pub = results['public_scores']
    check('all50_public_scores_unique', len(pub) == 50 and
          sorted(r['mlp_index'] for r in pub) == list(range(50)))
    check('public_scores_finite_nonnegative', all(numeric(v) and v >= 0
          for r in pub for v in r['scores'].values()))
    aggregate = results['aggregates']['public']
    check('public_aggregate_no_failure', aggregate['n_failed_mlps'] == 0 and
          all(v == 0 for v in aggregate['failure_breakdown'].values()), aggregate['failure_breakdown'])
    score = raw['submission_score']
    check('positive_finite_page_score', numeric(score) and score > 0)
    displayed = [r['scores']['adjusted_final_layer_score'] for r in pub]
    # The hosted public display provides four significant digits. Treat its
    # displayed values as rounding intervals, never as exact per-case scores.
    radii = [.5*10**(math.floor(math.log10(v))-3) if v > 0 else 0 for v in displayed]
    interval = [statistics.mean(v-d for v,d in zip(displayed,radii)),
                statistics.mean(v+d for v,d in zip(displayed,radii))]
    check('page_score_consistent_with_rounded_public50', interval[0] <= score <= interval[1], interval)
    all_ok = all(c['passed'] for c in checks)
    return {'verified_full100_valid': all_ok, 'checks': checks, 'failed_checks':
            [c['name'] for c in checks if not c['passed']], 'failures': failures,
            'all100_rows': sorted(row_checks, key=lambda r:r['index']),
            'runtime_summary': {k:stats([r['telemetry'][k] for r in rows if numeric(r['telemetry'][k])]) for k in METRICS},
            'runtime_missing_counts': {k:sum(r['telemetry'][k] is None for r in rows) for k in METRICS},
            'setups': setups, 'setup_duplicate_worker_generation_pairs': len(keys)-len(set(keys)),
            'setup_wall_summary': stats([r['wall_s'] for r in setups]),
            'headroom_within_ms_counts': {str(m):sum(r['headroom_ms'] <= m for r in row_checks)
                                       for m in (0,5,10,15,20,30)},
            'runtime_environment': e['runtime_environment'],
            'submission_score': score, 'public50_display_rounding_interval': interval,
            'score_vs_best466_pct': 100*(score/BEST466_EXACT-1),
            'score_vs_last_verified_J2W_pct': 100*(score/J2W_LAST_VERIFIED-1),
            'beats_best466': all_ok and score < BEST466_EXACT,
            'beats_last_verified_J2W': all_ok and score < J2W_LAST_VERIFIED}


def comparison(raw, old):
    ar = {r['mlp_index']:r for r in raw['evaluation']['results']['per_mlp']}
    br = {r['mlp_index']:r for r in old['evaluation']['results']['per_mlp']}
    assert set(ar) == set(br) == set(range(100))
    rows = []
    for i in range(100):
        a,b = ar[i],br[i]
        assert a['mlp_name'] == b['mlp_name']
        at,bt = a['telemetry'],b['telemetry']
        rows.append({'index':i, 'name':a['mlp_name'], 'candidate_state':a['state'],
                     'reference_state':b['state'], 'candidate_residual_s':at['residual_wall_time_s'],
                     'reference_residual_s':bt['residual_wall_time_s'],
                     'residual_delta_ms':1000*(at['residual_wall_time_s']-bt['residual_wall_time_s']),
                     'residual_delta_pct':100*(at['residual_wall_time_s']/bt['residual_wall_time_s']-1),
                     'flop_delta':at['flops_used']-bt['flops_used'],
                     'flop_delta_pct':100*(at['flops_used']/bt['flops_used']-1),
                     'candidate_setup_generation':at['setup_generation'],
                     'reference_setup_generation':bt['setup_generation']})
    return {'reference_submission_id':old['submission_id'],
            'interpretation':'DESCRIPTIVE_SAME_NAMED_CASES_ACROSS_HOSTED_EXECUTIONS_NOT_CAUSAL_OR_SAME_HOST_PROOF',
            'all100_names_matched':True, 'all100_rows':rows,
            'residual_delta_ms':stats([r['residual_delta_ms'] for r in rows]),
            'residual_delta_pct':stats([r['residual_delta_pct'] for r in rows]),
            'flop_delta_pct':stats([r['flop_delta_pct'] for r in rows]),
            'n_residual_lower':sum(r['residual_delta_ms'] < 0 for r in rows),
            'SDK_metadata_equal':raw['evaluation']['runtime_environment'] == old['evaluation']['runtime_environment']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'immutable output already exists'
    raw = json.loads(args.raw.read_text())
    assert raw['submission_id'] == 337506, 'wrong submission'
    assert raw['submission_status'] == 'GRADED', 'not final yet'
    pins = []
    for p,digest in {**FROZEN_ARTIFACTS, **{p:s for p,s in REFERENCE_INPUTS.values()}}.items():
        actual = sha(p)
        assert actual == digest, (p, actual)
        pins.append({'path':str(p),'sha256':actual})
    pins.extend([{'path':str(args.raw.resolve()),'sha256':sha(args.raw)},
                 {'path':str(Path(__file__).resolve()),'sha256':sha(__file__)}])
    report = evaluate(raw, 337506)
    report.update({'utc':subprocess.check_output(['date','-u','+%Y-%m-%dT%H:%M:%SZ'],text=True).strip(),
                   'status':'INDEPENDENT_FULL_HOSTED_RECEIPT_AUDIT', 'input_pins':pins,
                   'submission_id':337506, 'source_sha256':FROZEN_ARTIFACTS[H/'candidate_first128_paid_nocensus_release_v1.py'],
                   'reference_mutated':False, 'submission_action':False,
                   'comparisons':{str(i):comparison(raw,json.loads(p.read_text())) for i,(p,_) in REFERENCE_INPUTS.items()},
                   'limits':['Public scores expose50 rounded case scores; all100 runtime/bill rows are checked.',
                             'Historical timing comparisons do not prove causal changes or identical hosts.',
                             'J2W comparator is last independently verified1.416013781559781e-9; leaderboard refresh is separate.',
                             'Frozen local artifact identity is checked; public raw telemetry does not expose its archive hash.',
                             'Passing all100 hosted rows is evidence for this execution, not a future timing guarantee.']})
    args.output.write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps({'output':str(args.output),'sha256':sha(args.output),
                      'verified_full100_valid':report['verified_full100_valid'],
                      'failed_checks':report['failed_checks'],
                      'score':report['submission_score'],
                      'residual':report['runtime_summary']['residual_wall_time_s'],
                      'setup':report['setup_wall_summary']}))


if __name__ == '__main__':
    main()
