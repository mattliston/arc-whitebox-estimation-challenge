"""Reusable root-only VALID40 assessment; imports the frozen native panel engine.

NATIVE -B assess.py CONFIG CONFIG_SHA BROAD_ADMISSION_SHA
Only saved predictions/metadata are read. No estimator or truth evaluation.
"""
from pathlib import Path
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys

D = Path(__file__).resolve().parent
H2 = D.parent
SCREEN = H2/'native_screen_engine_v1/screen.py'
SCREEN_SHA = '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
S32_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
OFFICIAL_SHA = '6c4a0e3f3c84ab9756c391c0929bd4d51057a61bba4167b9d21c0a317e612fe6'
ROSTERS = {
    'b1': [37,39,42,48,59,63,66,67,71,73],
    'b2': [75,79,93,101,122,130,132,135,139,145],
    'b3': [146,149,150,151,155,159,163,169,173,175],
    'b4': [181,192,198,199,204,219,224,227,233,234],
}
GATES = dict(adjusted_pct_at_most=-1, wins_min=28, ci_upper_below=0,
             max_per_network_regression_pct=10)
LIMITS = dict(seconds=420, rss_bytes=7500000000)


def initialize():
    assert hashlib.sha256(SCREEN.read_bytes()).hexdigest() == SCREEN_SHA
    spec = importlib.util.spec_from_file_location('valid40_frozen_native', SCREEN)
    q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
    assert Path(sys.executable).absolute() == q.NATIVE and q.np.__version__ == '2.4.6'
    assert all(os.environ.get(k) == '1' for k in q.THREADS)
    q.ROSTERS.update(ROSTERS)  # panel() accepts these fixed process-order keys.
    return q


def load(q, record):
    return json.loads(q.checked(record))


def configure(q, path, digest):
    cp = q.pin(path); assert cp['sha256'] == digest
    c = load(q, cp)
    assert c['schema'] == 'valid40-screen-v1' and c['prospective_authorization'] is True
    assert c['future_only'] is True and c['raw_error_veto'] is False
    assert c['engine'] == q.pin(__file__) and c['runner'] == q.pin(D/'run.py')
    assert c['native_engine'] == q.pin(SCREEN) and c['score_gates'] == GATES
    assert c['assessment_limits'] == dict(seconds=120, rss_bytes=7500000000)
    assert c['projected_wall_s_max'] == 105
    for item in c['provenance'] + [c['engine'], c['runner'], c['native_engine'],
                                  c['guard_source'], c['policy'], c['official'], c['controls']]:
        q.checked(item)
    assert c['official']['sha256'] == OFFICIAL_SHA
    controls = load(q, c['controls'])
    base = controls['arm']; child = c['candidate']
    assert base['source']['sha256'] == S32_SHA
    for item in controls['provenance']: q.checked(item)
    for role, arm in (('base',base),('candidate',child)):
        q.checked(arm['source'])
        assert set(arm['blocks']) == set(ROSTERS)
        assert arm['setup']['seconds_max'] == 4.1
        assert type(arm['ubc_retired_required']) is bool
        if role == 'candidate': assert arm['ubc_retired_required'] is True
        for key, ranks in ROSTERS.items():
            b = arm['blocks'][key]
            assert b['ranks'] == ranks and Path(b['tag']).name == b['tag']
            assert type(b['no_bytecode']) is bool
            assert b['limits'] == (LIMITS if role == 'candidate' else dict(seconds=360,rss_bytes=7500000000))
            q.checked(b['driver'])
    c['base'] = base
    return cp, c


def prerequisites(q, cp, c, admission_sha):
    study = Path(cp['path']).parent
    ap = q.pin(study/'broad_admission.json'); assert ap['sha256'] == admission_sha
    a = load(q, ap)
    assert a['passed'] is True and a['config'] == cp and a['candidate'] == c['candidate']['source']
    nc_pin, nc = q.configure(Path(c['native_config']['path']), c['native_config']['sha256'])
    assert nc_pin == c['native_config'] and nc['engine'] == c['native_engine']
    assert nc['arms']['candidate']['source'] == c['candidate']['source']
    assert nc['arms']['base']['source'] == c['base']['source']
    assert nc['arms']['candidate']['setup'] == c['candidate']['setup']
    stage = nc['stages']['8']
    assert a['training'] == q.pin(stage['result'])
    training = load(q, a['training'])
    assert training['passed_integrity'] is True and training['admitted'] is True and training['n'] == 8
    assert training['candidate'] == c['candidate']['source'] and training['comparator'] == c['base']['source']
    assert training['config'] == nc_pin and training['engine'] == c['native_engine']
    assert training['gates'] and all(training['gates'].values())
    assert training['all16_predictions_finite'] is True
    q.guard(stage['assessment_guard'], stage['assessment_limits'],
            [str(q.NATIVE),'-B',c['native_engine']['path'],nc_pin['path'],nc_pin['sha256'],'8'])
    for item in training['evidence']: q.checked(item)
    # Re-read actual eight-network CPU rows bound by the admitted native receipt.
    cpu = {'candidate': [], 'base': []}
    assert [b['size'] for b in training['blocks']] == [2,6]
    for block in training['blocks']:
        for role in cpu:
            saved = block['arms'][role]
            expected = nc['arms'][role]['blocks'][str(block['size'])]
            assert saved['rows']['path'] == str(H2/expected['tag']/'rows.jsonl')
            rows = [json.loads(s) for s in q.checked(saved['rows']).decode().splitlines()]
            assert [r['rank'] for r in rows] == q.ROSTERS[block['size']]
            assert all(math.isfinite(r['cpu_s']) and r['cpu_s'] > 0 for r in rows)
            cpu[role].extend(r['cpu_s'] for r in rows)
    native8_ratio = statistics.mean(cpu['candidate'])/statistics.mean(cpu['base'])
    assert math.isclose(100*(native8_ratio-1),training['mean_cpu_pct'],rel_tol=1e-12,abs_tol=1e-10)
    req = nc['boundary_before8']
    assert a['boundary'] == q.pin(req['receipt']) == training['boundary']
    boundary = load(q,a['boundary'])
    assert boundary['passed'] is True and boundary['receipt_checks_passed'] is True
    assert boundary['candidate'] == c['candidate']['source'] and boundary['native_config'] == nc_pin
    assert boundary['assessor'] == req['assessor'] and boundary['native2'] == training['prior2']
    assert boundary['gates'] and all(boundary['gates'].values())
    q.checked(req['assessor']); q.checked(boundary['config'])
    q.guard(req['assessment_guard'],req['assessment_limits'],req['assessment_command'])
    for item in boundary['evidence']: q.checked(item)
    rt = boundary['runtime']; residual = boundary['residual']
    assert residual['gate_s'] == .35 and residual['projected_s'] <= .35
    assert rt['gate_s'] == 105 and rt['projected_worst_wall_s'] <= 105
    assert boundary['setup']['setup_wall_s'] <= 4.1
    growth = max(0.,native8_ratio-1,rt['native2_cpu_ratio']-1,rt['boundary_mean_cpu_ratio']-1)
    requests = rt['nonnegative_mean_request_delta']
    assert all(math.isfinite(x) and x >= 0 for x in (growth,requests))
    official = load(q,c['official'])
    assert official['submission_id'] == 332738
    ev = official['evaluation']
    assert ev['progress']['mlps_completed'] == 100 and ev['progress']['mlps_failed'] == 0
    telemetry = [r['telemetry'] for r in ev['results']['per_mlp']]; assert len(telemetry) == 100
    assert all(math.isfinite(t[k]) and t[k] >= 0 for t in telemetry
               for k in ('participant_predict_wall_s','flopscope_server_kernel_wall_s'))
    grid = [dict(native_fraction=f,request_cost_s=p,projected_worst_wall_s=max(
        t['participant_predict_wall_s']+f*growth*t['flopscope_server_kernel_wall_s']+requests*p
        for t in telemetry)) for f in (.60,.78,.95) for p in (.00030,.00041,.00050)]
    wall = max(r['projected_worst_wall_s'] for r in grid)
    runtime = dict(native8_cpu_ratio=native8_ratio,native2_cpu_ratio=rt['native2_cpu_ratio'],
        boundary_mean_cpu_ratio=rt['boundary_mean_cpu_ratio'],used_nonnegative_cpu_change=growth,
        nonnegative_mean_request_delta=requests,projection_grid=grid,projected_worst_wall_s=wall,
        gate_s=105,passed=wall<=105)
    assert runtime['passed'], runtime
    return ap, a, runtime


def assess(q, cp, c, admission_sha):
    ap, a, runtime = prerequisites(q,cp,c,admission_sha)
    study = Path(cp['path']).parent
    run_pin = q.pin(study/'broad_run.json'); run = load(q,run_pin)
    assert run['completed'] is True and run['all_guards_passed'] is True
    assert run['config'] == cp and run['admission'] == ap and run['candidate'] == c['candidate']['source']
    assert run['runner'] == c['runner'] and len(run['jobs']) == 4
    attempt = load(q,run['attempt'])
    assert attempt['config'] == cp and attempt['admission'] == ap and attempt['runner'] == c['runner']
    rows = {'base':[],'candidate':[]}; blocks=[]; guards=[]
    for key, job_pin in zip(ROSTERS,run['jobs']):
        job = load(q,job_pin); assert job['passed'] is True and job['block'] == key
        assert job['candidate'] == c['candidate']['source'] and job['config'] == cp and job['admission'] == ap
        block = {}
        for role,arm in (('base',c['base']),('candidate',c['candidate'])):
            rr, arrays, records = q.panel(arm,key,role)
            del arrays
            rows[role].extend(rr); block[role] = records
        assert job['guard'] == block['candidate']['guard']; q.checked(job['log'])
        guards.append(job['guard']); blocks.append(dict(block=key,arms=block))
    result = q.metrics(rows['base'],rows['candidate'])
    gates = dict(adjusted_gain=result['adjusted_pct']<=-1, wins=result['adjusted_wins']>=28,
        ci_upper_negative=result['adjusted_ci95_pct'][1]<0,
        max_regression=result['max_per_network_adjusted_regression_pct']<=10,
        native8_runtime_projection=runtime['passed'])
    for record in list(q.EVIDENCE.values()): q.checked(record)
    return dict(passed=all(gates.values()),receipt_checks_passed=True,validation_score_passed=all(gates.values()),
        stage='VALID40',candidate=c['candidate']['source'],parent=c['base']['source'],baseline_submission=332738,
        config=cp,admission=ap,assessor=c['engine'],runner=c['runner'],native_engine=c['native_engine'],
        training=a['training'],boundary=a['boundary'],runtime_recheck=runtime,run=run_pin,
        metrics=result,gates=gates,blocks=blocks,validation_guards=guards,evidence=list(q.EVIDENCE.values()),
        policy=c['policy'],bootstrap=dict(resamples=50000,seed=20260926),all16_predictions_finite=True,
        reused_validation_not_pristine=True,raw_error_veto=False,targets_read=False,model_executed=False,
        worker_qualified=False,release_qualified=False,submitted=False)


def main():
    assert len(sys.argv) == 4
    q = initialize(); cp,c = configure(q,Path(sys.argv[1]).resolve(),sys.argv[2])
    target = Path(cp['path']).parent/'broad_assessment.json'; assert not target.exists()
    try: result = assess(q,cp,c,sys.argv[3])
    except Exception as exc:
        q.once(target,dict(passed=False,receipt_checks_passed=False,validation_score_passed=False,
            config=cp,candidate=c['candidate']['source'],assessor=c['engine'],
            error=dict(type=type(exc).__name__,message=str(exc)),release_qualified=False,submitted=False))
        raise
    q.once(target,result)
    print(json.dumps({k:result[k] for k in ('passed','receipt_checks_passed','gates','metrics')}))


if __name__ == '__main__': main()
