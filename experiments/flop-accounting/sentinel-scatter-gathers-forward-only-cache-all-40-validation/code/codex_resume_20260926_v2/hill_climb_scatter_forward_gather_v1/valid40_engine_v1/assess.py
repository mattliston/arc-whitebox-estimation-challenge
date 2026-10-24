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
H2 = D.parent.parent
SCREEN = D.parent/'screen.py'
CONTROL_SCREEN = H2/'native_screen_engine_v1/screen.py'
CONTROL_SCREEN_SHA = '1fc332a52c18d2d348256364cc9fc0ca3e4a1e608e4bee6315bb3008429bf287'
SCREEN_SHA = 'ffbd26aa01839d6017fb3b2482d1667901864b5da4f02d37d7b99974626d6b8a'
S32_SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
S34_SHA = '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
RUNTIME_CONFIG_SHA = '5c4bf41c38012ee1a4e4af0d706d33f6f8902f52f4a32c899c66a3506aaf8c8d'
BOUNDARY_SHA = '0e1c5c61be3724ecb241d2b6d581e151c010e3bccf287fc90d1846fc483c8f11'
DRIVER_SHA = '8acc3e338ca5f991f778edaf0376bc8265ff69b330137c5bba4d079766ca8af5'
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
    assert hashlib.sha256(CONTROL_SCREEN.read_bytes()).hexdigest() == CONTROL_SCREEN_SHA
    spec = importlib.util.spec_from_file_location('valid40_original_control_native', CONTROL_SCREEN)
    control = importlib.util.module_from_spec(spec); spec.loader.exec_module(control)
    q.control_engine = control
    return q


def load(q, record):
    return json.loads(q.checked(record))


def score_bank(q, reference):
    assert reference['submission_id'] == 333050 and reference['source']['sha256'] == S34_SHA
    bank = load(q, reference['bank_receipt'])
    assert bank['submission_id'] == reference['submission_id']
    assert bank['candidate'] == reference['source'] and bank['source_sha256'] == S34_SHA
    assert bank['telemetry']['completed'] == 100 and bank['telemetry']['failed'] == 0
    poll = bank['authenticated_poll']
    assert poll['grading_status_cd'] == 'graded'
    assert all(math.isfinite(poll[k]) and poll[k] > 0 for k in ('score','score_secondary'))
    public = load(q, bank['evaluation_receipt']); ev = public['evaluation']
    assert public['submission_id'] == reference['submission_id'] == int(ev['submission_id'])
    assert ev['state'] == 'completed' and ev['error'] is None and ev['truncation'] is None
    assert ev['progress']['total_mlps'] == ev['progress']['mlps_completed'] == 100
    assert ev['progress']['mlps_failed'] == ev['progress']['mlps_in_flight'] == 0
    rows = ev['results']['per_mlp']; assert sorted(r['mlp_index'] for r in rows) == list(range(100))
    for row in rows:
        assert row['state'] == 'completed' and row['error_code'] is None
        assert all(row['telemetry'][k] is False for k in (
            'budget_exhausted','time_exhausted','residual_wall_time_exhausted','combined_budget_exhausted'))
    assert ev['results']['setup_executions'] and all(
        r['error_code'] == 'SUCCESS' for r in ev['results']['setup_executions'])
    for key in ('candidate','authenticated_snapshot','public_snapshot','submission_receipt',
                'archive_receipt','qualification','local_validation_reference'):
        q.checked(bank[key])
    return bank


def native_cpu_rows(q, arm, size, records):
    block = arm['blocks'][str(size)]; ranks = q.ROSTERS[size]
    assert block['ranks'] == ranks
    for key, path in (('manifest',H2/block['tag']/'manifest.json'),
                      ('rows',H2/block['tag']/'rows.jsonl'),('guard',Path(block['guard']))):
        assert records[key] == q.pin(path); q.checked(records[key])
        if 'artifacts' in block: assert records[key] == block['artifacts'][key]
    q.checked(arm['source']); q.checked(block['driver'])
    manifest = load(q, records['manifest'])
    assert manifest['candidate'] == arm['source']['path'] and manifest['sha256'] == arm['source']['sha256']
    assert manifest['ranks'] == ranks and manifest['runtime'] == str(q.NATIVE) and manifest['numpy'] == '2.4.6'
    assert manifest['thread_env'] == dict.fromkeys(q.THREADS,'1')
    assert block['limits'] == dict(seconds=150 if size == 2 else 300,rss_bytes=7500000000)
    command = [str(q.NATIVE)] + (['-B'] if block['no_bytecode'] else []) + [
        block['driver']['path'],arm['source']['path'],','.join(map(str,ranks)),block['tag']]
    q.guard(records['guard']['path'],block['limits'],command)
    assert 0 <= manifest['setup']['setup_s'] <= 4.1
    for key,count in arm['setup']['counts'].items():
        assert [r[1] for r in manifest['setup']['pb_log'] if r[0] == key] == [count]
    rows = [json.loads(line) for line in q.checked(records['rows']).decode().splitlines()]
    assert [r['rank'] for r in rows] == ranks
    assert all(math.isfinite(r['cpu_s']) and r['cpu_s'] > 0 for r in rows)
    return rows


def configure(q, path, digest):
    cp = q.pin(path); assert cp['sha256'] == digest
    c = load(q, cp)
    assert c['schema'] == 'valid40-screen-v2' and c['prospective_authorization'] is True
    assert c['future_only'] is True and c['raw_error_veto'] is False
    assert c['engine'] == q.pin(__file__) and c['runner'] == q.pin(D/'run.py')
    assert c['native_engine'] == q.pin(SCREEN) and c['control_native_engine'] == q.pin(CONTROL_SCREEN) and c['score_gates'] == GATES
    assert c['assessment_limits'] == dict(seconds=120, rss_bytes=7500000000)
    assert c['projected_wall_s_max'] == 105
    for item in c['provenance'] + [c['engine'], c['runner'], c['native_engine'],
                                  c['guard_source'], c['policy'], c['official'], c['controls']]:
        q.checked(item)
    assert c['official']['sha256'] == OFFICIAL_SHA
    c['score_bank'] = score_bank(q,c['score_reference'])
    reference = c['runtime_reference']
    assert reference['official_submission_id'] == 332738 and reference['native_arm'] == 'base'
    assert reference['source']['sha256'] == S32_SHA and reference['official'] == c['official']
    assert reference['native_config']['sha256'] == RUNTIME_CONFIG_SHA
    q.checked(reference['source']); q.checked(reference['native_config'])
    controls = load(q, c['controls'])
    base = controls['arm']; child = c['candidate']
    assert base['source'] == c['score_reference']['source']
    assert controls['score_reference'] == c['score_reference']
    assert controls['assessment'] == c['score_bank']['local_validation_reference']
    previous = load(q,controls['assessment'])
    assert previous['passed'] is True and previous['receipt_checks_passed'] is True
    assert previous['validation_score_passed'] is True and previous['candidate'] == base['source']
    assert [x['block'] for x in previous['blocks']] == list(ROSTERS)
    for block in previous['blocks']:
        assert base['blocks'][block['block']]['artifacts'] == block['arms']['candidate']
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
            assert b['limits'] == LIMITS and b['no_bytecode'] is True
            assert b['driver']['sha256'] == DRIVER_SHA
            q.checked(b['driver'])
    assert base['setup']['counts'] == dict(bound=682,lvfuse=649,psdb_fused=9)
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
    assert nc['arms']['base']['source'] == c['incremental_reference']
    assert nc['arms']['candidate']['setup'] == c['candidate']['setup']
    stage = nc['stages']['8']
    assert a['training'] == q.pin(stage['result'])
    training = load(q, a['training'])
    assert training['passed_integrity'] is True and training['admitted'] is True and training['n'] == 8
    assert training['candidate'] == c['candidate']['source'] and training['comparator'] == c['incremental_reference']
    assert training['config'] == nc_pin and training['engine'] == c['native_engine']
    assert training['gates'] and all(training['gates'].values())
    assert training['all16_predictions_finite'] is True and training['all16_parent_bytes_exact'] is True
    q.guard(stage['assessment_guard'], stage['assessment_limits'],
            [str(q.NATIVE),'-B',c['native_engine']['path'],nc_pin['path'],nc_pin['sha256'],'8'])
    for item in training['evidence']: q.checked(item)
    # Score-comparator timing stays descriptive; runtime uses direct S32 rows below.
    cpu = {'candidate': [], 'base': []}; candidate_records = {}
    assert [b['size'] for b in training['blocks']] == [2,6]
    for block in training['blocks']:
        size = block['size']; candidate_records[str(size)] = block['arms']['candidate']
        for role in cpu:
            rows = native_cpu_rows(q,nc['arms'][role],size,block['arms'][role])
            cpu[role].extend(r['cpu_s'] for r in rows)
    # Direct same-eight comparison with saved S34 controls, distinct from incremental parent.
    sp, sc = q.control_engine.configure(Path(c['native_s34_controls']['path']), c['native_s34_controls']['sha256'])
    assert sp == c['native_s34_controls'] and sc['arms']['base']['source'] == c['score_reference']['source']
    s34_rows = []; s34_inputs = []
    for size in (2, 6):
        sa = sc['arms']['base']
        records = {k: sa['blocks'][str(size)]['artifacts'][k] for k in ('manifest', 'rows', 'guard')}
        rows = native_cpu_rows(q, sa, size, records)
        for row in rows:
            assert math.isfinite(row['adjusted']) and row['adjusted'] > 0
            assert row['util'] == row['flops']/2**41
            assert math.isclose(row['adjusted'], row['mse']*max(.1,row['util']), rel_tol=1e-14, abs_tol=0)
            assert row['ubc']['retired'] and row['ubc']['retained_cache_bytes'] == row['ubc']['live_cache_bytes'] == 0
            assert 0 <= row['ubc']['max_resident_bytes'] <= 512000000
        s34_rows.extend(rows); s34_inputs.append(records['rows'])
    for record in q.control_engine.EVIDENCE.values(): q.checked(record)
    assert len(s34_rows) == 8 and sorted(row['rank'] for row in s34_rows) == sorted(training['ranks'])
    total = 100*(training['candidate_mean_adjusted']/(math.fsum(row['adjusted'] for row in s34_rows)/8)-1)
    assert math.isfinite(total) and total <= -.8
    assert a['headroom_required_gain_pct'] == .8 and a['headroom_reference'] == sp and a['headroom_rows'] == s34_inputs
    assert math.isclose(a['total_vs_S34_native8_pct'], total, rel_tol=0, abs_tol=1e-12)
    component_ratio = math.fsum(cpu['candidate'])/math.fsum(cpu['base'])
    assert math.isclose(100*(component_ratio-1),training['mean_cpu_pct'],rel_tol=1e-12,abs_tol=1e-10)
    req = nc['boundary_before8']
    assert a['boundary'] == q.pin(req['receipt']) == training['boundary']
    boundary = load(q,a['boundary'])
    assert boundary['passed'] is True and boundary['receipt_checks_passed'] is True
    assert boundary['candidate'] == c['candidate']['source'] and boundary['native_config'] == nc_pin
    assert boundary['assessor'] == req['assessor'] and boundary['native2'] == training['prior2']
    assert boundary['gates'] and all(boundary['gates'].values())
    assert req['assessor']['sha256'] == BOUNDARY_SHA
    q.checked(req['assessor']); q.checked(boundary['config'])
    q.guard(req['assessment_guard'],req['assessment_limits'],req['assessment_command'])
    for item in boundary['evidence']: q.checked(item)
    assert boundary['native8_runtime_recheck_contract'] == 'direct_candidate_over_pinned_S32_v2'
    reference = c['runtime_reference']; br = boundary['runtime_reference']
    for key in ('official_submission_id','source','native_config','native_arm'):
        assert br[key] == reference[key]
    _, rc = q.control_engine.configure(Path(reference['native_config']['path']),reference['native_config']['sha256'])
    for record in q.control_engine.EVIDENCE.values(): q.checked(record)
    arm = rc['arms'][reference['native_arm']]; assert arm['source'] == reference['source']
    s32_cpu = {}; runtime_inputs = {}
    for size in (2,6):
        records = {k:arm['blocks'][str(size)]['artifacts'][k] for k in ('manifest','rows','guard')}
        assert br['native_controls'][str(size)] == records
        s32_cpu[size] = [r['cpu_s'] for r in native_cpu_rows(q,arm,size,records)]
        runtime_inputs[str(size)] = dict(candidate=candidate_records[str(size)]['rows'],reference=records['rows'])
    native8_ratio = math.fsum(cpu['candidate'])/math.fsum(s32_cpu[2]+s32_cpu[6])
    native2_ratio = math.fsum(cpu['candidate'][:2])/math.fsum(s32_cpu[2])
    rt = boundary['runtime']; residual = boundary['residual']
    assert rt['native2_cpu_reference'] == reference['source']
    assert rt['native2_ratio_inputs']['candidate_rows'] == runtime_inputs['2']['candidate']
    assert rt['native2_ratio_inputs']['reference_rows'] == runtime_inputs['2']['reference']
    assert math.isclose(rt['native2_cpu_ratio'],native2_ratio,rel_tol=1e-12,abs_tol=0)
    assert boundary['gates']['cpu_agreement'] is True
    assert abs(100*(rt['boundary_mean_cpu_ratio']-native2_ratio)) <= 5
    assert residual['gate_s'] == .4 and residual['strict_upper_bound'] is True and residual['projected_s'] < .4
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
    runtime = dict(runtime_reference=reference,native8_ratio_inputs=runtime_inputs,
        native8_cpu_ratio_rule='sum(candidate TRAIN2+TRAIN6 cpu_s) / sum(pinned S32 TRAIN2+TRAIN6 cpu_s)',
        score_comparator=c['incremental_reference'],score_comparator_native8_cpu_ratio=component_ratio,
        native8_cpu_ratio=native8_ratio,native2_cpu_ratio=rt['native2_cpu_ratio'],
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
        stage='VALID40',candidate=c['candidate']['source'],parent=c['base']['source'],
        baseline_submission=c['score_reference']['submission_id'],
        baseline_adjusted=c['score_bank']['authenticated_poll']['score'],
        score_reference=c['score_reference'],runtime_reference=c['runtime_reference'],
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
