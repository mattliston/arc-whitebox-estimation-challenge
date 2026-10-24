"""Root-only final worker20/setup3/contract assessment; no new estimator jobs.

NATIVE -B assess.py CONFIG CONFIG_SHA VALIDATION_ADMISSION_SHA
Reuses the exact passed early boundary and actual-eight runtime recheck.
"""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.util
import io
import json
import math
import sys


def json_objects(text):
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != '{':
            continue
        try:
            value, _ = decoder.raw_decode(text[i:])
        except ValueError:
            continue
        if isinstance(value, dict):
            yield value


def setup(ctx, seconds, log):
    assert math.isfinite(seconds) and 0 <= seconds <= 4.1
    for label, count in ctx.c['setup_counts'].items():
        assert [row[1] for row in log if row[0] == label] == [count]


def assess(m, ctx):
    q, c, study = ctx.q, ctx.c, ctx.study
    candidate, prefix, tag = c['candidate'], ctx.PREFIX, c['release_tag']
    policy = json.loads(q.checked(c['policy']))
    assert policy['bank_gates'] == dict(adjusted_pct_at_most=-1, ci95_upper_below=0,
        wins_min_of40=28, max_per_network_regression_pct=10, serving_required=True)
    run_pin = q.pin(study/(prefix+'qualification_run.json'))
    run = json.loads(q.checked(run_pin))
    assert run['completed'] is True and run['all_guards_passed'] is True
    for key, expected in (('candidate', candidate), ('config', ctx.config_pin),
                         ('validation_admission', ctx.admission_pin), ('runner', c['runner']),
                         ('reused_boundary', ctx.broad['boundary'])):
        assert run[key] == expected
    assert run['assessment_not_run'] is True and run['release_admitted'] is False and run['submitted'] is False
    assert run['attempt'] == q.pin(study/(prefix+'qualification_attempt.json'))
    attempt = json.loads(q.checked(run['attempt']))
    for key in ('candidate', 'config', 'validation_admission', 'runner', 'reused_boundary'):
        assert attempt[key] == run[key]
    assert attempt['runtime_recheck'] == ctx.runtime
    assert attempt['serial'] is True and attempt['no_automatic_retry'] is True
    for record in attempt['inputs']:
        q.checked(record)
    jobs = m.schedule(ctx)
    assert attempt['jobs'] == [dict(name=n, seconds=s, command=cmd) for n, s, cmd in jobs]
    assert len(run['jobs']) == len(jobs) == 3
    guards = []
    for (name, seconds, command), record in zip(jobs, run['jobs']):
        assert record == q.pin(study/(prefix+name+'_job.json'))
        job = json.loads(q.checked(record))
        assert job['name'] == name and job['passed'] is True and job['failures'] == []
        assert job['guard_exit_code'] == 0 and job['error'] is None
        assert job['command'] == command and job['seconds'] == seconds
        for key in ('candidate', 'config', 'validation_admission'):
            assert job[key] == run[key]
        assert job['guard'] == q.pin(study/(prefix+name+'_guard.json'))
        assert job['log'] == q.pin(study/(prefix+name+'.log'))
        assert job['guarded_command'] == [str(ctx.LIGHT), c['guard_source']['path'],
            '--record', job['guard']['path'], '--seconds', str(seconds),
            '--rss-limit', '7500000000', *command]
        guards.append((job['guard'], m.guard(ctx, job['guard']['path'], seconds, command)))
        q.checked(job['log'])

    worker = ctx.H2/(tag+'_worker')
    manifest = q.read(worker/'manifest.json')
    launcher = q.read(worker/'launcher.json')
    worker_rows = [json.loads(line) for line in q.checked(q.pin(worker/'rows.jsonl')).decode().splitlines()]
    assert manifest['sha256'] == candidate['sha256'] and manifest['candidate'] == candidate['path']
    assert manifest['server_instrumented'] is True
    assert manifest['ranks'] == ctx.WORKER_RANKS == [row['rank'] for row in worker_rows]
    assert len(worker_rows) == len(set(manifest['ranks'])) == 20
    setup(ctx, manifest['setup_s'], manifest['pb_log'])
    caps = c['resource_gates']['index_caches']
    for row in worker_rows:
        assert row['finite_outputs'] == 16384 and type(row['flops']) is int and 0 < row['flops'] <= 2**41
        assert len(row['index_cache_entries']) == 3
        assert all(type(n) is int and 0 <= n <= cap for n, cap in zip(row['index_cache_entries'], caps))
    assert launcher['candidate_sha256'] == candidate['sha256']
    assert launcher['client_exit'] == 0 and launcher['server_retired'] is True
    handles = launcher['handles']
    assert handles['array_count_limit'] == 70000 and 0 <= handles['peak_handles'] < 70000
    assert handles['stores'] and all(store['clears'] == 0 for store in handles['stores'])
    assert q.read(worker/'server_handles.json') == handles

    # Source/guard/roster checks precede saved prediction decoding. Native evidence
    # comes from the same admitted eight-network receipt, never a proxy source.
    assert ctx.training['candidate'] == candidate and ctx.training['n'] == 8
    assert [block['size'] for block in ctx.training['blocks']] == [2, 6]
    native = ctx.training['blocks'][0]['arms']['candidate']
    nm = json.loads(q.checked(native['manifest']))
    nr = [json.loads(line) for line in q.checked(native['rows']).decode().splitlines()]
    assert nm['sha256'] == candidate['sha256'] and nm['candidate'] == candidate['path']
    assert nm['ranks'] == [35, 100] == [r['rank'] for r in nr]
    parity = []
    for index, rank in enumerate((35, 100)):
        meta = q.read(worker/f'pred_{rank}.json')
        assert meta['shape'] == [16, 1024] and q.np.dtype(meta['dtype']) == q.np.dtype('float32')
        rpc_pin = q.pin(worker/f'pred_{rank}.bin')
        prediction_pin = native['predictions'][str(rank)]
        assert prediction_pin['path'] == str(Path(native['manifest']['path']).parent/f'pred_{rank}.npy')
        rpc_bytes, native_bytes = q.checked(rpc_pin), q.checked(prediction_pin)
        assert len(rpc_bytes) == 16*1024*4
        rpc = q.np.frombuffer(rpc_bytes, dtype=q.np.float32).reshape(16, 1024)
        prediction = q.np.load(io.BytesIO(native_bytes), allow_pickle=False)
        assert prediction.shape == (16, 1024) and prediction.dtype == q.np.dtype('float64')
        assert q.np.isfinite(rpc).all() and q.np.isfinite(prediction).all()
        assert rpc.astype(q.np.float64).tobytes() == prediction.tobytes()
        assert worker_rows[index]['flops'] == nr[index]['flops']
        parity.append(dict(rank=rank, all16_promoted_bytes_exact=True,
            exact_flops=nr[index]['flops'], rpc=rpc_pin, native=prediction_pin))

    stress = q.read(ctx.H2/(tag+'_setup')/'summary.json')
    assert stress['candidate_sha256'] == candidate['sha256']
    assert len(stress['records']) == 3 and [r['repeat'] for r in stress['records']] == [0, 1, 2]
    for trial in stress['records']:
        assert trial['sha256'] == candidate['sha256'] and trial['candidate'] == candidate['path']
        assert trial['load_processes'] == 2
        original = q.read(ctx.H2/(tag+'_setup')/f"setup_{trial['repeat']}.json")
        assert {k:v for k,v in trial.items() if k not in ('repeat', 'load_processes', 'system_load')} == original
        setup(ctx, trial['setup_s'], trial['pb_log'])
    contract_log = q.checked(q.pin(study/(prefix+'contract.log'))).decode()
    contracts = [r for r in json_objects(contract_log) if r.get('ok') is True and
                 r.get('class_name') == 'Estimator' and r.get('output_shape') == [2, 4]]
    assert len(contracts) == 1, 'Full estimator contract missing or ambiguous'
    boundary = ctx.boundary
    setup(ctx, boundary['setup']['setup_wall_s'], boundary['setup']['setup_log'])
    assert boundary['passed'] is True and boundary['receipt_checks_passed'] is True
    assert boundary['candidate'] == candidate and all(boundary['gates'].values())
    assert boundary['residual']['projected_s'] < .4 and ctx.runtime['projected_worst_wall_s'] <= 105
    assert ctx.runtime['passed'] is True
    for record in list(q.EVIDENCE.values()):
        q.checked(record)
    metrics = ctx.broad['metrics']
    return dict(passed=True, receipt_checks_passed=True, status='QUALIFIED; not submitted',
        created_utc=datetime.now(timezone.utc).isoformat(), candidate=candidate,
        candidate_sha256=candidate['sha256'], parent=ctx.broad['parent'], policy=c['policy'],
        config=ctx.config_pin, validation_admission=ctx.admission_pin,
        broad_assessment=ctx.admission['broad_assessment'], native8=ctx.broad['training'],
        baseline_submission=ctx.broad['baseline_submission'], baseline_adjusted=ctx.broad['baseline_adjusted'],
        score_reference=c['score_reference'], runtime_reference=c['runtime_reference'],
        validation={key:metrics[key] for key in ('n', 'adjusted_pct', 'adjusted_ci95_pct',
            'adjusted_wins', 'raw_pct', 'raw_ci95_pct', 'mean_util_pct', 'mean_cpu_pct',
            'max_per_network_adjusted_regression_pct')},
        boundary=ctx.broad['boundary'], boundary_reused_without_rerun=True,
        runtime=ctx.runtime, residual_headroom=boundary['residual'],
        setup=dict(worker_s=manifest['setup_s'], boundary_s=boundary['setup']['setup_wall_s'],
            stress_trial_s=[r['setup_s'] for r in stress['records']], counts=c['setup_counts']),
        long_worker=dict(networks=20, measured_server_peak_handles=handles['peak_handles'],
            capacities=caps, enforced_handle_cap=70000, no_store_clear=True, prediction_parity=parity),
        full_estimator_contract=True,
        resources=dict(maximum_group_rss_bytes=max(g['max_rss_bytes'] for _, g in guards),
            maximum_swap_growth_bytes=0, guards=[p for p, _ in guards],
            earlier_native_boundary_validation_guards_verified=True),
        run=run_pin, assessor=c['assessor'], evidence=list(q.EVIDENCE.values()),
        worker_qualified=True, release_qualified=True, submitted=False)


def main():
    assert len(sys.argv) == 4
    path = Path(sys.argv[1]).resolve()
    raw = path.read_bytes(); assert hashlib.sha256(raw).hexdigest() == sys.argv[2]
    config = json.loads(raw)
    source = Path(__file__).resolve().parent/'common.py'
    assert config['common']['path'] == str(source)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == config['common']['sha256']
    spec = importlib.util.spec_from_file_location('serving_common', source)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    assessment = path.parent/(m.PREFIX+'qualification_assessment.json')
    fire = path.parent/'FIRE_CHECKS_v1.json'
    assert not assessment.exists() and not fire.exists(), 'Existing qualification; no overwrite'
    ctx = None
    try:
        ctx = m.load_context(path, sys.argv[2], sys.argv[3])
        result = assess(m, ctx)
    except Exception as error:
        result = dict(passed=False, receipt_checks_passed=False, candidate=config['candidate'],
            error=dict(type=type(error).__name__, message=str(error)),
            evidence=[] if ctx is None else list(ctx.q.EVIDENCE.values()),
            release_qualified=False, submitted=False)
    with assessment.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    if result['passed']:
        ctx.q.once(fire, result)
    print(json.dumps(dict(passed=result['passed'], fire_checks_written=result['passed'], error=result.get('error'))))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
