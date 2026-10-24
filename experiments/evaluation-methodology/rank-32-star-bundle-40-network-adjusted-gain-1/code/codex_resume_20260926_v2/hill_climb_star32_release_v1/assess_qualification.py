"""Assess frozen release evidence only; no estimator, packaging, or API calls."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import json
import math
import statistics
import numpy as np

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
S31 = '5d43930820d144db00aac86ace9281fe0232240e321f53701cfdd8fe1ec98f26'
EVIDENCE = {}
GUARDS = []


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pin(path):
    path = Path(path).resolve(); raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    actual = pin(record['path'])
    require(all(actual[k] == record[k] for k in ('path', 'sha256', 'bytes') if k in record),
            'Changed pinned evidence: '+record['path'])
    EVIDENCE[actual['path']] = actual
    return Path(actual['path']).read_bytes()


def load(path):
    record = pin(path)
    return json.loads(checked(record))


def rows(path):
    return [json.loads(line) for line in checked(pin(path)).decode().splitlines()]


def guard(record):
    g = json.loads(checked(record))
    require(g['completed'] and g['exit_code'] == 0 and g['error'] is None and
            g['whole_group_retired'] and not g['remaining_rows'], 'Guard did not retire cleanly: '+record['path'])
    require(0 <= g['max_rss_bytes'] < 7_500_000_000 and g['max_swap_growth_bytes'] == 0,
            'Guard RSS/swap violation: '+record['path'])
    GUARDS.append((record, g))
    return g


def setup(seconds, log):
    require(math.isfinite(seconds) and 0 <= seconds <= 4.1, 'Setup exceeds 4.1 seconds')
    pairs = [row[:2] for row in log]
    require(['bound', 624] in pairs and ['lvfuse', 600] in pairs, 'Setup binding/fusion count mismatch')


def boundary_rows(path):
    data = load(path)
    require([r['rank'] for r in data] == [35, 100], 'Boundary cold/warm roster mismatch')
    for row in data:
        require(row['n_out'] == 16384 and row['nonfinite'] == 0, 'Boundary nonfinite/incomplete outputs')
        require(row['flops'] == row['flops_ctx'] and row['flops'] > 0, 'Boundary billing mismatch')
        require(row['requests'] > 0 and math.isfinite(row['server_cpu_s']) and row['server_cpu_s'] > 0,
                'Invalid boundary CPU/requests')
        require(math.isfinite(row['residual_s']) and row['residual_s'] >= 0, 'Invalid boundary residual')
    return data


def json_objects(text):
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character == '{':
            try:
                value, _ = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def assess():
    config = load(HERE/'config.json'); candidate = config['candidate']; checked(candidate)
    require(config['baseline_submission'] == 332517 and config['baseline_adjusted'] == 2.1736465056908115e-9,
            'Wrong official baseline')
    policy = json.loads(checked(config['policy']))
    require(policy['minimum_adjusted_gain_pct'] == 1 and policy['submission_already_authorized'],
            'Current 1% submission policy missing')
    admission = load(HERE/'validation_admission.json')
    require(admission['passed'] and admission['candidate'] == candidate and admission['policy'] == config['policy'],
            'Validation admission does not bind this candidate/policy')
    legacy = json.loads(checked(admission['legacy_validation_result']))
    require(legacy['candidate'] == candidate and legacy['comparison'] == admission['comparison'],
            'Legacy validation record does not bind the comparison/candidate')
    require(admission['comparison']['path'] == config['validation_comparison'], 'Wrong validation comparison')
    report = json.loads(checked(admission['comparison']))
    for item in admission['validation_guards']:
        guard(item)
    expected_guards = {str(Path(config['validation_study'])/(name+'_guard.json'))
                       for name in ('valid_b1','valid_b2','valid_b3','valid_b4','compare')}
    require({p['path'] for p in admission['validation_guards']} == expected_guards,
            'Incomplete or wrong validation guard evidence')
    require(report['n'] == 40 and len(set(report['ranks'])) == 40 and report['adjusted_pct'] <= -1 and
            report['adjusted_wins'] >= 30 and report['adjusted_ci95_pct'][1] < 0, 'VALID40 policy gate failed')
    baseline_order = []
    for i in range(1, 5):
        manifest = load(H2/f'both_tail_valid40_b{i}_v6/manifest.json')
        require(manifest['sha256'] == S31, 'Wrong saved S31 validation source')
        baseline_order += manifest['ranks']
    require(report['ranks'] == sorted(baseline_order), 'VALID40 roster mismatch')
    for family, expected_sha in (('candidate_sources', candidate['sha256']), ('base_sources', S31)):
        source_order = []
        require(len(report[family]) == 4, 'Missing comparison source batches')
        for manifest in report[family]:
            require(manifest['sha256'] == expected_sha and pin(manifest['candidate'])['sha256'] == expected_sha,
                    'Comparison source hash mismatch')
            require(manifest['numpy'] == '2.4.6' and all(v == '1' for v in manifest['thread_env'].values()),
                    'Comparison native runtime mismatch')
            checked(pin(manifest['candidate'])); source_order += manifest['ranks']
        require(source_order == baseline_order, 'Comparison process-order mismatch')

    run = load(HERE/'qualification_run.json')
    require(run['completed'] and run['all_guards_passed'] and run['candidate'] == candidate and
            run['policy'] == config['policy'] and run['config'] == pin(HERE/'config.json') and
            run['validation_admission'] == pin(HERE/'validation_admission.json'), 'Qualification controller did not pass')
    attempt = json.loads(checked(run['attempt']))
    require(attempt['candidate'] == candidate and attempt['config'] == run['config'] and
            attempt['validation_admission'] == run['validation_admission'], 'Qualification attempt mismatch')
    for record in attempt['sources']:
        checked(record)
    names = ('worker', 'boundary', 'setup', 'contract')
    require(len(run['jobs']) == 4 and [r['name'] for r in attempt['jobs']] == list(names), 'Qualification job order mismatch')
    for name, job_pin, scheduled in zip(names, run['jobs'], attempt['jobs']):
        require(job_pin == pin(HERE/(name+'_job.json')), 'Qualification job pin mismatch')
        job = json.loads(checked(job_pin))
        require(job['passed'] and not job['failures'] and job['guard_exit_code'] == 0 and job['error'] is None and
                job['candidate'] == candidate and job['validation_admission'] == run['validation_admission'] and
                job['command'] == scheduled['command'], 'Qualification job failed or changed: '+name)
        require(job['guard'] == pin(HERE/(name+'_guard.json')) and job['log'] == pin(HERE/(name+'.log')),
                'Qualification evidence pin mismatch: '+name)
        g = guard(job['guard'])
        require(g['command'] == job['command'] and g['elapsed_s'] <= scheduled['seconds'], 'Qualification command/time mismatch')
        require(candidate['path'] in g['command'], 'Qualification guard used a different source: '+name)
        checked(job['log'])
    tag = config['release_tag']; worker = H2/(tag+'_worker')
    manifest = load(worker/'manifest.json'); launcher = load(worker/'launcher.json')
    worker_rows = rows(worker/'rows.jsonl')
    require(manifest['sha256'] == candidate['sha256'] and manifest['candidate'] == candidate['path'],
            'Worker source mismatch')
    roster = config['worker_ranks']
    require(len(roster) == 20 and roster[:2] == [35,100] and manifest['ranks'] == roster and
            [r['rank'] for r in worker_rows] == roster, 'Worker20 roster/order mismatch')
    require(manifest['server_instrumented'], 'Worker must measure transient server handles')
    setup(manifest['setup_s'], manifest['pb_log'])
    for row in worker_rows:
        require(row['finite_outputs'] == 16384 and len(row['index_cache_entries']) == 3 and
                all(0 <= value <= cap for value, cap in zip(row['index_cache_entries'], (320,352,320))),
                'Worker output/cache contract failed')
    handles = launcher['handles']
    require(launcher['candidate_sha256'] == candidate['sha256'] and launcher['client_exit'] == 0 and
            launcher['server_retired'], 'Worker launcher failed')
    require(handles['array_count_limit'] == 70000 and handles['peak_handles'] < 70000 and
            handles['stores'] and all(s['clears'] == 0 for s in handles['stores']),
            'Worker handle limit or no-reset contract failed')
    native_dir = Path(config['native_train2']); native_manifest = load(native_dir/'manifest.json')
    native_rows = rows(native_dir/'rows.jsonl')
    require(native_manifest['sha256'] == candidate['sha256'] and native_manifest['ranks'] == [35,100] and
            [r['rank'] for r in native_rows] == [35,100], 'Native TRAIN2 source/order mismatch')
    parity = []
    for index, rank in enumerate((35,100)):
        meta = load(worker/f'pred_{rank}.json')
        require(meta['shape'] == [16,1024] and np.dtype(meta['dtype']) == np.dtype('float32'), 'RPC output format mismatch')
        rpc = np.frombuffer(checked(pin(worker/f'pred_{rank}.bin')), dtype=np.dtype(meta['dtype'])).reshape(16,1024)
        native = np.load(io.BytesIO(checked(pin(native_dir/f'pred_{rank}.npy'))), allow_pickle=False)
        require(native.shape == (16,1024) and native.dtype == np.dtype('float64') and
                np.isfinite(rpc).all() and np.isfinite(native).all(), 'Invalid parity arrays')
        require(rpc.astype(np.float64).tobytes() == native.tobytes(), 'RPC/native all-layer bytes differ')
        require(worker_rows[index]['flops'] == native_rows[index]['flops'], 'RPC/native billed FLOPs differ')
        parity.append(dict(rank=rank, all16_promoted_bytes_exact=True, exact_flops=native_rows[index]['flops']))

    current = boundary_rows(H2/(tag+'_boundary_census.json'))
    reference = boundary_rows(H2/'both_tail_boundary_v6_census.json')
    reference_launcher = load(H2/'both_tail_boundary_v6_launcher.json')
    require(reference_launcher['candidate_sha256'] == S31 and reference_launcher['client_exit'] == 0 and
            reference_launcher['server_retired'], 'Cached boundary reference is not retired S31')
    boundary_launcher = load(H2/(tag+'_boundary_launcher.json'))
    require(boundary_launcher['candidate_sha256'] == candidate['sha256'] and
            boundary_launcher['candidate'] == candidate['path'] and boundary_launcher['client_exit'] == 0 and
            boundary_launcher['server_retired'], 'Boundary launcher mismatch')
    setup_records = [r for r in json_objects((HERE/'boundary.log').read_text()) if 'setup_wall_s' in r]
    require(len(setup_records) == 1, 'Boundary setup record missing/ambiguous')
    setup(setup_records[0]['setup_wall_s'], setup_records[0]['setup_log'])
    comparisons = []
    for index, (actual, base) in enumerate(zip(current, reference)):
        require(actual['flops'] == native_rows[index]['flops'], 'Boundary/native exact FLOPs differ')
        delta = actual['residual_s']-base['residual_s']
        require(delta <= .010, 'Boundary residual regression exceeds 0.010 seconds')
        comparisons.append(dict(rank=actual['rank'], residual_delta_s=delta,
            request_delta=actual['requests']-base['requests'], server_cpu_ratio=actual['server_cpu_s']/base['server_cpu_s']))
    stress = load(H2/(tag+'_setup')/'summary.json')
    require(stress['candidate_sha256'] == candidate['sha256'] and len(stress['records']) == 3, 'Setup stress incomplete')
    require([r['repeat'] for r in stress['records']] == [0,1,2], 'Setup repeat order mismatch')
    for trial in stress['records']:
        require(trial['sha256'] == candidate['sha256'] and trial['candidate'] == candidate['path'] and
                trial['load_processes'] == 2, 'Setup stress source/load mismatch')
        setup(trial['setup_s'], trial['pb_log'])
    contracts = [r for r in json_objects((HERE/'contract.log').read_text()) if r.get('ok') is True and
                 r.get('class_name') == 'Estimator' and r.get('output_shape') == [2,4]]
    require(len(contracts) == 1, 'Contract validation JSON did not pass')

    du = report['mean_cpu_pct']/100
    cached_ratio = statistics.mean(r['server_cpu_ratio'] for r in comparisons)
    mismatch = abs(100*(cached_ratio-1)-report['mean_cpu_pct'])
    threshold = config['resource_gates']['native_boundary_cpu_disagreement_percentage_points_max']
    require(threshold == 5, 'Runtime disagreement policy changed')
    resolution = None; used_du = du; matched_ratio = None
    if mismatch > threshold:
        path = HERE/'runtime_resolution.json'
        require(path.exists(), 'Material CPU disagreement requires a pinned matched-control resolution')
        resolution = load(path)
        require(resolution['passed'] and resolution['candidate'] == candidate and resolution['reason'].strip(),
                'CPU resolution is not explicit and source-bound')
        require(resolution['current_boundary'] == pin(H2/(tag+'_boundary_census.json')), 'Resolution uses stale current boundary')
        checked(resolution['current_boundary']); control_guard = guard(resolution['matched_control_guard'])
        control_launcher = json.loads(checked(resolution['matched_control_launcher']))
        require(control_launcher['candidate_sha256'] == S31 and pin(control_launcher['candidate'])['sha256'] == S31 and
                control_launcher['client_exit'] == 0 and control_launcher['server_retired'], 'Matched control is not retired S31')
        checked(resolution['matched_control_census'])
        control_tag = Path(resolution['matched_control_census']['path']).name.removesuffix('_census.json')
        require(control_guard['command'][-3:] == [control_launcher['candidate'], '35,100', control_tag],
                'Matched-control guard does not bind the census source/roster/tag')
        control = boundary_rows(resolution['matched_control_census']['path'])
        for actual, base in zip(control, reference):
            require(actual['flops'] == base['flops'], 'Matched S31 control billing changed')
        matched_ratio = statistics.mean(a['server_cpu_s']/b['server_cpu_s'] for a,b in zip(current,control))
        require(math.isclose(resolution['computed_mean_server_cpu_ratio'], matched_ratio, rel_tol=1e-12),
                'CPU resolution ratio disagrees with pinned measurements')
        used_du = max(du, matched_ratio-1)
    official = load(H2/'eval_332517.json')['evaluation']
    require(official['progress']['mlps_completed'] == 100 and official['progress']['mlps_failed'] == 0,
            'Official S31 reference did not complete')
    worst = max(official['results']['per_mlp'], key=lambda r:r['telemetry']['participant_predict_wall_s'])['telemetry']
    wall = worst['participant_predict_wall_s']; kernel = worst['flopscope_server_kernel_wall_s']
    request_delta = statistics.mean(r['request_delta'] for r in comparisons)
    def projection(change):
        return sorted(wall+t*change*kernel+request_delta*o for t in (.60,.78,.95) for o in (.00030,.00041,.00050))
    native_band = projection(du); band = projection(used_du)
    require(all(math.isfinite(v) for v in band) and band[-1] <= 90, 'Conservative projected runtime exceeds 90 seconds')
    runtime = dict(official_s31_max_wall_s=wall, official_s31_kernel_at_max_s=kernel,
        native_cpu_ratio=1+du, cached_boundary_mean_cpu_ratio=cached_ratio,
        native_cached_disagreement_percentage_points=mismatch, materiality_percentage_points=threshold,
        matched_boundary_mean_cpu_ratio=matched_ratio, used_cpu_ratio=1+used_du,
        mean_boundary_request_delta=request_delta, native_projection_low_s=native_band[0], native_projection_high_s=native_band[-1],
        projected_worst_low_s=band[0], projected_worst_high_s=band[-1], gate_s=90,
        resolution=None if resolution is None else pin(HERE/'runtime_resolution.json'),
        note='Projection only; official runtime for this candidate is not measured.')
    for name in ('prepare_submission.py', 'submit_once.py'):
        checked(pin(HERE/name))
    for record in list(EVIDENCE.values()):
        checked(record)
    return dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(), status='QUALIFIED; not submitted',
        candidate=candidate, candidate_sha256=candidate['sha256'], policy=config['policy'], config=pin(HERE/'config.json'),
        validation_admission=pin(HERE/'validation_admission.json'), baseline_submission=332517,
        baseline_adjusted=config['baseline_adjusted'],
        validation={k:report[k] for k in ('n','adjusted_pct','adjusted_ci95_pct','adjusted_wins','raw_pct','raw_ci95_pct','mean_util_pct','mean_cpu_pct')},
        resources=dict(maximum_group_rss_bytes=max(g['max_rss_bytes'] for _,g in GUARDS), maximum_swap_growth_bytes=0,
                       guards=[p for p,_ in GUARDS]),
        setup=dict(boundary_s=setup_records[0]['setup_wall_s'], stress_trial_s=[r['setup_s'] for r in stress['records']], bound_entries=624, fused_entries=600),
        long_worker=dict(networks=20, measured_server_peak_handles=handles['peak_handles'], capacities=[320,352,320],
                         enforced_handle_cap=70000, no_store_clear=True, prediction_parity=parity),
        boundary=comparisons, runtime=runtime, standing_authorization=policy['user_answer'],
        evidence=list(EVIDENCE.values()), assessor=pin(__file__))


def main():
    require(not (HERE/'FIRE_CHECKS_v1.json').exists(), 'FIRE_CHECKS already exists')
    require(not (HERE/'qualification_assessment.json').exists(), 'Assessment already exists')
    try:
        result = assess()
    except Exception as error:
        result = dict(passed=False, failure_type=type(error).__name__, reason=str(error),
                      evidence=list(EVIDENCE.values()), assessor=pin(__file__))
    with (HERE/'qualification_assessment.json').open('x') as output:
        json.dump(result, output, indent=2, allow_nan=False); output.write('\n')
    if result['passed']:
        with (HERE/'FIRE_CHECKS_v1.json').open('x') as output:
            json.dump(result, output, indent=2, allow_nan=False); output.write('\n')
    print(json.dumps(dict(passed=result['passed'], reason=result.get('reason'),
                         fire_checks_written=result['passed'])))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
