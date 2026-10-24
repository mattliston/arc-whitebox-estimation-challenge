"""Root-only worker2/boundary2/setup3 after the separately completed native gate."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess

HERE = Path(__file__).resolve().parent
H2, P2 = HERE.parent, HERE.parent.parent
ROOT = P2.parents[3]
LIGHT = ROOT / '.venv/bin/python'
NATIVE_PY = P2 / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
GUARD = P2 / 'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
BASE_NATIVE = H2 / 'hill_climb_star_projection32_train2_v1'
RELEASE = H2 / 'hill_climb_star32_release_v1'
TAG = HERE.name
NATIVE_OUTPUT = H2 / 'hill_climb_exact_bundle_native2_v1'
NATIVE_DRIVER = H2 / 'hill_climb_uvp_base_bcache_probe_v1/native_probe.py'


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def checked(record):
    data = Path(record['path']).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
    return data


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def guard_check(record, command, seconds):
    assert record['command'] == command
    assert record['completed'] and record['exit_code'] == 0 and record['error'] is None
    assert record['whole_group_retired'] and record['remaining_rows'] == []
    assert record['rss_trigger_bytes'] == 7_500_000_000
    assert 0 <= record['max_rss_bytes'] < 7_500_000_000
    assert record['max_swap_growth_bytes'] == 0
    assert 0 <= record['elapsed_s'] <= seconds


def native_guard_check(record, candidate):
    command = record['command']
    guard_check(record, command, 150)
    args = [x for x in command if x != '-B']
    assert len(args) == 5 and args[3:] == ['35,100', NATIVE_OUTPUT.name]
    def absolute(value):
        path = Path(value)
        return (path if path.is_absolute() else ROOT/path).resolve()
    assert absolute(args[0]) == NATIVE_PY.resolve()
    assert absolute(args[1]) == NATIVE_DRIVER.resolve()
    assert absolute(args[2]) == Path(candidate['path']).resolve()


def bundle_inputs(build, check=checked):
    """Verify the actual UBC core plus two unchanged suffixes, without imports."""
    candidate, parent = build['candidate'], build['parent_s32']
    assert candidate['path'] == str(HERE/'candidate.py')
    assert candidate['sha256'] == '3651f482c3b4369228aab6741d9f40c3011b3ee155db78a51d247bfb6f215ba7'
    assert parent['sha256'] == 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
    records = [candidate, parent, build['base'], build['builder'], build['protocol'],
               *build['fragments'], *build['component_sources'], *build['component_receipts']]
    for record in records:
        check(record)
    assert all(build['inverse_exact'].values()) and build['no_duplicate_setup_keys']
    assert build['setup_keys'] == build['setup_fused_expected'] == 636
    assert build['setup_bound_expected'] == 660 and build['cache_caps'] == [320,352,320]
    assert build['uvp_copy_limit_bytes'] == 512000000
    source = {record['path']: record for record in build['component_sources']}
    d3, pb = [check(record) for record in build['fragments']]
    base, original = check(build['base']), check(parent)
    assert check(candidate) == base + d3 + pb
    assert check(source[str(H2/'hill_climb_star32_fused_d3_v1/candidate.py')]) == original + d3
    assert check(source[str(H2/'hill_climb_s32_prebind27_v1/candidate.py')]) == original + pb
    assert check(source[str(H2/'hill_climb_s32_prebind27_v1/fragment.py')]) == pb
    ubc = json.loads(check(source[str(H2/'hill_climb_uvp_base_bcache_v1/build.json')]))
    assert ubc['candidate'] == build['base'] and ubc['parent'] == parent and ubc['inverse_exact']
    records.extend([ubc['fragment'], ubc['build'], ubc['protocol']])
    fragment = check(ubc['fragment'])
    assert base.endswith(b'\n\n' + fragment)
    inverse = base[:-len(fragment)-2].decode()
    assert len(ubc['edits']) == 6
    for edit in reversed(ubc['edits']):
        assert inverse.count(edit['after']) == 1
        inverse = inverse.replace(edit['after'], edit['before'], 1)
    assert inverse.encode() == original
    for record in records:
        check(record)
    return records


def jobs(candidate):
    return [
        ('worker', 180, [str(LIGHT), str(H2/'run_handle_worker_v3.py'), candidate, '35,100', TAG+'_worker']),
        ('boundary', 240, [str(LIGHT), str(H2/'run_boundary_census.py'), candidate, '35,100', TAG+'_boundary']),
        ('setup', 120, [str(LIGHT), str(H2/'setup_stress_driver_v1.py'), candidate, TAG+'_setup']),
    ]


def main():
    build_pin = pin(HERE/'build.json')
    build = json.loads(checked(build_pin))
    candidate, parent = build['candidate'], build['parent_s32']
    composition_inputs = bundle_inputs(build)
    native_assessment_pin = pin(HERE/'native2_assessment.json')
    native_assessment = json.loads(checked(native_assessment_pin))
    assert native_assessment['passed'] and native_assessment['candidate'] == candidate
    assert all(native_assessment['checks'].values())
    admission_pin = pin(HERE/'native_admission.json')
    admission = json.loads(checked(admission_pin))
    assert admission['candidate'] == candidate and admission['parent'] == parent
    assert admission['tag'] == NATIVE_OUTPUT.name and admission['ranks'] == [35,100]
    assert admission['driver'] == pin(NATIVE_DRIVER)
    assert native_assessment['admission'] == admission_pin
    assert admission['setup_bound'] == 660 and admission['setup_fused'] == 636
    assert admission['cache_max_bytes'] == 512000000 and admission['cache_retired_each_predict']
    native_guard = pin(HERE/'native2_guard.json')
    native_guard_data = json.loads(checked(native_guard))
    native_guard_check(native_guard_data, candidate)
    assert native_assessment['guard'] == native_guard
    current_native = {name: pin(NATIVE_OUTPUT/name) for name in ('manifest.json', 'rows.jsonl', 'pred_35.npy', 'pred_100.npy')}
    assert native_assessment['evidence'] == list(current_native.values())
    assert native_assessment['comparison'] == pin(HERE/'native2_comparison.json')
    resolution_pin = pin(RELEASE/'residual_resolution_v1.json')
    resolution = json.loads(checked(resolution_pin))
    assert resolution['passed'] and resolution['candidate'] == parent
    baseline = resolution['fresh_candidate']
    names = ['native_eval.py', 'run_handle_worker_v3.py', 'handle_server_v2.py', 'handle_client_v2.py',
             'run_boundary_census.py', 'setup_stress_driver_v1.py', 'setup_only_client_v1.py']
    sources = [pin(path) for path in [Path(__file__), HERE/'assess_probe.py', GUARD,
        P2/'fable_porter_v1/census_boundary_setup_v2.py', H2/'census_boundary_setup_snapshot_v1.py',
        *[H2/n for n in names]]]
    assert checked(sources[3]) == checked(sources[4])
    parent_native = {name: pin(BASE_NATIVE/name) for name in ('manifest.json', 'rows.jsonl', 'pred_35.npy', 'pred_100.npy')}
    manifest = json.loads(checked(parent_native['manifest.json']))
    assert manifest['sha256'] == parent['sha256'] and manifest['candidate'] == parent['path']
    assert manifest['ranks'] == [35, 100]
    inputs = [build_pin, *composition_inputs, native_assessment_pin, admission_pin, native_guard,
              admission['driver'], native_assessment['comparison'], *current_native.values(),
              resolution_pin, *baseline.values(), *parent_native.values(), *sources]
    for record in inputs:
        checked(record)
    schedule = jobs(candidate['path'])
    outputs = [HERE/n for n in ('probe_attempt.json', 'probe_run.json', 'probe_assessment.json', 'full_assess.log')]
    outputs += [H2/(TAG+'_'+n) for n in ('worker', 'setup')]
    outputs += [H2/(TAG+'_boundary'+s) for s in ('_census.json', '_launcher.json', '_server.log')]
    for name, _, _ in schedule:
        outputs += [HERE/(name+s) for s in ('.log', '_guard.json', '_job.json')]
    assert not any(p.exists() for p in outputs), 'Probe output exists; no retry'
    once(HERE/'probe_attempt.json', dict(candidate=candidate, parent=parent, build=build_pin,
        baseline_resolution=resolution_pin, baseline=baseline, parent_native=parent_native,
        native_assessment=native_assessment_pin, native_admission=admission_pin,
        native_guard=native_guard, current_native=current_native,
        sources=sources, inputs=inputs, jobs=[dict(name=n, seconds=s, command=c) for n,s,c in schedule],
        serial=True, no_retry=True, score_claim=False, submission_action=False))
    receipts = []
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    for name, seconds, command in schedule:
        for record in inputs:
            checked(record)
        log, guard = HERE/(name+'.log'), HERE/(name+'_guard.json')
        guarded = [str(LIGHT), str(GUARD), '--record', str(guard), '--seconds', str(seconds),
                   '--rss-limit', '7500000000', *command]
        error = None
        with log.open('x') as stream:
            process = subprocess.Popen(guarded, cwd=H2, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait()
            except BaseException as exc:
                error = repr(exc)
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                returncode = process.wait()
        failures = []
        guard_pin = pin(guard) if guard.exists() else None
        try:
            assert error is None and returncode == 0 and guard_pin is not None
            guard_check(json.loads(checked(guard_pin)), command, seconds)
            for record in inputs:
                checked(record)
        except Exception as exc:
            failures.append(repr(exc))
        receipt = HERE/(name+'_job.json')
        once(receipt, dict(name=name, passed=not failures, failures=failures, candidate=candidate,
             command=command, seconds=seconds, guard=guard_pin, log=pin(log), guard_exit_code=returncode, error=error))
        print(json.dumps(dict(job=name, passed=not failures, failures=failures)), flush=True)
        if failures:
            raise RuntimeError(name+' probe failed; partial receipts retained')
        receipts.append(pin(receipt))
    once(HERE/'probe_run.json', dict(completed=True, candidate=candidate, parent=parent,
         attempt=pin(HERE/'probe_attempt.json'), jobs=receipts, native_admission=native_assessment_pin,
         score_claim=False, submitted=False))
    with (HERE/'full_assess.log').open('x') as stream:
        subprocess.run([str(LIGHT), str(HERE/'assess_probe.py'), 'full'], cwd=H2, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, check=True)


if __name__ == '__main__':
    main()
