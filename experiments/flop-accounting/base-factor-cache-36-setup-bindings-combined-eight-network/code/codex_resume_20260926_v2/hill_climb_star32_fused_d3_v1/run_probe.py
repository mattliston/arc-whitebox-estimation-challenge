"""Root-only serial overhead probe; no score, worker20, or submission action."""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
ROOT = P2.parents[3]
LIGHT = ROOT / '.venv/bin/python'
GUARD = P2 / 'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
NATIVE = H2 / 'hill_climb_star_projection32_train2_v1'
RELEASE = H2 / 'hill_climb_star32_release_v1'
TAG = HERE.name


def pin(path):
    path = Path(path).resolve(); data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def checked(record):
    data = Path(record['path']).read_bytes()
    assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256'], record['path']
    return data


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def guard_check(record, command, seconds):
    assert record['command'] == command
    assert record['completed'] and record['exit_code'] == 0 and record['error'] is None
    assert record['whole_group_retired'] and record['remaining_rows'] == []
    assert record['rss_trigger_bytes'] == 7_500_000_000
    assert 0 <= record['max_rss_bytes'] < 7_500_000_000
    assert record['max_swap_growth_bytes'] == 0
    assert 0 <= record['elapsed_s'] <= seconds


def jobs(candidate):
    return [
        ('worker', 180, [str(LIGHT), str(H2/'run_handle_worker_v3.py'), candidate, '35,100', TAG+'_worker']),
        ('boundary', 240, [str(LIGHT), str(H2/'run_boundary_census.py'), candidate, '35,100', TAG+'_boundary']),
        ('setup', 120, [str(LIGHT), str(H2/'setup_stress_driver_v1.py'), candidate, TAG+'_setup']),
    ]


def main():
    protocol_pin = pin(HERE/'protocol.json'); protocol = json.loads(checked(protocol_pin))
    candidate, parent = protocol['candidate'], protocol['parent']
    assert candidate['path'] == str(HERE/'candidate.py')
    assert checked(candidate).startswith(checked(parent))
    resolution_pin = pin(RELEASE/'residual_resolution_v1.json')
    resolution = json.loads(checked(resolution_pin))
    assert resolution['passed'] and resolution['candidate'] == parent
    baseline = resolution['fresh_candidate']
    source_names = ['run_handle_worker_v3.py', 'handle_server_v2.py', 'handle_client_v2.py',
                    'run_boundary_census.py', 'setup_stress_driver_v1.py', 'setup_only_client_v1.py']
    source_paths = [Path(__file__), HERE/'assess_probe.py', GUARD,
                    P2/'fable_porter_v1/census_boundary_setup_v2.py',
                    H2/'census_boundary_setup_snapshot_v1.py', *[H2/n for n in source_names]]
    assert source_paths[3].read_bytes() == source_paths[4].read_bytes()
    sources = [pin(path) for path in source_paths]
    native = {name: pin(NATIVE/name) for name in ('manifest.json', 'rows.jsonl', 'pred_35.npy', 'pred_100.npy')}
    manifest = json.loads(checked(native['manifest.json']))
    assert manifest['sha256'] == parent['sha256'] and manifest['candidate'] == parent['path']
    assert manifest['ranks'] == [35, 100]
    inputs = [protocol_pin, candidate, parent, resolution_pin, *baseline.values(), *native.values(), *sources]
    for record in inputs: checked(record)
    schedule = jobs(candidate['path'])
    outputs = [HERE/n for n in ('probe_attempt.json', 'probe_run.json', 'probe_assessment.json')]
    outputs += [H2/(TAG+'_worker'), H2/(TAG+'_setup')]
    outputs += [H2/(TAG+'_boundary'+suffix) for suffix in ('_census.json', '_launcher.json', '_server.log')]
    for name, _, _ in schedule:
        outputs += [HERE/(name+suffix) for suffix in ('.log', '_guard.json', '_job.json')]
    assert not any(path.exists() for path in outputs), 'Probe outputs already exist; no retry'
    once(HERE/'probe_attempt.json', dict(candidate=candidate, parent=parent, protocol=protocol_pin,
        baseline_resolution=resolution_pin, baseline=baseline, native=native, sources=sources,
        jobs=[dict(name=n, seconds=s, command=c) for n,s,c in schedule],
        serial=True, no_retry=True, score_claim=False, submission_action=False))
    receipts = []
    for name, seconds, command in schedule:
        for record in inputs: checked(record)
        log = HERE/(name+'.log'); guard = HERE/(name+'_guard.json')
        guarded = [str(LIGHT), str(GUARD), '--record', str(guard), '--seconds', str(seconds),
                   '--rss-limit', '7500000000', *command]
        error = None; returncode = None
        with log.open('x') as stream:
            process = subprocess.Popen(guarded, cwd=H2,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'), stdout=stream, stderr=subprocess.STDOUT)
            try:
                returncode = process.wait()
            except BaseException as exc:
                error = repr(exc)
                if process.poll() is None: process.send_signal(signal.SIGINT)
                returncode = process.wait()
        failures = []
        guard_pin = pin(guard) if guard.exists() else None
        try:
            assert error is None and returncode == 0, 'Guard process interrupted or failed'
            assert guard_pin is not None, 'Missing guard receipt'
            guard_check(json.loads(checked(guard_pin)), command, seconds)
            for record in inputs: checked(record)
        except Exception as exc:
            failures.append(repr(exc))
        receipt = HERE/(name+'_job.json')
        once(receipt, dict(name=name, passed=not failures, failures=failures, candidate=candidate,
             command=command, seconds=seconds, guard=guard_pin, log=pin(log),
             guard_exit_code=returncode, error=error))
        print(json.dumps(dict(job=name, passed=not failures, failures=failures)), flush=True)
        if failures: raise RuntimeError(name+' probe failed; retained partial receipts')
        receipts.append(pin(receipt))
    once(HERE/'probe_run.json', dict(completed=True, candidate=candidate, parent=parent,
         attempt=pin(HERE/'probe_attempt.json'), jobs=receipts, score_claim=False,
         assessment_run=False, submitted=False))


if __name__ == '__main__':
    main()
