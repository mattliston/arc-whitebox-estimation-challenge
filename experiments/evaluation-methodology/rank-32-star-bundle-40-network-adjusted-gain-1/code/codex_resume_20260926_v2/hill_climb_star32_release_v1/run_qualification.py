"""Run the existing release harnesses serially after root's validation admission.

This records execution and guard results only. The independent assessment
owns worker parity, setup, contract, runtime, and final release decisions.
"""
from pathlib import Path
import hashlib
import json
import os
import signal
import subprocess
import time


HERE = Path(__file__).resolve().parent
H2 = HERE.parent
P2 = H2.parent
ROOT = P2.parents[3]
LIGHT = ROOT / '.venv/bin/python'
WHEST = ROOT / '.venv/bin/whest'
GUARD = P2 / 'codex_resume_20260919_v1/learnable_joint_basis_training_qualification_v1/guard.py'
WORKER_RANKS = [35, 100, 37, 39, 42, 48, 59, 63, 66, 67, 71, 73,
                76, 80, 84, 88, 92, 96, 104, 108]


def pin(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def checked(record):
    raw = Path(record['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == record['sha256'], record['path']
    assert len(raw) == record['bytes'], record['path']
    return raw


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def guard_errors(record, command, seconds):
    checks = dict(
        exact_command=record.get('command') == command,
        completed=record.get('completed') is True,
        exit_zero=record.get('exit_code') == 0,
        no_error=record.get('error') is None,
        group_retired=record.get('whole_group_retired') is True,
        no_remaining_rows=record.get('remaining_rows') == [],
        exact_rss_limit=record.get('rss_trigger_bytes') == 7_500_000_000,
        rss_below_limit=0 <= record.get('max_rss_bytes', float('inf')) < 7_500_000_000,
        zero_added_swap=record.get('max_swap_growth_bytes') == 0,
        time_limit=0 <= record.get('elapsed_s', float('inf')) <= seconds,
    )
    return [name for name, passed in checks.items() if not passed]


def main():
    config_pin = pin(HERE / 'config.json')
    config = json.loads(checked(config_pin))
    admission_pin = pin(HERE / 'validation_admission.json')
    admission = json.loads(checked(admission_pin))
    assert admission['passed'] is True
    assert admission['candidate'] == config['candidate']
    assert admission['policy'] == config['policy']
    checked(config['candidate'])
    policy = json.loads(checked(config['policy']))
    assert policy['minimum_adjusted_gain_pct'] == 1
    assert policy['submission_already_authorized'] is True
    assert policy['other_validation_resource_and_contract_gates_unchanged'] is True
    assert config['validation_gates'] == dict(adjusted_pct_at_most=-1, wins_min=30,
                                             ci95_upper_below=0)
    assert config['resource_gates']['rss_bytes_below'] == 7_500_000_000
    assert config['resource_gates']['added_swap_bytes'] == 0
    assert config['worker_ranks'] == WORKER_RANKS
    tag = config['release_tag']
    assert tag == HERE.name and Path(tag).name == tag
    candidate = config['candidate']['path']
    jobs = [
        ('worker', 1050, [str(LIGHT), str(H2 / 'run_handle_worker_v3.py'), candidate,
                         ','.join(map(str, WORKER_RANKS)), tag + '_worker'],
         ['run_handle_worker_v3.py', 'handle_server_v2.py', 'handle_client_v2.py']),
        ('boundary', 240, [str(LIGHT), str(H2 / 'run_boundary_census.py'), candidate,
                          '35,100', tag + '_boundary'], ['run_boundary_census.py']),
        ('setup', 120, [str(LIGHT), str(H2 / 'setup_stress_driver_v1.py'), candidate,
                       tag + '_setup'], ['setup_stress_driver_v1.py', 'setup_only_client_v1.py']),
        ('contract', 120, [str(WHEST), 'validate', '--estimator', candidate,
                          '--class', 'Estimator', '--json'], []),
    ]
    boundary_source = P2 / 'fable_porter_v1/census_boundary_setup_v2.py'
    source_paths = [Path(__file__), GUARD, WHEST, boundary_source]
    for _, _, _, files in jobs:
        source_paths.extend(H2 / name for name in files)
    snapshot = H2 / 'census_boundary_setup_snapshot_v1.py'
    if snapshot.exists():
        assert snapshot.read_bytes() == boundary_source.read_bytes()
        source_paths.append(snapshot)
    sources = [pin(path) for path in dict.fromkeys(source_paths)]
    inputs = [config_pin, admission_pin, config['candidate'], config['policy'], *sources]
    outputs = [HERE / 'qualification_attempt.json', HERE / 'qualification_run.json',
               H2 / (tag + '_worker'), H2 / (tag + '_setup'),
               H2 / (tag + '_boundary_census.json'),
               H2 / (tag + '_boundary_server.log'), H2 / (tag + '_boundary_launcher.json')]
    for name, _, _, _ in jobs:
        outputs.extend(HERE / (name + suffix) for suffix in ('.log', '_guard.json', '_job.json'))
    assert not any(path.exists() for path in outputs), 'qualification outputs already exist; no retry'
    once(HERE / 'qualification_attempt.json', dict(
        candidate=config['candidate'], policy=config['policy'], config=config_pin,
        validation_admission=admission_pin, sources=sources,
        jobs=[dict(name=name, seconds=seconds, command=command)
              for name, seconds, command, _ in jobs],
        serial=True, no_automatic_retry=True, assessment_not_run=True))
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    completed = []
    for name, seconds, command, _ in jobs:
        # The admitted source, policy, and harnesses must still be the pinned bytes.
        for record in inputs:
            checked(record)
        log_path = HERE / (name + '.log')
        guard_path = HERE / (name + '_guard.json')
        guarded = [str(LIGHT), str(GUARD), '--record', str(guard_path),
                   '--seconds', str(seconds), '--rss-limit', '7500000000', *command]
        assert not guard_path.exists()
        started = time.monotonic()
        error = None
        returncode = None
        with log_path.open('x') as log:
            process = subprocess.Popen(guarded, cwd=H2, env=env, stdout=log,
                                       stderr=subprocess.STDOUT)
            try:
                returncode = process.wait()
            except BaseException as exc:
                error = repr(exc)
                # Let the existing guard perform its own process-group retirement.
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)
                returncode = process.wait()
        failures = []
        guard_pin = None
        if guard_path.exists():
            guard_pin = pin(guard_path)
            try:
                failures.extend(guard_errors(json.loads(checked(guard_pin)), command, seconds))
            except Exception as exc:
                failures.append('invalid_guard_receipt:' + repr(exc))
        else:
            failures.append('missing_guard_receipt')
        if returncode != 0:
            failures.append('guard_process_exit')
        if error is not None:
            failures.append('orchestrator_interrupted')
        try:
            for record in inputs:
                checked(record)
        except Exception as exc:
            failures.append('input_pin_changed:' + repr(exc))
        receipt = dict(name=name, passed=not failures, failures=failures,
                       command=command, guarded_command=guarded,
                       guard_exit_code=returncode, error=error,
                       orchestrator_elapsed_s=time.monotonic() - started,
                       candidate=config['candidate'], policy=config['policy'],
                       config=config_pin, validation_admission=admission_pin,
                       sources=sources, guard=guard_pin, log=pin(log_path))
        once(HERE / (name + '_job.json'), receipt)
        print(json.dumps(dict(job=name, passed=not failures, failures=failures)), flush=True)
        if failures:
            raise RuntimeError(f'{name} qualification job failed: {failures}')
        completed.append(pin(HERE / (name + '_job.json')))
    once(HERE / 'qualification_run.json', dict(
        completed=True, all_guards_passed=True, candidate=config['candidate'],
        policy=config['policy'], config=config_pin, validation_admission=admission_pin,
        attempt=pin(HERE / 'qualification_attempt.json'), jobs=completed,
        assessment_not_run=True, release_admitted=False, submitted=False))


if __name__ == '__main__':
    main()
