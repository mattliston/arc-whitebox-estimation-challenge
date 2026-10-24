"""Package the qualified frozen E1/young-trace/PSDB-depth5 source; never submits or reads credentials."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
ROOT = H2.parents[4]
SHA = '42e50cdc80f49a562428f018afb3c20ae4bf42f4b48b1f63025bb384decd08eb'
POLICY_SHA = '26e99f9a54d97a4d6fca11ac477056219f7acf351d5f7327df9b6ff7cf898fa6'
AUTH_SHA = 'd069a04ed856331da19099e1ed86a3aa56bca81b56e8d2d09278fc7a6af793e7'
CONFIG_SHA = '822b6501d8e426e4cccfd77a83058f9ec8a19aedf95a5a50dff2b321a46d5d63'
NATIVE = H2.parent / 'astra_phase2_reexamination_v1/openblas_runtime_env_v1/bin/python'
WHest = ROOT / '.venv/bin/whest'


def read(path):
    return json.loads(Path(path).read_text())


def pin(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


def check_pin(record):
    actual = pin(record['path'])
    assert actual['sha256'] == record['sha256'], record['path']
    assert 'bytes' not in record or actual['bytes'] == record['bytes'], record['path']
    return actual


def check_tree(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            check_pin(value)
        for item in value.values():
            check_tree(item)
    elif isinstance(value, list):
        for item in value:
            check_tree(item)


def once(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def verify_gate():
    config_pin = pin(HERE / 'serving_config.json')
    assert config_pin['sha256'] == CONFIG_SHA
    config = read(config_pin['path'])
    gate = read(HERE / 'FIRE_CHECKS_v1.json')
    assert gate['passed'] is True and gate['receipt_checks_passed'] is True
    assert gate['worker_qualified'] is True and gate['release_qualified'] is True
    assert gate['submitted'] is False and config['prospective_authorization'] is True
    assert gate['config'] == config_pin
    assert gate['candidate'] == config['candidate'] == check_pin(config['candidate'])
    assert gate['candidate']['sha256'] == SHA and gate['candidate']['bytes'] == 962929
    assert gate['policy'] == config['policy'] == check_pin(config['policy'])
    assert gate['policy']['sha256'] == POLICY_SHA
    assert gate['assessor'] == config['assessor']
    assert gate['validation_admission'] == pin(HERE / 'validation_admission.json')
    check_tree(config); check_tree(gate)
    assessment = read(HERE / 'serving_official_limit_v1_qualification_assessment.json')
    assert assessment == gate
    guard = read(HERE / 'serving_official_limit_v1_assess_guard.json')
    assert guard['command'] == [str(NATIVE), '-B', config['assessor']['path'],
        config_pin['path'], CONFIG_SHA, gate['validation_admission']['sha256']]
    assert guard['completed'] is True and guard['exit_code'] == 0 and guard['error'] is None
    assert guard['whole_group_retired'] is True and guard['remaining_rows'] == []
    assert 0 < guard['elapsed_s'] <= 120 and guard['samples'] > 0
    assert 0 < guard['max_rss_bytes'] < guard['rss_trigger_bytes'] == 7500000000
    assert guard['max_swap_growth_bytes'] == 0
    policy = read(gate['policy']['path'])
    assert policy['bank_gates'] == dict(adjusted_pct_at_most=-1, ci95_upper_below=0,
        wins_min_of40=28, max_per_network_regression_pct=10, serving_required=True)
    authorization = pin(H2 / 'submission_policy_20260928_v1.json')
    assert authorization['sha256'] == AUTH_SHA
    auth = read(authorization['path'])
    assert auth['minimum_adjusted_gain_pct'] == 1 and auth['submission_already_authorized'] is True
    admission = read(gate['validation_admission']['path']); check_tree(admission)
    assert admission['passed'] is True and admission['candidate'] == gate['candidate']
    assert admission['config'] == config_pin and admission['policy'] == gate['policy']
    assert admission['broad_assessment'] == gate['broad_assessment']
    assert gate['baseline_submission'] == 332838
    validation = gate['validation']
    assert validation['n'] == 40 and validation['adjusted_pct'] <= -1
    assert validation['adjusted_wins'] >= 28 and validation['adjusted_ci95_pct'][1] < 0
    assert validation['max_per_network_adjusted_regression_pct'] <= 10
    assert gate['full_estimator_contract'] is True and gate['long_worker']['networks'] == 20
    assert gate['setup']['counts'] == dict(bound=682, lvfuse=649, psdb_fused=9)
    assert gate['runtime']['passed'] is True and gate['runtime']['projected_worst_wall_s'] <= 105
    assert gate['residual_headroom']['projected_s'] < .4
    assert gate['original_boundary_passed'] is False and gate['boundary_failure_preserved'] is True
    assert gate['final_residual_eligibility'] == dict(projected_s=gate['residual_headroom']['projected_s'],
        official_limit_s=.4, strict_below_official_limit=True, original_cutoff_s=.35)
    assert admission['official_limit_qualification_authorized'] is True
    assert admission['extension_policy'] == gate['research_extension']['policy']
    return gate, config

def inspect_archive(path, candidate):
    from whestbench.validation import validate_package
    assert validate_package(path).ok, 'Archive validation failed'
    members = []
    with tarfile.open(path, 'r:gz') as archive:
        entries = archive.getmembers()
        assert sorted(x.name for x in entries) == ['estimator.py', 'manifest.json']
        assert all(x.isfile() for x in entries)
        for entry in entries:
            assert entry.size <= 2_000_000
            data = archive.extractfile(entry).read()
            members.append(dict(name=entry.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
            if entry.name == 'estimator.py':
                assert len(data) == candidate['bytes'] and members[-1]['sha256'] == candidate['sha256']
            else:
                manifest = json.loads(data)
    assert manifest['entrypoint'] == dict(module='estimator', **{'class': 'Estimator'})
    assert manifest['files'] == [dict(name='estimator.py', sha256=candidate['sha256'])]
    return dict(members=sorted(members, key=lambda x: x['name']), manifest=manifest)


def main():
    gate, _ = verify_gate()
    assert not (HERE / 'archive_receipt.json').exists()
    assert not (HERE / 'upload_attempt_v1.json').exists()
    scripts = [pin(HERE / name) for name in ('prepare_submission.py', 'submit_once.py')]
    gate_pin = pin(HERE / 'FIRE_CHECKS_v1.json')
    once(HERE / 'package_attempt_v1.json', dict(utc=datetime.now(timezone.utc).isoformat(),
        candidate=gate['candidate'], qualification=gate_pin, scripts=scripts,
        qualification_guard=pin(HERE / 'serving_official_limit_v1_assess_guard.json'),
        authorization=pin(H2 / 'submission_policy_20260928_v1.json')))
    payload = HERE / 'payload'; payload.mkdir()
    source = Path(gate['candidate']['path'])
    estimator = payload / 'estimator.py'
    with estimator.open('xb') as stream:
        stream.write(source.read_bytes())
    assert pin(estimator)['sha256'] == SHA
    assert sorted(x.name for x in payload.iterdir()) == ['estimator.py']
    archive = HERE / 'submission.tar.gz'; assert not archive.exists()
    commands = [
        ('package_response_v1.json', [str(WHest), 'package', '--estimator', str(estimator),
          '--class', 'Estimator', '--output', str(archive), '--yes', '--json']),
        ('validate_package_response_v1.json', [str(WHest), 'validate-package', str(archive), '--json'])]
    for name, command in commands:
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
            capture_output=True, text=True, timeout=120)
        once(HERE / name, dict(command=command, returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr))
        assert result.returncode == 0, 'Packaging/validation failed; retained ' + name
    contents = inspect_archive(archive, gate['candidate'])
    verify_gate(); assert pin(HERE / 'FIRE_CHECKS_v1.json') == gate_pin
    for script in scripts:
        check_pin(script)
    record = dict(passed=True, validation='passed', utc=datetime.now(timezone.utc).isoformat(),
        candidate=gate['candidate'], policy=gate['policy'], config=gate['config'],
        validation_admission=gate['validation_admission'], qualification=gate_pin,
        qualification_guard=pin(HERE / 'serving_official_limit_v1_assess_guard.json'),
        authorization=pin(H2 / 'submission_policy_20260928_v1.json'),
        archive=pin(archive), estimator=pin(estimator), scripts=scripts,
        package_response=pin(HERE / 'package_response_v1.json'),
        validation_response=pin(HERE / 'validate_package_response_v1.json'), **contents)
    once(HERE / 'archive_receipt.json', record)
    print(json.dumps(dict(status='packaged and validated; not submitted', archive=record['archive'])), flush=True)


if __name__ == '__main__':
    main()
