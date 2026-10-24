"""Package the qualified frozen star32 source; never submits or reads credentials."""
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
SHA = 'f226a45750e6c3851e50967ba96f1742bb7bf563d43c1adfc2988fd674041aef'
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
    config = read(HERE / 'config.json')
    gate = read(HERE / 'FIRE_CHECKS_v1.json')
    assert gate['passed'] is True
    assert gate['config'] == pin(HERE / 'config.json')
    assert gate['candidate'] == config['candidate'] == check_pin(config['candidate'])
    assert gate['candidate']['sha256'] == SHA
    assert gate['policy'] == config['policy'] == check_pin(config['policy'])
    assert gate['validation_admission'] == pin(HERE / 'validation_admission.json')
    check_tree(gate)
    policy = read(gate['policy']['path'])
    admission = read(gate['validation_admission']['path'])
    check_tree(admission)
    assert policy['minimum_adjusted_gain_pct'] == 1
    assert policy['submission_already_authorized'] is True
    assert admission['passed'] is True
    assert admission['candidate'] == gate['candidate'] and admission['policy'] == gate['policy']
    assert gate['baseline_submission'] == config['baseline_submission'] == 332517
    limits = config['validation_gates']; validation = gate['validation']
    assert limits['adjusted_pct_at_most'] == -1
    assert validation['n'] == 40 and validation['adjusted_pct'] <= -1
    assert validation['adjusted_wins'] >= limits['wins_min'] == 30
    assert validation['adjusted_ci95_pct'][1] < limits['ci95_upper_below'] == 0
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
        candidate=gate['candidate'], qualification=gate_pin, scripts=scripts))
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
        archive=pin(archive), estimator=pin(estimator), scripts=scripts,
        package_response=pin(HERE / 'package_response_v1.json'),
        validation_response=pin(HERE / 'validate_package_response_v1.json'), **contents)
    once(HERE / 'archive_receipt.json', record)
    print(json.dumps(dict(status='packaged and validated; not submitted', archive=record['archive'])), flush=True)


if __name__ == '__main__':
    main()
