"""Submit the fully qualified E1/young-trace/PSDB-depth5 archive once; no polling or automatic retry."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from prepare_submission import HERE, ROOT, WHest, check_pin, check_tree, inspect_archive, once, pin, read, verify_gate


def main():
    from whestbench.aicrowd_client import extract_submission_id
    gate, config = verify_gate()
    record = read(HERE / 'archive_receipt.json')
    assert record['passed'] is True and record['validation'] == 'passed'
    check_tree(record)
    for key in ('candidate', 'policy', 'config', 'validation_admission'):
        assert record[key] == gate[key]
    assert record['qualification'] == pin(HERE / 'FIRE_CHECKS_v1.json')
    assert record['scripts'] == [pin(HERE / name) for name in ('prepare_submission.py', 'submit_once.py')]
    archive = Path(record['archive']['path'])
    assert archive == HERE / 'submission.tar.gz'
    contents = inspect_archive(archive, gate['candidate'])
    assert contents['members'] == record['members'] and contents['manifest'] == record['manifest']
    assert not (HERE / 'upload_attempt_v1.json').exists(), 'Attempt exists; inspect retained response before any repeat.'
    assert not (HERE / 'submission_receipt_v1.json').exists()
    v = gate['validation']
    description = ('Phase 2: E1/GNN batching, young negative trace, and PSDB depth5; '
        f"matched VALID40 adjusted improvement {-v['adjusted_pct']:.4f}% vs S33, "
        f"{v['adjusted_wins']}/40 wins; actual 20-network worker and resource/contract gates passed.")
    command = [str(WHest), 'submit', str(archive), '--challenge',
        'arc-white-box-estimation-challenge-2026', '--description', description, '--yes', '--json']
    once(HERE / 'upload_attempt_v1.json', dict(status='attempt started',
        utc=datetime.now(timezone.utc).isoformat(), candidate=gate['candidate'], archive=record['archive'],
        archive_receipt=pin(HERE / 'archive_receipt.json'), qualification=record['qualification'],
        policy=gate['policy'], validation_admission=gate['validation_admission'],
        scripts=record['scripts'], command=command))
    # Only this child receives the credential; neither the key nor environment is logged.
    key = (ROOT / 'key.txt').read_text().strip()
    assert key, 'Missing credential; attempt remains recorded'
    env = dict(os.environ, AICROWD_API_KEY=key, PYTHONDONTWRITEBYTECODE='1')
    try:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=180)
            response = dict(returncode=result.returncode, stdout=result.stdout.replace(key, '[REDACTED]'),
                stderr=result.stderr.replace(key, '[REDACTED]'), timed_out=False)
        except subprocess.TimeoutExpired as error:
            def text(value):
                return (value.decode(errors='replace') if isinstance(value, bytes) else value or '').replace(key, '[REDACTED]')
            response = dict(returncode=None, stdout=text(error.stdout), stderr=text(error.stderr), timed_out=True)
    finally:
        env.pop('AICROWD_API_KEY', None)
        del key, env
    once(HERE / 'upload_response_v1.json', response)
    assert not response['timed_out'] and response['returncode'] == 0, 'Submission response retained; no automatic retry.'
    payload = json.loads(response['stdout'])
    assert isinstance(payload, dict), 'Unrecognized retained response; inspect before any repeat.'
    sid = extract_submission_id(payload)
    if sid is None:
        for value in payload.values():
            if isinstance(value, dict):
                sid = extract_submission_id(value)
                if sid is not None:
                    break
    assert sid is not None and int(sid) > 0, 'No accepted ID extracted; response retained, do not repeat.'
    receipt = dict(status='submitted; grading pending', submission_id=int(sid),
        url=f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{sid}',
        utc=datetime.now(timezone.utc).isoformat(), candidate=gate['candidate'], archive=record['archive'],
        baseline_submission=gate['baseline_submission'], qualification=record['qualification'],
        policy=gate['policy'], response=pin(HERE / 'upload_response_v1.json'),
        attempt=pin(HERE / 'upload_attempt_v1.json'))
    once(HERE / 'submission_receipt_v1.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
