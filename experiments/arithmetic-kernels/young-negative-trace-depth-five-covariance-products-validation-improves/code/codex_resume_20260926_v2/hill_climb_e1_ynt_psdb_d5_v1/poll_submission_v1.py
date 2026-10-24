"""One authenticated status GET for this release's accepted ID; no retry/upload."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from prepare_submission import HERE, ROOT, check_tree, once, pin, read


CHILD = '''import json, os, sys
from whestbench import aicrowd_client as api
api.POLL_RETRY = api.RetryPolicy(max_attempts=1, base_delay=0, max_delay=0, deadline_s=60)
client = api.AIcrowdClient(api_key=os.environ['AICROWD_API_KEY'], timeout=60)
print(json.dumps(client.get_submission_status(int(sys.argv[1]))), flush=True)
'''


def summarize(payload, submission_id, url):
    data = payload.get('data', payload)
    assert isinstance(data, dict), 'Unrecognized retained status body'
    for name in ('id', 'submission_id'):
        if data.get(name) is not None:
            assert int(data[name]) == submission_id, 'Response ID mismatch'
    if data.get('challenge_id') is not None:
        assert int(data['challenge_id']) == 1174
    evaluation = data.get('evaluation') or {}
    assert isinstance(evaluation, dict)
    if evaluation.get('submission_id') is not None:
        assert int(evaluation['submission_id']) == submission_id
    progress = evaluation.get('progress') or {}
    public = ((evaluation.get('results') or {}).get('aggregates') or {}).get('public') or {}
    status = data.get('grading_status_cd') or evaluation.get('state') or 'unknown'
    score = data.get('score')
    if score is None:
        score = public.get('adjusted_final_layer_score', public.get('score'))
    return dict(submission_id=submission_id, status=status,
        completed=status in ('graded', 'completed'), failed=status in ('failed', 'error', 'errored'),
        score=score, completed_mlps=progress.get('mlps_completed'),
        failed_mlps=progress.get('mlps_failed', public.get('n_failed_mlps')), url=url)


def main():
    receipt_path = HERE / 'submission_receipt_v1.json'
    receipt = read(receipt_path)
    check_tree(receipt)
    submission_id = int(receipt['submission_id'])
    assert submission_id > 0
    url = f'https://www.aicrowd.com/challenges/arc-white-box-estimation-challenge-2026/submissions/{submission_id}'
    assert receipt['url'] == url
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ') + f'_{os.getpid()}'
    stem = f'eval_{submission_id}_{stamp}'
    python = ROOT / '.venv/bin/python'
    command = [str(python), '-B', '-c', CHILD, str(submission_id)]
    provenance = dict(submission_id=submission_id, url=url, receipt=pin(receipt_path),
        script=pin(__file__), client_source=pin(ROOT / '.venv/lib/python3.11/site-packages/whestbench/aicrowd_client.py'))
    once(HERE / f'{stem}_attempt.json', dict(**provenance,
        started_at_utc=datetime.now(timezone.utc).isoformat(), operation='one status GET', max_attempts=1))
    # Credential goes only in the authenticated child's environment, never its argv or receipts.
    key = (ROOT / 'key.txt').read_text().strip()
    assert key, 'Missing credential; status attempt remains recorded'
    env = dict(os.environ, AICROWD_API_KEY=key, PYTHONDONTWRITEBYTECODE='1')
    try:
        try:
            result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=75)
            response = dict(returncode=result.returncode, timed_out=False,
                stdout=result.stdout.replace(key, '[REDACTED]'), stderr=result.stderr.replace(key, '[REDACTED]'))
        except subprocess.TimeoutExpired as error:
            def clean(value):
                return (value.decode(errors='replace') if isinstance(value, bytes) else value or '').replace(key, '[REDACTED]')
            response = dict(returncode=None, timed_out=True, stdout=clean(error.stdout), stderr=clean(error.stderr))
    finally:
        env.pop('AICROWD_API_KEY', None)
        del env, key
    snapshot = dict(**provenance, observed_at_utc=datetime.now(timezone.utc).isoformat(),
        attempt=pin(HERE / f'{stem}_attempt.json'), response=response)
    summary = dict(submission_id=submission_id, status='fetch failed', completed=False,
        failed=None, score=None, url=url, fetch_passed=False)
    parse_error = None
    if response['returncode'] == 0 and not response['timed_out']:
        try:
            payload = json.loads(response['stdout'])
            assert isinstance(payload, dict), 'Unrecognized retained status response'
            snapshot['payload'] = payload
            summary = dict(**summarize(payload, submission_id, url), fetch_passed=True)
        except (ValueError, TypeError, AssertionError) as error:
            parse_error = dict(type=type(error).__name__, message=str(error))
            summary['status'] = 'unrecognized status response'
    if parse_error:
        snapshot['parse_error'] = parse_error
    once(HERE / f'{stem}.json', snapshot)
    summary.update(observed_at_utc=snapshot['observed_at_utc'], snapshot=pin(HERE / f'{stem}.json'))
    once(HERE / f'{stem}_summary.json', summary)
    print(json.dumps(summary), flush=True)
    if not summary['fetch_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
